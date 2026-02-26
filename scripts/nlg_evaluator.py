"""
================================================================================
nlg_evaluator.py — NLG Evaluation Suite for Vision-Language Model Outputs
================================================================================

Description:
    Evaluates the text output of a generative model against ground-truth
    reference answers using a comprehensive set of NLG metrics:

        Lexical  : BLEU-4, ROUGE-2, METEOR
        Semantic : BERTScore, LongCLIP Cosine, T5 Cosine, SBERT Cosine

    The script aligns predictions and references by matching the first image
    path in each sample's "images" field, then extracts assistant-role text
    from the "messages" field for scoring.

Command-line Arguments:
    --model_output  PATH    Path to the model predictions JSONL file.
    --reference     PATH    Path to the ground-truth reference JSONL file.
    --longclip_ckpt PATH    (Optional) Path to the LongCLIP checkpoint file.
                            Defaults to: ./checkpoints/longclip-L.pt
    --output_json   PATH    (Optional) Save results to a JSON file.

Expected JSONL Format (each line is a JSON object):
    {
        "images": ["path/to/image.jpg"],
        "messages": [
            {"role": "user",      "content": "What disease is shown?"},
            {"role": "assistant", "content": "The image shows powdery mildew."}
        ]
    }

    Notes:
    - The "images" field is used as the alignment key between the two files.
      Both files must use the same image paths.
    - The "content" field of assistant messages can be either a plain string
      or a list of dicts with {"type": "text", "text": "..."} structure
      (standard multi-modal message format).
    - Samples with empty predictions are excluded from embedding metrics but
      are counted in the total sample count.

Example Usage:
    python nlg_evaluator.py \\
        --model_output ./outputs/model_predictions.jsonl \\
        --reference    ./data/test.jsonl

    python nlg_evaluator.py \\
        --model_output ./outputs/model_predictions.jsonl \\
        --reference    ./data/test.jsonl \\
        --longclip_ckpt ./checkpoints/longclip-L.pt \\
        --output_json  ./results/scores.json

Dependencies:
    pip install nltk rouge-score bert-score sentence-transformers transformers torch scipy scikit-learn tqdm

    LongCLIP requires a custom installation. See: https://github.com/beichenzbc/Long-CLIP
================================================================================
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import warnings

import nltk
import numpy as np
import torch
import transformers
from sklearn.metrics.pairwise import paired_cosine_distances
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Silence noisy third-party loggers
# ---------------------------------------------------------------------------
transformers.logging.set_verbosity_error()
logging.getLogger("bert_score").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------
for _res, _kind in [("punkt", "tokenizers"), ("punkt_tab", "tokenizers"),
                    ("wordnet", "corpora"), ("omw-1.4", "corpora")]:
    try:
        nltk.data.find(f"{_kind}/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rouge_lib

# ---------------------------------------------------------------------------
# Performance knobs
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_WORKERS = 8
USE_AMP = True

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    from model import longclip
    LONGCLIP_AVAILABLE = True
except ImportError:
    LONGCLIP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from bert_score import score as bert_eval
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from transformers import T5EncoderModel, T5Tokenizer
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False

# Module-level ROUGE scorer (shared across worker processes)
_rouge_scorer = rouge_lib.RougeScorer(["rouge2"], use_stemmer=True)


# ===========================================================================
# Text utilities
# ===========================================================================

def normalize(text: str) -> str:
    """Lowercase and strip punctuation."""
    if not text:
        return ""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def simple_tokenize(text: str):
    """NLTK word tokeniser with a plain-split fallback."""
    if not text:
        return []
    try:
        return nltk.word_tokenize(text)
    except Exception:
        return text.split()


def extract_text(content) -> str:
    """
    Extract plain text from an assistant message content field.
    Handles both a raw string and the multi-modal list-of-dicts format.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return ""


def get_image_key(record: dict):
    """Return the first image path as the alignment key, or None."""
    images = record.get("images", [])
    if isinstance(images, list) and images:
        return images[0].replace("\\", "/").strip()
    return None


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Paired cosine similarity between two equal-length embedding arrays."""
    if emb1.shape != emb2.shape:
        raise ValueError(f"Shape mismatch: {emb1.shape} vs {emb2.shape}")
    distances = np.clip(paired_cosine_distances(emb1, emb2), 0.0, 2.0)
    return 1.0 - distances


# ===========================================================================
# PyTorch Dataset helper
# ===========================================================================

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text if text.strip() else " "


# ===========================================================================
# Encoder classes
# ===========================================================================

class LongCLIPEncoder:
    """Text encoder backed by the LongCLIP model (handles up to ~248 tokens)."""

    MAX_TOKENS = 248

    def __init__(self, checkpoint_path: str, device: str = None):
        if not LONGCLIP_AVAILABLE:
            raise ImportError("LongCLIP module not found. See installation notes.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = longclip.load(checkpoint_path, device=self.device)
        self.model.eval()

    def _truncate(self, text: str) -> str:
        if not text:
            return ""
        words = text.split()
        limit = int(self.MAX_TOKENS * 0.7)
        return " ".join(words[:limit]) if len(words) > limit else text

    def encode(self, texts) -> np.ndarray:
        embeddings = []
        loader = DataLoader(
            TextDataset(texts),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            collate_fn=lambda b: [self._truncate(t) for t in b],
            pin_memory=True,
        )
        with torch.inference_mode():
            for batch in loader:
                try:
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        tokens = longclip.tokenize(batch).to(self.device)
                        feats = self.model.encode_text(tokens)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                except RuntimeError:
                    shorter = [self._truncate(t)[:77] for t in batch]
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        tokens = longclip.tokenize(shorter).to(self.device)
                        feats = self.model.encode_text(tokens)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                embeddings.append(feats.cpu().float().numpy())
        return np.vstack(embeddings) if embeddings else np.array([])


class T5Encoder:
    """Mean-pooled T5 encoder for semantic similarity."""

    def __init__(self, model_name: str = "t5-base", device: str = None):
        if not T5_AVAILABLE:
            raise ImportError("transformers T5 not available.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts) -> np.ndarray:
        embeddings = []

        def collate_fn(batch):
            return self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )

        loader = DataLoader(
            TextDataset(texts),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        with torch.inference_mode():
            for inputs in loader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    hidden = self.model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                embeddings.append(pooled.cpu().float().numpy())
        return np.vstack(embeddings) if embeddings else np.array([])


class SBERTEncoder:
    """Sentence-BERT encoder for semantic similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers not available.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


# ===========================================================================
# Row-level metric worker  (runs in a separate process)
# ===========================================================================

def _calc_row_metrics(packet: dict) -> dict:
    """Compute METEOR and ROUGE-2 for a single reference/prediction pair."""
    ref, pred = packet["ref"], packet["pred"]
    rt = simple_tokenize(ref)
    pt = simple_tokenize(pred)
    result = {"METEOR": 0.0, "ROUGE-2": 0.0}

    if not rt:
        return result

    try:
        result["METEOR"] = meteor_score([rt], pt)
    except Exception:
        pass

    try:
        result["ROUGE-2"] = _rouge_scorer.score(ref, pred)["rouge2"].fmeasure
    except Exception:
        pass

    return result


# ===========================================================================
# Main evaluator
# ===========================================================================

class NLGEvaluator:
    """
    Loads a model-output JSONL and a reference JSONL, aligns them by image key,
    and computes BLEU-4, ROUGE-2, METEOR, BERTScore, LongCLIP, T5, and SBERT scores.
    """

    def __init__(self, longclip_ckpt: str = "./checkpoints/longclip-L.pt"):
        self.longclip_ckpt = longclip_ckpt
        self.encoders: dict = {}
        self._encoders_ready = False

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: str) -> list:
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _align(refs: list, preds: list) -> list:
        """
        Match reference and prediction records by their first image path.
        Returns a list of {"ref": str, "pred": str} dicts.
        """
        ref_map = {get_image_key(r): r for r in refs if get_image_key(r)}
        pred_map = {get_image_key(p): p for p in preds if get_image_key(p)}

        aligned = []
        for key in set(ref_map) & set(pred_map):
            r_msgs = ref_map[key].get("messages", [])
            p_msgs = pred_map[key].get("messages", [])
            for r_msg, p_msg in zip(r_msgs, p_msgs):
                if r_msg.get("role") == "assistant" and p_msg.get("role") == "assistant":
                    r_text = extract_text(r_msg.get("content"))
                    p_text = extract_text(p_msg.get("content"))
                    if r_text:
                        aligned.append({"ref": r_text, "pred": p_text})
        return aligned

    # ------------------------------------------------------------------
    # Encoder initialisation (lazy, on first embedding call)
    # ------------------------------------------------------------------

    def _init_encoders(self):
        if self._encoders_ready:
            return
        print("\nInitialising embedding encoders …")
        torch.cuda.empty_cache()

        if LONGCLIP_AVAILABLE and os.path.exists(self.longclip_ckpt):
            try:
                self.encoders["LongCLIP"] = LongCLIPEncoder(self.longclip_ckpt)
                print("  ✓ LongCLIP")
            except Exception as exc:
                print(f"  ✗ LongCLIP — {exc}")

        if T5_AVAILABLE:
            try:
                self.encoders["T5"] = T5Encoder()
                print("  ✓ T5")
            except Exception as exc:
                print(f"  ✗ T5 — {exc}")

        if SBERT_AVAILABLE:
            try:
                self.encoders["SBERT"] = SBERTEncoder()
                print("  ✓ SBERT")
            except Exception as exc:
                print(f"  ✗ SBERT — {exc}")

        self._encoders_ready = True
        print()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model_output_path: str, reference_path: str) -> dict:
        """
        Run the full evaluation pipeline.

        Parameters
        ----------
        model_output_path : str
            Path to the model predictions JSONL file.
        reference_path : str
            Path to the ground-truth reference JSONL file.

        Returns
        -------
        dict with keys: num_samples, num_valid, scores
        """
        print(f"\nLoading files …")
        print(f"  Reference    : {reference_path}")
        print(f"  Model output : {model_output_path}")

        refs_data  = self._load_jsonl(reference_path)
        preds_data = self._load_jsonl(model_output_path)

        aligned = self._align(refs_data, preds_data)
        if not aligned:
            raise ValueError(
                "No aligned samples found. Ensure both files share the same image paths."
            )

        ref_list  = [x["ref"]  for x in aligned]
        pred_list = [x["pred"] for x in aligned]

        valid_idx   = [i for i, t in enumerate(pred_list) if t.strip()]
        valid_refs  = [ref_list[i]  for i in valid_idx]
        valid_preds = [pred_list[i] for i in valid_idx]

        print(f"\n  Total aligned samples : {len(aligned)}")
        print(f"  Non-empty predictions : {len(valid_idx)}")

        scores = {}

        # ── Lexical metrics (parallelised) ────────────────────────────
        with tqdm(total=3, desc="Lexical metrics", ncols=80) as pbar:

            pbar.set_postfix_str("METEOR + ROUGE-2")
            n_cores = max(1, min((os.cpu_count() or 1) - 2, 8))
            with multiprocessing.Pool(n_cores) as pool:
                row_results = list(pool.imap(_calc_row_metrics, aligned))
            scores["METEOR"]  = float(np.mean([r["METEOR"]  for r in row_results]))
            scores["ROUGE-2"] = float(np.mean([r["ROUGE-2"] for r in row_results]))
            pbar.update(2)

            pbar.set_postfix_str("BLEU-4")
            ref_tok  = [[simple_tokenize(r)] for r in ref_list]
            pred_tok = [simple_tokenize(p)   for p in pred_list]
            scores["BLEU-4"] = float(corpus_bleu(ref_tok, pred_tok))
            pbar.update(1)

        # ── Embedding-based metrics ───────────────────────────────────
        use_embeddings = any([
            BERTSCORE_AVAILABLE, LONGCLIP_AVAILABLE,
            T5_AVAILABLE, SBERT_AVAILABLE
        ])

        if use_embeddings and valid_idx:
            self._init_encoders()

            # BERTScore
            if BERTSCORE_AVAILABLE:
                print("  Computing BERTScore …")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                collected = []
                for i in range(0, len(valid_preds), BATCH_SIZE):
                    bp = valid_preds[i: i + BATCH_SIZE]
                    br = valid_refs[i: i + BATCH_SIZE]
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        _, _, f1 = bert_eval(
                            bp, br, lang="en", verbose=False,
                            device=device, batch_size=len(bp)
                        )
                    collected.extend(f1.cpu().numpy().tolist())
                full = np.zeros(len(pred_list), dtype=np.float32)
                full[valid_idx] = collected
                scores["BERTScore"] = float(np.mean(full))
            else:
                scores["BERTScore"] = 0.0

            # Cosine-similarity encoders
            for name, encoder in self.encoders.items():
                print(f"  Computing {name} cosine similarity …")
                full = np.zeros(len(pred_list), dtype=np.float32)
                try:
                    emb_r = encoder.encode(valid_refs)
                    emb_p = encoder.encode(valid_preds)
                    if emb_r.size and emb_p.size:
                        full[valid_idx] = compute_cosine_similarity(emb_p, emb_r)
                except Exception as exc:
                    print(f"    ✗ {name} encoding failed: {exc}")
                scores[f"{name} Cosine"] = float(np.mean(full))

            torch.cuda.empty_cache()
        else:
            scores["BERTScore"] = 0.0
            for name in ("LongCLIP", "T5", "SBERT"):
                scores[f"{name} Cosine"] = 0.0

        return {
            "num_samples": len(aligned),
            "num_valid":   len(valid_idx),
            "scores":      scores,
        }


# ===========================================================================
# Result display helpers
# ===========================================================================

METRIC_ORDER = [
    "BLEU-4", "ROUGE-2", "METEOR",
    "BERTScore", "LongCLIP Cosine", "T5 Cosine", "SBERT Cosine",
]


def print_results(result: dict, model_output_path: str):
    label   = os.path.basename(model_output_path)
    scores  = result["scores"]
    n_total = result["num_samples"]
    n_valid = result["num_valid"]

    width = 60
    print("\n" + "=" * width)
    print("  EVALUATION RESULTS")
    print("=" * width)
    print(f"  File    : {label}")
    print(f"  Samples : {n_total} total, {n_valid} with predictions")
    print("-" * width)
    for metric in METRIC_ORDER:
        val = scores.get(metric)
        if val is not None:
            flag = "✓" if val > 0 else "✗"
            print(f"  {flag}  {metric:<22} {val:.4f}")
        else:
            print(f"  -  {metric:<22} N/A")
    print("=" * width)

    # Dependency warnings
    notes = []
    if not BERTSCORE_AVAILABLE:
        notes.append("BERTScore unavailable  →  pip install bert-score")
    if not LONGCLIP_AVAILABLE:
        notes.append("LongCLIP unavailable   →  custom install required")
    if not T5_AVAILABLE:
        notes.append("T5 unavailable         →  pip install transformers")
    if not SBERT_AVAILABLE:
        notes.append("SBERT unavailable      →  pip install sentence-transformers")
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  • {note}")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="NLG Evaluation Suite for Vision-Language Model Outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_output", required=True,
        metavar="PATH",
        help="Path to the model predictions JSONL file.",
    )
    parser.add_argument(
        "--reference", required=True,
        metavar="PATH",
        help="Path to the ground-truth reference JSONL file.",
    )
    parser.add_argument(
        "--longclip_ckpt",
        default="./checkpoints/longclip-L.pt",
        metavar="PATH",
        help="Path to the LongCLIP checkpoint (default: ./checkpoints/longclip-L.pt).",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        metavar="PATH",
        help="(Optional) Save the results dict to a JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for path, label in [(args.model_output, "--model_output"),
                        (args.reference,     "--reference")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    evaluator = NLGEvaluator(longclip_ckpt=args.longclip_ckpt)
    result = evaluator.evaluate(
        model_output_path=args.model_output,
        reference_path=args.reference,
    )

    print_results(result, args.model_output)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n  Results saved to: {args.output_json}")

    print("\n✓ Evaluation complete.\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
