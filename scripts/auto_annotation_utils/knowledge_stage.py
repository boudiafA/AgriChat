"""
Stage II: verified class-level knowledge generation with Google Search grounding.

This module queries the Google GenAI SDK with structured output enabled to
produce one knowledge record per class. The output is designed to be resumable
and directly consumable by the QA synthesis stage.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .common import load_class_names, load_jsonl_index

DEFAULT_KNOWLEDGE_MODEL = "gemini-3-pro-preview"

KNOWLEDGE_PROMPT_TEMPLATE = """
Generate a single scientifically grounded agricultural knowledge paragraph for the class "{class_name}".

The paragraph should concisely cover:
- taxonomic identity
- distinctive visual or morphological characteristics
- habitat or cultivation context
- relevant agricultural use or impact
- common health or management considerations when applicable

Requirements:
- use precise, professional language
- avoid markdown, bullets, and citations
- do not invent dataset-specific facts
- keep the text useful for downstream question-answer generation
""".strip()

KNOWLEDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "class_name": {"type": "string"},
        "knowledge": {"type": "string"},
    },
    "required": ["class_name", "knowledge"],
}


def build_genai_client():
    """Create a Google GenAI client using environment-based authentication."""
    from google import genai

    return genai.Client()


def request_class_knowledge(client, class_name: str, model_id: str) -> dict[str, str]:
    """Request one structured knowledge record for *class_name*."""
    from google.genai import types

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    response = client.models.generate_content(
        model=model_id,
        contents=KNOWLEDGE_PROMPT_TEMPLATE.format(class_name=class_name),
        config=types.GenerateContentConfig(
            tools=[grounding_tool],
            response_mime_type="application/json",
            response_json_schema=KNOWLEDGE_RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )
    record = response.parsed if getattr(response, "parsed", None) else json.loads(response.text)
    record["class_name"] = class_name
    return record


def run_knowledge_stage(
    image_root: Path,
    class_names_file: Path | None,
    output_path: Path,
    model_id: str,
    resume: bool,
) -> dict[str, int | str]:
    """Run Stage II and persist class knowledge records."""
    class_names = load_class_names(class_names_file=class_names_file, image_root=image_root)
    if not class_names:
        raise ValueError(
            "No class names were found. Provide a class-organised image root or --class-names-file."
        )

    existing = load_jsonl_index(output_path, "class_name") if resume else {}
    pending_classes = [class_name for class_name in class_names if class_name not in existing]

    if not pending_classes:
        return {
            "stage": "knowledge",
            "total_classes": len(class_names),
            "processed": 0,
            "skipped_existing": len(class_names),
            "output_path": str(output_path),
        }

    print(f"[Stage II] Loading Google GenAI client for model: {model_id}")
    client = build_genai_client()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resume else "w"
    with output_path.open(write_mode, encoding="utf-8") as handle:
        for class_name in tqdm(pending_classes, desc="Stage II - knowledge", unit="class"):
            try:
                record = request_class_knowledge(client=client, class_name=class_name, model_id=model_id)
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[Stage II][WARN] Failed on {class_name}: {exc}")
                continue

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    return {
        "stage": "knowledge",
        "total_classes": len(class_names),
        "processed": len(pending_classes),
        "skipped_existing": len(class_names) - len(pending_classes),
        "output_path": str(output_path),
    }
