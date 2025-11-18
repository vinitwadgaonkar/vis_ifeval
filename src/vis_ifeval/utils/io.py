"""I/O utilities for loading prompts and saving results."""

import json
from collections.abc import Iterable
from pathlib import Path


def load_prompts(path: str) -> list[dict]:
    """Load prompts from a JSONL file.

    Args:
        path: Path to the JSONL file containing prompts.

    Returns:
        List of prompt dictionaries, one per line in the file.
    """
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def save_jsonl(path: str, records: Iterable[dict]) -> None:
    """Save records to a JSONL file.

    Args:
        path: Path where the JSONL file should be written.
        records: Iterable of dictionaries to write as JSON lines.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

