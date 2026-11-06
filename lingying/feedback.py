from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .fusion import apply_feedback


def record_selection_feedback(
    result: dict,
    out_dir: str | Path,
    kept_ids: set[str],
) -> dict:
    """Persist keep/drop labels for later reranker and weight experiments."""
    out_dir = Path(out_dir)
    original_ids = set(result.get("selected_ids", []))
    labels = {segment_id: int(segment_id in kept_ids) for segment_id in original_ids}
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "video": result.get("input", {}).get("name", ""),
        "template_id": result.get("config", {}).get("template_id", ""),
        "labels": labels,
        "weights_before": result.get("config", {}).get("weights", {}),
    }
    path = out_dir / "feedback.json"
    history = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            history = raw if isinstance(raw, list) else [raw]
        except Exception:
            history = []
    history.append(record)
    path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def aggregate_feedback(result: dict, labels: dict[str, int]) -> dict:
    raw = result.get("config", {}).get("weights", {})
    from .config import Weights

    base = Weights(
        text=float(raw.get("text", 0.45)),
        audio=float(raw.get("audio", 0.30)),
        visual=float(raw.get("visual", 0.25)),
    )
    adjusted = apply_feedback(result.get("segments", []), labels, base)
    return asdict(adjusted)
