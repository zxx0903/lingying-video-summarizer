from __future__ import annotations

import json
from pathlib import Path

from .text_features import jaccard, tokenize


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def normalize_ground_truth(raw) -> list[dict]:
    if isinstance(raw, dict):
        items = raw.get("important_segments") or raw.get("segments") or []
    else:
        items = raw
    result = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", item.get("start_time")))
            end = float(item.get("end", item.get("end_time")))
        except (TypeError, ValueError):
            continue
        if end > start:
            result.append(
                {
                    "start": start,
                    "end": end,
                    "label": str(item.get("label") or item.get("title") or ""),
                }
            )
    return sorted(result, key=lambda item: item["start"])


def evaluate_result(result: dict, ground_truth) -> dict:
    selected = sorted(
        (
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": str(seg.get("text", "") or ""),
            }
            for seg in result.get("selected", [])
        ),
        key=lambda item: item["start"],
    )
    gt = normalize_ground_truth(ground_truth)
    gt_total = _union_duration([(item["start"], item["end"]) for item in gt])
    selected_duration = sum(item["end"] - item["start"] for item in selected)

    covered_intervals = []
    for item in gt:
        for seg in selected:
            overlap = _overlap(item["start"], item["end"], seg["start"], seg["end"])
            if overlap > 0:
                covered_intervals.append(
                    (
                        max(item["start"], seg["start"]),
                        min(item["end"], seg["end"]),
                    )
                )
    covered_duration = _union_duration(covered_intervals)

    matched = 0
    strictly_matched = 0
    for item in gt:
        duration = max(1e-6, item["end"] - item["start"])
        best_overlap = max(
            (
                _overlap(item["start"], item["end"], seg["start"], seg["end"])
                for seg in selected
            ),
            default=0.0,
        )
        min_hit = min(10.0, max(2.0, duration * 0.10))
        if best_overlap >= min_hit:
            matched += 1
        if best_overlap / duration >= 0.3:
            strictly_matched += 1

    overlapping_selected = 0.0
    for seg in selected:
        duration = seg["end"] - seg["start"]
        overlap = sum(
            _overlap(seg["start"], seg["end"], item["start"], item["end"])
            for item in gt
        )
        if duration > 0:
            overlapping_selected += min(duration, overlap)

    token_sets = [set(tokenize(seg["text"])) for seg in selected]
    redundant_duration = 0.0
    for i, seg in enumerate(selected):
        if any(
            jaccard(token_sets[i], token_sets[j]) > 0.65
            for j in range(i)
        ):
            redundant_duration += seg["end"] - seg["start"]

    target = result.get("metrics", {}).get("target_seconds")
    target_error = None
    if target:
        target_error = (selected_duration - float(target)) / float(target)
    return {
        "ground_truth_segments": len(gt),
        "selected_segments": len(selected),
        "selected_duration": round(selected_duration, 3),
        "target_seconds": target,
        "target_error": round(target_error, 4) if target_error is not None else None,
        "ground_truth_coverage": round(covered_duration / max(1e-6, gt_total), 4),
        "segment_recall": round(matched / max(1, len(gt)), 4),
        "strict_topic_recall": round(strictly_matched / max(1, len(gt)), 4),
        "segment_precision": round(overlapping_selected / max(1e-6, selected_duration), 4),
        "content_redundancy": round(
            redundant_duration / max(1e-6, selected_duration), 4
        ),
        "chapter_count": len(result.get("summary", {}).get("chapters", [])),
    }


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
