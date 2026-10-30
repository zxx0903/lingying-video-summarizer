from __future__ import annotations

from .text_features import jaccard


def _overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] - 1e-6 and b["start"] < a["end"] - 1e-6


def _duration(seg: dict) -> float:
    return max(0.0, seg["end"] - seg["start"])


def _clean_end(seg: dict, proposed_end: float) -> float:
    """Snap a trimmed end to the nearest subtitle cue boundary."""
    cues = seg.get("cues") or []
    best = proposed_end
    for cue in cues:
        end = float(cue.get("end", 0.0))
        if seg["start"] + 6.5 <= end <= proposed_end + 0.8:
            best = end
    return best


def _fit_target(selected: list[dict], target_seconds: float) -> list[dict]:
    total = sum(_duration(s) for s in selected)
    if total <= target_seconds * 1.08:
        return selected
    selected = sorted(selected, key=lambda s: s["start"])
    scale = target_seconds / max(1e-6, total)
    for seg in selected:
        dur = _duration(seg)
        keep = max(7.0, dur * scale)
        if keep < dur - 0.3:
            seg["end"] = round(_clean_end(seg, seg["start"] + keep), 2)
    total = sum(_duration(s) for s in selected)
    for seg in reversed(selected):
        total = sum(_duration(s) for s in selected)
        excess = total - target_seconds
        if excess <= 0.3:
            break
        dur = _duration(seg)
        cut = min(excess, max(0.0, dur - 7.0))
        if cut <= 0.2:
            continue
        seg["end"] = round(_clean_end(seg, seg["end"] - cut), 2)
    return selected


def select_segments(
    segments: list[dict],
    target_seconds: float,
    max_similarity: float = 0.65,
    min_segments: int = 3,
    min_clip: float = 9.0,
) -> list[dict]:
    if not segments:
        return []
    ranked = sorted(segments, key=lambda s: s.get("score", 0.0), reverse=True)
    max_score = max(1e-6, ranked[0]["score"])

    scenes: dict[int, list[dict]] = {}
    for seg in segments:
        scenes.setdefault(int(seg.get("scene_id", 0)), []).append(seg)
    scene_best = [
        (sid, max(group, key=lambda s: s.get("score", 0.0)))
        for sid, group in scenes.items()
    ]
    scene_best.sort(key=lambda item: item[1].get("score", 0.0), reverse=True)

    max_scenes = max(1, int(target_seconds // max(6.0, min_clip)))
    chosen_scenes = scene_best if len(scene_best) <= max_scenes else scene_best[:max_scenes]

    selected: list[dict] = []
    total = 0.0

    # Pass 1: one representative per scene/section. For short lecture videos
    # this keeps every slide; for long videos it keeps the strongest sections.
    for _sid, seg in chosen_scenes:
        if any(_overlap(seg, s) for s in selected):
            continue
        selected.append(seg)
        total += _duration(seg)

    # Pass 2: fill remaining time with the next best non-overlapping segments.
    for seg in ranked:
        if total >= target_seconds and len(selected) >= min_segments:
            break
        if any(seg is s for s in selected):
            continue
        if any(_overlap(seg, s) for s in selected):
            continue
        toks = set(seg.get("tokens", []))
        too_similar = False
        for s in selected:
            sim = jaccard(toks, set(s.get("tokens", [])))
            if sim > max_similarity and seg["score"] < s["score"] * 1.15:
                too_similar = True
                break
        if too_similar:
            continue
        selected.append(seg)
        total += _duration(seg)

    if len(selected) < min_segments:
        for seg in ranked:
            if len(selected) >= min_segments:
                break
            if any(seg is s for s in selected):
                continue
            if any(_overlap(seg, s) for s in selected):
                continue
            selected.append(seg)

    selected = _fit_target(selected, target_seconds)
    selected.sort(key=lambda s: s["start"])
    return selected
