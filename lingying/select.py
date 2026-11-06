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
    for seg in selected:
        seg.setdefault("_original_start", float(seg["start"]))
        seg.setdefault("_original_end", float(seg["end"]))
    total = sum(_duration(s) for s in selected)
    if total <= target_seconds * 1.05:
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


def _extend_to_target(selected: list[dict], target_seconds: float) -> None:
    """Restore trimmed clip tails before adding lower-ranked new segments."""
    total = sum(_duration(s) for s in selected)
    if total >= target_seconds * 0.98:
        return
    for seg in sorted(selected, key=lambda item: item.get("score", 0.0), reverse=True):
        need = target_seconds - total
        if need <= 0.3:
            break
        original_end = float(seg.get("_original_end", seg["end"]))
        can_extend = max(0.0, original_end - float(seg["end"]))
        if can_extend <= 0.1:
            continue
        extension = min(need, can_extend)
        seg["end"] = round(float(seg["end"]) + extension, 2)
        total += extension


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

    # Shot detection often returns dozens of cuts in edited talking-head videos.
    # Keep semantic scene coverage, but cap representatives so short clips do
    # not automatically become dozens of chapters.
    if len(scene_best) <= 10:
        max_scenes = len(scene_best)
    else:
        # Roughly one semantically distinct section per 12 seconds of summary,
        # capped to avoid shot cuts exploding into dozens of tiny summaries.
        max_scenes = min(18, max(8, int((target_seconds + 11.999) // 12)))
    if len(scene_best) <= max_scenes:
        chosen_scenes = scene_best
    else:
        # Pick the best representative in each temporal band. Top-K alone can
        # put the entire summary in the first half of an edited lecture.
        by_time = sorted(
            scene_best,
            key=lambda item: (
                item[1]["start"] + item[1]["end"]
            )
            / 2.0,
        )
        chosen_scenes = []
        for band in range(max_scenes):
            lo = band * len(by_time) // max_scenes
            hi = (band + 1) * len(by_time) // max_scenes
            candidates = by_time[lo:hi]
            if candidates:
                chosen_scenes.append(
                    max(candidates, key=lambda item: item[1].get("score", 0.0))
                )

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

    # If lexical de-duplication blocked every remaining candidate, relax it in
    # small steps to get within the specified timing tolerance. Overlap is still
    # forbidden, so this cannot reuse the same video interval.
    relaxed_similarity = max_similarity
    while total < target_seconds * 0.95 and relaxed_similarity <= 1.0:
        for seg in ranked:
            if total >= target_seconds * 0.98:
                break
            if any(seg is s for s in selected):
                continue
            if any(_overlap(seg, s) for s in selected):
                continue
            toks = set(seg.get("tokens", []))
            if any(
                jaccard(toks, set(s.get("tokens", []))) > relaxed_similarity
                for s in selected
            ):
                continue
            selected.append(seg)
            total += _duration(seg)
        relaxed_similarity += 0.1

    if total < target_seconds * 0.95:
        for seg in ranked:
            if total >= target_seconds * 0.98:
                break
            if any(seg is s for s in selected):
                continue
            if any(_overlap(seg, s) for s in selected):
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
    total = sum(_duration(s) for s in selected)
    if total < target_seconds * 0.95:
        for seg in ranked:
            if total >= target_seconds * 0.98:
                break
            if any(seg is s for s in selected):
                continue
            if any(_overlap(seg, s) for s in selected):
                continue
            selected.append(seg)
            total += _duration(seg)
    selected = _fit_target(selected, target_seconds)
    _extend_to_target(selected, target_seconds)
    selected.sort(key=lambda s: s["start"])
    return selected
