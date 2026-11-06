from __future__ import annotations

import time
from pathlib import Path

from .config import PipelineConfig, Weights
from .ffmpeg_utils import concat_clips
from .render import render_selected_clips
from .summarize import build_summary


def config_from_result(result: dict) -> PipelineConfig:
    raw = dict(result.get("config", {}))
    w = raw.pop("weights", {}) if isinstance(raw, dict) else {}
    return PipelineConfig(weights=Weights(**w), **raw)


def rerender_selection(result: dict, out_dir: Path, ids: set[str]):
    selected = [s for s in result["segments"] if s["id"] in ids]
    selected.sort(key=lambda s: s["start"])
    stamp = str(int(time.time()))
    config = config_from_result(result)
    clips = render_selected_clips(
        Path(result["input"]["path"]),
        selected,
        out_dir / "clips_adjusted" / stamp,
        output_format=config.output_format,
        output_height=config.output_height,
    )
    preview = out_dir / f"preview_adjusted_{stamp}.{config.output_format}"
    if len(clips) > 1:
        concat_clips([c["path"] for c in clips], preview)
    elif clips:
        preview = Path(clips[0]["path"])
    summary = build_summary(
        result["input"]["duration"],
        selected,
        [
            s.get("semantic_text") or s.get("text", "")
            for s in result["segments"]
        ],
        config,
        outline=result.get("outline"),
    )
    return preview, summary, clips
