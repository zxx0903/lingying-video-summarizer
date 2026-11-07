from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .asr import Cue, cues_to_srt
from .ffmpeg_utils import concat_clips, extract_frame_at, render_clip
from .visual_features import frame_stats

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _fmt(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def make_contact_sheet(
    frames: list[Path],
    labels: list[str],
    out_path: Path,
    cols: int = 3,
    thumb_w: int = 360,
) -> None:
    if not frames:
        return
    thumb_h = int(thumb_w * 9 / 16)
    rows = math.ceil(len(frames) / cols)
    pad = 12
    header = 40
    width = cols * thumb_w + (cols + 1) * pad
    height = rows * (thumb_h + 34) + (rows + 1) * pad + header
    canvas = Image.new("RGB", (width, height), "#f4f6fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), "关键帧预览", fill="#1f2a44", font=_font(22))
    for i, path in enumerate(frames):
        r, c = divmod(i, cols)
        x = pad + c * (thumb_w + pad)
        y = header + pad + r * (thumb_h + 34)
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            canvas.paste(img, (x, y))
        except Exception:
            draw.rectangle([x, y, x + thumb_w, y + thumb_h], fill="#dde3ef")
        draw.rectangle([x, y + thumb_h, x + thumb_w, y + thumb_h + 28], fill="#ffffff")
        draw.text((x + 8, y + thumb_h + 4), labels[i], fill="#33415c", font=_font(15))
    canvas.save(out_path, quality=92)


def contact_sheet_as_pdf(image_path: Path, pdf_path: Path) -> Path:
    with Image.open(image_path) as image:
        image.convert("RGB").save(pdf_path, "PDF", resolution=120.0)
    return pdf_path


def make_timeline(
    segments: list[dict],
    selected_ids: set[str],
    duration: float,
    out_path: Path,
) -> None:
    width, height = 1400, 300
    margin_l, margin_r, margin_t, margin_b = 70, 40, 50, 60
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin_l, 14),
        "多模态重要度时间轴（橙色为入选片段）",
        fill="#1f2a44",
        font=_font(22),
    )
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    base_y = margin_t + plot_h
    draw.line([(margin_l, base_y), (margin_l + plot_w, base_y)], fill="#9aa5b8", width=2)

    def x_of(t: float) -> float:
        return margin_l + min(1.0, max(0.0, t / max(1e-6, duration))) * plot_w

    tick_step = 60 if duration <= 600 else (300 if duration <= 1800 else 600)
    t = 0
    while t <= duration + 1e-6:
        x = x_of(t)
        draw.line([(x, base_y), (x, base_y + 6)], fill="#9aa5b8", width=1)
        label = _fmt(t)
        draw.text((x - 18, base_y + 10), label, fill="#5b6578", font=_font(13))
        t += tick_step

    for seg in segments:
        x1, x2 = x_of(seg["start"]), x_of(seg["end"])
        h = 24 + seg.get("score", 0.0) * (plot_h - 50)
        color = "#f28c28" if seg["id"] in selected_ids else "#c9d2e3"
        draw.rounded_rectangle(
            [x1 + 1, base_y - h, max(x1 + 3, x2 - 1), base_y],
            radius=3,
            fill=color,
        )
        if seg["id"] in selected_ids:
            draw.text(
                (x1 + 2, base_y - h - 18),
                f"S{seg['index']}",
                fill="#b45a00",
                font=_font(13),
            )
    canvas.save(out_path, quality=92)


def render_selected_clips(
    video: Path,
    selected: list[dict],
    clips_dir: Path,
    output_format: str = "mp4",
    output_height: int | None = None,
    progress=None,
) -> list[dict]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    output_format = output_format.lower().lstrip(".")
    if output_format not in {"mp4", "mov", "avi"}:
        output_format = "mp4"
    results = []
    for i, seg in enumerate(selected, 1):
        out = (
            clips_dir
            / f"clip_{i:02d}_{int(seg['start'])}-{int(seg['end'])}.{output_format}"
        )
        if progress:
            progress(i - 1, len(selected), f"渲染片段 {i}/{len(selected)}")
        render_clip(video, seg["start"], seg["end"], out, height=output_height)
        seg["clip"] = str(out)
        results.append({"index": i, "path": str(out), "segment_id": seg["id"]})
    if progress:
        progress(len(selected), max(1, len(selected)), "片段渲染完成")
    return results


def render_keyframes(
    video: Path,
    selected: list[dict],
    frames_dir: Path,
    source_frames: list[Path] | None = None,
    frame_fps: float = 1.0,
    per_clip: int = 3,
) -> list[dict]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    out = []
    stats_cache: dict[int, dict] = {}

    def cached_stats(index: int) -> dict:
        if index not in stats_cache:
            stats_cache[index] = frame_stats(source_frames[index])
        return stats_cache[index]

    for i, seg in enumerate(selected, 1):
        dur = seg["end"] - seg["start"]
        candidates: list[tuple[float, int, Path]] = []
        if source_frames:
            for index, frame_path in enumerate(source_frames):
                t = index / max(0.1, frame_fps)
                if seg["start"] + 0.15 <= t <= seg["end"] - 0.15:
                    candidates.append((t, index, frame_path))
        if candidates:
            buckets: list[list[tuple[float, int, Path]]] = [[] for _ in range(per_clip)]
            count = len(candidates)
            for pos, item in enumerate(candidates):
                bucket = min(per_clip - 1, int(pos * per_clip / max(1, count)))
                buckets[bucket].append(item)
            selected_times: list[float] = []
            for j, bucket in enumerate(buckets):
                if not bucket:
                    ratio = (j + 0.5) / per_clip
                    selected_times.append(seg["start"] + dur * ratio)
                    continue
                scored = []
                for t, _index, path in bucket:
                    stats = cached_stats(_index)
                    brightness = float(stats.get("brightness", 0.0))
                    exposure = max(0.0, 1.0 - abs(brightness - 0.52) * 2.2)
                    saturation = float(stats.get("saturation", 0.0))
                    colorfulness = float(stats.get("colorfulness", 0.0))
                    artifact_penalty = min(
                        1.0,
                        max(0.0, saturation - 0.65) * 2.0
                        + max(0.0, colorfulness - 0.30) * 1.5,
                    )
                    transition_penalty = 0.0
                    gray = stats.get("gray")
                    if gray is not None and source_frames:
                        diffs = []
                        for neighbor in (_index - 1, _index + 1):
                            if 0 <= neighbor < len(source_frames):
                                neighbor_gray = cached_stats(neighbor).get("gray")
                                if neighbor_gray is not None:
                                    diffs.append(
                                        float(np.mean(cv2.absdiff(gray, neighbor_gray)) / 255.0)
                                    )
                        if diffs:
                            transition_penalty = min(1.0, float(np.mean(diffs)) / 0.30)
                    score = (
                        0.45 * float(stats.get("edge_density", 0.0))
                        + 0.25 * float(stats.get("contrast", 0.0))
                        + 0.15 * colorfulness
                        + 0.15 * exposure
                        - 1.25 * transition_penalty
                        - 0.75 * artifact_penalty
                    )
                    scored.append((score, t))
                selected_times.append(max(scored)[1])
        else:
            selected_times = [
                seg["start"] + dur * ((j + 0.5) / per_clip)
                for j in range(per_clip)
            ]

        raw_scene_start = seg.get("scene_start")
        raw_scene_end = seg.get("scene_end")
        scene_start = (
            float(raw_scene_start) if raw_scene_start is not None else seg["start"]
        )
        scene_end = (
            float(raw_scene_end) if raw_scene_end is not None else seg["end"]
        )
        window_start = max(seg["start"] + 0.1, scene_start + 0.3)
        window_end = min(seg["end"] - 0.1, scene_end - 0.1)
        if window_start > window_end:
            window_start = seg["start"] + 0.1
            window_end = seg["end"] - 0.1

        keyframe_times = [
            min(max(t, window_start), window_end)
            for t in selected_times[:per_clip]
        ]
        if len({round(t, 2) for t in keyframe_times}) < len(keyframe_times):
            # Short detected scenes can collapse multiple buckets onto the same
            # timestamp. Spread them across the selected clip in that case.
            keyframe_times = [
                seg["start"] + dur * ((j + 0.5) / per_clip)
                for j in range(per_clip)
            ]

        for j, t in enumerate(keyframe_times, 1):
            path = frames_dir / f"kf_{i:02d}_{j}.jpg"
            extract_frame_at(video, t, path)
            out.append(
                {
                    "segment_id": seg["id"],
                    "clip_index": i,
                    "time": round(t, 2),
                    "path": str(path),
                    "label": f"S{i} · {_fmt(t)}",
                }
            )
    return out


def build_selected_srt(
    cues: list[Cue], selected: list[dict], out_path: Path
) -> list[Cue]:
    """Build a continuous SRT whose timeline follows the concatenated clips."""
    out_cues: list[Cue] = []
    offset = 0.0
    for seg in selected:
        dur = seg["end"] - seg["start"]
        for cue in cues:
            s = max(cue.start, seg["start"])
            e = min(cue.end, seg["end"])
            if e - s <= 0.1:
                continue
            out_cues.append(
                Cue(
                    start=offset + (s - seg["start"]),
                    end=offset + (e - seg["start"]),
                    text=cue.text,
                )
            )
        offset += dur
    out_path.write_text(cues_to_srt(out_cues), encoding="utf-8")
    return out_cues
