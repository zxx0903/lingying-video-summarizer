from __future__ import annotations

import json
import time
import traceback
from bisect import bisect_right
from pathlib import Path
from typing import Callable

import numpy as np

from .asr import (
    Cue,
    find_sidecar_srt,
    parse_srt,
    transcribe_with_whisper,
)
from .audio_features import read_wav, segment_audio_stats
from .config import PipelineConfig, ensure_dir
from .ffmpeg_utils import (
    concat_clips,
    extract_audio,
    extract_frames,
    probe,
    write_json,
)
from .fusion import apply_template_prior, score_segments
from .ocr import ocr_frames
from .render import (
    build_selected_srt,
    contact_sheet_as_pdf,
    make_contact_sheet,
    make_timeline,
    render_keyframes,
    render_selected_clips,
)
from .exports import write_summary_exports
from .select import select_segments
from .summarize import build_summary, load_outline
from .templates import resolve_template_keywords
from .text_features import (
    keyword_hit_ratio,
    keyword_scores,
    split_sentences,
    text_richness,
    cue_word_score,
    tokenize,
)
from .visual_features import detect_scene_boundaries, segment_visual_stats
from .vlm import (
    attach_visual_evidence,
    combine_semantic_text,
    describe_video_frames,
    vlm_availability,
)

ProgressFn = Callable[[str, float], None]


def _noop(stage: str, pct: float) -> None:
    pass


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def adaptive_fps(duration: float, requested: float, max_frames: int = 1400) -> float:
    if duration <= 0:
        return requested
    return max(0.1, min(requested, max_frames / duration))


def scene_id_for(t: float, outline: list[dict], scene_times: list[float]) -> int:
    if outline:
        for i, item in enumerate(outline):
            if item["start"] - 0.5 <= t < item["end"] + 0.5:
                return i
        return len(outline)
    return max(0, bisect_right(scene_times, t) - 1)


def build_segments(
    cues: list[Cue],
    duration: float,
    segment_seconds: float = 24.0,
    min_seconds: float = 7.0,
    max_segments: int = 120,
    preserve_boundaries: bool = False,
) -> list[dict]:
    segments: list[dict] = []
    if cues:
        if preserve_boundaries:
            segments = [_cue_group([cue]) for cue in cues]
        else:
            current: list[Cue] = []
            for cue in cues:
                if not current:
                    current = [cue]
                    continue
                gap = cue.start - current[-1].end
                span = cue.end - current[0].start
                if gap > 1.8 or span + cue.duration > segment_seconds:
                    segments.append(_cue_group(current))
                    current = [cue]
                else:
                    current.append(cue)
            if current:
                segments.append(_cue_group(current))
    else:
        step = segment_seconds
        t = 0.0
        while t < duration - 1.0:
            end = min(duration, t + step)
            segments.append(
                {
                    "start": round(t, 2),
                    "end": round(end, 2),
                    "text": "",
                    "cues": [],
                }
            )
            t = end

    # Drop tiny tails, then cap the number of segments by merging the shortest neighbours.
    segments = [s for s in segments if s["end"] - s["start"] >= min_seconds] or segments
    while len(segments) > max_segments:
        idx = min(range(len(segments) - 1), key=lambda i: segments[i]["end"] - segments[i]["start"])
        a, b = segments[idx], segments[idx + 1]
        segments[idx] = {
            "start": a["start"],
            "end": b["end"],
            "text": (a["text"] + " " + b["text"]).strip(),
            "cues": a["cues"] + b["cues"],
        }
        del segments[idx + 1]

    for i, seg in enumerate(segments, 1):
        seg["id"] = f"s{i:03d}"
        seg["index"] = i
        seg["duration"] = round(seg["end"] - seg["start"], 2)
    return segments


def _cue_group(group: list[Cue]) -> dict:
    return {
        "start": round(group[0].start, 2),
        "end": round(group[-1].end, 2),
        "text": " ".join(c.text.strip() for c in group).strip(),
        "cues": [c.to_dict() for c in group],
    }


def attach_features(
    segments: list[dict],
    keywords: dict[str, float],
    global_kws: list[str],
    audio_samples: np.ndarray | None,
    sample_rate: int,
    frames: list[Path],
    frame_fps: float,
) -> None:
    visual_cache: dict[str, dict] = {}
    for seg in segments:
        text = seg.get("semantic_text") or seg.get("text", "")
        toks = tokenize(text)
        chars = len(text.replace(" ", ""))
        dur = max(0.1, seg["end"] - seg["start"])
        local_kw = []
        for tok in toks:
            if tok in keywords and tok not in local_kw:
                local_kw.append(tok)
        local_kw.sort(key=lambda k: keywords.get(k, 0.0), reverse=True)
        seg["tokens"] = toks
        seg["keywords"] = local_kw[:4]
        seg["text_stats"] = {
            "chars": chars,
            "richness": round(text_richness(text), 4),
            "cue": round(cue_word_score(text), 4),
            "keyword_hit": round(keyword_hit_ratio(text, global_kws[:6]), 4),
            "speech_rate": round(chars / dur, 3),
            "sentences": len(split_sentences(text)),
        }
        if audio_samples is not None:
            seg["audio_stats"] = segment_audio_stats(
                audio_samples, sample_rate, seg["start"], seg["end"]
            )
        else:
            seg["audio_stats"] = {
                "rms": 0.0,
                "peak": 0.0,
                "zcr": 0.0,
                "silence_ratio": 1.0,
                "energy_var": 0.0,
                "voiced_ratio": 0.0,
            }
        seg["visual_stats"] = segment_visual_stats(
            frames, frame_fps, seg["start"], seg["end"], visual_cache
        )


def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    config: PipelineConfig | None = None,
    srt_path: str | Path | None = None,
    outline_path: str | Path | None = None,
    progress: ProgressFn | None = None,
) -> dict:
    config = config or PipelineConfig()
    progress = progress or _noop
    input_path = Path(input_path)
    out_dir = ensure_dir(out_dir)
    started = time.time()
    warnings: list[str] = []

    progress("读取视频信息", 0.02)
    info = probe(input_path)
    if outline_path is None:
        for candidate in (
            input_path.with_suffix(".outline.json"),
            input_path.with_suffix(".json"),
        ):
            if candidate.exists():
                outline_path = candidate
                break
    outline = load_outline(outline_path)

    audio_dir = ensure_dir(out_dir / "audio")
    frames_dir = ensure_dir(out_dir / "frames")
    clips_dir = ensure_dir(out_dir / "clips")
    kf_dir = ensure_dir(out_dir / "keyframes")

    progress("提取音轨", 0.08)
    audio_samples = None
    sample_rate = 16000
    if info.has_audio:
        wav = audio_dir / "audio.wav"
        try:
            extract_audio(input_path, wav)
            sample_rate, audio_samples = read_wav(wav)
        except Exception as exc:
            warnings.append(f"音轨提取失败: {exc}")
    else:
        warnings.append("视频没有音轨，音频模态将被降权。")

    progress("抽取视频帧", 0.15)
    fps = adaptive_fps(info.duration, config.sample_fps)
    frames: list[Path] = []
    try:
        frames = extract_frames(input_path, frames_dir, fps=fps, width=config.frame_width)
    except Exception as exc:
        warnings.append(f"抽帧失败: {exc}")
    try:
        scene_times = detect_scene_boundaries(frames, fps)
    except Exception:
        scene_times = [0.0]

    progress("识别字幕/语音", 0.28)
    cues: list[Cue] = []
    asr_mode = "none"
    chosen_srt = Path(srt_path) if srt_path else find_sidecar_srt(input_path)
    if chosen_srt and chosen_srt.exists():
        cues = parse_srt(chosen_srt)
        asr_mode = "srt"
    elif config.asr_enabled and info.has_audio:
        try:
            cues = transcribe_with_whisper(
                audio_dir / "audio.wav", model_size=config.asr_model
            )
            asr_mode = "whisper"
        except Exception as exc:
            warnings.append(
                "Whisper 不可用或转写失败，已退化为音频+视觉模式: "
                + str(exc).splitlines()[0][:160]
            )
    if not cues and config.ocr_enabled:
        try:
            cues = ocr_frames(
                frames,
                fps,
                info.duration,
                max_frames=config.ocr_max_frames,
                min_confidence=config.ocr_min_confidence,
            )
            if cues:
                asr_mode = "ocr"
        except Exception as exc:
            warnings.append(
                "OCR 不可用或识别失败，已退化为音频+视觉模式: "
                + str(exc).splitlines()[0][:160]
            )
    if not cues:
        asr_mode = "none"

    progress("切分片段", 0.36)
    segments = build_segments(
        cues,
        info.duration,
        segment_seconds=config.segment_seconds,
        min_seconds=2.0 if asr_mode == "ocr" else 7.0,
        max_segments=config.max_segments,
        preserve_boundaries=asr_mode == "ocr",
    )
    for index, seg in enumerate(segments):
        midpoint = (seg["start"] + seg["end"]) / 2.0
        seg["scene_id"] = (
            index
            if asr_mode == "ocr" and not outline
            else scene_id_for(midpoint, outline, scene_times)
        )
        sid = int(seg["scene_id"])
        if outline and 0 <= sid < len(outline):
            seg["scene_start"] = outline[sid]["start"]
            seg["scene_end"] = outline[sid]["end"]
        elif scene_times:
            seg["scene_start"] = scene_times[sid] if sid < len(scene_times) else 0.0
            seg["scene_end"] = (
                scene_times[sid + 1] if sid + 1 < len(scene_times) else info.duration
            )

    progress("理解画面语义", 0.48)
    vlm_mode = "none"
    if config.vlm_enabled and frames:
        available, availability_message = vlm_availability(config)
        if not available:
            warnings.append("VLM 未启用: " + availability_message)
        else:
            try:
                evidence = describe_video_frames(
                    frames,
                    fps,
                    info.duration,
                    scene_times,
                    config,
                    cache_path=out_dir / "vlm_frames.json",
                    progress=lambda done, total, msg: progress(
                        msg, 0.48 + 0.08 * (done / max(1, total))
                    ),
                )
                attach_visual_evidence(segments, evidence)
                if evidence:
                    vlm_mode = config.vlm_backend
            except Exception as exc:
                warnings.append(
                    "VLM 画面理解失败，已回退到字幕/语音/OCR: "
                    + str(exc).splitlines()[0][:200]
                )

    for seg in segments:
        semantic_text = combine_semantic_text(seg)
        if semantic_text:
            seg["semantic_text"] = semantic_text
    if not cues and vlm_mode == "none":
        warnings.append("没有字幕、语音或画面语义，文本模态将被降权。")
    content_mode = asr_mode
    if vlm_mode != "none":
        content_mode = (
            "vlm" if asr_mode == "none" else f"{asr_mode}+vlm"
        )

    progress("计算多模态特征", 0.56)
    texts = [s.get("semantic_text") or s.get("text", "") for s in segments]
    kw_scores = keyword_scores([t for t in texts if t], topk=15)
    global_kws = list(kw_scores.keys())
    attach_features(
        segments,
        kw_scores,
        global_kws,
        audio_samples,
        sample_rate,
        frames,
        fps,
    )

    progress("融合评分", 0.66)
    score_segments(segments, config.weights, has_audio=bool(info.has_audio and audio_samples is not None))
    template_keywords = resolve_template_keywords(
        config.template_id, config.custom_keywords
    )
    apply_template_prior(segments, template_keywords)
    target = config.resolve_target(info.duration)
    selected = select_segments(segments, target)
    for seg in segments:
        seg.pop("_original_start", None)
        seg.pop("_original_end", None)
    selected_ids = {s["id"] for s in selected}
    for seg in segments:
        seg["selected"] = seg["id"] in selected_ids

    progress("生成章节与摘要", 0.7)
    summary = build_summary(info.duration, selected, texts, config, outline=outline)

    progress("渲染片段", 0.76)
    clips: list[dict] = []
    keyframes: list[dict] = []
    preview_path: Path | None = None
    srt_out: Path | None = None
    if selected:
        try:
            clips = render_selected_clips(
                input_path,
                selected,
                clips_dir,
                output_format=config.output_format,
                output_height=config.output_height,
                progress=lambda done, total, msg: progress(
                    msg, 0.76 + 0.16 * (done / max(1, total))
                ),
            )
            keyframes = render_keyframes(
                input_path,
                selected,
                kf_dir,
                source_frames=frames,
                frame_fps=fps,
            )
            clip_paths = [c["path"] for c in clips]
            if len(clip_paths) > 1:
                preview_path = out_dir / f"preview.{config.output_format}"
                concat_clips(clip_paths, preview_path)
            elif clip_paths:
                preview_path = Path(clip_paths[0])
            if cues:
                srt_out = out_dir / "summary.srt"
                build_selected_srt(cues, selected, srt_out)
        except Exception as exc:
            warnings.append(f"片段渲染失败: {exc}")
            traceback.print_exc()

    progress("生成可视化", 0.94)
    timeline_path = out_dir / "timeline.png"
    try:
        make_timeline(segments, selected_ids, info.duration, timeline_path)
    except Exception as exc:
        warnings.append(f"时间轴生成失败: {exc}")
        timeline_path = None

    sheet_path = out_dir / "keyframes.jpg"
    keyframe_pdf_path: Path | None = None
    try:
        if keyframes:
            make_contact_sheet(
                [Path(k["path"]) for k in keyframes],
                [k["label"] for k in keyframes],
                sheet_path,
            )
            keyframe_pdf_path = contact_sheet_as_pdf(
                sheet_path, out_dir / "keyframes.pdf"
            )
    except Exception as exc:
        warnings.append(f"关键帧拼图失败: {exc}")
        sheet_path = None

    for seg in segments:
        seg.pop("cues", None)
        seg.pop("tokens", None)
    for seg in selected:
        seg.pop("cues", None)
        seg.pop("tokens", None)

    elapsed = time.time() - started
    selected_duration = sum(s["end"] - s["start"] for s in selected)
    result = {
        "version": "0.2.0",
        "input": {
            "name": input_path.name,
            "path": str(input_path),
            "duration": round(info.duration, 2),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "has_audio": info.has_audio,
        },
        "config": config.to_dict(),
        "asr_mode": asr_mode,
        "vlm_mode": vlm_mode,
        "content_mode": content_mode,
        "outline": outline,
        "summary": summary,
        "segments": segments,
        "selected_ids": sorted(selected_ids),
        "selected": [
            {
                "id": s["id"],
                "index": s["index"],
                "start": s["start"],
                "end": s["end"],
                "score": s["score"],
                "text": s.get("text", ""),
                "semantic_text": s.get("semantic_text", ""),
                "visual_description": s.get("visual_description", ""),
                "visual_evidence": s.get("visual_evidence", []),
                "keywords": s.get("keywords", []),
                "modality": s.get("modality", {}),
                "gate": s.get("gate", {}),
                "template_hits": s.get("template_hits", []),
                "template_bonus": s.get("template_bonus", 0.0),
                "clip": _rel(Path(s["clip"]), out_dir) if s.get("clip") else None,
            }
            for s in selected
        ],
        "clips": [
            {**c, "path": _rel(Path(c["path"]), out_dir)} for c in clips
        ],
        "keyframes": [
            {**k, "path": _rel(Path(k["path"]), out_dir)} for k in keyframes
        ],
        "artifacts": {
            "timeline": _rel(timeline_path, out_dir) if timeline_path else None,
            "contact_sheet": _rel(sheet_path, out_dir) if sheet_path else None,
            "keyframes_pdf": (
                _rel(keyframe_pdf_path, out_dir) if keyframe_pdf_path else None
            ),
            "srt": _rel(srt_out, out_dir) if srt_out else None,
            "preview": _rel(preview_path, out_dir) if preview_path else None,
            "vlm_frames": (
                "vlm_frames.json"
                if vlm_mode != "none" and (out_dir / "vlm_frames.json").exists()
                else None
            ),
        },
        "metrics": {
            "duration": round(info.duration, 2),
            "target_seconds": round(target, 2),
            "selected_duration": round(selected_duration, 2),
            "compression_ratio": round(selected_duration / max(1e-6, info.duration), 4),
            "segment_count": len(segments),
            "selected_count": len(selected),
            "frame_fps": round(fps, 3),
            "frame_count": len(frames),
            "visual_evidence_count": sum(
                len(seg.get("visual_evidence", [])) for seg in segments
            ),
            "processing_seconds": round(elapsed, 2),
            "realtime_factor": round(elapsed / max(1e-6, info.duration), 3),
        },
        "warnings": warnings,
    }
    try:
        result["artifacts"].update(write_summary_exports(result, out_dir))
    except Exception as exc:
        warnings.append(f"文本摘要导出失败: {exc}")
        result["warnings"] = warnings

    progress("保存结果", 0.98)
    result = _jsonable(result)
    write_json(out_dir / "result.json", result)
    progress("完成", 1.0)
    return result
