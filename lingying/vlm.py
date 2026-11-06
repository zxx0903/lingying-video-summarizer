from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

from .config import PipelineConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_QWEN_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}

SYSTEM_PROMPT = """你是教育视频的画面理解器。只描述画面中直接可见的信息。
不要猜测身份、地点、意图或未显示的内容。
如果画面包含公式、代码、幻灯片、板书或屏幕文字，请尽量准确抄录。
输出一个 JSON 对象，字段为：
description（一句中文概括）
objects（可见对象数组）
actions（可见动作数组）
on_screen_text（可见文字，没有则为空字符串）
content_type（slide/code/whiteboard/lab/person/scene/other）
confidence（0 到 1）
不要输出 Markdown。"""


@dataclass
class VisualSample:
    time: float
    path: Path


@dataclass
class VisualEvidence:
    time: float
    description: str
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    on_screen_text: str = ""
    content_type: str = "other"
    confidence: float = 0.0
    backend: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["time"] = round(float(self.time), 2)
        data["confidence"] = round(float(self.confidence), 4)
        return data


def _clean_json_text(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_visual_response(raw: str, backend: str = "") -> dict:
    """Parse a model response while tolerating fenced or partially structured JSON."""
    cleaned = _clean_json_text(raw)
    try:
        data = json.loads(cleaned)
    except (TypeError, json.JSONDecodeError):
        data = {"description": (raw or "").strip()}
    if not isinstance(data, dict):
        data = {"description": str(data)}

    def string_list(value) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "description": str(data.get("description") or "").strip(),
        "objects": string_list(data.get("objects")),
        "actions": string_list(data.get("actions")),
        "on_screen_text": str(data.get("on_screen_text") or "").strip(),
        "content_type": str(data.get("content_type") or "other").strip(),
        "confidence": min(1.0, max(0.0, confidence)),
        "backend": backend,
    }


def select_visual_samples(
    frames: list[Path],
    fps: float,
    duration: float,
    scene_times: list[float] | None = None,
    max_frames: int = 80,
) -> list[VisualSample]:
    """Select scene representatives plus uniform coverage without decoding all frames."""
    if not frames or duration <= 0:
        return []
    max_frames = max(1, int(max_frames))
    boundaries = sorted(
        {max(0.0, min(duration, float(t))) for t in (scene_times or [0.0])}
    )
    boundaries.append(duration)
    scene_points = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 0.5:
            continue
        # Avoid transitions at the start and end of a detected scene.
        scene_points.append(start + (end - start) * 0.45)

    uniform_count = max(1, max_frames - len(scene_points))
    uniform_points = [
        duration * (i + 0.5) / uniform_count for i in range(uniform_count)
    ]
    chosen_times = sorted(scene_points + uniform_points)
    if len(chosen_times) > max_frames:
        chosen_times = [
            chosen_times[round(i * (len(chosen_times) - 1) / (max_frames - 1))]
            for i in range(max_frames)
        ] if max_frames > 1 else [chosen_times[len(chosen_times) // 2]]

    samples: list[VisualSample] = []
    seen_indices: set[int] = set()
    for time_value in chosen_times:
        index = min(
            len(frames) - 1,
            max(0, round(time_value * max(0.1, fps))),
        )
        if index in seen_indices:
            continue
        seen_indices.add(index)
        samples.append(VisualSample(index / max(0.1, fps), frames[index]))
    return samples


def resolve_vlm_model(config: PipelineConfig) -> str:
    configured = os.environ.get("LINGYING_VLM_MODEL", "").strip()
    if configured:
        return configured
    pointer = _PROJECT_ROOT / "runtime" / "vlm_model_path.txt"
    if pointer.exists():
        configured = pointer.read_text(encoding="utf-8-sig").strip()
        if configured:
            return configured
    return config.vlm_model


def vlm_availability(config: PipelineConfig) -> tuple[bool, str]:
    if config.vlm_backend == "openai-compatible":
        return True, "OpenAI-compatible visual endpoint"
    if config.vlm_backend != "qwen-vl":
        return False, f"Unsupported VLM backend: {config.vlm_backend}"
    missing = [
        name
        for name in ("torch", "transformers")
        if importlib.util.find_spec(name) is None
    ]
    if importlib.util.find_spec("qwen_vl_utils") is None:
        missing.append("qwen_vl_utils")
    if missing:
        return False, "Missing Python packages: " + ", ".join(missing)
    model = resolve_vlm_model(config)
    if Path(model).exists() or "/" in model:
        return True, model
    return False, f"VLM model not found: {model}"


def _load_qwen(config: PipelineConfig):
    source = resolve_vlm_model(config)
    key = (source, config.vlm_device, config.vlm_quantization)
    if key in _QWEN_CACHE:
        return _QWEN_CACHE[key]

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    kwargs: dict[str, Any] = {
        "device_map": config.vlm_device,
        "low_cpu_mem_usage": True,
    }
    if config.vlm_quantization == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(source, **kwargs)
    processor = AutoProcessor.from_pretrained(source)
    _QWEN_CACHE[key] = (model, processor)
    return model, processor


def _qwen_describe(image_path: Path, config: PipelineConfig) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _load_qwen(config)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = getattr(model, "device", None)
    if device is not None:
        inputs = inputs.to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max(32, int(config.vlm_max_new_tokens)),
            do_sample=False,
        )
    prompt_length = inputs.input_ids.shape[1]
    output = generated[:, prompt_length:]
    text = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    return parse_visual_response(text, "qwen-vl")


def _openai_describe(image_path: Path, config: PipelineConfig) -> dict:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    api_key = os.environ.get(config.vlm_api_key_env, "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        config.vlm_api_base_url.rstrip("/") + "/chat/completions",
        headers=headers,
        json={
            "model": config.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{encoded}",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
        },
        timeout=config.vlm_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    raw = data["choices"][0]["message"]["content"]
    if isinstance(raw, list):
        raw = " ".join(
            str(item.get("text", ""))
            for item in raw
            if isinstance(item, dict)
        )
    return parse_visual_response(str(raw), "openai-compatible")


def describe_frame(image_path: Path, config: PipelineConfig) -> dict:
    if config.vlm_backend == "openai-compatible":
        return _openai_describe(image_path, config)
    if config.vlm_backend == "qwen-vl":
        return _qwen_describe(image_path, config)
    raise ValueError(f"Unsupported VLM backend: {config.vlm_backend}")


def describe_video_frames(
    frames: list[Path],
    fps: float,
    duration: float,
    scene_times: list[float],
    config: PipelineConfig,
    cache_path: str | Path | None = None,
    progress=None,
) -> list[VisualEvidence]:
    samples = select_visual_samples(
        frames,
        fps,
        duration,
        scene_times=scene_times,
        max_frames=config.vlm_max_frames,
    )
    cache_file = Path(cache_path) if cache_path else None
    cached: dict[str, dict] = {}
    if cache_file and cache_file.exists():
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            for item in raw if isinstance(raw, list) else []:
                if isinstance(item, dict) and item.get("description"):
                    cached[str(round(float(item.get("time", 0.0)), 2))] = item
        except Exception:
            cached = {}

    results: list[VisualEvidence] = []
    for index, sample in enumerate(samples, 1):
        key = str(round(sample.time, 2))
        data = cached.get(key)
        if not data:
            data = describe_frame(sample.path, config)
            data["time"] = sample.time
        results.append(
            VisualEvidence(
                time=float(data.get("time", sample.time)),
                description=str(data.get("description", "")),
                objects=list(data.get("objects") or []),
                actions=list(data.get("actions") or []),
                on_screen_text=str(data.get("on_screen_text", "")),
                content_type=str(data.get("content_type", "other")),
                confidence=float(data.get("confidence", 0.0)),
                backend=str(data.get("backend", config.vlm_backend)),
            )
        )
        if progress:
            progress(index, len(samples), f"视觉语义 {index}/{len(samples)}")
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(
                    [item.to_dict() for item in results],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    return results


def combine_semantic_text(seg: dict) -> str:
    parts = [str(seg.get("text", "") or "").strip()]
    visual = str(seg.get("visual_text", "") or "").strip()
    if visual:
        parts.append("画面：" + visual)
    return "\n".join(part for part in parts if part)


def attach_visual_evidence(
    segments: list[dict],
    evidence: list[VisualEvidence],
) -> None:
    for seg in segments:
        seg["visual_evidence"] = []
    seen_descriptions: set[str] = set()
    for item in evidence:
        target = next(
            (
                seg
                for seg in segments
                if seg["start"] - 0.5 <= item.time < seg["end"] + 0.5
            ),
            None,
        )
        if target is None or not item.description:
            continue
        signature = re.sub(r"\W+", "", item.description).lower()
        if signature in seen_descriptions:
            continue
        seen_descriptions.add(signature)
        target["visual_evidence"].append(item.to_dict())

    for seg in segments:
        items = seg.get("visual_evidence", [])
        descriptions: list[str] = []
        visible_text: list[str] = []
        for item in items:
            desc = str(item.get("description", "")).strip()
            if desc and desc not in descriptions:
                descriptions.append(desc)
            text = str(item.get("on_screen_text", "")).strip()
            if text and text not in visible_text:
                visible_text.append(text)
        seg["visual_description"] = "\n".join(descriptions)
        seg["visual_text"] = "\n".join(
            part for part in ("\n".join(descriptions), "\n".join(visible_text)) if part
        )
        seg["semantic_text"] = combine_semantic_text(seg)
