from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Weights:
    text: float = 0.45
    audio: float = 0.30
    visual: float = 0.25

    def normalized(self) -> "Weights":
        total = self.text + self.audio + self.visual
        if total <= 0:
            return Weights()
        return Weights(self.text / total, self.audio / total, self.visual / total)


@dataclass
class PipelineConfig:
    target_ratio: float = 0.25
    target_seconds: float | None = None
    min_clip: float = 12.0
    max_clip: float = 45.0
    sample_fps: float = 1.0
    frame_width: int = 480
    segment_seconds: float = 24.0
    max_segments: int = 120
    weights: Weights = field(default_factory=Weights)
    asr_enabled: bool = True
    asr_model: str = "small"
    ocr_enabled: bool = True
    ocr_max_frames: int = 80
    ocr_min_confidence: float = 0.5
    llm_enabled: bool = False
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key_env: str = "OPENAI_API_KEY"
    seed: int = 42

    def resolve_target(self, duration: float) -> float:
        if self.target_seconds and self.target_seconds > 0:
            return min(float(self.target_seconds), max(10.0, duration * 0.9))
        ratio_target = duration * float(self.target_ratio)
        # Short lectures should still get a usable summary instead of a 30 s teaser.
        return max(15.0, min(duration * 0.45, max(ratio_target, 60.0)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        weights = Weights(**raw.pop("weights", {}))
        return cls(weights=weights, **raw)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
