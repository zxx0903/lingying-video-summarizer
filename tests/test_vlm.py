from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from lingying.config import PipelineConfig
from lingying.summarize import build_summary
from lingying.vlm import (
    VisualEvidence,
    attach_visual_evidence,
    describe_video_frames,
    parse_visual_response,
    select_visual_samples,
)


class VlmEvidenceTest(unittest.TestCase):
    def test_parse_fenced_json(self):
        data = parse_visual_response(
            '```json\n{"description":"讲师运行终端命令","objects":["讲师"],'
            '"actions":["运行命令"],"on_screen_text":"python demo.py",'
            '"content_type":"code","confidence":0.9}\n```',
            "test",
        )
        self.assertEqual(data["description"], "讲师运行终端命令")
        self.assertEqual(data["objects"], ["讲师"])
        self.assertAlmostEqual(data["confidence"], 0.9)

    def test_parse_truncated_json(self):
        data = parse_visual_response(
            '```json\n{"description":"实验者连接电路",'
            '"objects":["实验者","电路"],"actions":["连接导线"],'
            '"on_screen_text":"存储器实验","confidence":0.82',
            "test",
        )
        self.assertEqual(data["description"], "实验者连接电路")
        self.assertEqual(data["objects"], ["实验者", "电路"])
        self.assertEqual(data["on_screen_text"], "存储器实验")
        self.assertAlmostEqual(data["confidence"], 0.82)

    def test_select_scene_and_uniform_frames(self):
        frames = [Path(f"frame_{i}.jpg") for i in range(10)]
        samples = select_visual_samples(
            frames,
            fps=1.0,
            duration=10.0,
            scene_times=[0.0, 4.0],
            max_frames=4,
        )
        self.assertGreaterEqual(len(samples), 3)
        self.assertTrue(all(0.0 <= item.time <= 10.0 for item in samples))

    def test_attach_visual_text_without_speech(self):
        segments = [
            {
                "id": "s001",
                "index": 1,
                "start": 0.0,
                "end": 10.0,
                "score": 0.5,
                "scene_id": 0,
                "text": "",
                "keywords": [],
                "tokens": [],
            }
        ]
        attach_visual_evidence(
            segments,
            [
                VisualEvidence(
                    time=5.0,
                    description="实验者正在连接电路",
                    on_screen_text="实验一 存储器实验",
                    confidence=0.8,
                )
            ],
        )
        self.assertIn("连接电路", segments[0]["semantic_text"])
        self.assertIn("存储器", segments[0]["semantic_text"])
        summary = build_summary(
            10.0,
            segments,
            [segments[0]["semantic_text"]],
            PipelineConfig(),
        )
        self.assertIn("实验", summary["overview"])

    def test_describe_video_frames_writes_cache(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [root / f"frame_{i}.jpg" for i in range(4)]
            for frame in frames:
                frame.write_bytes(b"image")
            cache = root / "vlm_frames.json"
            config = PipelineConfig(
                vlm_enabled=True,
                vlm_max_frames=2,
                vlm_backend="openai-compatible",
            )
            fake = {
                "description": "讲师在运行代码",
                "objects": ["讲师", "屏幕"],
                "actions": ["运行代码"],
                "on_screen_text": "python demo.py",
                "content_type": "code",
                "confidence": 0.8,
                "backend": "mock",
            }
            with patch("lingying.vlm.describe_frame", return_value=fake):
                evidence = describe_video_frames(
                    frames,
                    fps=1.0,
                    duration=4.0,
                    scene_times=[0.0, 2.0],
                    config=config,
                    cache_path=cache,
                )
            self.assertTrue(evidence)
            self.assertTrue(cache.exists())
            self.assertIn("python demo.py", cache.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
