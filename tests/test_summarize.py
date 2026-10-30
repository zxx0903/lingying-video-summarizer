from __future__ import annotations

import unittest

from lingying.config import PipelineConfig
from lingying.summarize import build_summary


class BuildSummaryTest(unittest.TestCase):
    def test_builds_chapter_from_screen_text(self):
        selected = [
            {
                "id": "s001",
                "start": 2.0,
                "end": 10.0,
                "score": 0.8,
                "keywords": ["存储器", "地址", "读写"],
                "text": (
                    "实验一存储器实验 实验目的 掌握随机存储器RAM的读写方法。"
                    "实验注意事项 严禁带电插拔接线。"
                ),
                "scene_id": 0,
            }
        ]
        result = build_summary(
            20.0,
            selected,
            [selected[0]["text"]],
            PipelineConfig(),
        )
        self.assertEqual(len(result["chapters"]), 1)
        self.assertEqual(result["chapters"][0]["title"], "实验注意事项")
        self.assertIn("存储器", result["overview"])


if __name__ == "__main__":
    unittest.main()
