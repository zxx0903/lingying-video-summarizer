from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from lingying.config import PipelineConfig
from lingying.evaluation import evaluate_result
from lingying.exports import write_docx_summary
from lingying.fusion import apply_template_prior
from lingying.select import select_segments
from lingying.summarize import build_summary
from lingying.templates import (
    SUMMARY_TEMPLATES,
    load_custom_templates,
    resolve_template_keywords,
    save_custom_template,
)


def _segments(count: int = 12, duration: float = 10.0) -> list[dict]:
    result = []
    for i in range(count):
        start = i * 15.0
        result.append(
            {
                "id": f"s{i + 1:03d}",
                "index": i + 1,
                "start": start,
                "end": start + duration,
                "score": 0.9 - i * 0.02,
                "scene_id": i,
                "cues": [],
                "tokens": [f"topic{i}", "knowledge"],
                "keywords": [f"topic{i}", "knowledge"],
                "text": f"topic{i} knowledge lesson explanation {i}",
            }
        )
    return result


class QualityRegressionTest(unittest.TestCase):
    def test_has_at_least_twenty_templates(self):
        self.assertGreaterEqual(len(SUMMARY_TEMPLATES), 20)
        self.assertIn("步骤", resolve_template_keywords("education_steps"))

    def test_custom_template_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "templates.json"
            saved = save_custom_template("算法复习", ["算法", "复杂度"], path=path)
            loaded = load_custom_templates(path)
            self.assertEqual(loaded[0].id, saved.id)
            self.assertIn("复杂度", loaded[0].keywords)

    def test_selection_reaches_target_tolerance(self):
        selected = select_segments(_segments(), target_seconds=100.0)
        total = sum(seg["end"] - seg["start"] for seg in selected)
        self.assertGreaterEqual(total, 95.0)
        self.assertLessEqual(total, 105.0)

    def test_chapter_count_is_bounded(self):
        selected = _segments(14)
        summary = build_summary(
            210.0,
            selected,
            [seg["text"] for seg in selected],
            PipelineConfig(max_chapters=4),
        )
        self.assertGreaterEqual(len(summary["chapters"]), 1)
        self.assertLessEqual(len(summary["chapters"]), 4)

    def test_template_prior_is_capped(self):
        segments = _segments(2)
        segments[0]["text"] = "重点 定义 总结 注意"
        before = segments[0]["score"]
        apply_template_prior(segments, ["重点", "定义", "总结", "注意"])
        self.assertGreater(segments[0]["score"], before)
        self.assertLessEqual(segments[0]["template_bonus"], 0.10)

    def test_docx_export_is_valid_zip_with_document(self):
        result = {
            "summary": {"overview": "摘要", "keywords": ["计算"], "chapters": []},
            "metrics": {"duration": 20, "selected_duration": 10, "processing_seconds": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_docx_summary(result, Path(tmp) / "summary.docx")
            with zipfile.ZipFile(path) as archive:
                self.assertIn("word/document.xml", archive.namelist())
                self.assertIn("视频摘要", archive.read("word/document.xml").decode("utf-8"))

    def test_temporal_evaluation_metrics(self):
        result = {
            "selected": [
                {"start": 10, "end": 20, "text": "alpha beta"},
                {"start": 40, "end": 50, "text": "alpha beta"},
            ],
            "metrics": {"target_seconds": 20},
            "summary": {"chapters": [{"title": "a"}]},
        }
        gt = {"important_segments": [{"start": 12, "end": 22}]}
        metrics = evaluate_result(result, gt)
        self.assertGreater(metrics["ground_truth_coverage"], 0.7)
        self.assertEqual(metrics["segment_recall"], 1.0)
        self.assertGreater(metrics["content_redundancy"], 0.0)


if __name__ == "__main__":
    unittest.main()
