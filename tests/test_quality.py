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

    def test_target_ratio_is_honored_on_short_video(self):
        target = PipelineConfig(target_ratio=0.25).resolve_target(36.85)
        self.assertAlmostEqual(target, 9.2125)
        segments = _segments()
        for seg in segments:
            seg["cues"] = [{"start": seg["start"], "end": seg["end"]}]
        selected = select_segments(segments, target_seconds=target)
        total = sum(seg["end"] - seg["start"] for seg in selected)
        self.assertGreaterEqual(total, target * 0.95)
        self.assertLessEqual(total, target * 1.05)

    def test_summary_is_bounded_and_has_specific_title(self):
        text = (
            "Volumetric techniques. Measuring, essentially. "
            "Have you ever added too much pepper to a recipe? "
            "I know, it is not so great afterwards, but it is an innocent mistake. "
            "But in the lab, you have to be sure to have precise measurements. "
            "So we are going to learn how to use a pipette and a burette. "
            "Pipettes measure small quantities of volume, usually up to 10 ml. "
            "They work on the principle of creating a vacuum. "
            "The bulb is the wide part of a volumetric pipette."
        )
        selected = [
            {
                "id": "s001",
                "index": 1,
                "start": 0.0,
                "end": 30.0,
                "score": 0.8,
                "scene_id": 0,
                "cues": [],
                "tokens": ["volume", "pipette", "technique"],
                "keywords": ["volume", "pipette", "technique"],
                "text": text,
            }
        ]
        summary = build_summary(
            30.0,
            selected,
            [text],
            PipelineConfig(max_chapters=1),
        )
        chapter = summary["chapters"][0]
        self.assertIn("Volumetric", chapter["title"])
        self.assertLessEqual(len(chapter["summary"]), 320)
        self.assertLess(len(chapter["summary"]), len(text))
        self.assertNotEqual(chapter["title"].lower(), "sure")

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
