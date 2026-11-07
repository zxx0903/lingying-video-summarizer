from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lingying.asr import (
    Cue,
    deduplicate_rolling_cues,
    find_sidecar_subtitle,
    parse_subtitle_text,
)


class SubtitleParsingTest(unittest.TestCase):
    def test_parses_vtt_without_hour_component(self):
        cues = parse_subtitle_text(
            """WEBVTT

00:01.250 --> 00:03.500 align:start
<c.yellow>First line</c>

00:03.500 --> 00:05.000
Second &amp; final
"""
        )
        self.assertEqual([cue.text for cue in cues], ["First line", "Second & final"])
        self.assertAlmostEqual(cues[0].start, 1.25)
        self.assertAlmostEqual(cues[1].end, 5.0)

    def test_deduplicates_rolling_caption_windows(self):
        cues = deduplicate_rolling_cues(
            [
                Cue(0.0, 2.0, "Concordances and KIC."),
                Cue(2.0, 4.0, "Concordances and KIC. A word rarely comes alone."),
                Cue(4.0, 6.0, "A word rarely comes alone."),
                Cue(6.0, 8.0, "A word rarely comes alone. In addition to methods,"),
                Cue(8.0, 10.0, "In addition to methods,"),
            ]
        )
        self.assertEqual(
            [cue.text for cue in cues],
            [
                "Concordances and KIC.",
                "A word rarely comes alone.",
                "In addition to methods,",
            ],
        )

    def test_deduplicates_single_word_rolling_caption(self):
        cues = deduplicate_rolling_cues(
            [
                Cue(0.0, 1.0, "Wrap your hand around the burette as"),
                Cue(1.0, 2.0, "shown."),
                Cue(2.0, 2.1, "shown."),
                Cue(2.1, 4.0, "shown. Gently push forward."),
            ]
        )
        self.assertEqual(
            [cue.text for cue in cues],
            [
                "Wrap your hand around the burette as",
                "shown.",
                "Gently push forward.",
            ],
        )

    def test_discovers_language_suffixed_srt_and_vtt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "lecture.mp4"
            video.touch()
            vtt = root / "lecture.en.vtt"
            vtt.write_text("WEBVTT\n\n00:01.000 --> 00:02.000\nHello\n", encoding="utf-8")
            self.assertEqual(find_sidecar_subtitle(video), vtt)

            srt = root / "lecture.en.srt"
            srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")
            self.assertEqual(find_sidecar_subtitle(video), srt)


if __name__ == "__main__":
    unittest.main()
