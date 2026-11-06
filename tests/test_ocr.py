from __future__ import annotations

import unittest

import numpy as np

from lingying.ocr import OcrSample, _screen_crops, merge_ocr_samples


class MergeOcrSamplesTest(unittest.TestCase):
    def test_merges_repeated_slide_text(self):
        cues = merge_ocr_samples(
            [
                OcrSample(1.0, "实验目的；掌握存储器读写"),
                OcrSample(2.0, "实验目的；掌握存储器读写。"),
                OcrSample(3.0, "实验步骤；写地址"),
            ],
            12.0,
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].start, 1.0)
        self.assertEqual(cues[0].end, 3.0)
        self.assertIn("存储器", cues[0].text)

    def test_last_slide_extends_to_video_end(self):
        cues = merge_ocr_samples([OcrSample(5.0, "地址锁存器")], 20.0)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].end, 20.0)

    def test_detects_bright_screen_region(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[90:310, 120:500] = 245
        crops = _screen_crops(image)
        self.assertTrue(crops)
        self.assertGreater(crops[0].shape[1], 500)


if __name__ == "__main__":
    unittest.main()
