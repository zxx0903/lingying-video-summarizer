import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "funasr-paraformer"
    / "tools"
    / "funasr_json_to_srt.py"
)


@pytest.fixture(scope="module")
def converter_module():
    spec = importlib.util.spec_from_file_location("funasr_json_to_srt", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_short_segment_preserves_all_wrapped_text(converter_module):
    cues = converter_module.split_segment_to_cues(
        text="aa bb cc dd ee ff",
        start_ms=0,
        end_ms=600,
        max_line_length=2,
        lines_per_cue=1,
    )

    assert [cue_text for _, _, cue_text in cues] == ["aa", "bb", "cc", "dd", "ee", "ff"]
    assert all(cue_end > cue_start for cue_start, cue_end, _ in cues)
    assert cues[-1][1] == 600


def test_too_short_segment_regroups_text_without_dropping_lines(converter_module):
    cues = converter_module.split_segment_to_cues(
        text="aa bb cc dd ee ff",
        start_ms=0,
        end_ms=3,
        max_line_length=2,
        lines_per_cue=1,
    )

    assert len(cues) == 3
    assert all(cue_end > cue_start for cue_start, cue_end, _ in cues)
    assert [line for _, _, cue_text in cues for line in cue_text.splitlines()] == [
        "aa",
        "bb",
        "cc",
        "dd",
        "ee",
        "ff",
    ]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_non_positive_values(converter_module, value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        converter_module.positive_int(value)


def test_wrap_text_rejects_non_positive_max_line_length(converter_module):
    with pytest.raises(ValueError, match="max_line_length"):
        converter_module.wrap_text("text", 0)


def test_split_rejects_non_positive_lines_per_cue(converter_module):
    with pytest.raises(ValueError, match="lines_per_cue"):
        converter_module.split_segment_to_cues("text", 0, 1000, 20, 0)
