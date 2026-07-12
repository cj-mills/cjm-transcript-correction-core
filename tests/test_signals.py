"""Tests for cjm_transcript_correction_core.signals — pure deterministic Tier-1 signals.

Projected from the signals notebook's smoke-check cell at the golden-reference flip."""
from cjm_transcript_correction_core.models import SpineSegment
from cjm_transcript_correction_core.signals import (
    boundary_punct_caps_flags,
    cluster_variants,
    compute_signal_flags,
    detect_empty_segments,
    fa_coverage_flags,
    levenshtein,
    phonetic_key,
    variant_divergence,
)

SEGS = [
    SpineSegment(id="0", index=0, text="The art of war", start_time=0.0, end_time=1.0),
    SpineSegment(id="1", index=1, text="", start_time=1.0, end_time=1.2),
    SpineSegment(id="2", index=2, text="is of vital importance.", start_time=1.2, end_time=2.0),
    SpineSegment(id="3", index=3, text="the general who wins", start_time=2.0, end_time=3.0),
]


def test_empty_and_coverage_flags():
    assert detect_empty_segments(SEGS) == [1]
    assert "empty-text" in fa_coverage_flags(SEGS)[1]


def test_boundary_punct_caps_flags():
    # 2->3: "...importance." terminal, "the general..." lowercase -> terminal-then-lowercase
    b = boundary_punct_caps_flags(SEGS)
    assert "boundary-terminal-then-lowercase" in b.get(2, [])


def test_clustering_primitives():
    assert levenshtein("nickel", "nccl") >= 1
    assert phonetic_key("nickel") == phonetic_key("nichol")  # like-sounding bucket
    assert isinstance(cluster_variants(["ChatGPT", "Chachi", "unrelated"]), list)


def test_variant_divergence_within_segment():
    # stage 5: divergence is WITHIN-SEGMENT (variant slices), not a second spine
    variants = {
        "0": {"voxtral": "The art of war", "whisper": "The art of war"},   # agreement
        "2": {"voxtral": "is of vital importance.", "whisper": "is of VITAL stuff."},  # divergence
    }
    d = variant_divergence(SEGS, variants)
    assert 2 in d and 0 not in d
    assert d[2][1] == "is of VITAL stuff."


def test_compute_signal_flags_combined():
    variants = {
        "0": {"voxtral": "The art of war", "whisper": "The art of war"},
        "2": {"voxtral": "is of vital importance.", "whisper": "is of VITAL stuff."},
    }
    flags = compute_signal_flags(SEGS, variants=variants)
    assert 1 in flags and "transcriber-divergence" in flags.get(2, [])
    assert "transcriber-divergence" not in flags.get(0, [])
