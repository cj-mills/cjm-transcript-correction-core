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


def test_speaker_turn_proposals_dominance_and_gaps():
    from cjm_transcript_correction_core.signals import speaker_turn_proposals
    segs = [
        SpineSegment(id="s0", index=0, text="a", start_time=0.0, end_time=10.0),
        SpineSegment(id="s1", index=1, text="b", start_time=10.0, end_time=20.0),
        SpineSegment(id="s2", index=2, text="c", start_time=100.0, end_time=110.0),  # no turn coverage
        SpineSegment(id="s3", index=3, text="d", start_time=None, end_time=None),    # no time span
    ]
    turns = [
        {"start": 0.0, "end": 8.0, "speaker": "SPEAKER_00"},
        {"start": 8.0, "end": 11.0, "speaker": "SPEAKER_01"},
        # overlapping speech: both turns cover 12-20; S01 dominates s1
        {"start": 12.0, "end": 20.0, "speaker": "SPEAKER_01"},
        {"start": 12.0, "end": 14.0, "speaker": "SPEAKER_00"},
    ]
    p = speaker_turn_proposals(segs, turns)
    assert p["s0"]["cluster"] == "SPEAKER_00" and p["s0"]["overlap"] == 8.0
    assert p["s0"]["coverage"] == 0.8
    assert p["s1"]["cluster"] == "SPEAKER_01"
    assert p["s1"]["overlap"] == 9.0  # 10-11 plus 12-20
    assert "s2" not in p and "s3" not in p
    assert speaker_turn_proposals(segs, []) == {}
    # empty-text chunks NEVER propose (drive ask 2026-07-27): text is the unit
    # of attribution supervision — silence chunks and inhale/bookend inserts
    # stay ∅ even under full turn coverage; a text-bearing split half proposes
    empty = SpineSegment(id="s4", index=4, text="  ", start_time=1.0, end_time=3.0)
    texty = SpineSegment(id="s5", index=4, text="split tail", start_time=1.0, end_time=3.0)
    p2 = speaker_turn_proposals([empty, texty], turns)
    assert "s4" not in p2 and p2["s5"]["cluster"] == "SPEAKER_00"
