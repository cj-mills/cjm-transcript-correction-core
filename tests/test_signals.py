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


def test_attention_tier_signals_and_compose():
    """The walk-lane attention tier (item 3758f6cb, narrowed by ruling 328c48ae on evidence
    1dbbc512): each pure signal derives the mark rows its class promises — mid-word
    boundary, trailing audio (overlap-clipped, so a straddling word fakes no tail), speech
    inside a gap, numeral adjacency, cut in continuous speech (silenced by a nearby event
    insert), speaker change — the three classes the second walk dismissed whenever they
    stood alone (fa-stretched-word, unexplained-gap, asr-extra-words) derive NOTHING and
    `divergence` is no longer a tier signal (the pure helper stays for other consumers),
    and `attention_marks` composes only the requested signals, sorted by time,
    deduplicated by key, loud on an unknown signal."""
    import pytest
    from cjm_transcript_correction_core.signals import (ATTENTION_DEFAULT_SIGNALS, ATTENTION_SIGNALS,
                                                        attention_boundary_marks,
                                                        attention_divergence_marks,
                                                        attention_fa_marks, attention_marks,
                                                        attention_speaker_marks)

    segs = [
        SpineSegment(id="a", index=0, text="we launch the kernel", start_time=0.0, end_time=2.0),
        SpineSegment(id="b", index=1, text="with 128", start_time=2.0, end_time=4.0),
        SpineSegment(id="c", index=2, text="threads per block", start_time=4.6, end_time=6.0),
        SpineSegment(id="d", index=3, text="", start_time=6.0, end_time=6.3),           # empty: never anchors
        SpineSegment(id="e", index=4, text="and then sync", start_time=6.3, end_time=9.0),
        SpineSegment(id="f", index=5, text="the grid", start_time=9.0, end_time=10.0),
        SpineSegment(id="g", index=6, text="done", start_time=12.0, end_time=13.0),      # 2.0 s silence before it
    ]
    words = [{"s": 0.1, "e": 0.4, "text": "we"}, {"s": 0.5, "e": 0.9, "text": "launch"},
             {"s": 1.0, "e": 1.3, "text": "the"}, {"s": 1.8, "e": 2.3, "text": "kernel"},   # straddles a|b
             {"s": 2.4, "e": 2.7, "text": "with"}, {"s": 2.8, "e": 3.9, "text": "128"},
             {"s": 4.2, "e": 4.4, "text": "stray"},                                          # inside the b|c gap
             {"s": 4.7, "e": 4.9, "text": "threads"}, {"s": 5.0, "e": 5.9, "text": "per block"},
             {"s": 6.4, "e": 6.6, "text": "and"}, {"s": 6.7, "e": 6.9, "text": "then"},
             {"s": 8.8, "e": 8.95, "text": "sync"}, {"s": 9.05, "e": 9.4, "text": "the"},
             {"s": 9.5, "e": 9.9, "text": "grid"}, {"s": 12.1, "e": 12.5, "text": "done"}]
    fa = attention_fa_marks(segs, words)
    by = {(r["mark_class"], tuple(sorted(r["anchor"].get(k) for k in ("boundary_after", "right_segment_id", "segment_id")
                                        if r["anchor"].get(k)))) for r in fa}
    assert ("fa-mid-word-boundary", ("a", "b")) in by          # 'kernel' 1.8-2.3 contains the 2.0 boundary
    assert not any(c == "fa-trailing-audio" for c, _ in by)   # a: the straddling word covers its tail; e: 'sync' ends 0.05 s before the end
    assert not any(c == "fa-leading-audio" for c, _ in by)

    b = attention_boundary_marks(segs, fa_words=words)
    bc = {(r["mark_class"], r["anchor"]["boundary_after"], r["anchor"]["right_segment_id"]) for r in b}
    assert ("speech-in-gap", "b", "c") in bc                    # 'stray' lies wholly inside the 0.6 s gap
    assert not any(c == "speech-in-gap" and (l, r) == ("a", "b") for c, l, r in bc)   # 'kernel' straddles: mid-word's finding
    assert not any(c == "unexplained-gap" for c, _, _ in bc)   # the 2.0 s silence before g no longer marks (ruling 328c48ae)
    assert ("numeral-adjacency", "b", "c") in bc                # '...128' | 'threads'
    assert ("cut-in-speech", "e", "f") in bc                    # 'sync' ends 8.95, 'the' starts 9.05, no breath
    assert not any(c == "cut-in-speech" and (l, r) == ("a", "b") for c, l, r in bc)
    b2 = attention_boundary_marks(segs, fa_words=words,
                                  events=[{"start": 8.97, "end": 9.02, "label": "inhale"},
                                          {"start": 10.2, "end": 11.9, "label": "background-noise"}])
    assert not any(r["mark_class"] == "cut-in-speech" for r in b2)   # the inhale explains the cut
    b3 = attention_boundary_marks(segs, fa_words=words,
                                  events=[{"start": 4.15, "end": 4.45, "label": "inhale"}])
    assert not any(r["mark_class"] == "speech-in-gap" for r in b3)   # 'stray' is the breath the aligner absorbed

    stretched = attention_fa_marks(
        [SpineSegment(id="p", index=0, text="we run", start_time=0.0, end_time=1.0),
         SpineSegment(id="q", index=1, text="performance", start_time=1.0, end_time=2.3)],
        [{"s": 0.1, "e": 0.3, "text": "we"}, {"s": 0.4, "e": 0.6, "text": "run"},
         {"s": 0.7, "e": 1.9, "text": "performance"}])                      # 1.2 s across the boundary
    assert [r["mark_class"] for r in stretched] == ["fa-mid-word-boundary"]   # a long word is just a word the boundary sits in

    d = attention_divergence_marks(segs, {"a": {"whisper": "we launch the colonel"},            # substitution: fidelity lane
                                          "c": {"whisper": "threads per block uh like"},         # fillers only
                                          "e": {"whisper": "and then sync the threads"},         # extra content words at the END edge
                                          "f": {"whisper": "the mighty grid"},                   # extra word MID-segment: fidelity lane
                                          "b": {"whisper": "with 128 128 128 128 128 128 128"}})  # runaway
    assert [r["anchor"]["segment_id"] for r in d] == ["e"]      # the pure helper still derives its rows ...
    assert "divergence" not in ATTENTION_SIGNALS               # ... but the tier no longer composes them

    turns = [{"start": 0.0, "end": 4.5, "speaker": "S0"}, {"start": 4.5, "end": 13.0, "speaker": "S1"}]
    sp = attention_speaker_marks(segs, turns)
    assert [(r["anchor"]["boundary_after"], r["anchor"]["right_segment_id"]) for r in sp] == [("b", "c")]

    rows = attention_marks(segs, fa_words=words, turns=turns)
    assert "speaker" not in ATTENTION_DEFAULT_SIGNALS and "cut" not in ATTENTION_DEFAULT_SIGNALS
    assert not any(r["mark_class"] in ("speaker-change", "cut-in-speech") for r in rows)
    assert {r["mark_class"] for r in rows} == {"fa-mid-word-boundary", "speech-in-gap", "numeral-adjacency"}
    assert any(r["mark_class"] == "cut-in-speech"
               for r in attention_marks(segs, fa_words=words, signals=("cut",)))   # still there when asked for
    assert [r["t"] for r in rows] == sorted(r["t"] for r in rows)
    assert len({r["key"] for r in rows}) == len(rows)
    rows2 = attention_marks(segs, fa_words=words, turns=turns, signals=("gap", "speaker"))
    assert {r["mark_class"] for r in rows2} == {"speech-in-gap", "speaker-change"}
    with pytest.raises(ValueError):
        attention_marks(segs, signals=("fa", "nope"))
    with pytest.raises(ValueError):
        attention_marks(segs, signals=("divergence",))          # retired from the tier: unknown now

    fa_trail = attention_fa_marks([SpineSegment(id="t", index=0, text="hello", start_time=0.0, end_time=3.0)],
                                  [{"s": 0.1, "e": 0.5, "text": "hello"}])
    assert [r["mark_class"] for r in fa_trail] == ["fa-trailing-audio"]
    fa_trail_ok = attention_fa_marks([SpineSegment(id="t", index=0, text="hello", start_time=0.0, end_time=3.0)],
                                     [{"s": 0.1, "e": 0.5, "text": "hello"}],
                                     events=[{"start": 0.6, "end": 2.9, "label": "background-noise"}])
    assert fa_trail_ok == []                                   # the insert explains the tail
