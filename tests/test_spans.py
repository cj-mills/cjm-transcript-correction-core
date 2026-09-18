"""Tests for cjm_transcript_correction_core.spans — the overlay-proposal lane's
domain half (design bbf8bafd): the label slate, the span pack (row_kind span,
digest, brief), quoted-word resolution (whole tokens, nth, refusals), the span
proposal set (own format, loader gating both ways), the lexicon tier, the
pending worklist, the derived-verdict bench, and snap-at-accept with
re-anchoring. Pure — no runtime."""
import pytest

from cjm_transcript_correction_core.graph import build_speech_overlay_correction
from cjm_transcript_correction_core.models import RECOMMENDED_OVERLAY_LABELS, SpineSegment
from cjm_transcript_correction_core.spans import (FILTERABLE_OVERLAY_LABELS, HESITATION_LEXICON,
                                                  KEEP_OVERLAY_LABELS, LEXICON_ACTOR,
                                                  OVERLAY_GLOSSES, SPAN_LANE,
                                                  SPAN_PROPOSAL_SET_FORMAT, bench_span_proposals,
                                                  build_span_pack, find_token_span, is_span_pack,
                                                  lexicon_span_rows, load_span_proposal_sets,
                                                  overlay_label_slate, pending_span_proposals,
                                                  render_span_pack, render_span_propset_markdown,
                                                  snap_span_proposal, span_proposals_from_rows,
                                                  validate_span_rows, write_span_propset)
from cjm_transcript_correction_core.strata import (build_filter_pack, load_filter_proposal_sets,
                                                   validate_proposal_rows)

SEGS = [
    SpineSegment(id="s0", index=0, text="Um, so we have the the data center.", start_time=0.0, end_time=4.0),
    SpineSegment(id="s1", index=1, text="", start_time=4.0, end_time=4.4),   # wordless — left out
    SpineSegment(id="s2", index=2, text="You know, it's much, much larger, uh, than before.",
                 start_time=4.4, end_time=9.0),
    SpineSegment(id="s3", index=3, text="I think we— what we found was that that works.",
                 start_time=9.0, end_time=13.0),
]


def _overlay(oid, seg_id, cs, ce, snap_text, label, start, end, proposal_id=None, actor="human"):
    return {"id": oid, "correction_type": "annotation", "status": "applied", "actor": actor,
            "created_at": 1.0, "session_id": "sess",
            "payload": {"operation": "speech_overlay", "source_id": "src",
                        "anchor": {"kind": "span", "segment_id": seg_id, "char_start": cs,
                                   "char_end": ce, "text_snapshot": snap_text},
                        "label": label, "start_time": start, "end_time": end,
                        "text": snap_text, "words": [], "snap": "fa-word",
                        **({"proposal_id": proposal_id} if proposal_id else {})}}


def _pack(**kw):
    return build_span_pack("src", "Episode", "sha256:skel", SEGS, content_hash="sha256:media", **kw)


# ---- slate ----

def test_slate_states_the_keep_boundary_and_is_glossed():
    slate = overlay_label_slate()
    assert [v["label"] for v in slate] == list(FILTERABLE_OVERLAY_LABELS) + list(KEEP_OVERLAY_LABELS)
    assert all(v["gloss"] for v in slate)
    assert {v["label"] for v in slate if v["keep"]} == set(KEEP_OVERLAY_LABELS)
    assert set(OVERLAY_GLOSSES) >= set(RECOMMENDED_OVERLAY_LABELS)   # the app's menu slate is glossed too
    assert set(FILTERABLE_OVERLAY_LABELS) | set(KEEP_OVERLAY_LABELS) == set(RECOMMENDED_OVERLAY_LABELS)


# ---- pack ----

def test_span_pack_is_a_filter_pack_with_span_rows_and_a_distinct_digest():
    pack = _pack()
    assert is_span_pack(pack)
    assert [r["id"] for r in pack["segments"]] == ["s0", "s2", "s3"]   # the wordless line is out
    assert pack["window"] == {"start": 0.0, "end": 13.0}
    stratum = build_filter_pack("src", "Episode", "sha256:skel", SEGS, content_hash="sha256:media")
    assert not is_span_pack(stratum)
    assert pack["digest"] != stratum["digest"]   # a different READ of the same lines
    # the stratum lane refuses a span pack; the span lane refuses a stratum pack
    with pytest.raises(ValueError, match="span-ingest"):
        validate_proposal_rows([], pack)
    with pytest.raises(ValueError, match="not a span pack"):
        validate_span_rows([], stratum)


def test_span_pack_renders_existing_overlays_as_do_not_re_propose_lines():
    ov = _overlay("o1", "s0", 0, 3, "Um,", "hesitation-marker", 0.0, 0.3)
    foreign = _overlay("o9", "sX", 0, 2, "so", "discourse-marker", 99.0, 99.5)   # not on this spine
    pack = _pack(overlays=[ov, foreign], margin=1, window=(4.0, None))
    assert [e["overlay_id"] for e in pack["existing_overlays"]] == []   # s0 is before the window
    pack = _pack(overlays=[ov, foreign])
    assert [(e["i"], e["label"], e["text"]) for e in pack["existing_overlays"]] == [
        (0, "hesitation-marker", "Um,")]
    brief = render_span_pack(pack)
    assert "line 0: “Um,” — `hesitation-marker` (by human) — do not re-propose" in brief
    assert "**(KEEP — never filtered)**" in brief and "`emphasis-repeat`" in brief
    assert '"nth"' in brief and "[2] 00:09.0–00:13.0  I think we—" in brief


# ---- quoted-word resolution ----

def test_find_token_span_matches_whole_words_case_and_punctuation_insensitive():
    from cjm_transcript_correction_core.spine import segment_word_tokens
    toks = segment_word_tokens(SEGS[2].text)   # You know, it's much, much larger, uh, than before.
    assert find_token_span(toks, "you know") == (0, 1)
    assert find_token_span(toks, "You know,") == (0, 1)
    assert find_token_span(toks, "uh") == (6, 6)
    assert find_token_span(toks, "much, much") == (3, 4)
    with pytest.raises(ValueError, match="occurs 2 times"):
        find_token_span(toks, "much")
    assert find_token_span(toks, "much", nth=2) == (4, 4)
    with pytest.raises(ValueError, match="occurs 2 time"):
        find_token_span(toks, "much", nth=3)
    with pytest.raises(ValueError, match="not found as whole words"):
        find_token_span(toks, "kno")   # a substring is not a word
    with pytest.raises(ValueError, match="no matchable words"):
        find_token_span(toks, "—")


def test_validate_span_rows_resolves_and_refuses_with_row_numbers():
    pack = _pack()
    rows = [
        {"label": "hesitation-marker", "i": 0, "text": "Um", "tier": 1, "confidence": 0.9},
        {"label": "word-repeat", "i": 0, "text": "the the", "tier": 1},
        {"label": "emphasis-repeat", "i": 1, "text": "much, much", "tier": 2, "rationale": "rhetorical"},
        {"label": "hesitation-marker", "i": 1, "text": "uh", "nth": 1},
        {"label": "false-start", "i": 2, "text": "I think we—"},
    ]
    valid = validate_span_rows(rows, pack)
    assert [(v["i"], v["char_start"], v["char_end"], v["text_snapshot"]) for v in valid] == [
        (0, 0, 3, "Um,"), (0, 15, 22, "the the"), (1, 15, 25, "much, much"), (1, 34, 37, "uh,"),
        (2, 0, 11, "I think we—")]
    assert valid[2]["tier"] == 2 and valid[0]["confidence"] == 0.9
    with pytest.raises(ValueError, match="row 1: label 'stutter' is not in the pack's slate"):
        validate_span_rows([{"label": "stutter", "i": 0, "text": "the the"}], pack)
    with pytest.raises(ValueError, match="row 1: line 7 outside"):
        validate_span_rows([{"label": "word-repeat", "i": 7, "text": "the the"}], pack)
    with pytest.raises(ValueError, match="row 1: line 1: text 'much' occurs 2 times"):
        validate_span_rows([{"label": "word-repeat", "i": 1, "text": "much"}], pack)
    with pytest.raises(ValueError, match="row 2: duplicate word-repeat span on line 0"):
        validate_span_rows([{"label": "word-repeat", "i": 0, "text": "the the"},
                            {"label": "word-repeat", "i": 0, "text": "The the"}], pack)
    # different labels may nest (a hesitation inside a repeat is the human's practice)
    assert len(validate_span_rows([{"label": "word-repeat", "i": 0, "text": "the the"},
                                   {"label": "hesitation-marker", "i": 0, "text": "the", "nth": 1}],
                                  pack)) == 2
    with pytest.raises(ValueError, match="row 1: tier must be 1 or 2"):
        validate_span_rows([{"label": "word-repeat", "i": 0, "text": "the the", "tier": 3}], pack)


def test_span_proposals_carry_the_anchor_and_estimated_times_in_line_order():
    pack = _pack()
    valid = validate_span_rows([{"label": "hesitation-marker", "i": 1, "text": "uh"},
                                {"label": "discourse-marker", "i": 1, "text": "you know"},
                                {"label": "hesitation-marker", "i": 0, "text": "um"}], pack)
    props = span_proposals_from_rows(valid, pack)
    assert [p["evidence"]["i"] for p in props] == [0, 1, 1]
    assert [p["label"] for p in props] == ["hesitation-marker", "discourse-marker", "hesitation-marker"]
    p = props[0]
    assert p["anchor"] == {"kind": "span", "segment_id": "s0", "char_start": 0, "char_end": 3,
                           "text_snapshot": "Um,"}
    assert p["snap"] == "estimated" and p["category"] == p["label"] and p["index"] == 0
    assert p["start_time"] == 0.0 and 0.0 < p["end_time"] < 1.0   # char fraction over 0..4 s
    assert props[1]["start_time"] < props[2]["start_time"]         # you know before uh


# ---- set write / load — format-gated both ways ----

def test_span_set_round_trip_is_never_a_stratum_set(tmp_path):
    pack = _pack()
    valid = validate_span_rows([{"label": "hesitation-marker", "i": 0, "text": "um"}], pack)
    props = span_proposals_from_rows(valid, pack)
    res = write_span_propset(pack, props, out_root=tmp_path / "proposals",
                             proposer={"kind": "claude-code-subagent", "name": "span-pass"})
    assert res["counts"] == {"hesitation-marker": 1}
    sets = load_span_proposal_sets(str(tmp_path), "src", skeleton_hash="sha256:skel")
    assert len(sets) == 1
    m = sets[0]["manifest"]
    assert m["format"] == SPAN_PROPOSAL_SET_FORMAT
    assert m["config"] == {"lane": SPAN_LANE,
                           "vocabulary": list(FILTERABLE_OVERLAY_LABELS) + list(KEEP_OVERLAY_LABELS)}
    assert m["pack"]["digest"] == pack["digest"]
    assert sets[0]["proposals"][0]["anchor"]["text_snapshot"] == "Um,"
    assert load_filter_proposal_sets(str(tmp_path), "src") == []   # the stratum lane never sees it
    md = render_span_propset_markdown(m, props, pack)
    assert "**⟦Um,⟧** so we have" in md and "hesitation-marker" in md


# ---- the lexicon tier ----

def test_lexicon_rows_propose_bare_hesitations_with_nth_and_validate():
    pack = build_span_pack("src", "t", None, [
        SpineSegment(id="a", index=0, text="Um, uh, we, um, like tensors.", start_time=0.0, end_time=2.0),
        SpineSegment(id="b", index=1, text="Umbrella time.", start_time=2.0, end_time=3.0),
        SpineSegment(id="c", index=2, text="Uh, um.", start_time=3.0, end_time=3.5)])   # wholly filler: not an overlay
    rows = lexicon_span_rows(pack)
    assert [(r["i"], r["text"], r["nth"]) for r in rows] == [(0, "Um,", 1), (0, "uh,", 1), (0, "um,", 2)]
    assert all(r["tier"] == 1 and r["label"] == "hesitation-marker" for r in rows)
    assert "like" not in HESITATION_LEXICON and LEXICON_ACTOR.startswith("capability:")
    valid = validate_span_rows(rows, pack)
    assert [(v["char_start"], v["char_end"]) for v in valid] == [(0, 3), (4, 7), (12, 15)]


# ---- pending + bench ----

def _props():
    pack = _pack()
    valid = validate_span_rows([
        {"label": "hesitation-marker", "i": 0, "text": "um"},              # accepted by id carry
        {"label": "word-repeat", "i": 0, "text": "the the"},              # accepted: same words, hand-made
        {"label": "discourse-marker", "i": 1, "text": "you know"},        # relabeled by the human
        {"label": "hesitation-marker", "i": 1, "text": "uh"},             # rejected (below watermark)
        {"label": "false-start", "i": 2, "text": "I think"},              # edited (human took 'I think we—')
        {"label": "coincidental-repeat", "i": 2, "text": "that that", "tier": 2},   # unvisited-ish tier 2
    ], pack)
    return pack, span_proposals_from_rows(valid, pack)


def test_pending_hides_materialized_by_id_or_same_label_overlap():
    pack, props = _props()
    pid = props[0]["proposal_id"]
    overlays = [_overlay("o1", "s0", 0, 3, "Um,", "hesitation-marker", 0.0, 0.3, proposal_id=pid),
                _overlay("o2", "s0", 15, 22, "the the", "word-repeat", 1.5, 2.2)]
    pend = pending_span_proposals(props, overlays)
    assert [p["label"] for p in pend] == ["discourse-marker", "hesitation-marker", "false-start"]
    assert len(pending_span_proposals(props, overlays, show_tier2=True)) == 4


def test_bench_derives_the_span_verdicts_and_missed():
    pack, props = _props()
    pid = props[0]["proposal_id"]
    overlays = [
        _overlay("o1", "s0", 0, 3, "Um,", "hesitation-marker", 0.0, 0.3, proposal_id=pid),
        _overlay("o2", "s0", 15, 22, "the the", "word-repeat", 1.5, 2.2),
        _overlay("o3", "s2", 0, 9, "You know,", "hesitation-marker", 4.4, 4.9),   # human's label differs
        _overlay("o4", "s3", 0, 11, "I think we—", "false-start", 9.0, 9.9),       # human widened it
        _overlay("o5", "s2", 26, 33, "larger,", "word-repeat", 7.0, 7.4),          # nothing proposed here
    ]
    b = bench_span_proposals(props, overlays, (0.0, None), watermark=13.0)
    by = {v["label"] + "@" + str(v["segment_id"]): v["verdict"] for v in b["verdicts"]}
    assert by["hesitation-marker@s0"] == "accepted"
    assert by["word-repeat@s0"] == "accepted"
    assert by["discourse-marker@s2"] == "relabeled"
    assert by["hesitation-marker@s2"] == "rejected"
    assert by["false-start@s3"] == "edited"
    assert by["coincidental-repeat@s3"] == "unaccepted"
    assert b["counts"]["tier1"] == {"accepted": 2, "edited": 1, "relabeled": 1, "rejected": 1, "unvisited": 0}
    assert b["rates"] == {"accepted": 0.4, "edited": 0.2, "relabeled": 0.2, "rejected": 0.2}
    assert [m["overlay_id"] for m in b["missed"]] == ["o5"]
    # no watermark = nothing visited: the unmatched tier-1 row reads unvisited, not rejected
    b2 = bench_span_proposals(props, overlays, (0.0, None))
    assert b2["counts"]["tier1"]["unvisited"] == 1 and b2["counts"]["tier1"]["rejected"] == 0


# ---- snap at accept ----

FA = [{"s": 0.0, "e": 0.3, "text": "um"}, {"s": 0.3, "e": 0.5, "text": "so"},
      {"s": 0.5, "e": 0.7, "text": "we"}, {"s": 0.7, "e": 0.9, "text": "have"},
      {"s": 0.9, "e": 1.1, "text": "the"}, {"s": 1.1, "e": 1.3, "text": "the"},
      {"s": 1.3, "e": 1.8, "text": "data"}, {"s": 1.8, "e": 2.4, "text": "center"}]


def test_snap_span_proposal_snaps_from_fa_words_and_reanchors_after_an_edit():
    pack, props = _props()
    rep = props[1]   # word-repeat 'the the' on s0
    rec = snap_span_proposal(rep, SEGS[0], FA)
    assert rec["anchor"]["text_snapshot"] == "the the" and rec["label"] == "word-repeat"
    assert (rec["start_time"], rec["end_time"], rec["snap"]) == (0.9, 1.3, "fa-word")
    assert [w["text"] for w in rec["words"]] == ["the", "the"]
    # the line was edited since the pack (a word inserted before the span): re-anchor by snapshot
    edited = SpineSegment(id="s0", index=0, text="Um, so we now have the the data center.",
                          start_time=0.0, end_time=4.0)
    rec2 = snap_span_proposal(rep, edited, None, label="emphasis-repeat")
    assert rec2["anchor"]["char_start"] == 19 and rec2["anchor"]["text_snapshot"] == "the the"
    assert rec2["label"] == "emphasis-repeat" and rec2["snap"] == "estimated"
    # the words are gone: refuse (the human already subtracted them by hand)
    gone = SpineSegment(id="s0", index=0, text="Um, so we have the data center.", start_time=0.0, end_time=4.0)
    with pytest.raises(ValueError, match="no longer on #0"):
        snap_span_proposal(rep, gone, FA)
    with pytest.raises(ValueError, match="anchors segment s0"):
        snap_span_proposal(rep, SEGS[2], FA)


def test_overlay_correction_carries_proposal_provenance():
    node, edges = build_speech_overlay_correction(
        "src", {"kind": "span", "segment_id": "s0", "char_start": 15, "char_end": 22,
                "text_snapshot": "the the"}, "word-repeat", 0.9, 1.3, "the the", "sess",
        snap="fa-word", proposal_id="p-1", proposal_set_id="set-1")
    pl = node["properties"]["payload"]
    assert pl["proposal_id"] == "p-1" and pl["proposal_set_id"] == "set-1"
    node2, _ = build_speech_overlay_correction(
        "src", {"kind": "span", "segment_id": "s0", "char_start": 15, "char_end": 22,
                "text_snapshot": "the the"}, "word-repeat", 0.9, 1.3, "the the", "sess")
    assert "proposal_id" not in node2["properties"]["payload"]   # a hand overlay is byte-compatible
