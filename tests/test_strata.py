"""Tests for cjm_transcript_correction_core.strata — the filtering lane's domain
half (DECs 304fd984 + 9d4c0a38): pack build/render, proposer-row validation,
proposal-set write/load round-trip, the stratum op, the pending worklist, the
derived-verdict bench join, the per-consumer exclusion query, and the gate's
lane fold. Pure — no runtime."""
import json

import pytest

from cjm_transcript_correction_core.graph import (build_stratum_correction,
                                                  build_extraction_gate_assertion,
                                                  build_text_correction,
                                                  corrections_to_edits,
                                                  latest_extraction_gates)
from cjm_transcript_correction_core.models import RECOMMENDED_STRATUM_CLASSES, SpineSegment
from cjm_transcript_correction_core.strata import (FILTER_LANE, FILTER_MARK_GLOSSES,
                                                   FILTER_PACK_FORMAT,
                                                   FILTER_PACK_VERSION,
                                                   FILTER_PACK_VERSION_LADDER,
                                                   FILTER_PROPOSAL_SET_FORMAT,
                                                   active_strata, bench_filter_proposals,
                                                   build_filter_pack, exclude_strata,
                                                   load_filter_proposal_sets,
                                                   merge_filter_proposals, pack_digest,
                                                   pending_filter_proposals, plan_pack_windows,
                                                   proposals_from_rows, render_filter_pack,
                                                   render_filter_propset_markdown,
                                                   select_span_segments,
                                                   STRATUM_GLOSSES, validate_proposal_rows,
                                                   write_filter_propset)

SEGS = [
    SpineSegment(id="s0", index=0, text="Opening credits, read by the author.", start_time=0.0, end_time=3.0),
    SpineSegment(id="s1", index=1, text="", start_time=3.0, end_time=3.4),   # wordless — left out of packs
    SpineSegment(id="s2", index=2, text="Today we talk about learning.", start_time=3.4, end_time=6.0),
    SpineSegment(id="s3", index=3, text="By the way, I use Notion for this.", start_time=6.0, end_time=8.5),
    SpineSegment(id="s4", index=4, text="Back to the main idea.", start_time=8.5, end_time=11.0),
    SpineSegment(id="s5", index=5, text="This episode is sponsored by X.", start_time=11.0, end_time=14.0),
]


def _pack(**kw):
    return build_filter_pack("src", "Chapter 1", "sha256:skel", SEGS, content_hash="sha256:media", **kw)


def _stratum(sid, cat, ids, start, end, proposal_id=None, created_at=1.0, status="applied"):
    return {"id": sid, "correction_type": "stratum", "status": status, "actor": "human",
            "created_at": created_at,
            "payload": {"operation": "classify", "source_id": "src", "category": cat,
                        "segment_ids": ids, "start_time": start, "end_time": end,
                        "proposal_id": proposal_id}}


# ---- vocabulary ----

def test_recommended_stratum_classes_are_glossed_class_tokens():
    assert all(c[:1].isalnum() for c in RECOMMENDED_STRATUM_CLASSES)
    assert len(set(RECOMMENDED_STRATUM_CLASSES)) == len(RECOMMENDED_STRATUM_CLASSES)
    assert set(RECOMMENDED_STRATUM_CLASSES) <= set(STRATUM_GLOSSES)
    assert "main-topic" not in RECOMMENDED_STRATUM_CLASSES   # absence IS main-topic
    # quotation: proposer-minted on LG ch04 (8 block quotes), ratified by the user 2026-09-01
    assert "quotation" in RECOMMENDED_STRATUM_CLASSES and "delimiters" in STRATUM_GLOSSES["quotation"]
    # finding 353394c8: disfluency marks content-bearing runs; filler is the wholly elidable class
    assert "filler" in RECOMMENDED_STRATUM_CLASSES and "NO content" in STRATUM_GLOSSES["filler"]
    assert "content stays" in STRATUM_GLOSSES["disfluency"]


def test_select_span_segments_is_containment_over_text_segments():
    """The span-edit gesture: the human re-states a run as a time span over the
    CURRENT effective view — contained text segments only (a neighbour touching
    the edge stays out; empties carry nothing to classify), spine order."""
    run = select_span_segments(SEGS, 3.4, 8.5)
    assert [s.id for s in run] == ["s2", "s3"]                # s1 empty, s4 starts AT 8.5 (out)
    assert [s.id for s in select_span_segments(SEGS, 3.0, 8.5)] == ["s2", "s3"]   # s1 in-span but empty
    assert [s.id for s in select_span_segments(SEGS, 3.43, 8.47)] == ["s2", "s3"]  # tolerance absorbs 0.05 jitter
    assert [s.id for s in select_span_segments(SEGS, 3.6, 8.5)] == ["s3"]         # s2 not contained
    assert select_span_segments(SEGS, 20.0, 30.0) == []
    shuffled = list(reversed(SEGS))
    assert [s.id for s in select_span_segments(shuffled, 0.0, 14.0)] == ["s0", "s2", "s3", "s4", "s5"]


# ---- pack ----

def test_pack_numbers_text_segments_only_and_binds_source():
    pack = _pack()
    assert pack["format"] == FILTER_PACK_FORMAT
    rows = pack["segments"]
    assert [r["id"] for r in rows] == ["s0", "s2", "s3", "s4", "s5"]   # s1 (empty) skipped
    assert [r["i"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[1]["index"] == 2                               # spine index rides along
    assert pack["source"] == {"source_id": "src", "title": "Chapter 1",
                              "content_hash": "sha256:media", "skeleton_hash": "sha256:skel"}
    assert pack["window"] == {"start": 0.0, "end": 14.0}
    assert [v["category"] for v in pack["vocabulary"]] == list(RECOMMENDED_STRATUM_CLASSES)
    assert pack["digest"] == pack_digest(pack) and pack["digest"].startswith("sha256:")


def test_pack_window_clips_and_existing_strata_render_in_pack_coordinates():
    strata = [_stratum("st1", "apparatus", ["s0"], 0.0, 3.0),
              _stratum("st2", "sponsor", ["s5"], 11.0, 14.0)]
    pack = _pack(window=(3.0, 9.0), strata=strata)
    assert [r["id"] for r in pack["segments"]] == ["s2", "s3", "s4"]
    assert pack["window"] == {"start": 3.0, "end": 9.0}
    assert pack["existing_strata"] == []       # both strata fall outside the window
    full = _pack(strata=strata)
    assert [(e["category"], e["from_i"], e["to_i"]) for e in full["existing_strata"]] \
        == [("apparatus", 0, 0), ("sponsor", 4, 4)]


def test_pack_digest_ignores_ids_and_timestamps():
    a, b = _pack(), _pack()
    assert a["pack_id"] != b["pack_id"] and a["digest"] == b["digest"]


def test_render_filter_pack_carries_brief_contract_and_lines():
    strata = [_stratum("st1", "apparatus", ["s0"], 0.0, 3.0)]
    md = render_filter_pack(_pack(strata=strata))
    assert "## Vocabulary" in md and "`tangent`" in md and STRATUM_GLOSSES["tangent"] in md
    assert "## Output contract" in md and '"from_i"' in md
    assert "`apparatus` lines 0–0" in md and "do not re-propose" in md
    assert "[2] 00:06.0–00:08.5  By the way, I use Notion for this." in md
    late = build_filter_pack("src", "t", None, [SpineSegment(id="x", index=0, text="late", start_time=59.96, end_time=119.99)])
    assert "[0] 01:00.0–02:00.0  late" in render_filter_pack(late)   # never 00:60.0
    assert md.index("## Vocabulary") < md.index("## Output contract") < md.index("## Transcript")


# ---- proposer rows ----

def test_validate_rows_normalizes_and_rejects_contract_breaks():
    pack = _pack()
    ok = validate_proposal_rows([
        {"category": "tool-mention", "from_i": 2, "to_i": 2, "confidence": 0.9,
         "rationale": "Notion", "quote": "I use Notion"},
        {"category": "sponsor", "from_i": 4, "to_i": 4, "tier": 2},
    ], pack)
    assert ok[0]["tier"] == 1 and ok[0]["confidence"] == 0.9
    assert ok[1]["tier"] == 2 and ok[1]["confidence"] is None and ok[1]["rationale"] is None
    with pytest.raises(ValueError, match="row 1: category"):
        validate_proposal_rows([{"category": "-bad", "from_i": 0, "to_i": 0}], pack)
    with pytest.raises(ValueError, match="outside the pack"):
        validate_proposal_rows([{"category": "tangent", "from_i": 0, "to_i": 9}], pack)
    with pytest.raises(ValueError, match="inverted"):
        validate_proposal_rows([{"category": "tangent", "from_i": 3, "to_i": 1}], pack)
    with pytest.raises(ValueError, match="tier"):
        validate_proposal_rows([{"category": "tangent", "from_i": 0, "to_i": 0, "tier": 3}], pack)
    with pytest.raises(ValueError, match="confidence"):
        validate_proposal_rows([{"category": "tangent", "from_i": 0, "to_i": 0, "confidence": 1.5}], pack)
    with pytest.raises(ValueError, match="overlaps"):
        validate_proposal_rows([{"category": "tangent", "from_i": 0, "to_i": 2},
                                {"category": "tangent", "from_i": 2, "to_i": 3}], pack)
    # different categories may overlap (a sponsor read that is also a tool mention)
    validate_proposal_rows([{"category": "sponsor", "from_i": 4, "to_i": 4},
                            {"category": "tool-mention", "from_i": 4, "to_i": 4}], pack)


def test_proposals_from_rows_resolve_pack_positions_to_spine_identity():
    pack = _pack()
    rows = validate_proposal_rows([
        {"category": "sponsor", "from_i": 4, "to_i": 4, "quote": "sponsored by X"},
        {"category": "apparatus", "from_i": 0, "to_i": 1, "tier": 2, "confidence": 0.4},
    ], pack)
    props = proposals_from_rows(rows, pack)
    assert [p["category"] for p in props] == ["apparatus", "sponsor"]   # time order
    a, s = props
    assert a["segment_ids"] == ["s0", "s2"] and a["start_time"] == 0.0 and a["end_time"] == 6.0
    assert s["segment_ids"] == ["s5"] and s["evidence"] == {
        "pack_id": pack["pack_id"], "from_i": 4, "to_i": 4, "quote": "sponsored by X"}
    assert s["label"] == "sponsor" and a["score"] == 0.4 and a["tier"] == 2
    assert len({p["proposal_id"] for p in props}) == 2


# ---- proposal set round-trip ----

def test_write_and_load_filter_propset(tmp_path):
    pack = _pack()
    props = proposals_from_rows(validate_proposal_rows([
        {"category": "sponsor", "from_i": 4, "to_i": 4},
        {"category": "tool-mention", "from_i": 2, "to_i": 2, "tier": 2},
    ], pack), pack)
    root = tmp_path / "proposals"
    res = write_filter_propset(pack, props, out_root=root,
                               proposer={"kind": "claude-code-subagent", "name": "reader-1"})
    m = json.loads((tmp_path / "proposals" / res["set_id"] / "manifest.json").read_text())
    assert m["format"] == FILTER_PROPOSAL_SET_FORMAT
    assert m["source"]["skeleton_hash"] == "sha256:skel" and m["window"]["end"] == 14.0
    assert m["pack"] == {"pack_id": pack["pack_id"], "digest": pack["digest"], "segments": 5}
    assert m["model"]["kind"] == "claude-code-subagent"
    assert m["counts"] == {"sponsor": 1} and m["tier2_counts"] == {"tool-mention": 1}
    assert m["classes"] == ["sponsor", "tool-mention"]
    # a foreign-format set in the same root is ignored; another source's set too
    (root / "other").mkdir()
    (root / "other" / "manifest.json").write_text(json.dumps({"format": "x/other"}))
    write_filter_propset(build_filter_pack("src2", "t", None, SEGS), [], out_root=root,
                         proposer={"kind": "api"})
    sets = load_filter_proposal_sets(str(tmp_path), "src")
    assert len(sets) == 1 and sets[0]["manifest"]["proposal_set_id"] == res["set_id"]
    assert [p["category"] for p in sets[0]["proposals"]] == ["tool-mention", "sponsor"]  # time order
    assert load_filter_proposal_sets(str(tmp_path), "src", skeleton_hash="sha256:else") == []
    assert load_filter_proposal_sets(str(tmp_path / "nowhere"), "src") == []


def test_render_filter_propset_markdown_joins_pack_runs(tmp_path):
    pack = _pack()
    props = proposals_from_rows(validate_proposal_rows([
        {"category": "sponsor", "from_i": 4, "to_i": 4, "confidence": 0.8,
         "rationale": "A sponsor read.", "quote": "sponsored by X"},
        {"category": "apparatus", "from_i": 0, "to_i": 1, "tier": 2},
    ], pack), pack)
    res = write_filter_propset(pack, props, out_root=tmp_path / "proposals",
                               proposer={"kind": "api", "name": "m"})
    manifest = json.loads((tmp_path / "proposals" / res["set_id"] / "manifest.json").read_text())
    md = render_filter_propset_markdown(manifest, props, pack)
    assert md.startswith("# Filtering proposals — Chapter 1")
    # time order: apparatus (0.0s) before sponsor (11.0s); spine index range from the pack
    assert md.index("**apparatus**") < md.index("**sponsor**")
    assert "`??` **apparatus** · 00:00.0–00:06.0 · spine 0–2" in md
    assert "`?` **sponsor** · 00:11.0–00:14.0 · spine 5–5 · c=0.80" in md
    assert "**Why:** A sponsor read." in md and "**Quote:** “sponsored by X”" in md
    # the run is bold, the context line either side is italic
    assert "> **[4]** 00:11.0 · spine 5 — This episode is sponsored by X." in md
    assert "> [3] 00:08.5 · spine 4 — _Back to the main idea._" in md
    # without a pack the set still renders, by pack lines
    assert "pack lines 4..4" in render_filter_propset_markdown(manifest, props, None)


# ---- the stratum op ----

def test_build_stratum_correction_shape_and_no_effective_edit():
    node, edges = build_stratum_correction(
        "src", ["s3"], "tool-mention", "sess", skeleton_hash="sha256:skel",
        start_time=6.0, end_time=8.5, proposal_id="p1", proposal_set_id="set1",
        actor="human", note="Notion")
    props = node["properties"]
    assert node["label"] == "Correction" and props["correction_type"] == "stratum"
    assert props["payload"]["operation"] == "classify"
    assert props["payload"]["category"] == "tool-mention"
    assert props["payload"]["segment_ids"] == ["s3"] and props["payload"]["proposal_id"] == "p1"
    assert props["rationale"] == "Notion"
    assert [(e["relation_type"], e["target_id"]) for e in edges] == [("CORRECTS", "s3")]
    # a stratum never touches the effective view
    d = dict(props); d["id"] = node["id"]
    assert corrections_to_edits([d]) == []
    # reclassify = supersession
    _n2, e2 = build_stratum_correction("src", ["s3"], "tangent", "sess",
                                       supersedes_id=node["id"])
    assert ("SUPERSEDES", node["id"]) in [(e["relation_type"], e["target_id"]) for e in e2]
    with pytest.raises(ValueError):
        build_stratum_correction("src", [], "tangent", "sess")
    with pytest.raises(ValueError):
        build_stratum_correction("src", ["s3"], "-", "sess")


def test_active_strata_and_exclusion_query():
    a = _stratum("a", "sponsor", ["s5"], 11.0, 14.0)
    b = _stratum("b", "tool-mention", ["s3"], 6.0, 8.5)
    c = _stratum("c", "tangent", ["s3"], 6.0, 8.5, created_at=2.0)   # superseded below
    p = _stratum("p", "apparatus", ["s0"], 0.0, 3.0, status="proposed")
    live = active_strata([a, b, c, p, {"id": "m", "correction_type": "mark"}], {"c"})
    assert [s["id"] for s in live] == ["b", "a"]                      # time order, c + p out
    notes = exclude_strata(SEGS, live, ["sponsor", "tangent", "apparatus"])
    assert [s.id for s in notes] == ["s0", "s1", "s2", "s3", "s4"]     # sponsor dropped, tool kept
    research = exclude_strata(SEGS, live, ["tool-mention"])
    assert "s3" not in [s.id for s in research]


# ---- worklist + derived verdicts ----

def _props():
    pack = _pack()
    return pack, proposals_from_rows(validate_proposal_rows([
        {"category": "apparatus", "from_i": 0, "to_i": 0},
        {"category": "tool-mention", "from_i": 2, "to_i": 2},
        {"category": "sponsor", "from_i": 4, "to_i": 4},
        {"category": "tangent", "from_i": 3, "to_i": 3, "tier": 2},
    ], pack), pack)


def test_pending_hides_tier2_and_drops_materialized():
    _pack_, props = _props()
    by_cat = {p["category"]: p for p in props}
    strata = [_stratum("x", "apparatus", ["s0"], 0.0, 3.0, proposal_id=by_cat["apparatus"]["proposal_id"]),
              _stratum("y", "sponsor", ["s5"], 11.2, 14.0)]   # same category, overlapping, no id carry
    pend = pending_filter_proposals(props, strata)
    assert [p["category"] for p in pend] == ["tool-mention"]
    assert [p["category"] for p in pending_filter_proposals(props, strata, show_tier2=True)] \
        == ["tool-mention", "tangent"]


def test_mark_family_routing_materializes_and_benches_as_accepted():
    """Class-family routing: a proposer's mark-family row accepted AS a mark
    carries the proposal id on the mark payload — the worklist drops it and the
    bench reads it ACCEPTED (family mark), never rejected below the watermark."""
    from cjm_transcript_correction_core.graph import build_mark_correction
    from cjm_transcript_correction_core.strata import materialized_mark_ids
    _pack_, props = _props()
    by_cat = {p["category"]: p for p in props}
    pid = by_cat["tool-mention"]["proposal_id"]
    node, _edges = build_mark_correction("src", {"kind": "segment", "segment_id": "s3"},
                                         "homophone-substitution", "sess",
                                         proposal_id=pid, proposal_set_id="set1")
    assert node["properties"]["payload"]["proposal_id"] == pid
    plain, _ = build_mark_correction("src", {"kind": "segment", "segment_id": "s3"}, "suspect", "sess")
    assert "proposal_id" not in plain["properties"]["payload"]
    d = dict(node["properties"]); d["id"] = node["id"]
    mids = materialized_mark_ids([d], set())
    assert mids == {pid} and materialized_mark_ids([d], {node["id"]}) == set()
    pend = pending_filter_proposals(props, [], materialized=mids)
    assert "tool-mention" not in [p["category"] for p in pend]
    b = bench_filter_proposals(props, [], (0.0, 14.0), watermark=14.0, mark_ids=mids)
    v = {r["category"]: r for r in b["verdicts"]}
    assert v["tool-mention"]["verdict"] == "accepted" and v["tool-mention"]["family"] == "mark"
    assert v["apparatus"]["verdict"] == "rejected"


def test_replacement_rows_validate_carry_through_and_render():
    """The fidelity-edit apply path's ROW CONTRACT: `replacement` is the full
    corrected text of ONE packed line, differing from it; it rides the proposal
    row with the packed line as evidence.text (the drift check) and the human
    view prints the fix; rows without it are untouched."""
    pack = _pack()   # pack line 2 = s3 "By the way, I use Notion for this." (s1 is wordless, left out)
    ok = validate_proposal_rows([
        {"category": "asr-error", "from_i": 2, "to_i": 2, "tier": 2, "confidence": 0.6,
         "rationale": "Notion, not motion", "quote": "I use Notion",
         "replacement": "By the way, I use Notion for this, daily. "},
        {"category": "sponsor", "from_i": 4, "to_i": 4},
    ], pack)
    assert ok[0]["replacement"] == "By the way, I use Notion for this, daily."   # stripped
    assert "replacement" not in ok[1]
    with pytest.raises(ValueError, match="ONE line"):
        validate_proposal_rows([{"category": "asr-error", "from_i": 2, "to_i": 3,
                                 "replacement": "x"}], pack)
    with pytest.raises(ValueError, match="equals line 2"):
        validate_proposal_rows([{"category": "asr-error", "from_i": 2, "to_i": 2,
                                 "replacement": "By the way, I use Notion for this."}], pack)
    with pytest.raises(ValueError, match="non-empty"):
        validate_proposal_rows([{"category": "asr-error", "from_i": 2, "to_i": 2,
                                 "replacement": "   "}], pack)
    props = proposals_from_rows(ok, pack)
    fix = next(p for p in props if p["category"] == "asr-error")
    assert fix["replacement"] == "By the way, I use Notion for this, daily."
    assert fix["evidence"]["text"] == "By the way, I use Notion for this."
    assert fix["segment_ids"] == ["s3"]
    plain = next(p for p in props if p["category"] == "sponsor")
    assert "replacement" not in plain and "text" not in plain["evidence"]
    manifest = {"proposal_set_id": "set1", "source": {"title": "Chapter 1"}, "model": {},
                "window": {"start": 0.0, "end": 14.0}}
    md = render_filter_propset_markdown(manifest, props, pack)
    assert "**Fix:** ~~By the way, I use Notion for this.~~ → By the way, I use Notion for this, daily." in md
    assert "## Output contract" in render_filter_pack(pack) and "`replacement`" in render_filter_pack(pack)


def test_fix_family_materializes_and_benches_as_accepted():
    """The apply path's materialization: an APPLIED row is a text_content
    correction carrying the proposal id — the worklist drops it, the bench reads
    it ACCEPTED (family fix, ahead of a mark carrying the same id), and a later
    human re-edit that supersedes the applied correction un-materializes it."""
    from cjm_transcript_correction_core.strata import materialized_fix_ids, materialized_mark_ids
    _pack_, props = _props()
    by_cat = {p["category"]: p for p in props}
    pid = by_cat["tool-mention"]["proposal_id"]
    node, edges = build_text_correction("src", "s3", "By the way, I use Notion for this, daily.",
                                        "sess", old_text="By the way, I use Notion for this.",
                                        rationale="applied from proposal (reader-1): Notion",
                                        proposal_id=pid, proposal_set_id="set1")
    pl = node["properties"]["payload"]
    assert pl["operation"] == "replace_text" and pl["proposal_id"] == pid and pl["proposal_set_id"] == "set1"
    assert node["properties"]["rationale"] == "applied from proposal (reader-1): Notion"
    assert [e["relation_type"] for e in edges] == ["CORRECTS"]
    hand, _ = build_text_correction("src", "s3", "x", "sess")
    assert "proposal_id" not in hand["properties"]["payload"]
    d = dict(node["properties"]); d["id"] = node["id"]
    fids = materialized_fix_ids([d], set())
    assert fids == {pid} and materialized_fix_ids([d], {node["id"]}) == set()
    assert materialized_mark_ids([d], set()) == set()          # a text edit is not a mark
    pend = pending_filter_proposals(props, [], materialized=fids)
    assert "tool-mention" not in [p["category"] for p in pend]
    b = bench_filter_proposals(props, [], (0.0, 14.0), watermark=14.0, mark_ids={pid}, fix_ids=fids)
    v = {r["category"]: r for r in b["verdicts"]}
    assert v["tool-mention"]["verdict"] == "accepted" and v["tool-mention"]["family"] == "fix"
    assert b["counts"]["tier1"]["accepted"] == 1
    # a proposed (not applied) text correction does not materialize
    d2 = dict(d); d2["status"] = "proposed"
    assert materialized_fix_ids([d2], set()) == set()


def test_bench_filter_proposals_derives_verdicts_below_watermark():
    _pack_, props = _props()
    by_cat = {p["category"]: p for p in props}
    strata = [
        _stratum("x", "apparatus", ["s0"], 0.0, 3.0, proposal_id=by_cat["apparatus"]["proposal_id"]),
        _stratum("y", "research-mark", ["s3"], 6.0, 8.5),          # tool-mention relabeled
        _stratum("z", "sponsor", ["s4", "s5"], 8.5, 14.0),         # sponsor edited (grown)
        _stratum("h", "disfluency", ["s2"], 3.4, 6.0),             # human-minted, unproposed = missed
    ]
    b = bench_filter_proposals(props, strata, (0.0, 14.0), watermark=14.0)
    v = {r["category"]: r["verdict"] for r in b["verdicts"]}
    assert v == {"apparatus": "accepted", "tool-mention": "relabeled",
                 "sponsor": "edited", "tangent": "unaccepted"}
    assert b["counts"]["tier1"] == {"accepted": 1, "edited": 1, "relabeled": 1,
                                    "rejected": 0, "unvisited": 0}
    assert b["rates"] == {"accepted": 0.333, "edited": 0.333, "relabeled": 0.333, "rejected": 0.0}
    assert [m["stratum_id"] for m in b["missed"]] == ["h"]
    # no watermark = nothing visited: an unmatched tier-1 row is UNVISITED, not rejected
    b2 = bench_filter_proposals(props, [], (0.0, 14.0))
    assert b2["counts"]["tier1"]["unvisited"] == 3 and b2["rates"] == {}
    # a watermark mid-source rejects only below it
    b3 = bench_filter_proposals(props, [], (0.0, 14.0), watermark=7.0)
    assert {r["category"]: r["verdict"] for r in b3["verdicts"] if r["tier"] == 1} \
        == {"apparatus": "rejected", "tool-mention": "rejected", "sponsor": "unvisited"}


# ---- the gate's lane fold ----

def test_extraction_gate_lanes_fold_separately():
    n_main, _ = build_extraction_gate_assertion("src", "h", "in_progress", 100.0)
    n_lane, _ = build_extraction_gate_assertion("src", "h", "in_progress", 40.0, lane=FILTER_LANE)
    assert "lane" not in n_main["properties"] and n_lane["properties"]["lane"] == FILTER_LANE
    rows = []
    for k, n in enumerate((n_main, n_lane)):
        d = dict(n["properties"]); d["id"] = n["id"]; d["created_at"] = float(k + 1)
        rows.append(d)
    main = latest_extraction_gates(rows)
    lane = latest_extraction_gates(rows, lane=FILTER_LANE)
    assert main["h"]["annotated_through"] == 100.0        # the newer LANE row did not displace it
    assert lane["h"]["annotated_through"] == 40.0
    assert latest_extraction_gates(rows, lane="other") == {}


# ---- the reading-ladder pack fields (design 6752db0a): speakers, margins, class-scoped passes ----

SPEAKERS = {"s0": "Host", "s1": "Host", "s2": "Host", "s3": "Guest", "s4": "Guest", "s5": None}


def test_pack_without_ladder_fields_stays_0_1_0():
    pack = _pack()
    assert pack["version"] == FILTER_PACK_VERSION
    assert all("speaker" not in r for r in pack["segments"])
    assert "context" not in pack and "closed_vocabulary" not in pack and "mark_vocabulary" not in pack


def test_pack_rows_carry_speakers_and_the_brief_names_each_turn_once():
    pack = _pack(speakers=SPEAKERS)
    assert pack["version"] == FILTER_PACK_VERSION_LADDER
    assert [r["speaker"] for r in pack["segments"]] == ["Host", "Host", "Guest", "Guest", None]
    assert pack["digest"] != _pack()["digest"]            # the attribution was READ
    assert pack["digest"] != _pack(speakers={**SPEAKERS, "s3": "Host"})["digest"]
    md = render_filter_pack(pack)
    assert "Speakers (from the human's assignment pass): Host · Guest" in md
    body = md[md.index("## Transcript"):]
    assert body.count("— Host —") == 1 and body.count("— Guest —") == 1
    assert body.index("— Guest —") < body.index("[2] ") < body.index("— (unassigned) —") < body.index("[4] ")


def test_pack_margin_keeps_unnumbered_read_only_context():
    pack = _pack(window=(3.0, 9.0), margin=1, speakers=SPEAKERS)
    assert [r["id"] for r in pack["segments"]] == ["s2", "s3", "s4"]
    assert [r["id"] for r in pack["context"]["before"]] == ["s0"]   # s1 is wordless — never context
    assert [r["id"] for r in pack["context"]["after"]] == ["s5"]
    assert all("i" not in r for side in pack["context"].values() for r in side)
    assert pack["digest"] != _pack(window=(3.0, 9.0), speakers=SPEAKERS)["digest"]
    md = render_filter_pack(pack)
    assert "(ctx) 00:00.0–00:03.0  Opening credits, read by the author." in md
    assert "(ctx) 00:11.0–00:14.0  This episode is sponsored by X." in md
    assert md.index("Context BEFORE") < md.index("[0] ") < md.index("Context AFTER")
    # rows still validate against the numbered window only
    with pytest.raises(ValueError):
        validate_proposal_rows([{"category": "sponsor", "from_i": 3, "to_i": 3}], pack)


def test_closed_vocabulary_pass_forbids_minting_and_names_the_mark_outlet():
    pack = _pack(vocabulary=["disfluency"], closed=True, mark_vocabulary=["asr-error", "seam-suspect"])
    assert [v["category"] for v in pack["vocabulary"]] == ["disfluency"]
    assert pack["closed_vocabulary"] is True
    assert [v["gloss"] for v in pack["mark_vocabulary"]] == [FILTER_MARK_GLOSSES["asr-error"],
                                                             FILTER_MARK_GLOSSES["seam-suspect"]]
    md = render_filter_pack(pack)
    assert "class-scoped pass" in md and "NEW kebab-case class" not in md
    assert "`tangent`" not in md and "## Mark-family classes" in md and "`seam-suspect`" in md
    open_md = render_filter_pack(_pack())
    assert "NEW kebab-case class" in open_md and "## Mark-family classes" not in open_md
    assert {"qa", "logistics"} <= set(STRATUM_GLOSSES) and "qa" not in RECOMMENDED_STRATUM_CLASSES


def _line(i, start, end, text="words"):
    return SpineSegment(id=f"w{i}", index=i, text=text, start_time=start, end_time=end)


def test_plan_pack_windows_tiles_the_spine_at_mechanical_seams():
    # 12 lines, 1 s each; a long silence before w5, a speaker turn at w7
    segs, t = [], 0.0
    for i in range(12):
        t += 3.0 if i == 5 else 0.2
        segs.append(_line(i, t, t + 1.0))
        t += 1.0
    who = {f"w{i}": ("A" if i < 7 else "B") for i in range(12)}
    assert plan_pack_windows(segs, 1) == [(0.0, None)]
    gap_cut = plan_pack_windows(segs, 2, slack=0.4)
    assert len(gap_cut) == 2 and gap_cut[0][1] == gap_cut[1][0] and gap_cut[1][1] is None
    assert segs[4].end_time < gap_cut[0][1] < segs[5].start_time      # the longest silence near the middle
    turn_cut = plan_pack_windows(segs, 2, speakers=who, slack=0.4)
    assert segs[6].end_time < turn_cut[0][1] < segs[7].start_time     # a speaker turn outranks the silence
    for windows in (gap_cut, turn_cut, plan_pack_windows(segs, 4, speakers=who)):
        packed = [r["id"] for w in windows
                  for r in build_filter_pack("src", "t", None, segs, window=w)["segments"]]
        assert packed == [s.id for s in segs]                          # every line exactly once, in order
    with pytest.raises(ValueError):
        plan_pack_windows(segs, 0)
    with pytest.raises(ValueError):
        plan_pack_windows(segs, 13)


def _set(tmp_path, pack, rows, name):
    res = write_filter_propset(pack, proposals_from_rows(validate_proposal_rows(rows, pack), pack),
                               out_root=tmp_path / "proposals",
                               proposer={"kind": "claude-code-subagent", "name": name})
    return [e for e in load_filter_proposal_sets(str(tmp_path), "src") if e["manifest"]["proposal_set_id"] == res["set_id"]][0]


def test_merge_folds_window_sets_and_rival_arms_into_target_coordinates(tmp_path):
    whole = _pack()
    late = _pack(window=(6.0, 14.0))                                   # s3 s4 s5 -> i 0 1 2
    arm_a = _set(tmp_path, whole, [
        {"category": "sponsor", "from_i": 4, "to_i": 4, "tier": 1, "confidence": 0.9, "quote": "sponsored by X"},
        {"category": "tangent", "from_i": 2, "to_i": 2, "tier": 2, "confidence": 0.4}], "whole")
    arm_b = _set(tmp_path, late, [
        {"category": "sponsor", "from_i": 2, "to_i": 2, "tier": 2, "confidence": 0.6},
        {"category": "tangent", "from_i": 0, "to_i": 1, "tier": 1, "confidence": 0.7},   # overlaps, disagrees
        {"category": "tool-mention", "from_i": 0, "to_i": 0, "tier": 1, "confidence": 0.8}], "late-window")
    merged = merge_filter_proposals([arm_a, arm_b], whole)
    by = {(r["category"], r["evidence"]["from_i"], r["evidence"]["to_i"]): r for r in merged}
    assert set(by) == {("sponsor", 4, 4), ("tangent", 2, 2), ("tangent", 2, 3), ("tool-mention", 2, 2)}
    agreed = by[("sponsor", 4, 4)]
    assert [o["proposer"] for o in agreed["origins"]] == ["whole", "late-window"]
    assert agreed["tier"] == 1 and agreed["segment_ids"] == ["s5"] and agreed["evidence"]["pack_id"] == whole["pack_id"]
    assert agreed["proposal_id"] not in {o["proposal_id"] for o in agreed["origins"]}
    assert by[("tangent", 2, 3)]["segment_ids"] == ["s3", "s4"] and len(by[("tangent", 2, 3)]["origins"]) == 1
    assert [r["start_time"] for r in merged] == sorted(r["start_time"] for r in merged)
    # each ORIGIN set still benches on its own against the live strata — no second walk
    live = [_stratum("st", "sponsor", ["s5"], 11.0, 14.0, proposal_id=agreed["proposal_id"])]
    for arm, window in ((arm_a, (0.0, None)), (arm_b, (6.0, None))):
        verdicts = {v["category"]: v["verdict"] for v in
                    bench_filter_proposals(arm["proposals"], live, window, watermark=99.0)["verdicts"]}
        assert verdicts["sponsor"] == "accepted"
    early = _set(tmp_path, whole, [{"category": "apparatus", "from_i": 0, "to_i": 0}], "early")
    with pytest.raises(ValueError):                                    # a row the target pack cannot hold
        merge_filter_proposals([early], late)
    other = build_filter_pack("src2", "t", None, SEGS)
    with pytest.raises(ValueError):
        merge_filter_proposals([arm_a], other)
