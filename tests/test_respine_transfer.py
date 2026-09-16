"""The chunk-scoped transfer a chunk respine calls (7a5e9c84; 0b4d5cfa (5), 4a7ec4f8):
the speaker carry by time with the straddle guard, the turn-shaped grouping, the live
reader predicate on every spine query, and the entry-point registration."""

import asyncio
import importlib.metadata
from types import SimpleNamespace

from cjm_transcript_graph_schema.schema import SEGMENT_SUPERSEDED_BY_PROP

from cjm_transcript_correction_core import respine_transfer as RT
from cjm_transcript_correction_core.graph import _spine_query
from cjm_transcript_correction_core.models import SpineSegment


def _seg(i, s, e, sid=None):
    return SpineSegment(id=sid or f"seg{i}", index=i, text="x", start_time=s, end_time=e)


def test_speaker_spans_reads_only_assigned_units_in_time_order():
    old = [_seg(12, 300.0, 302.0, "o2"), _seg(10, 296.0, 298.0, "o0"), _seg(11, 298.0, 300.0, "o1")]
    assigns = {"o0": {"entity_id": "mark", "verdict": "name", "correction_id": "c0"},
               "o2": {"entity_id": "guest", "verdict": "accept", "correction_id": "c2"},
               "o1": {"entity_id": None}}
    spans = RT.speaker_spans(old, assigns)
    assert [(r["start"], r["entity_id"], r["correction_id"]) for r in spans] == [
        (296.0, "mark", "c0"), (300.0, "guest", "c2")]


def test_plan_speaker_carry_covers_straddles_and_uncovered():
    spans = [{"start": 296.0, "end": 299.0, "entity_id": "mark", "verdict": "name", "correction_id": "c0"},
             {"start": 299.0, "end": 302.0, "entity_id": "guest", "verdict": "accept", "correction_id": "c2"}]
    new = [_seg(10, 296.0, 297.5, "n0"),      # inside mark
           _seg(11, 297.5, 299.02, "n1"),     # mark, a 20 ms sliver into guest -> still mark
           _seg(12, 298.0, 300.5, "n2"),      # 1 s mark + 1.5 s guest -> STRADDLE (donors deduped per entity)
           _seg(13, 300.5, 302.0, "n3"),      # guest
           _seg(14, 302.0, 303.0, "n4"),      # after every span -> uncovered
           SpineSegment(id="n5", index=15, text="", start_time=None, end_time=None)]  # no times -> uncovered
    rows, straddles, uncovered = RT.plan_speaker_carry(spans, new)
    assert [(r["segment_id"], r["entity_id"], r["verdict"], r["donors"]) for r in rows] == [
        ("n0", "mark", "name", ["c0"]), ("n1", "mark", "name", ["c0"]), ("n3", "guest", "accept", ["c2"])]
    assert straddles == ["n2"] and uncovered == ["n4", "n5"]
    # a sub-eps segment takes any real overlap (never uncovered by its own shortness)
    tiny = [_seg(20, 298.99, 299.0, "t")]
    assert RT.plan_speaker_carry(spans, tiny)[0][0]["entity_id"] == "mark"


def test_group_speaker_rows_folds_contiguous_runs_into_turns():
    rows = [{"segment_id": "n0", "entity_id": "mark", "verdict": "name", "donors": ["c0"]},
            {"segment_id": "n1", "entity_id": "mark", "verdict": "name", "donors": ["c0", "c1"]},
            {"segment_id": "n3", "entity_id": "guest", "verdict": "accept", "donors": ["c2"]},
            {"segment_id": "n4", "entity_id": "mark", "verdict": "name", "donors": ["c0"]}]
    turns = RT.group_speaker_rows(rows)
    assert [(t["entity_id"], t["segment_ids"], t["donors"]) for t in turns] == [
        ("mark", ["n0", "n1"], ["c0", "c1"]), ("guest", ["n3"], ["c2"]), ("mark", ["n4"], ["c0"])]
    assert RT.group_speaker_rows([]) == []


def test_spine_query_carries_the_live_predicate_and_ands_callers_where():
    from cjm_context_graph_primitives.query import PropertyPredicate
    base = _spine_query(["r1"]).to_dict()
    assert base["where"] == [{"prop": SEGMENT_SUPERSEDED_BY_PROP, "op": "is_null", "value": None}]
    scoped = _spine_query(["r1"], where=[PropertyPredicate("skeleton_hash", "eq", "h")]).to_dict()
    assert [w["prop"] for w in scoped["where"]] == [SEGMENT_SUPERSEDED_BY_PROP, "skeleton_hash"]
    counted = _spine_query(["r1"], order_by=None, project=None, count=True, where=[]).to_dict()
    assert counted["count"] and counted["where"][0]["prop"] == SEGMENT_SUPERSEDED_BY_PROP


def test_entry_point_registers_the_handler():
    eps = [ep for ep in importlib.metadata.entry_points(group="cjm_transcript_decomp_core.chunk_transfer")
           if ep.name == "correction"]
    assert eps, "pyproject registers the chunk-transfer entry point (editable install refreshed?)"
    assert eps[0].load() is RT.chunk_respine_transfer


def test_chunk_respine_transfer_plans_and_commits_both_classes(monkeypatch):
    # Old chunk: o0 (296-298, mark) · o1 (298-300, mark) · o2 (300-302, guest), an accepted
    # inhale insert after o1 at 299.9-300.1 (wordless) and a word-bearing insert (stays).
    old = {"o0": _seg(10, 296.0, 298.0, "o0"), "o1": _seg(11, 298.0, 300.0, "o1"), "o2": _seg(12, 300.0, 302.0, "o2")}
    new = {"n0": _seg(10, 296.0, 299.0, "n0"), "n1": _seg(11, 299.0, 299.97, "n1"), "n2": _seg(12, 299.97, 302.0, "n2")}
    corrections = [
        {"id": "ins", "correction_type": "insertion", "status": "applied", "created_at": 5.0,
         "payload": {"operation": "chunk_insert", "source_id": "src", "after_segment_id": "o1",
                     "before_segment_id": "o2", "start_time": 299.9, "end_time": 300.1, "label": "inhale", "text": ""}},
        {"id": "ins-words", "correction_type": "insertion", "status": "applied", "created_at": 6.0,
         "payload": {"operation": "chunk_insert", "source_id": "src", "after_segment_id": "o0",
                     "before_segment_id": "o1", "start_time": 297.0, "end_time": 297.4, "label": "empty", "text": "hi"}},
        {"id": "spk-mark", "correction_type": "speaker", "created_at": 7.0,
         "payload": {"operation": "speaker_assign", "source_id": "src", "segment_ids": ["o0", "o1"],
                     "entity_id": "mark", "verdict": "name"}},
        {"id": "spk-guest", "correction_type": "speaker", "created_at": 8.0,
         "payload": {"operation": "speaker_assign", "source_id": "src", "segment_ids": ["o2"],
                     "entity_id": "guest", "verdict": "accept"}},
    ]

    async def fake_load(queue, gid, ids):
        pool = {**old, **new}
        return sorted((pool[i] for i in ids if i in pool), key=lambda s: s.index)

    async def fake_corr(queue, gid, source_id):
        return corrections, set()

    async def fake_props(queue, gid, sid):
        return {"skeleton_hash": "H"}

    async def fake_prev(queue, gid, source_id, h, index):
        assert (h, index) == ("H", 9)
        return _seg(9, 290.0, 296.0, "prev")

    committed = []

    async def fake_session(queue, gid, scope, journal_path=None, purpose=None, actor="human"):
        committed.append(("session", purpose, actor))
        return SimpleNamespace(id="sess-1")

    async def fake_insert(queue, gid, source_id, after_id, start, end, sess_id, before_segment_id=None,
                          label=None, rank=0.0, actor="human", journal_path=None):
        committed.append(("insert", after_id, before_segment_id, start, end, label))
        return f"ins-{len(committed)}"

    async def fake_assign(queue, gid, source_id, segment_ids, entity_id, sess_id, verdict="name",
                          proposal=None, supersedes_id=None, actor="human", journal_path=None):
        committed.append(("assign", tuple(segment_ids), entity_id, verdict, proposal["transfer"],
                          tuple(proposal["donor_correction_ids"])))
        return f"spk-{len(committed)}"

    async def fake_status(queue, gid, sess_id, status, journal_path=None, actor="human"):
        committed.append(("status", status))

    monkeypatch.setattr(RT, "load_segments_by_ids", fake_load)
    monkeypatch.setattr(RT, "load_source_corrections", fake_corr)
    monkeypatch.setattr(RT, "_segment_props", fake_props)
    monkeypatch.setattr(RT, "live_neighbour", fake_prev)
    monkeypatch.setattr(RT, "start_session", fake_session)
    monkeypatch.setattr(RT, "commit_chunk_insert_correction", fake_insert)
    monkeypatch.setattr(RT, "commit_speaker_assign_correction", fake_assign)
    monkeypatch.setattr(RT, "set_session_status", fake_status)

    r = asyncio.run(RT.chunk_respine_transfer(None, "g", source_id="src", old_segment_ids=["o0", "o1", "o2"],
                                              new_segment_ids=["n0", "n1", "n2"], journal_path=None,
                                              actor="human:t", run_id="decomp_r"))
    assert r["session_id"] == "sess-1" and r["events"] == 1 and r["speakers"] == 2
    assert r["word_bearing"] == 1 and r["dups"] == 0 and r["unanchored"] == 0
    # the inhale lands after n1 (starts at or before 299.9) with n2 as the right flank
    assert ("insert", "n1", "n2", 299.9, 300.1, "inhale") in committed
    # speakers: n0 + n1 are mark (n1 ends 30 ms before guest's span -> covered by mark only);
    # n2 spans 299.97-302: a 30 ms sliver of mark (< eps) + 2 s of guest -> guest, not a straddle
    assigns = [c for c in committed if c[0] == "assign"]
    assert assigns == [("assign", ("n0", "n1"), "mark", "name", "chunk-respine", ("spk-mark",)),
                       ("assign", ("n2",), "guest", "accept", "chunk-respine", ("spk-guest",))]
    assert r["straddles"] == [] and r["uncovered"] == []
    assert committed[0] == ("session", RT.TRANSFER_PURPOSE, "human:t") and committed[-1] == ("status", "completed")
    assert len(r["carried"]) == 3


def test_chunk_respine_transfer_nothing_to_carry_mints_no_session(monkeypatch):
    async def fake_load(queue, gid, ids):
        return [_seg(1, 0.0, 1.0, i) for i in ids]

    async def fake_corr(queue, gid, source_id):
        return [], set()

    async def fake_props(queue, gid, sid):
        return {}

    async def boom(*a, **k):
        raise AssertionError("no session when nothing carries")

    monkeypatch.setattr(RT, "load_segments_by_ids", fake_load)
    monkeypatch.setattr(RT, "load_source_corrections", fake_corr)
    monkeypatch.setattr(RT, "_segment_props", fake_props)
    monkeypatch.setattr(RT, "start_session", boom)
    r = asyncio.run(RT.chunk_respine_transfer(None, "g", source_id="src", old_segment_ids=["a"],
                                              new_segment_ids=["b"], journal_path=None, actor="t"))
    assert r["session_id"] is None and r["carried"] == [] and r["uncovered"] == ["b"]
