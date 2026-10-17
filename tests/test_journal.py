"""The correction journal: sidecar path + op envelope (pure half; replay is e2e-proven)."""

import asyncio
from types import SimpleNamespace

import pytest
from cjm_context_graph_layer.journal import sidecar_journal_path, UnjournaledWrite
from cjm_context_graph_layer.testing import MemGraph
from cjm_context_graph_primitives.journal import read_journal
from cjm_transcript_correction_core.journal import correction_replay_handlers, journal_correction_op


def test_sidecar_journal_path():
    assert (sidecar_journal_path("/data/context_graph.db")
            == "/data/context_graph.writes.jsonl")
    assert sidecar_journal_path("/data/odd_name") == "/data/odd_name.writes.jsonl"


def test_journal_correction_op_envelope(tmp_path):
    """Envelope: op id = minted Correction id, set = session, anchor + exact wires ride along --
    and the op is JOURNALED, then APPLIED through its verb's handler (ruling 6683f49e (1))."""
    j = str(tmp_path / "w.jsonl")
    g = MemGraph()
    g.nodes["seg-1"] = {"id": "seg-1", "label": "Segment", "sources": [],
                        "properties": {"index": 0, "source_id": "src-envelope", "start_time": 1.0,
                                       "end_time": 2.0, "text": "t"}}
    g.nodes["src-envelope"] = {"id": "src-envelope", "label": "Source", "properties": {},
                               "sources": [SimpleNamespace(content_hash="sha256:x")]}
    node = {"id": "corr-1", "label": "Correction", "properties": {"correction_type": "grouping"}}
    edge = {"id": "e-1", "source_id": "corr-1", "target_id": "seg-1", "relation_type": "CORRECTS"}

    def write():
        return asyncio.run(journal_correction_op(
            g, "gid", "boundary-shift", actor="human", session_id="sess-1",
            args={"direction": "pull"}, nodes=[node], edges=[edge], anchor_segments=["seg-1"],
            op_id="corr-1", journal_path=j))

    landed = write()
    [rec] = read_journal(j)  # the row as appended (the apply then stamps the wires' times in place)
    assert landed["id"] == rec["id"]
    assert rec["id"] == "corr-1" and rec["set"] == "sess-1" and rec["actor"] == "human"
    assert rec["wires"]["nodes"][0]["id"] == "corr-1" and rec["anchor"]["segments"][0]["end"] == 2.0
    assert rec["anchor"]["sources"] == [{"id": "src-envelope", "content_hash": "sha256:x"}]
    assert "corr-1" in g.nodes and "e-1" in g.edges  # applied after the append, by apply_wires
    # No content dedup (17daa86b): a re-write lands as a second LINE, and the apply collides the
    # duplicate wires into verified no-ops (live corrections mint a fresh uuid per commit anyway).
    write()
    assert len(read_journal(j)) == 2 and len(g.nodes) == 3
    # every journaled verb has a replay handler (replay_journal raises LOUDLY on
    # unregistered verbs — time-nudge shipped without one and would have crashed
    # any rebuild over a journal holding nudge ops; caught 2026-07-24)
    assert set(correction_replay_handlers()) == {"session-start", "boundary-shift", "text-correction",
                                                 "prune-amendment", "mark", "mark-dismiss",
                                                 "mark-discharge",
                                                 "review-markers", "session-status", "session-purpose",
                                                 "time-nudge", "chunk-insert", "chunk-insert-remove",
                                                 "chunk-split", "chunk-split-remove",
                                                 "speech-overlay", "speech-overlay-remove",
                                                 "speaker-entity", "speaker-assign",
                                                 "entity-rename", "extraction-gate",
                                                 "stratum", "stratum-retract", "prune"}


def test_replay_handlers_cover_mark_verbs():
    """Marks are BORN JOURNALED (DEC 2a231843): both verbs replay as wire ops."""
    handlers = correction_replay_handlers()
    assert "mark" in handlers and "mark-dismiss" in handlers


def test_session_lifecycle_rows_carry_the_callers_actor(tmp_path):
    """Finding ac878d68: session-start / session-status / session-purpose rows used to
    stamp actor="human" regardless of the verb's --actor. The actor threads through
    now (default "human" keeps every existing caller byte-identical). The status and
    purpose merges are their ops' applies (ruling 6683f49e (1))."""
    from cjm_transcript_correction_core import graph as g

    store = MemGraph()
    j = str(tmp_path / "w.jsonl")

    async def run():
        sess = await g.start_session(store, "gid", ["src"], journal_path=j,
                                     purpose="feature-test", actor="agent:smoke")
        await g.set_session_status(store, "gid", sess.id, "completed", journal_path=j,
                                   actor="agent:smoke")
        await g.set_session_purpose(store, "gid", sess.id, None, journal_path=j)
        return sess

    sess = asyncio.run(run())
    rows = read_journal(j)
    assert [r["verb"] for r in rows] == ["session-start", "session-status", "session-purpose"]
    assert [r["actor"] for r in rows] == ["agent:smoke", "agent:smoke", "human"]
    props = store.nodes[sess.id]["properties"]
    assert props["status"] == "completed" and props["purpose"] is None


def test_a_commit_with_no_journal_refuses_unless_explicitly_unjournaled(tmp_path):
    """Design detail 65f4a6c9 (3): a journal-sourced graph is never written unjournaled by
    default -- refused before anything is read or written; `unjournaled=True` applies the op
    through the same handler and journals nothing."""
    from cjm_transcript_correction_core import graph as g

    store = MemGraph()
    with pytest.raises(UnjournaledWrite, match="session-start"):
        asyncio.run(g.start_session(store, "gid", ["src"]))
    assert store.methods == [] and store.nodes == {}
    sess = asyncio.run(g.start_session(store, "gid", ["src"], unjournaled=True))
    assert sess.id in store.nodes
