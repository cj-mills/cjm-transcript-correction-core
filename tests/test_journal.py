"""The correction journal: sidecar path + op envelope (pure half; replay is e2e-proven)."""

from cjm_context_graph_primitives.journal import read_journal
from cjm_transcript_correction_core.journal import correction_replay_handlers, journal_correction_op
from cjm_context_graph_layer.journal import sidecar_journal_path


def test_sidecar_journal_path():
    assert (sidecar_journal_path("/data/context_graph.db")
            == "/data/context_graph.writes.jsonl")
    assert sidecar_journal_path("/data/odd_name") == "/data/odd_name.writes.jsonl"


def test_journal_correction_op_envelope(tmp_path):
    """Envelope: op id = minted Correction id, set = session, anchor + exact wires ride along."""
    j = str(tmp_path / "w.jsonl")
    node = {"id": "corr-1", "label": "Correction", "properties": {"correction_type": "grouping"}}
    edge = {"id": "e-1", "source_id": "corr-1", "target_id": "seg-1", "relation_type": "CORRECTS"}
    anchor = {"sources": [{"id": "src", "content_hash": "sha256:x"}],
              "segments": [{"id": "seg-1", "start": 1.0, "end": 2.0, "text": "t"}]}
    assert journal_correction_op(j, "boundary-shift", actor="human", session_id="sess-1",
                                 args={"direction": "pull"}, nodes=[node], edges=[edge],
                                 anchor=anchor, op_id="corr-1")
    rec = read_journal(j)[0]
    assert rec["id"] == "corr-1" and rec["set"] == "sess-1" and rec["actor"] == "human"
    assert rec["wires"]["nodes"][0]["id"] == "corr-1" and rec["anchor"]["segments"][0]["end"] == 2.0
    # Live appends ride the bulk lane (dedup=False — the 111ms/op rescan priced the TUI):
    # a re-append lands as a duplicate LINE, and exactness is REPLAY's job (extend
    # collides duplicate wires into verified no-ops).
    assert journal_correction_op(j, "boundary-shift", actor="human", session_id="sess-1",
                                 args={"direction": "pull"}, nodes=[node], edges=[edge],
                                 anchor=anchor, op_id="corr-1")
    assert len(read_journal(j)) == 2
    # every journaled verb has a replay handler (replay_journal raises LOUDLY on
    # unregistered verbs — time-nudge shipped without one and would have crashed
    # any rebuild over a journal holding nudge ops; caught 2026-07-24)
    assert set(correction_replay_handlers()) == {"session-start", "boundary-shift", "text-correction",
                                                 "prune-amendment", "mark", "mark-dismiss",
                                                 "review-markers", "session-status", "session-purpose",
                                                 "time-nudge", "chunk-insert", "chunk-insert-remove",
                                                 "chunk-split", "chunk-split-remove",
                                                 "speech-overlay", "speech-overlay-remove",
                                                 "speaker-entity", "speaker-assign",
                                                 "entity-rename", "extraction-gate",
                                                 "stratum", "stratum-retract"}


def test_replay_handlers_cover_mark_verbs():
    """Marks are BORN JOURNALED (DEC 2a231843): both verbs replay as wire ops."""
    handlers = correction_replay_handlers()
    assert "mark" in handlers and "mark-dismiss" in handlers


def test_session_lifecycle_rows_carry_the_callers_actor(tmp_path, monkeypatch):
    """Finding ac878d68: session-start / session-status / session-purpose rows used to
    stamp actor="human" regardless of the verb's --actor. The actor threads through
    now (default "human" keeps every existing caller byte-identical)."""
    import asyncio
    from cjm_transcript_correction_core import graph as g

    async def fake_commit(queue, graph_id, nodes, edges):
        return {"nodes": len(nodes), "edges": len(edges)}

    async def fake_task(queue, graph_id, op, **kw):
        return None

    monkeypatch.setattr(g, "commit_nodes_edges", fake_commit)
    monkeypatch.setattr(g, "graph_task", fake_task)
    j = str(tmp_path / "w.jsonl")

    async def run():
        sess = await g.start_session(None, "gid", ["src"], journal_path=j,
                                     purpose="feature-test", actor="agent:smoke")
        await g.set_session_status(None, "gid", sess.id, "completed", journal_path=j,
                                   actor="agent:smoke")
        await g.set_session_purpose(None, "gid", sess.id, None, journal_path=j)
        return sess

    asyncio.run(run())
    rows = read_journal(j)
    assert [r["verb"] for r in rows] == ["session-start", "session-status", "session-purpose"]
    assert [r["actor"] for r in rows] == ["agent:smoke", "agent:smoke", "human"]
