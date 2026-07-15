"""The correction journal: sidecar path + op envelope (pure half; replay is e2e-proven)."""

from cjm_context_graph_primitives.journal import read_journal
from cjm_transcript_correction_core.journal import (correction_replay_handlers,
                                                    journal_correction_op, sidecar_journal_path)


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
    # every registered replay verb has a handler
    assert set(correction_replay_handlers()) == {"session-start", "boundary-shift", "text-correction",
                                                 "prune-amendment", "review-markers", "session-status"}
