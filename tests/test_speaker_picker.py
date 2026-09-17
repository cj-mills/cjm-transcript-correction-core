"""The scalable assign-lane speaker picker (DEC 774dbe40, item 75bc0384): digit
slots from two tiers (this source's assigned speakers, then the collection's),
the rest of the registry behind typed search, and the graph read that feeds the
collection tier (speaker_assign Corrections over the sibling Sources)."""
import asyncio
import json
from types import SimpleNamespace

from cjm_substrate.core.queue import JobStatus
from cjm_transcript_correction_core.graph import speaker_assignment_sources
from cjm_transcript_correction_core.spine import layer_speaker_menu, match_speaker_query


def _ent(eid, name, provisional=False):
    return {"id": eid, "properties": {"canonical_name": name, "kind": "person",
                                      "provisional": provisional}}


REGISTRY = [_ent("mark", "Mark Saroufim"), _ent("jacob", "Jacob Hemstad"),
            _ent("jeremy", "Jeremy Howard"), _ent("ti", "Ti Morse"),
            _ent("chris", "Chris Wright"), _ent("scott", "Scott Nolan"),
            _ent("un", "unnamed"), _ent("hh", "HH montage narrator", provisional=True)]
NAMES = {d["id"]: d["properties"]["canonical_name"] for d in REGISTRY}


def test_layer_speaker_menu_source_tier_is_stable_and_collection_tier_ranks_by_breadth():
    menu = layer_speaker_menu(assigned=["jacob", "mark"],
                              collection_counts={"jeremy": 1, "mark": 7, "chris": 0, "un": 3},
                              recents=["mark", "jeremy"], names=NAMES)
    # tier 1 keeps first-appearance order even though mark is the most recent pick
    assert menu[:2] == [("jacob", "src"), ("mark", "src")]
    # tier 2: breadth first (un=3 over jeremy=1); zero-count entities never enter; mark not repeated
    assert menu[2:] == [("un", "coll"), ("jeremy", "coll")]


def test_layer_speaker_menu_recents_break_breadth_ties_then_name():
    menu = layer_speaker_menu(assigned=[], collection_counts={"ti": 2, "chris": 2, "scott": 2},
                              recents=["scott"], names=NAMES)
    assert [e for e, _ in menu] == ["scott", "chris", "ti"]   # recent first, then Chris < Ti by name


def test_layer_speaker_menu_unfiled_source_is_source_tier_only():
    assert layer_speaker_menu(["un"], {}, [], NAMES) == [("un", "src")]


def test_match_speaker_query_token_prefix_then_substring_then_tier():
    tiers = {"mark": "src", "jeremy": "coll"}
    assert [e for e, _, _ in match_speaker_query("sar", REGISTRY, tiers, [])] == ["mark"]
    assert [e for e, _, _ in match_speaker_query("m s", REGISTRY, tiers, [])] == ["mark"]
    # substring rank comes after prefix rank: "ar" prefixes nothing, but is inside Mark / Chris... only rank-1 hits
    subs = match_speaker_query("orse", REGISTRY, tiers, [])
    assert [e for e, _, _ in subs] == ["ti"]
    # the j-prefixed pair: coll tier (jeremy) sorts before reg tier (jacob)
    js = match_speaker_query("j", REGISTRY, tiers, [])
    assert [(e, t) for e, _, t in js] == [("jeremy", "coll"), ("jacob", "reg")]
    # recents break ties inside a tier
    assert [e for e, _, _ in match_speaker_query("j", REGISTRY, {}, ["jacob"])][0] == "jacob"


def test_match_speaker_query_empty_and_mint_grammar_never_search():
    assert match_speaker_query("", REGISTRY, {}, []) == []
    assert match_speaker_query("   ", REGISTRY, {}, []) == []
    assert match_speaker_query("? HH montage narrator", REGISTRY, {}, []) == []
    # a distinct full name that shares a first name matches nothing -> the caller mints
    assert match_speaker_query("Mark Chen", REGISTRY, {}, []) == []
    # limit honours the digit budget
    assert len(match_speaker_query("a", REGISTRY, {}, [], limit=2)) == 2


class _Queue:
    def __init__(self, rows):
        self.rows, self.queries = rows, []

    async def submit(self, graph_id, **kw):
        self.queries.append(kw)
        return "1"

    async def wait_for_job(self, jid):
        return SimpleNamespace(status=JobStatus.completed, error=None,
                               result=SimpleNamespace(rows=self.rows))


def test_speaker_assignment_sources_filters_by_sibling_ids_and_parses_serialized_payload():
    rows = [
        {"id": "c1", "payload": json.dumps({"operation": "speaker_assign", "source_id": "s2", "entity_id": "mark"})},
        {"id": "c2", "payload": json.dumps({"operation": "speaker_assign", "source_id": "s2", "entity_id": "mark"})},
        {"id": "c3", "payload": {"operation": "speaker_assign", "source_id": "s3", "entity_id": "mark"}},
        {"id": "c4", "payload": {"operation": "speaker_assign", "source_id": "s3", "entity_id": "jeremy"}},
        {"id": "c5", "payload": "not json"},
        {"id": "c6", "payload": {"operation": "speaker_assign", "source_id": "s3"}},
    ]
    q = _Queue(rows)
    out = asyncio.run(speaker_assignment_sources(q, "g", ["s2", "s3"]))
    assert out == {"mark": ["s2", "s3"], "jeremy": ["s3"]}
    where = q.queries[0]["query"]["where"]
    assert [(w["prop"], w["op"]) for w in where] == [("payload.operation", "eq"), ("payload.source_id", "in")]
    assert where[1]["value"] == ["s2", "s3"]
    assert q.queries[0]["query"]["project"] == ["payload"]


def test_speaker_assignment_sources_empty_sibling_set_reads_nothing_and_none_reads_all():
    q = _Queue([])
    assert asyncio.run(speaker_assignment_sources(q, "g", [])) == {}
    assert q.queries == []
    asyncio.run(speaker_assignment_sources(q, "g", None))
    assert [w["prop"] for w in q.queries[0]["query"]["where"]] == ["payload.operation"]
