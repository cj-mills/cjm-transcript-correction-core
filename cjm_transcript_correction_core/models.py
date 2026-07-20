"""Overlay data shapes for the transcript-correction workflow: the Correction / CorrectionSession graph nodes + their relation registry, the read view of a committed spine segment, the worklist item, run configuration, and the correction run manifest (proto-bundle that chains decomp -> correction)."""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from cjm_context_graph_primitives.graph import GraphNode


class CorrectionRelations:
    """Registry of edge types the correction overlay adds to the spine graph."""
    CORRECTS = "CORRECTS"          # Correction -> the layer-0 Segment(s) it corrects
    SUPERSEDES = "SUPERSEDES"      # Correction -> the prior Correction it replaces (undo/update chain)
    DERIVED_FROM = "DERIVED_FROM"  # grouping Correction -> the layer-0 Segments it regroups/prunes
    REVIEWED = "REVIEWED"          # CorrectionSession -> Segment (carries a `decision` property)

    @classmethod
    def all(cls) -> list:  # All relation type strings
        """Return all defined relation types."""
        return [v for k, v in cls.__dict__.items()
                if not k.startswith('_') and isinstance(v, str)]


@dataclass
class Correction:
    """A single non-destructive correction over the committed spine (overlay node).

    Layer-0 spine nodes are immutable; every correction is a supersede-able
    overlay. Defined IN-CORE (the C6 pattern, kept at stage 5 after
    cjm-graph-domains dissolved): a plain dataclass mapping itself onto the
    generic GraphNode. Corrections are DECISIONS (asserted events) — they keep
    GENERATED ids, the FLIP-TRIGGER-protected class.
    """
    correction_type: str                                   # "text_content" | "punctuation" | "grouping" | "review" | "mark"
    status: str = "applied"                                # "proposed" | "applied" | "superseded"
    session_id: str = ""                                   # Owning CorrectionSession id
    payload: Dict[str, Any] = field(default_factory=dict)  # Type-specific data (new text, prune set, ...)
    actor: str = "human"                                   # "human" | "agent:<id>" | "capability:<name>"
    canonical_form: Optional[str] = None                   # Optional entity key (cross-transcript matching)
    rationale: Optional[str] = None                        # Optional human/agent note
    created_at: float = field(default_factory=time.time)   # Unix timestamp
    id: str = field(default_factory=lambda: str(uuid4()))  # Generated node id (decision = event)

    def to_graph_node(self) -> GraphNode:  # Generic graph node (label = class name)
        """Map onto a generic GraphNode (None-valued fields excluded from properties)."""
        props = {k: v for k, v in asdict(self).items() if k != "id" and v is not None}
        return GraphNode(id=self.id, label="Correction", properties=props, sources=[])


@dataclass
class CorrectionSession:
    """A resumable, reopen-able correction review over one or more sources."""
    status: str = "in_progress"                            # "in_progress" | "completed" | "reopened"
    scope: List[str] = field(default_factory=list)         # Source node ids in scope
    started_at: float = field(default_factory=time.time)   # Unix timestamp at session start
    updated_at: float = field(default_factory=time.time)   # Unix timestamp of last activity
    id: str = field(default_factory=lambda: str(uuid4()))  # Generated node id (session = event)

    def to_graph_node(self) -> GraphNode:  # Generic graph node
        """Map onto a generic GraphNode (None-valued fields excluded from properties)."""
        props = {k: v for k, v in asdict(self).items() if k != "id" and v is not None}
        return GraphNode(id=self.id, label="CorrectionSession", properties=props, sources=[])


@dataclass
class SpineSegment:
    """A committed layer-0 Segment loaded from the graph (read view).

    Stage 5 (Source-rooted schema): segments carry an audio `TimeSlice` ref
    (the stable anchor) + per-transcriber `CharSlice` refs into Transcript
    nodes; `content_hash` is the AUTHORITATIVE text's hash (the `text_from`
    slice) — the cross-transcript cache key."""
    id: str                                   # Graph Segment node id
    index: int                                # 0-based position in the source spine
    text: str                                 # Layer-0 text (may be empty for silence VAD chunks)
    start_time: Optional[float] = None        # Source-coordinate start (seconds)
    end_time: Optional[float] = None          # Source-coordinate end (seconds)
    source_locator: Optional[str] = None      # Audio SourceRef locator URI (the stable provenance anchor)
    content_hash: Optional[str] = None        # Authoritative text slice's content_hash (None when empty)
    text_from: Optional[str] = None           # Authoritative Transcript node id (provenance designation)
    text_slices: List[Dict[str, Any]] = field(default_factory=list)  # [{transcript, start, end, content_hash}]

    @property
    def is_empty(self) -> bool:  # True when the segment has no non-whitespace text
        """Empty-text segment (silence VAD chunk with no aligned words; decomp D14)."""
        return not (self.text or "").strip()


@dataclass
class WorklistItem:
    """One spine segment surfaced for review, with its deterministic Tier-1 flags."""
    segment: SpineSegment                            # The segment under review
    flags: List[str] = field(default_factory=list)   # Tier-1 signal flags (empty, boundary, divergence, ...)

    @property
    def index(self) -> int:  # Segment spine index
        """Spine index of the underlying segment."""
        return self.segment.index


@dataclass
class CorrectionConfig:
    """Configuration for one correction run."""
    graph_capability: str = "cjm-capability-graph-sqlite"  # Graph-storage capability id
    graph_db_path: Optional[str] = None            # Graph DB the spine lives in (from the decomp manifest)
    actor: str = "human"                           # Actor recorded on corrections + review markers
    assume_yes: bool = False                       # Auto-accept HITL seams (headless mode)
    prune_empty: bool = True                       # Run the D14 empty-segment prune as the first operation
    rendition_selector: Optional[str] = None       # Which AudioRendition spine to correct ("raw" | preprocessing substring); None = auto-select the populated one (error if ambiguous)

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict snapshot for the manifest
        """Serialize to a plain dict."""
        return asdict(self)


@dataclass
class CorrectionManifest:
    """Durable record of one correction run (proto-bundle; chainable, CR-20).

    Schema 0.2.0 (stage 5): `documents` became `sources` (Document dissolved
    into Source); the cross-transcriber diff is intra-graph now, so the
    secondary-manifest pointer is gone."""
    run_id: str             # Unique run identifier
    created_at: float       # Unix timestamp at run start
    config: Dict[str, Any]  # CorrectionConfig snapshot
    decomp_manifest: str    # Path to the consumed decomp run manifest
    graph_db_path: str      # The shared graph DB the spine + overlay live in
    session_id: str         # CorrectionSession node id this run used
    source_format: str = ""   # Upstream manifest format tag (interchange contract)
    source_version: str = ""  # Upstream manifest schema version
    signals_used: List[str] = field(default_factory=list)  # Deterministic signals active this run
    sources: List[Dict[str, Any]] = field(default_factory=list)  # Per-source outcome records

    FORMAT: str = field(default="cjm-transcript-correction-core/run-manifest", repr=False)  # Format tag
    VERSION: str = field(default="0.2.0", repr=False)                                       # Schema version

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict form for JSON serialization
        """Serialize to a plain dict."""
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": self.config,
            "decomp_manifest": self.decomp_manifest,
            "graph_db_path": self.graph_db_path,
            "session_id": self.session_id,
            "source_format": self.source_format,
            "source_version": self.source_version,
            "signals_used": list(self.signals_used),
            "sources": list(self.sources),
        }

    def save(
        self,
        path: Union[str, Path],  # Destination JSON file (parent dirs created)
    ) -> Path:  # The written path
        """Write the manifest as pretty-printed JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2))
        return out


def new_run_id() -> str:  # e.g. "correct_20260608_153000_1a2b3c4d"
    """Generate a unique, sortable correction run id."""
    return f"correct_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


# The RECOMMENDED mark-class slate (DEC 2a231843) — an OPEN vocabulary: classes are
# DATA, not schema. A mark commits with any non-empty class string; this tuple is
# the evidence-derived starting set (census dc31c33c + the drive evidence chain)
# that pickers and status hints surface, so a new class found mid-walk is a journal
# entry, never a core release.
RECOMMENDED_MARK_CLASSES = (
    "hesitation-omission",     # single Um / 'you know' slots dropped (fill dominates)
    "repeat-omission",         # dropped repeated words/stutters — the omission-entangled boundary case
    "meta-speech-omission",    # spoken 'Quote' / 'End Quote' markers dropped
    "meta-speech-executed",    # meta-speech rendered AS punctuation instead
    "homophone-substitution",  # context-vs-acoustics substitution (where/were)
    "proper-noun-suspect",     # entity spelling suspect (Hiroo/Hiro/Hero)
    "orthographic-drift",      # decoder-state capitalization/orthography decay
    "granularity-mismatch",    # VAD split lands mid-word/mid-token
    "foreign-speech",          # non-English speech garbled into English (montage/quote cases; drive-minted 2026-07-19)
    "suspect",                 # free-note catch-all — flag now, judge later
)
