"""Overlay data shapes for the transcript-correction workflow: the Correction / CorrectionSession graph nodes + their relation registry, the read view of a committed spine segment, the worklist item, run configuration, and the correction run manifest (proto-bundle that chains decomp -> correction)."""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from cjm_context_graph_primitives.graph import GraphNode
from cjm_substrate.core.workspace import relativize_recorded


class CorrectionRelations:
    """Registry of edge types the correction overlay adds to the spine graph."""
    CORRECTS = "CORRECTS"          # Correction -> the layer-0 Segment(s) it corrects
    SUPERSEDES = "SUPERSEDES"      # Correction -> the prior Correction it replaces (undo/update chain)
    DERIVED_FROM = "DERIVED_FROM"  # grouping Correction -> the layer-0 Segments it regroups/prunes
    REVIEWED = "REVIEWED"          # CorrectionSession -> Segment (carries a `decision` property)
    ASSIGNS = "ASSIGNS"            # speaker Correction -> the Entity it assigns (DEC d6df3a8e)
    GATES = "GATES"                # ExtractionGate assertion -> the Source whose spine it gates (DEC 8e05b87b)

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
    correction_type: str                                   # "text_content" | "punctuation" | "grouping" | "review" | "mark" | "timing" | "insertion" | "speaker"
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
    purpose: Optional[str] = None                          # None = genuine pass; "feature-test" = structurally excludable from flywheel datasets (open vocabulary, DEC c86714a4)
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
    skeleton_selector: Optional[str] = None        # Which SKELETON spine ("legacy" | skeleton-hash prefix, DEC f1024568); None = auto (error when several coexist)

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
        workspace=None,  # Active Workspace; owned paths record as ${WS}/<rel> (5daadfc4 rung f)
    ) -> Path:  # The written path
        """Write the manifest as pretty-printed JSON.

        With `workspace`, recorded paths under its root take the ${WS}/ token
        form (relativize_recorded), so the manifest relocates with the
        workspace; readers resolve via resolve_recorded_tree at load."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(relativize_recorded(self.to_dict(), workspace), indent=2))
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
    "false-start",             # speaker restarts mid-segment (self-correction; words often unpinnable — distinct from full repeats; user-minted 2026-07-28)
    "meta-speech-omission",    # spoken 'Quote' / 'End Quote' markers dropped
    "meta-speech-executed",    # meta-speech rendered AS punctuation instead
    "homophone-substitution",  # context-vs-acoustics substitution (where/were)
    "proper-noun-suspect",     # entity spelling suspect (Hiroo/Hiro/Hero)
    "orthographic-drift",      # decoder-state capitalization/orthography decay
    "granularity-mismatch",    # VAD split lands mid-word/mid-token
    "foreign-speech",          # non-English speech garbled into English (montage/quote cases; drive-minted 2026-07-19)
    "speaker-merge",           # two speakers fused into one sentence by punctuation (a9cadfec)
    "voiced-quote",            # speaker performing another voice — quote/character (DEC 44afb2df)
    "persona-shift",           # deviation from the source's persona default (DEC 44afb2df)
    "speaker-unresolved",      # cannot individuate the voice — layered audio (DEC 484e2d74)
    "suspect",                 # free-note catch-all — flag now, judge later
)

# The RECOMMENDED insert-label slate (DEC 3d3fa2a8 + the C.1 drive) — the same
# OPEN-vocabulary regime as mark classes: labels are DATA, not schema; this
# tuple seeds the I-editor's numbered menu, and a new label typed mid-walk is a
# journal entry, never a core release. "um" and friends land as TEXT (e-edit)
# under hesitation-marker — label = phenomenon class, text = verbatim content.
RECOMMENDED_INSERT_LABELS = (
    "inhale",              # audible breath bookend (the isolation pattern's anchor case)
    "hesitation-marker",   # um / uh / you-know slots the transcript dropped
    "throat-clear",        # non-speech vocalization
    "missed-speech",       # a chunk VAD never cut (the de994164 dispatch class)
)


@dataclass
class Entity:
    """A source-spanning identity in the shared entity substrate (DEC 4ec6a49c).

    Minted by the speaker-assignment lane (physical speakers — DEC 44afb2df) and
    shared with the proper-noun correction lexicon: the same person as SPEAKER
    and as corrected proper noun in text resolves to ONE node, so assignment
    pre-populates the layer the pass-2 assist tier queries. `provisional=True`
    records a DESCRIPTIVE handle ("HH montage narrator"), not an identification
    (DEC 484e2d74: individuation and identification are separate acts) —
    identifying later is a RENAME on this stable id, propagating corpus-wide.
    Minted-once identities with GENERATED ids, referenced by every assignment.
    """
    canonical_name: str                                    # Display name, or a descriptive handle when provisional
    kind: str = "person"                                   # Entity kind (speakers are persons; the lexicon adds more)
    provisional: bool = False                              # True = handle is a description, NOT an identification
    variants: List[str] = field(default_factory=list)      # Observed surface forms (the lexicon convergence)
    actor: str = "human"                                   # Who minted it
    created_at: float = field(default_factory=time.time)   # Unix timestamp
    id: str = field(default_factory=lambda: str(uuid4()))  # Generated node id (the stable identity handle)

    def to_graph_node(self) -> GraphNode:  # Generic graph node
        """Map onto a generic GraphNode (None-valued fields excluded from properties)."""
        props = {k: v for k, v in asdict(self).items() if k != "id" and v is not None}
        return GraphNode(id=self.id, label="Entity", properties=props, sources=[])


# The extraction-gate status vocabulary (DEC 8e05b87b): in_progress is the DEFAULT
# for any spine with no assertion — a bare boolean would either poison frame-level
# training with false-negative frames from the unvisited tail or exclude every
# partially-annotated spine; the annotated_through watermark carries the real boundary.
EXTRACTION_STATUSES = ("in_progress", "signed_off", "excluded")


@dataclass
class ExtractionGate:
    """One per-spine extraction-gate ASSERTION (DEC 8e05b87b — flywheel build leg 1).

    Spine-level state, not a correction: extraction_status gates whether a spine's
    overlay feeds dataset extraction, and `annotated_through` is the LOAD-BEARING
    watermark — label absence means true-negative only BELOW it (above = unvisited).
    Append-only like every overlay verb: rescind/update = a NEW assertion; the read
    is latest-wins per (source_id, skeleton_hash) and the chain is the full history.
    The spine is named by (source_id, skeleton_hash) — None hash = the pre-split
    legacy spine, matching the f1024568 spine-identity vocabulary."""
    source_id: str                                         # Source whose spine is gated
    skeleton_hash: Optional[str] = None                    # Which SKELETON spine (None = legacy pre-split)
    extraction_status: str = "in_progress"                 # EXTRACTION_STATUSES member
    annotated_through: Optional[float] = None              # Watermark (source-coordinate seconds); None = nothing visited
    session_id: Optional[str] = None                       # CorrectionSession context (None = CLI assert)
    actor: str = "human"                                   # Who asserted
    created_at: float = field(default_factory=time.time)   # Unix timestamp (latest-wins key)
    id: str = field(default_factory=lambda: str(uuid4()))  # Generated node id (assertion = event)

    def to_graph_node(self) -> GraphNode:  # Generic graph node
        """Map onto a generic GraphNode (None-valued fields excluded from properties,
        EXCEPT skeleton_hash — spine identity must round-trip: absent-vs-None cannot
        be ambiguous when the legacy spine is a real gate target)."""
        props = {k: v for k, v in asdict(self).items() if k != "id" and v is not None}
        props["skeleton_hash"] = self.skeleton_hash
        return GraphNode(id=self.id, label="ExtractionGate", properties=props, sources=[])


@dataclass
class DatasetManifest:
    """Durable record of one dataset extraction (chainable; DEC 16159e09).

    BORN WORKFLOW-LOCAL: format `cjm-transcript-correction-core/dataset-manifest`,
    generalized to a substrate-generic seam only at n=2 dataset-producing
    workflows. Extends the CorrectionManifest pattern (format tag + version +
    consumed pointers + WS-token paths): records the extraction config, the
    consumed graph db + write-journal family (the source of truth this dataset
    is a regenerable projection of), the session-purpose policy (load-bearing
    for ALL state-derived datasets — supersede-discipline does not self-clean,
    finding 493b8b9e), the observed OPEN class vocabulary, AUGMENTATION and
    SPLIT policy as per-dataset DATA (v1 split IS the tail reservation:
    annotated head = train, reserved tail = live bench, DEC 8cf12c22), and
    each consumed spine's extraction_status + annotated_through at extraction
    time (rescind detection). The chain: source -> decomp manifest ->
    correction journal -> THIS -> training-run manifest -> model (DEC e047beee)."""
    dataset_id: str          # Unique dataset identifier (sortable, run-id pattern)
    created_at: float        # Unix timestamp at extraction start
    config: Dict[str, Any]   # Extraction config snapshot (capability, selectors, purpose args)
    graph_db_path: str       # The shared graph DB the overlay was folded from
    journals: List[str] = field(default_factory=list)  # Consumed write-journal family (the source of truth)
    session_purpose_policy: Dict[str, Any] = field(default_factory=dict)  # Which session purposes feed EXAMPLES
    split_policy: Dict[str, Any] = field(default_factory=dict)            # Per-dataset DATA (v1: tail-reservation)
    augmentation_policy: Dict[str, Any] = field(default_factory=dict)     # Per-dataset DATA (v1: none; provenance tags real/augmented/spliced/synthetic)
    class_vocabulary: Dict[str, int] = field(default_factory=dict)  # Observed OPEN vocabulary (label -> example count)
    spines: List[Dict[str, Any]] = field(default_factory=list)      # Per consumed spine: gate state @ extraction + counts
    files: Dict[str, str] = field(default_factory=dict)             # Dataset-relative data files (events/regions)
    counts: Dict[str, int] = field(default_factory=dict)            # Grand totals (examples, regions, skipped)

    FORMAT: str = field(default="cjm-transcript-correction-core/dataset-manifest", repr=False)  # Format tag
    VERSION: str = field(default="0.1.0", repr=False)                                            # Schema version

    def to_dict(self) -> Dict[str, Any]:  # Plain-dict form for JSON serialization
        """Serialize to a plain dict."""
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "config": self.config,
            "graph_db_path": self.graph_db_path,
            "journals": list(self.journals),
            "session_purpose_policy": self.session_purpose_policy,
            "split_policy": self.split_policy,
            "augmentation_policy": self.augmentation_policy,
            "class_vocabulary": dict(self.class_vocabulary),
            "spines": list(self.spines),
            "files": dict(self.files),
            "counts": dict(self.counts),
        }

    def save(
        self,
        path: Union[str, Path],  # Destination JSON file (parent dirs created)
        workspace=None,  # Active Workspace; owned paths record as ${WS}/<rel> (5daadfc4 rung f)
    ) -> Path:  # The written path
        """Write the manifest as pretty-printed JSON (WS-token recorded paths,
        the CorrectionManifest.save discipline — datasets relocate with the
        workspace, DEC a5883992)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(relativize_recorded(self.to_dict(), workspace), indent=2))
        return out


def new_dataset_id() -> str:  # e.g. "dataset_20260729_153000_1a2b3c4d"
    """Generate a unique, sortable dataset id (the new_run_id pattern, dataset kind)."""
    return f"dataset_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
