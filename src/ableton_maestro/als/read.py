"""Read-only parser for Ableton project (.als) and device preset (.adg) files.

Extracts typed project structures directly from gzip-compressed XML, including tracks,
device chains, clip notes, clip/track automation layers, drum pad mappings, and sidechain
routings.

Parser observations and structure rules:
- Automation layers: reports clip envelopes (Session clips) and track automation
  (Arrangement timeline) as separate layers.
- Parameter slots: 128 slots allocated per VST instance; unconfigured slots carry the
  placeholder value 0.1234567687 (measured on 3rd-party VST plugin: 128 slots, 2 named).
- Note events: pitch is stored on the parent <KeyTrack> container (<MidiKey Value="N"/>).
- Event pairs: breakpoint events in <FloatEvent> occur in time-duplicated pairs.
- Lead-in: events at Time = -63072000 are lead-in pre-roll markers and filtered by
  PRE_ROLL_TIME.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import math
import os
import re
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from xml.etree.ElementTree import Element, fromstring

__all__ = [
    "AutomationReport",
    "Clip",
    "Device",
    "DrumPad",
    "Envelope",
    "EnvelopeRef",
    "Macro",
    "Note",
    "NoteStats",
    "Project",
    "Sidechain",
    "Track",
    "automation_of",
    "beats_per_bar",
    "format_notes_report",
    "format_report",
    "load_xml",
    "note_name",
    "read_clip_notes",
    "read_project",
    "read_track",
    "to_dict",
    "unique_note_clips",
]

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
"""Pitch class names with C = 0. note_name formats scientific pitch (C4 = 60)."""

PRE_ROLL_TIME = -1e6
"""Lead-in threshold in beats; events at or below this value are pre-roll markers."""

TRACK_TAGS = (
    "MidiTrack",
    "AudioTrack",
    "GroupTrack",
    "ReturnTrack",
    "MasterTrack",
    "MainTrack",
)
"""XML element tags representing Ableton project tracks.

``MainTrack`` is Live 12's name for ``MasterTrack`` and both have to be listed. Without
``MainTrack``, every Live 12 project loses its master track and its device chain, which
is 49 of the 174 corpus projects. Counted over the corpus and against a Live 12
project's XML, where ``<MainTrack>`` appears once and ``<MasterTrack>`` never.
"""

NOT_A_DEVICE = frozenset(
    {
        "MidiTrack",
        "AudioTrack",
        "GroupTrack",
        "ReturnTrack",
        "MasterTrack",
        "MainTrack",
        "MainSequencer",
        "FreezeSequencer",
        "ClipSlot",
        "Devices",
        "MidiClip",
        "AudioClip",
        "Mixer",
        "AudioInputRouting",
        "AudioOutputRouting",
        "MidiInputRouting",
        "MidiOutputRouting",
    }
)
"""Element tags in device chain hierarchy that represent non-device containers."""

DRUM_BRANCH_TAGS = ("DrumBranch", "DrumBranchPreset")
"""XML tags for drum branch containers across .als and .adg files.

A drum rack branch is ``DrumBranch`` in an .als and ``DrumBranchPreset`` in an .adg.
Searching for only one of them finds zero pads in the other file type and reports a
full kit as empty (measured on three factory kits).
"""

SAMPLE_SUFFIXES = (".wav", ".aif", ".aiff", ".flac")
"""Supported audio file extensions for sample identification in drum pads."""

GRID: tuple[tuple[str, float], ...] = (
    ("1/4", 1.0),
    ("1/8", 0.5),
    ("1/16", 0.25),
    ("1/32", 0.125),
    ("1/8T", 1.0 / 3.0),
    ("1/16T", 1.0 / 6.0),
)
"""Metric grid divisions in descending coarseness, prioritizing binary divisions over triplets."""

GRID_TOL = 1e-4
"""Float tolerance in beats for matching onset timestamps to grid divisions."""

GRID_NAMES: tuple[str, ...] = (*(name for name, _ in GRID), "off")

GRID_PLAIN = ("1/4", "1/8", "1/16")
"""Standard binary divisions considered clean grid alignments."""

IV_CLASSES = ("0", "1-2", "3-4", "5", "6-11", ">=12")
"""Semitone distance bins for interval classification."""

RECUR_BARS = (1, 2, 4, 8)
"""Bar intervals tested for exact pitch recurrence."""

Layer = Literal["clip", "track"]
"""Automation layer origin classification ('clip' or 'track')."""

EventKind = Literal["float", "bool", "enum"]
"""Data type of automation event values."""


# ------------------------------------------------------------------ XML helpers
def load_xml(path: str | os.PathLike[str]) -> Element:
    """Decompress gzip-compressed .als or .adg project file and parse XML root.

    Args:
        path: Filesystem path to Ableton project or preset file.

    Returns:
        Parsed xml.etree.ElementTree.Element root.
    """
    with gzip.open(path, "rb") as fh:
        return fromstring(fh.read())


def _val(node: Element | None, xpath: str, default: str | None = None) -> str | None:
    """Return Value attribute of first matching XPath element, or default."""
    if node is None:
        return default
    el = node.find(xpath)
    return el.get("Value") if el is not None else default


def _first(node: Element | None, *xpaths: str) -> str | None:
    """Return first non-empty Value attribute found across multiple XPath candidates."""
    for xpath in xpaths:
        got = _val(node, xpath)
        if got:
            return got
    return None


def _fval(node: Element | None, xpath: str, default: float | None = None) -> float | None:
    """Return Value attribute converted to float, or default on error or absence."""
    try:
        return float(_val(node, xpath))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _bool(text: str | None) -> bool | None:
    """Parse Live boolean string ('true'/'false') into bool or None."""
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _number(text: str | None) -> float | None:
    """Convert attribute string to float, returning None on failure."""
    try:
        return float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parents(root: Element) -> dict[Element, Element]:
    """Construct mapping from each child Element to its parent Element in the subtree."""
    out: dict[Element, Element] = {}
    for parent in root.iter():
        for child in parent:
            out[child] = parent
    return out


def _within(node: Element, tag: str, parents: dict[Element, Element]) -> bool:
    """Return True if node is a descendant of an element with specified tag."""
    current = node
    while current in parents:
        current = parents[current]
        if current.tag == tag:
            return True
    return False


def note_name(pitch: int) -> str:
    """Format MIDI note number as scientific pitch string (C4 = 60).

    Args:
        pitch: Integer MIDI key number.

    Returns:
        Note name with scientific octave index.
    """
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


# -------------------------------------------------------------------- the model
@dataclass(frozen=True, slots=True)
class DrumPad:
    """Decoded drum rack pad assignment.

    Attributes:
        note: Decoded MIDI pitch number (128 - receiving_note).
        receiving_note: Raw receiving note attribute from XML.
        name: Optional effective or user-specified pad label.
        sample: Base filename of referenced audio sample.
    """

    note: int
    receiving_note: int
    name: str | None = None
    sample: str | None = None


@dataclass(frozen=True, slots=True)
class Sidechain:
    """Sidechain routing configuration for an audio effect or plugin device.

    Attributes:
        on: State of sidechain toggle.
        target: Routing target path identifier.
        source: Display label of routing source track.
        channel: Selected audio channel or tap point.
        gain: Linear input gain factor (1.0 = 0 dB).
        mix: Dry/wet blend factor.
        listen: Sidechain listen mode state.
        mono: Whether sidechain input is summed to mono.
        eq_on: State of sidechain filter stage.
        eq_freq: Filter cutoff frequency.
        eq_gain: Filter gain adjustment.
        eq_q: Filter resonance/bandwidth factor.
        eq_mode: Filter topology mode.
        configured: True if sidechain is active with a valid non-empty target.

    Note:
        Measured 2026-08-30 against Live 12.4.5: the routing sits on the device as
        ``input_routing_type`` / ``input_routing_channel``, both readable and writable,
        with the sources enumerated in ``available_input_routing_types``. This class
        reads the same wiring out of a saved file, which is what a project Live does not
        have open leaves you.

        ``configured`` is the only field worth branching on. Live attaches a sidechain
        input to every capable device and ``source`` keeps the last known track name
        long after the routing is gone, so only ``on`` together with a real target is a
        finding.

        ``gain`` is a raw factor, not decibels: 1 = 0.00 dB (measured), and 790 of 794
        occurrences across 174 corpus projects are exactly 1. What unit it is beyond
        that point is unverified; ``20*log10`` is a guess, not a measurement.
    """

    on: bool | None
    target: str | None
    source: str | None
    channel: str | None
    gain: float | None
    mix: float | None
    listen: str | None
    mono: bool | None
    eq_on: bool | None
    eq_freq: float | None
    eq_gain: float | None
    eq_q: float | None
    eq_mode: str | None
    configured: bool


@dataclass(frozen=True, slots=True)
class Device:
    """Audio or MIDI device in track processing chain.

    Attributes:
        tag: XML element tag of device.
        name: Resolved device or preset name.
        kind: Classification as stock Live device or third-party plugin.
        on: Enabled state of device.
        param_count: Total parameter slot count.
        parameters: Sequence of (parameter_name, raw_value) tuples.
        sidechain: Optional Sidechain routing configuration.
        depth: Nesting level inside rack branches.
        pads: List of drum pad mappings if device is a drum rack.
    """

    tag: str
    name: str
    kind: Literal["plugin", "stock"]
    on: bool | None
    param_count: int
    parameters: list[tuple[str, str | None]]
    sidechain: Sidechain | None
    depth: int
    pads: list[DrumPad] = field(default_factory=list)

    @property
    def configured_parameters(self) -> list[tuple[str, str | None]]:
        """Return parameter entries that have non-empty assigned names."""
        return [(name, value) for name, value in self.parameters if name and name.strip()]


@dataclass(frozen=True, slots=True)
class Macro:
    """Rack macro knob configuration.

    Attributes:
        index: Macro index (0 to 7).
        name: Configured macro display label.
        value: Current raw value string.
        default: Default parameter value string.
        note: Annotation or description text.

    Note:
        What a macro points at is not in here. In an .adg preset every
        ``ModulationTarget`` carries id 0, so the wiring is not saved at all. To find
        out which parameter hangs on which knob, open the rack in Live. Measured on six
        third-party racks.
    """

    index: int
    name: str
    value: str | None
    default: str | None
    note: str = ""


@dataclass(frozen=True, slots=True)
class Envelope:
    """Summarized automation curve extracted from clip or track timeline.

    Attributes:
        layer: Envelope origin layer ('clip' or 'track').
        event_kind: Type of automation events ('float', 'bool', 'enum').
        pointee_id: Internal target ID referenced by envelope.
        events: Raw count of XML event elements encountered.
        points: Count of distinct time/value support breakpoints.
        time_from: Earliest breakpoint beat position.
        time_to: Latest breakpoint beat position.
        value_min: Lowest value among interior curve points.
        value_max: Highest value among interior curve points.
        largest_gap_beats: Maximum duration between adjacent breakpoints.
        target: Resolved device/parameter path description.

    Note:
        ``pointee_id`` is a set-internal id, not a parameter index. The same id appears
        as the ``Id`` attribute of the ``AutomationTarget`` inside the parameter it
        belongs to, and the parameter's name is that node's parent tag.

        Measured at the artefact 2026-08-29: in one project all four track envelopes
        resolve to ``AutoFilter2 / Filter_Frequency`` on the four tracks that carry a
        filter move, with value ranges in hertz. Not verified against running Live,
        which is why an unresolved id is left as a bare id instead of being guessed at.
        ``pointee_id`` is always reported alongside ``target``.
    """

    layer: Layer
    event_kind: EventKind
    pointee_id: str | None
    events: int
    points: int
    time_from: float
    time_to: float
    value_min: float
    value_max: float
    largest_gap_beats: float
    target: str | None = None

    @property
    def span_beats(self) -> float:
        """Beats covered from first to last support point."""
        return self.time_to - self.time_from

    @property
    def gap_share(self) -> float | None:
        """Largest gap as fraction of total curve span, or None if span is zero."""
        span = self.span_beats
        return (self.largest_gap_beats / span) if span > 0 else None


@dataclass(frozen=True, slots=True)
class Note:
    """Individual MIDI note event.

    Attributes:
        time: Clip-local start position in beats.
        pitch: MIDI pitch number (0 to 127).
        duration: Duration of note in beats.
        velocity: Note velocity (0.0 to 127.0).
    """

    time: float
    pitch: int
    duration: float
    velocity: float


@dataclass(frozen=True, slots=True)
class NoteStats:
    """Statistical metrics computed across MIDI clip note events.

    Attributes:
        notes: Active note event count.
        notes_disabled: Muted or disabled note count.
        beats_per_bar: Metric beats per bar according to meter.
        bars: Length of analyzed section in bars.
        bars_source: Whether bar length was declared by clip bounds or content.
        notes_per_bar: Average note density per bar.
        onsets: Count of distinct onset timestamps.
        poly_share: Fraction of onsets with polyphonic note stacks.
        grid: Counts of onsets matching specific grid divisions.
        grid_share: Fraction of onsets per grid division.
        grid_clean: True if all onsets land cleanly on standard divisions (1/4 to 1/16).
        dur_median: Median note duration in beats.
        dur_common: Most frequent note durations and occurrence counts.
        q_pairs: Count of evaluated successive note pairs.
        q_median: Median articulation ratio (duration / inter-onset interval).
        q_legato_share: Fraction of transitions with legato articulation (q >= 0.95).
        q_staccato_share: Fraction of transitions with staccato articulation (q < 0.6).
        q_common: Most common articulation ratios.
        pitch_min: Minimum MIDI pitch observed.
        pitch_max: Maximum MIDI pitch observed.
        ambitus: Semitone span between pitch_min and pitch_max.
        pitches_distinct: Count of unique pitches used.
        pitch_classes_distinct: Count of unique pitch classes (0-11) used.
        interval_share: Distribution across interval distance categories.
        interval_median: Median semitone leap distance.
        interval_max: Maximum single semitone leap.
        octave_leaps: Count of melodic leaps >= 12 semitones.
        direction_changes: Melodic contour direction reversal rate.
        vel_min: Lowest note velocity.
        vel_max: Highest note velocity.
        vel_span: Velocity range span.
        vel_sd: Population standard deviation of velocities.
        recurrence: Exact bar-periodic pitch pattern repetition fractions.

    Note:
        Derived from a corpus measurement of 103 projects, 833 tracks and 73,539 notes,
        and re-checked per clip against it over 85 clips with no deviation. Everything
        here is measured and carries no target value: judging what the numbers mean for
        a given piece is not this class's business.

        Time signature is the clip's own, else the project's (from the master track),
        else 4/4. In 40 re-checked projects every one of the 2,177 MIDI clips carried
        its own time signature, 4/4 without exception, so the project default is in
        practice a pure fallback and anything other than 4/4 is unmeasured rather than
        excluded.

        ``bars_source`` is ``"declared"`` (``CurrentEnd - CurrentStart`` in the
        Arrangement, ``LoopEnd - LoopStart`` in the Session) or ``"from content"`` when
        the declared length is shorter than the notes reach, because Live keeps notes
        past the clip edge. In that case the note content is rounded up to whole bars;
        an invented length would be worse than a rounded one.
    """

    notes: int
    notes_disabled: int
    beats_per_bar: float
    bars: float | None
    bars_source: Literal["declared", "from content"]
    notes_per_bar: float | None
    onsets: int
    poly_share: float
    grid: dict[str, int]
    grid_share: dict[str, float]
    grid_clean: bool
    dur_median: float
    dur_common: list[tuple[float, int]]
    q_pairs: int
    q_median: float | None
    q_legato_share: float | None
    q_staccato_share: float | None
    q_common: list[tuple[float, int]]
    pitch_min: int
    pitch_max: int
    ambitus: int
    pitches_distinct: int
    pitch_classes_distinct: int
    interval_share: dict[str, float] | None
    interval_median: float | None
    interval_max: int | None
    octave_leaps: int
    direction_changes: float | None
    vel_min: float
    vel_max: float
    vel_span: float
    vel_sd: float
    recurrence: dict[str, float]


@dataclass(frozen=True, slots=True)
class Clip:
    """Session or Arrangement clip metadata and content.

    Attributes:
        name: Name label of clip.
        start: Clip start offset in beats.
        end: Clip end offset in beats.
        is_audio: True for audio clips, False for MIDI clips.
        note_count: Total note count including muted notes.
        pitch_classes: Histogram of pitch class occurrences.
        pitch_low: Lowest pitch present.
        pitch_high: Highest pitch present.
        envelopes: List of automation envelopes stored within clip.
        in_arranger: True if clip resides on Arrangement timeline.
        in_freeze: True if clip resides within a freeze sequencer.
        notes: Optional full sequence of active Note events.
        note_stats: Optional computed NoteStats metrics.
        fingerprint: Content-based hash identifier for duplicate detection.
    """

    name: str | None
    start: float
    end: float
    is_audio: bool
    note_count: int
    pitch_classes: dict[int, int]
    pitch_low: int | None
    pitch_high: int | None
    envelopes: list[Envelope]
    in_arranger: bool | None = None
    in_freeze: bool | None = None
    notes: list[Note] | None = None
    note_stats: NoteStats | None = None
    fingerprint: str | None = None

    @property
    def where(self) -> str:
        """Return compact location string ('Ses' or 'Arr', with '+Frz' if frozen)."""
        if self.in_arranger is None:
            return "?"
        return ("Arr" if self.in_arranger else "Ses") + ("+Frz" if self.in_freeze else "")


@dataclass(frozen=True, slots=True)
class Track:
    """Track configuration, device chain, and clips.

    Attributes:
        name: Track name.
        type: Track category tag (e.g. MidiTrack, AudioTrack, ReturnTrack).
        group_id: Parent group track identifier if grouped.
        volume: Linear mixer volume factor (1.0 = 0 dB).
        panning: Pan position (-1.0 to 1.0).
        sends: Sequence of send levels.
        devices: Devices in processing chain.
        session_clips: Clips located in Session slots.
        arrangement_clips: Clips placed on Arrangement timeline.
        track_automation: Track-level automation envelopes.
    """

    name: str
    type: str
    group_id: str | None
    volume: float | None
    panning: float | None
    sends: list[float | None]
    devices: list[Device]
    session_clips: list[Clip]
    arrangement_clips: list[Clip]
    track_automation: list[Envelope]


@dataclass(frozen=True, slots=True)
class Project:
    """Parsed Ableton project (.als) or device rack preset (.adg).

    Attributes:
        file: Filename of parsed source.
        path: Absolute filesystem path of file.
        live_version: Ableton Live version string.
        creator: Creator field from XML header.
        tempo: Project tempo in BPM.
        tracks: List of parsed tracks.
        macros: List of macro controls if file is a rack preset.
        beats_per_bar: Primary time signature beats per bar.
        is_rack: True if file is a device rack preset rather than full set.
    """

    file: str
    path: str
    live_version: str | None
    creator: str | None
    tempo: float | None
    tracks: list[Track]
    macros: list[Macro] = field(default_factory=list)
    beats_per_bar: float | None = None
    is_rack: bool = False

    def track(self, selector: int | str) -> Track:
        """Retrieve track by integer index or exact name.

        Args:
            selector: Track index or name string.

        Returns:
            Matching Track instance.

        Raises:
            IndexError: If integer index is out of bounds.
            KeyError: If track name is not found.
        """
        if isinstance(selector, int):
            return self.tracks[selector]
        for track in self.tracks:
            if track.name == selector:
                return track
        raise KeyError(f"no track named {selector!r} in {self.file}")


@dataclass(frozen=True, slots=True)
class EnvelopeRef:
    """Envelope reference associated with its enclosing track and clip."""

    track: str
    clip: str | None
    envelope: Envelope


@dataclass(frozen=True, slots=True)
class AutomationReport:
    """Segregated report of clip-level envelopes and track-level automation.

    The two lists are never added together. Measured over 174 foreign projects: 52
    (30 %) have clip envelopes but 159 (91 %) have track automation, so counting clip
    envelopes alone would have reported 110 of them as unautomated. Only
    ``clip_envelopes`` is script-writable, and only inside Session clips.
    """

    clip_envelopes: list[EnvelopeRef]
    track_automation: list[EnvelopeRef]

    @property
    def has_clip_envelopes(self) -> bool:
        """Return True if any clip envelopes are present."""
        return bool(self.clip_envelopes)

    @property
    def has_track_automation(self) -> bool:
        """Return True if any track automation curves are present."""
        return bool(self.track_automation)

    @property
    def automated(self) -> bool:
        """Return True if project contains any automation on either layer."""
        return self.has_clip_envelopes or self.has_track_automation


# -------------------------------------------------------------------- devices
def _device_params(dev: Element) -> tuple[list[tuple[str, str | None]], Literal["plugin", "stock"]]:
    """Extract parameter definitions and classify device as plugin or stock."""
    out: list[tuple[str, str | None]] = []
    for param in dev.findall(".//ParameterList/PluginFloatParameter"):
        name = _val(param, "./ParameterName")
        value = _val(param, ".//Manual")
        if name is not None:
            out.append((name, value))
    if out:
        return out, "plugin"
    for child in dev:
        manual = child.find("./Manual")
        if manual is not None:
            out.append((child.tag, manual.get("Value")))
    return out, "stock"


def _sidechain_of(dev: Element) -> Sidechain | None:
    """Extract Sidechain routing parameters from device element, if present."""
    sc = dev.find("./SideChain")
    if sc is None:
        return None
    routed = sc.find("./RoutedInput")

    def under(node: Element | None, tag: str) -> str | None:
        if node is None:
            return None
        el = node.find(f"./{tag}")
        if el is None:
            el = node.find(f".//{tag}")
        return el.get("Value") if el is not None else None

    on = _bool(_val(sc, "./OnOff/Manual"))
    target = under(routed, "Target")
    configured = bool(on and target and target not in ("AudioIn/None", "None"))
    return Sidechain(
        on=on,
        target=target,
        source=under(routed, "UpperDisplayString"),
        channel=under(routed, "LowerDisplayString"),
        gain=_fval(routed, "./Volume/Manual"),
        mix=_fval(sc, "./DryWet/Manual"),
        listen=_first(dev, ".//SideChainListen/Manual", ".//SideListen/Manual"),
        mono=_bool(_val(dev, ".//SideChainMono/Manual")),
        eq_on=_bool(_first(dev, ".//SideChainEq_On/Manual", ".//SideChainEq/OnOff/Manual")),
        eq_freq=_number(_first(dev, ".//SideChainEq_Freq/Manual", ".//SideChainEq//Freq/Manual")),
        eq_gain=_number(_first(dev, ".//SideChainEq_Gain/Manual", ".//SideChainEq//Gain/Manual")),
        eq_q=_number(_first(dev, ".//SideChainEq_Q/Manual", ".//SideChainEq//Q/Manual")),
        eq_mode=_first(dev, ".//SideChainEq_Mode/Manual", ".//SideChainEq//Mode/Manual"),
        configured=configured,
    )


def _drum_branches(dev: Element) -> list[Element]:
    """Locate drum rack branch elements under Branches or BranchPresets containers.

    Direct containers are searched first (``Branches`` in an .als, ``BranchPresets`` in
    an .adg) so that a drum rack nested inside another rack does not have its pads
    pulled up into the outer one. Falls back to a full subtree search when the layout is
    something else, because an unmeasured layout should yield pads rather than silence.
    """
    for holder in ("./Branches", "./BranchPresets"):
        node = dev.find(holder)
        if node is None:
            continue
        found = [branch for branch in node if branch.tag in DRUM_BRANCH_TAGS]
        if found:
            return found
    return [branch for tag in DRUM_BRANCH_TAGS for branch in dev.iter(tag)]


def _drum_pads_of(dev: Element) -> list[DrumPad]:
    """Extract decoded drum pad assignments from drum rack element."""
    pads: dict[int, DrumPad] = {}
    for branch in _drum_branches(dev):
        raw = _val(branch, ".//ReceivingNote")
        if raw is None:
            for node in branch.iter("ReceivingNote"):
                raw = node.get("Value")
                break
        try:
            receiving = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        sample = None
        for path_el in branch.iter("Path"):
            value = path_el.get("Value", "")
            if value.lower().endswith(SAMPLE_SUFFIXES):
                sample = os.path.basename(value.replace("\\", "/"))
                break
        note = 128 - receiving
        pads[note] = DrumPad(
            note=note,
            receiving_note=receiving,
            name=_first(branch, "./Name/EffectiveName", "./Name/UserName"),
            sample=sample,
        )
    return [pads[key] for key in sorted(pads)]


def _device_name(dev: Element) -> str:
    """Resolve display name of device from user name, plug name, or plugin metadata."""
    name = _first(dev, "./UserName", "./PlugName")
    if name:
        return name
    for info in ("Vst3PluginInfo", "VstPluginInfo", "AuPluginInfo"):
        name = _val(dev, f".//{info}/Name")
        if name:
            return name
    return ""


def _walk_chain(container: Element, out: list[tuple[Element, Device]], prefix: str = "") -> None:
    """Traverse device chain hierarchy recursively, resolving nested rack branches.

    Devices nested inside racks are included, which matters because that is where most
    of the craft sits. In one measured template project all sidechained compressors were
    inside racks: an analysis that reads only the top chain reports "no sidechain" for a
    project that is full of it (failure mode 2 in the module docstring).

    Nesting: ``AudioEffectGroupDevice`` / ``InstrumentGroupDevice`` -> ``Branches`` ->
    ``*Branch`` -> ``DeviceChain`` -> ``*DeviceChain`` -> ``Devices``.
    """
    for dev in list(container):
        if dev.tag in NOT_A_DEVICE:
            continue
        name = _device_name(dev)
        params, kind = _device_params(dev)
        out.append(
            (
                dev,
                Device(
                    tag=dev.tag,
                    name=prefix + (name or dev.tag),
                    kind=kind,
                    on=_bool(_val(dev, "./On/Manual")),
                    param_count=len(params),
                    parameters=params,
                    sidechain=_sidechain_of(dev),
                    depth=prefix.count(">"),
                    pads=_drum_pads_of(dev),
                ),
            )
        )
        branches = dev.find("./Branches")
        if branches is None:
            continue
        for branch in list(branches):
            label = _first(branch, "./Name/EffectiveName", "./Name/UserName") or branch.tag
            for sub in branch.iter("Devices"):
                _walk_chain(sub, out, prefix=f"{prefix}{label} > ")
                break


def _devices_of(track: Element) -> list[tuple[Element, Device]]:
    """Extract ordered device chain for a track, including nested rack devices."""
    chain = track.find(".//DeviceChain/DeviceChain/Devices")
    if chain is None:
        chain = track.find(".//Devices")
    if chain is None:
        return []
    out: list[tuple[Element, Device]] = []
    _walk_chain(chain, out)
    return out


def _macros_of(node: Element) -> list[Macro]:
    """Extract configured macro knob definitions from a rack device element."""
    out: list[Macro] = []
    for index in range(8):
        name = _val(node, f"./MacroDisplayNames.{index}")
        if not name or not name.strip() or re.match(r"^Macro \d+$", name.strip()):
            continue
        out.append(
            Macro(
                index=index,
                name=name.strip(),
                value=_val(node, f"./MacroControls.{index}/Manual"),
                default=_val(node, f"./MacroDefaults.{index}"),
                note=(_val(node, f"./MacroAnnotations.{index}") or "").strip(),
            )
        )
    return out


# ------------------------------------------------------------------- envelopes
def _points(events: list[Element]) -> list[tuple[float, float]]:
    """Parse XML automation events into unique (time, value) breakpoint pairs."""
    raw: list[tuple[float, float]] = []
    for event in events:
        time = event.get("Time")
        value: str | float | None = event.get("Value")
        if time is None:
            continue
        if value in ("true", "false"):
            value = 1.0 if value == "true" else 0.0
        try:
            raw.append((float(time), float(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    body = [(t, v) for t, v in raw if t > PRE_ROLL_TIME]
    seen: set[float] = set()
    unique: list[tuple[float, float]] = []
    for time, value in body:
        key = round(time, 4)
        if key not in seen:
            seen.add(key)
            unique.append((time, value))
    return unique


def _summarise(
    unique: list[tuple[float, float]],
    events_count: int,
    pointee_id: str | None,
    *,
    layer: Layer,
    kind: EventKind,
    target: str | None,
) -> Envelope | None:
    """Calculate summary metrics for parsed breakpoint sequence."""
    if not unique:
        return None
    inner = unique[1:-1] if len(unique) > 2 else unique
    values = [v for _, v in inner] or [v for _, v in unique]
    times = [t for t, _ in unique]
    gap = 0.0
    for i in range(1, len(times)):
        gap = max(gap, times[i] - times[i - 1])
    return Envelope(
        layer=layer,
        event_kind=kind,
        pointee_id=pointee_id,
        events=events_count,
        points=len(unique),
        time_from=min(times),
        time_to=max(times),
        value_min=min(values),
        value_max=max(values),
        largest_gap_beats=gap,
        target=target,
    )


def _automation_targets(
    track: Element,
    devices: list[tuple[Element, Device]],
    parents: dict[Element, Element],
) -> dict[str, str]:
    """Map AutomationTarget Id to readable 'Device / Parameter' descriptor."""
    owner = {element: device.name for element, device in devices}
    out: dict[str, str] = {}
    for parent in track.iter():
        for child in parent:
            if child.tag != "AutomationTarget":
                continue
            ident = child.get("Id")
            if ident is None:
                continue
            node: Element | None = parent
            holder: str | None = None
            while node is not None:
                if node in owner:
                    holder = owner[node]
                    break
                if node.tag == "Mixer":
                    holder = "Mixer"
                    break
                node = parents.get(node)
            out[ident] = f"{holder} / {parent.tag}" if holder else parent.tag
    return out


def _track_automation_of(track: Element, targets: dict[str, str]) -> list[Envelope]:
    """Extract timeline track automation envelopes from track element."""
    out: list[Envelope] = []
    for envelope in track.iter("AutomationEnvelope"):
        pointee = _val(envelope, ".//PointeeId")
        floats = envelope.findall(".//FloatEvent")
        enums = envelope.findall(".//EnumEvent")
        bools = envelope.findall(".//BoolEvent")
        events = floats or enums or bools
        if not events:
            continue
        kind: EventKind = "float" if floats else "enum" if enums else "bool"
        record = _summarise(
            _points(events),
            len(events),
            pointee,
            layer="track",
            kind=kind,
            target=targets.get(pointee or ""),
        )
        if record is not None:
            out.append(record)
    return out


def _clip_envelopes_of(clip: Element, targets: dict[str, str]) -> list[Envelope]:
    """Extract clip automation envelopes from clip element."""
    out: list[Envelope] = []
    for envelope in clip.iter("ClipEnvelope"):
        pointee = _val(envelope, ".//PointeeId")
        events = envelope.findall(".//FloatEvent")
        if not events:
            continue
        record = _summarise(
            _points(events),
            len(events),
            pointee,
            layer="clip",
            kind="float",
            target=targets.get(pointee or ""),
        )
        if record is not None:
            out.append(record)
    return out


# ----------------------------------------------------------------------- notes
def beats_per_bar(node: Element | None, fallback: float = 4.0) -> float:
    """Calculate beats (quarter notes) per bar from TimeSignature element.

    Args:
        node: Element containing TimeSignature definitions.
        fallback: Default value if time signature is absent or invalid.

    Returns:
        Beats per bar calculated as numerator * 4.0 / denominator.

    Note:
        Without this conversion every "per bar" density is wrong for anything but 4/4.

        The first ``RemoteableTimeSignature`` below the node is used, so a project with
        a meter change gets its first meter here. The measured corpus was 4/4 without
        exception; the arithmetic is general anyway, and a meter change is unmeasured
        rather than excluded.
    """
    if node is None:
        return fallback
    ts = node.find(".//TimeSignature/TimeSignatures/RemoteableTimeSignature")
    if ts is None:
        return fallback
    num = _fval(ts, "./Numerator")
    den = _fval(ts, "./Denominator")
    if not num or not den:
        return fallback
    return num * 4.0 / den


def _notes_of(clip: Element) -> tuple[list[Note], int]:
    """Extract all active Note events and count of disabled notes from clip element."""
    out: list[Note] = []
    muted = 0
    for key_track in clip.iter("KeyTrack"):
        midi_key = key_track.find("./MidiKey")
        if midi_key is None:
            continue
        try:
            pitch = int(midi_key.get("Value"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        for event in key_track.iter("MidiNoteEvent"):
            if str(event.get("IsEnabled", "true")).lower() == "false":
                muted += 1
                continue
            try:
                out.append(
                    Note(
                        time=float(event.get("Time")),  # type: ignore[arg-type]
                        pitch=pitch,
                        duration=float(event.get("Duration")),  # type: ignore[arg-type]
                        velocity=float(event.get("Velocity")),  # type: ignore[arg-type]
                    )
                )
            except (TypeError, ValueError):
                continue
    out.sort(key=lambda n: (n.time, n.pitch))
    return out, muted


def _grid_of(time: float) -> str:
    """Determine finest matching metric grid division for onset beat timestamp."""
    for name, step in GRID:
        quotient = time / step
        if abs(quotient - round(quotient)) * step < GRID_TOL:
            return name
    return "off"


def _iv_class(semitones: float) -> str:
    """Classify absolute semitone distance into standard interval bracket."""
    size = abs(semitones)
    if size == 0:
        return "0"
    if size <= 2:
        return "1-2"
    if size <= 4:
        return "3-4"
    if size == 5:
        return "5"
    if size <= 11:
        return "6-11"
    return ">=12"


def _shares(
    values: list[float],
    classes: tuple[str, ...],
    classify: Callable[[float], str],
) -> dict[str, float]:
    """Compute fractional distribution across specified categorical classes."""
    total = float(len(values)) or 1.0
    counted = collections.Counter(classify(value) for value in values)
    return {name: counted.get(name, 0) / total for name in classes}


def _mono_line(notes: list[Note]) -> tuple[list[Note], float]:
    """Reduce note sequence to top voice per onset and compute polyphony share."""
    by_time: collections.OrderedDict[int, list[Note]] = collections.OrderedDict()
    for note in notes:
        by_time.setdefault(round(note.time * 96.0), []).append(note)
    stacked = sum(1 for group in by_time.values() if len(group) > 1)
    line = sorted(
        (max(group, key=lambda n: n.pitch) for group in by_time.values()),
        key=lambda n: n.time,
    )
    return line, (float(stacked) / len(by_time) if by_time else 0.0)


def _recurrence(notes: list[Note], bpb: float, bars: float | None) -> dict[str, float]:
    """Calculate fraction of notes recurring at identical pitch and grid position."""
    if not notes or not bars:
        return {}
    slots: dict[int, set[int]] = {}
    for note in notes:
        slots.setdefault(round(note.time * 4.0), set()).add(note.pitch)
    out: dict[str, float] = {}
    for bar in RECUR_BARS:
        if bars < 2 * bar:
            continue
        offset = round(bar * bpb * 4.0)
        limit = round((bars - bar) * bpb * 4.0)
        base = [key for key in slots if key < limit]
        if len(base) < 4:
            continue
        hits = sum(1 for key in base if slots.get(key + offset) == slots[key])
        out[str(bar)] = float(hits) / len(base)
    return out


def _fingerprint(notes: list[Note]) -> str:
    """Generate compact hash fingerprint of clip note content for deduplication."""
    digest = hashlib.md5(usedforsecurity=False)
    for note in sorted(notes, key=lambda n: (n.time, n.pitch, n.duration, n.velocity)):
        digest.update(
            f"{note.time:.4f}|{note.pitch:d}|{note.duration:.4f}|{int(note.velocity):d};".encode(
                "ascii"
            )
        )
    return digest.hexdigest()[:12]


def _note_stats(
    notes: list[Note],
    disabled: int,
    clip: Element,
    bpb_default: float = 4.0,
    arrangement: bool = False,
) -> NoteStats | None:
    """Calculate musical and statistical metrics for note content in clip."""
    if not notes:
        return None
    bpb = beats_per_bar(clip, bpb_default)
    current_start = _fval(clip, "./CurrentStart", 0.0) or 0.0
    current_end = _fval(clip, "./CurrentEnd", 0.0) or 0.0
    loop_start = _fval(clip, "./Loop/LoopStart", 0.0) or 0.0
    loop_end = _fval(clip, "./Loop/LoopEnd", 0.0) or 0.0
    visible = (current_end - current_start) if arrangement else (loop_end - loop_start)
    span = max(n.time + n.duration for n in notes) - min(n.time for n in notes)
    source: Literal["declared", "from content"]
    if visible > 0 and visible >= span - 1e-6:
        beats, source = visible, "declared"
    else:
        beats, source = max(bpb, math.ceil(span / bpb - 1e-6) * bpb), "from content"
    bars = beats / bpb if bpb else None

    line, poly = _mono_line(notes)
    line_pitches = [n.pitch for n in line]
    intervals = [line_pitches[i + 1] - line_pitches[i] for i in range(len(line_pitches) - 1)]
    directions = [1 if step > 0 else -1 for step in intervals if step != 0]
    turns = sum(1 for i in range(len(directions) - 1) if directions[i] != directions[i + 1])

    ratios: list[float] = []
    for i in range(len(line) - 1):
        distance = line[i + 1].time - line[i].time
        if distance > 1e-9:
            ratios.append(line[i].duration / distance)
    ratio_count = float(len(ratios))

    starts = [n.time for n in notes]
    durations = [n.duration for n in notes]
    pitches = [n.pitch for n in notes]
    velocities = [n.velocity for n in notes]
    share = _shares(starts, GRID_NAMES, _grid_of)
    return NoteStats(
        notes=len(notes),
        notes_disabled=disabled,
        beats_per_bar=bpb,
        bars=bars,
        bars_source=source,
        notes_per_bar=(len(notes) / bars) if bars else None,
        onsets=len(line),
        poly_share=poly,
        grid=dict(collections.Counter(_grid_of(t) for t in starts)),
        grid_share=share,
        grid_clean=all(share[name] == 0.0 for name in GRID_NAMES if name not in GRID_PLAIN),
        dur_median=statistics.median(durations),
        dur_common=collections.Counter(round(d, 4) for d in durations).most_common(4),
        q_pairs=len(ratios),
        q_median=statistics.median(ratios) if ratios else None,
        q_legato_share=(sum(1 for q in ratios if q >= 0.95) / ratio_count) if ratios else None,
        q_staccato_share=(sum(1 for q in ratios if q < 0.6) / ratio_count) if ratios else None,
        q_common=collections.Counter(round(q, 3) for q in ratios).most_common(4),
        pitch_min=min(pitches),
        pitch_max=max(pitches),
        ambitus=max(pitches) - min(pitches),
        pitches_distinct=len(set(pitches)),
        pitch_classes_distinct=len({p % 12 for p in pitches}),
        interval_share=(_shares(intervals, IV_CLASSES, _iv_class) if len(intervals) >= 4 else None),
        interval_median=statistics.median([abs(s) for s in intervals]) if intervals else None,
        interval_max=max(abs(s) for s in intervals) if intervals else None,
        octave_leaps=sum(1 for s in intervals if abs(s) >= 12),
        direction_changes=(float(turns) / (len(directions) - 1) if len(directions) > 1 else None),
        vel_min=min(velocities),
        vel_max=max(velocities),
        vel_span=max(velocities) - min(velocities),
        vel_sd=statistics.pstdev(velocities) if len(velocities) > 1 else 0.0,
        recurrence=_recurrence(notes, bpb, bars),
    )


# ----------------------------------------------------------------------- clips
def _clips_of(
    track: Element,
    arrangement: bool,
    targets: dict[str, str],
    parents: dict[Element, Element],
    with_notes: bool = False,
    bpb_default: float = 4.0,
) -> list[Clip]:
    """Extract Session or Arrangement clips from track element."""
    root = track.find(".//ArrangerAutomation") if arrangement else track
    if root is None:
        return []
    out: list[Clip] = []
    for clip in root.iter():
        if clip.tag not in ("MidiClip", "AudioClip"):
            continue
        pitch_classes: collections.Counter[int] = collections.Counter()
        low, high = 200, -1
        for key_track in clip.iter("KeyTrack"):
            midi_key = key_track.find("./MidiKey")
            if midi_key is None:
                continue
            try:
                pitch = int(midi_key.get("Value"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            count = len(key_track.findall(".//MidiNoteEvent"))
            if count:
                pitch_classes[pitch % 12] += count
                low, high = min(low, pitch), max(high, pitch)
        notes: list[Note] | None = None
        stats: NoteStats | None = None
        fingerprint: str | None = None
        in_arranger: bool | None = None
        in_freeze: bool | None = None
        if with_notes:
            notes, muted = _notes_of(clip)
            in_arranger = _within(clip, "ArrangerAutomation", parents)
            in_freeze = _within(clip, "FreezeSequencer", parents)
            stats = _note_stats(notes, muted, clip, bpb_default, in_arranger)
            fingerprint = _fingerprint(notes) if notes else None
        out.append(
            Clip(
                name=_val(clip, "./Name"),
                start=float(_val(clip, "./CurrentStart", "0") or 0),
                end=float(_val(clip, "./CurrentEnd", "0") or 0),
                is_audio=clip.tag == "AudioClip",
                note_count=sum(pitch_classes.values()),
                pitch_classes=dict(pitch_classes),
                pitch_low=low if high >= 0 else None,
                pitch_high=high if high >= 0 else None,
                envelopes=_clip_envelopes_of(clip, targets),
                in_arranger=in_arranger,
                in_freeze=in_freeze,
                notes=notes,
                note_stats=stats,
                fingerprint=fingerprint,
            )
        )
    return out


# -------------------------------------------------------------------- the read
def _rack_track(root: Element, path: str) -> tuple[Track, list[Macro]]:
    """Construct synthetic track representing a standalone .adg device rack preset."""
    group = root.find(".//GroupDevicePreset/Device/")
    devices: list[Device] = []
    if group is not None:
        params, kind = _device_params(group)
        devices.append(
            Device(
                tag=group.tag,
                name=_val(group, "./UserName") or group.tag,
                kind=kind,
                on=_bool(_val(group, "./On/Manual")),
                param_count=len(params),
                parameters=params,
                sidechain=_sidechain_of(group),
                depth=0,
                pads=_drum_pads_of(group),
            )
        )
    for index, branch in enumerate(root.iter("AudioEffectBranchPreset")):
        label = (_val(branch, "./Name") or "").strip() or f"Chain {index + 1}"
        for preset in branch.findall("./DevicePresets/AbletonDevicePreset/Device/"):
            params, kind = _device_params(preset)
            name = _val(preset, "./UserName") or _device_name(preset)
            devices.append(
                Device(
                    tag=preset.tag,
                    name=f"{label} > {name or preset.tag}",
                    kind=kind,
                    on=_bool(_val(preset, "./On/Manual")),
                    param_count=len(params),
                    parameters=params,
                    sidechain=_sidechain_of(preset),
                    depth=1,
                    pads=_drum_pads_of(preset),
                )
            )
    track = Track(
        name=os.path.splitext(os.path.basename(path))[0] + " (rack)",
        type="RackFile",
        group_id=None,
        volume=None,
        panning=None,
        sends=[],
        devices=devices,
        session_clips=[],
        arrangement_clips=[],
        track_automation=[],
    )
    return track, (_macros_of(group) if group is not None else [])


def read_project(
    path: str | os.PathLike[str],
    *,
    with_notes: bool = False,
    resolve_targets: bool = True,
) -> Project:
    """Parse Ableton project (.als) or rack preset (.adg) into a Project structure.

    Args:
        path: Filesystem path to Ableton file.
        with_notes: Whether to parse note events and compute musical statistics.
        resolve_targets: Whether to resolve automation target IDs to parameter names.

    Returns:
        Project instance representing parsed file contents.
    """
    file_path = os.fspath(path)
    root = load_xml(file_path)
    tempo: float | None = None
    # Do not enter through MasterTrack; that path does not hold up reliably.
    for element in root.iter("Tempo"):
        manual = element.find("./Manual")
        if manual is not None:
            tempo = _number(manual.get("Value"))
            break

    # Read project meter from MasterTrack or MainTrack if notes requested.
    bpb = 4.0
    if with_notes:
        for tag in ("MasterTrack", "MainTrack"):
            master = next(root.iter(tag), None)
            if master is not None:
                bpb = beats_per_bar(master, 4.0)
                break

    is_rack = not any(element.tag in TRACK_TAGS for element in root.iter())
    if is_rack:
        track, macros = _rack_track(root, file_path)
        return Project(
            file=os.path.basename(file_path),
            path=file_path,
            live_version=root.get("MinorVersion"),
            creator=root.get("Creator"),
            tempo=tempo,
            tracks=[track],
            macros=macros,
            beats_per_bar=bpb if with_notes else None,
            is_rack=True,
        )

    tracks: list[Track] = []
    for element in root.iter():
        if element.tag not in TRACK_TAGS:
            continue
        devices = _devices_of(element)
        parents = _parents(element) if (with_notes or resolve_targets) else {}
        targets = _automation_targets(element, devices, parents) if resolve_targets else {}
        mixer = element.find(".//Mixer")
        sends: list[float | None] = []
        if mixer is not None:
            for holder in mixer.findall(".//Sends/TrackSendHolder"):
                sends.append(_fval(holder, ".//Send/Manual"))
        tracks.append(
            Track(
                name=_first(element, "./Name/EffectiveName", "./Name/UserName") or element.tag,
                type=element.tag,
                group_id=_val(element, "./TrackGroupId"),
                volume=_fval(element, ".//Mixer/Volume/Manual"),
                panning=_fval(element, ".//Mixer/Pan/Manual"),
                sends=sends,
                devices=[device for _, device in devices],
                session_clips=_clips_of(element, False, targets, parents, with_notes, bpb),
                arrangement_clips=_clips_of(element, True, targets, parents, with_notes, bpb),
                track_automation=_track_automation_of(element, targets),
            )
        )
    return Project(
        file=os.path.basename(file_path),
        path=file_path,
        live_version=root.get("MinorVersion"),
        creator=root.get("Creator"),
        tempo=tempo,
        tracks=tracks,
        beats_per_bar=bpb if with_notes else None,
        is_rack=False,
    )


def read_track(
    path: str | os.PathLike[str],
    track: int | str,
    *,
    with_notes: bool = True,
    resolve_targets: bool = True,
) -> Track:
    """Parse a single track from an Ableton project file by index or name.

    Args:
        path: Filesystem path to Ableton project file.
        track: Track index (0-based) or exact track name string.
        with_notes: Whether to parse note events for clips in this track.
        resolve_targets: Whether to resolve automation target IDs.

    Returns:
        Parsed Track instance.

    Raises:
        IndexError: If track index is out of bounds.
        KeyError: If track name is not found.
    """
    project = read_project(path, with_notes=with_notes, resolve_targets=resolve_targets)
    return project.track(track)


def read_clip_notes(
    path: str | os.PathLike[str],
    track: int | str,
    clip: int,
    *,
    arrangement: bool = False,
) -> list[Note]:
    """Read active MIDI notes from a specific clip in an Ableton project.

    Args:
        path: Filesystem path to Ableton project file.
        track: Track index or name.
        clip: Clip index within the specified track.
        arrangement: If True, index into arrangement_clips; else session_clips.

    Returns:
        List of Note instances sorted by time and pitch.
    """
    found = read_track(path, track, with_notes=True, resolve_targets=False)
    clips = found.arrangement_clips if arrangement else found.session_clips
    return list(clips[clip].notes or [])


# --------------------------------------------------------------- both layers
def automation_of(project: Project) -> AutomationReport:
    """Collate clip envelopes and track automation layers across project tracks.

    Args:
        project: Parsed Project instance.

    Returns:
        AutomationReport segregating clip envelopes and track automation.
    """
    clip_envelopes: list[EnvelopeRef] = []
    track_automation: list[EnvelopeRef] = []
    for track in project.tracks:
        for clip in track.session_clips:
            for envelope in clip.envelopes:
                clip_envelopes.append(EnvelopeRef(track.name, clip.name, envelope))
        for envelope in track.track_automation:
            track_automation.append(EnvelopeRef(track.name, None, envelope))
    return AutomationReport(clip_envelopes=clip_envelopes, track_automation=track_automation)


def unique_note_clips(project: Project) -> list[tuple[str, Clip]]:
    """Extract list of (track_name, clip) tuples deduplicated by content fingerprint.

    Args:
        project: Project instance parsed with with_notes=True.

    Returns:
        List of unique (track_name, clip) pairs containing notes.
    """
    rows: list[tuple[str, Clip]] = []
    seen: set[tuple[str, str | None, float, float, str | None]] = set()
    for track in project.tracks:
        for clip in track.session_clips + track.arrangement_clips:
            if not clip.note_stats:
                continue
            key = (track.name, clip.name, clip.start, clip.end, clip.fingerprint)
            if key in seen:
                continue
            seen.add(key)
            rows.append((track.name, clip))
    return rows


def to_dict(project: Project) -> dict[str, Any]:
    """Serialize Project dataclass tree to a standard dictionary."""
    return asdict(project)


# --------------------------------------------------------------------- reports
def format_report(project: Project, *, show_devices: bool = False, show_params: int = 0) -> str:
    """Format summary text report of project tracks, routing, and automation.

    Args:
        project: Target Project instance.
        show_devices: Whether to include detailed device chains.
        show_params: Number of device parameters to display per device.

    Returns:
        Multi-line formatted summary string.

    Note:
        Clip envelopes and track automation are never added together here, because they
        are not the same thing and only one of them can be written by a script. Measured
        over 174 foreign projects: 52 (30 %) have clip envelopes, 159 (91 %) have track
        automation, so counting clip envelopes alone would have reported 110 projects as
        unautomated.

        Only ``Track.session_clips`` is walked, deliberately: it is already every clip
        of the track, the Arrangement's included (see the module docstring). Adding
        ``Track.arrangement_clips`` on top would list the same clips twice.
    """
    lines: list[str] = []
    add = lines.append
    tempo = f"{project.tempo:g}" if project.tempo is not None else "?"
    add("=" * 78)
    add(
        f"{project.file}  |  Live {project.live_version}  |  {tempo} BPM  "
        f"|  {len(project.tracks)} tracks"
    )
    add("=" * 78)
    add("")

    add("TRACKS")
    for index, track in enumerate(project.tracks):
        group = "" if track.group_id in (None, "-1") else f"  in group {track.group_id}"
        add(
            f"{index:3d} {track.name[:20]:<20} {track.type.replace('Track', ''):<12} "
            f"{len(track.devices):2d} devices  {len(track.session_clips):2d} session  "
            f"{len(track.arrangement_clips):2d} arrangement{group}"
        )

    spans: dict[tuple[float, float], set[str]] = {}
    for track in project.tracks:
        for clip in track.arrangement_clips:
            label = (clip.name or "?").split()
            spans.setdefault((clip.start, clip.end), set()).add(label[0] if label else "?")
    if spans:
        add("")
        add("SECTIONS (from the arrangement clips)")
        for start, end in sorted(spans):
            names = " ".join(sorted(spans[(start, end)]))[:52]
            add(
                f"  beat {start:6.0f}..{end:6.0f}  bar {start / 4 + 1:5.1f}..{end / 4 + 1:5.1f}  "
                f"{(end - start) / 4:2.0f} bars  {names}"
            )

    add("")
    add(_sidechain_section(project))
    add(_automation_sections(project))

    if project.macros:
        add("")
        add("MACRO KNOBS")
        for macro in project.macros:
            add(
                f"  {macro.index}  {macro.name[:22]:<22} value {macro.value!s:<8} "
                f"default {macro.default!s:<8} {macro.note[:26]}"
            )
        add("  (What they point at is not in the .adg. Every ModulationTarget")
        add("   carries id 0. Open the rack in Live for that.)")

    pads = [(t.name, d) for t in project.tracks for d in t.devices if d.pads]
    if pads:
        add("")
        add("DRUM PADS (decoded as 128 - ReceivingNote; raw values are mirrored)")
        for track_name, device in pads:
            add(f"  {track_name[:18]:<18} {device.name[:24]:<24} {len(device.pads)} pads")
            for pad in device.pads:
                add(
                    f"      {pad.note:3d} {note_name(pad.note):<4} "
                    f"(file {pad.receiving_note:3d})  {pad.sample or pad.name or '-'}"
                )

    if show_devices:
        add("")
        add("DEVICE CHAINS")
        for track in project.tracks:
            if not track.devices:
                continue
            add(f"  {track.name}")
            for index, device in enumerate(track.devices):
                off = "  (off)" if device.on is False else ""
                add(
                    f"    {index:2d} {device.name[:24]:<24} {device.kind:<7} "
                    f"{device.param_count:3d} slots, "
                    f"{len(device.configured_parameters):3d} named{off}"
                )
                for name, value in device.parameters[:show_params]:
                    add(f"        {name[:30]:<30} {value}")

    pitch_classes: collections.Counter[int] = collections.Counter()
    for track in project.tracks:
        for clip in track.session_clips:
            for pitch_class, count in clip.pitch_classes.items():
                pitch_classes[int(pitch_class)] += count
    if pitch_classes:
        add("")
        add(f"PITCH-CLASS CONTENT over all session clips ({sum(pitch_classes.values())} notes)")
        add("  " + "  ".join(f"{NOTE_NAMES[p]}:{n}" for p, n in pitch_classes.most_common()))
        used = sorted(pitch_classes)
        add(f"  {len(used)} distinct pitch classes: " + " ".join(NOTE_NAMES[p] for p in used))
    return "\n".join(lines) + "\n"


def _sidechain_section(project: Project) -> str:
    """Format sidechain routing section for report."""
    lines = ["SIDECHAIN"]
    found: list[tuple[str, str, Sidechain]] = []
    for track in project.tracks:
        for device in track.devices:
            if device.sidechain is not None and device.sidechain.configured:
                found.append((track.name, device.name, device.sidechain))
    if not found:
        lines.append("  no wired sidechain in this file.")
        lines.append("  (Live attaches a sidechain input to EVERY plugin, usually with")
        lines.append("   OnOff=true and target AudioIn/None. Only a routed source counts:")
        lines.append("   otherwise every project on earth reports sidechain.)")
    for track_name, device_name, sidechain in found:
        lines.append(
            f"  {track_name[:18]:<18} {device_name[:20]:<20} "
            f"source {sidechain.source} / {sidechain.channel}"
        )
        lines.append(
            f"  {'':<18} {'':<20} mix {sidechain.mix}  input gain {sidechain.gain} "
            f"(1 = neutral, a factor and not dB)  mono {sidechain.mono}  "
            f"listen {sidechain.listen}"
        )
        if sidechain.eq_on:
            lines.append(
                f"  {'':<18} {'':<20} sidechain EQ: freq {sidechain.eq_freq}  "
                f"gain {sidechain.eq_gain}  Q {sidechain.eq_q}  mode {sidechain.eq_mode}"
            )
    return "\n".join(lines)


def _automation_sections(project: Project) -> str:
    """Format track automation and clip envelope sections for report."""
    report = automation_of(project)
    lines = ["", f"TRACK AUTOMATION in the arrangement ({len(report.track_automation)})"]
    if not report.track_automation:
        lines.append("  none. Automation then sits in clip envelopes (below) or nowhere.")
    for ref in report.track_automation:
        env = ref.envelope
        target = env.target or f"id {env.pointee_id}"
        lines.append(
            f"  {ref.track[:16]:<16} {target[:28]:<28} {env.points:5d} points  "
            f"{env.event_kind:<5} {env.value_min:10.3f}..{env.value_max:<10.3f} "
            f"beat {env.time_from:.0f}..{env.time_to:.0f}"
        )

    lines.append("")
    lines.append(f"CLIP ENVELOPES ({len(report.clip_envelopes)})")
    for ref in report.clip_envelopes:
        env = ref.envelope
        target = env.target or f"id {env.pointee_id}"
        flag = ""
        share = env.gap_share
        if share is not None and share > 0.5:
            flag = (
                f"  ? gap {env.largest_gap_beats:.1f} of {env.span_beats:.0f} beats: "
                "either intent (a move at the section end) or truncated"
            )
        lines.append(
            f"  {ref.track[:16]:<16} {(ref.clip or '?')[:18]:<18} {target[:28]:<28} "
            f"{env.points:5d} points  {env.value_min:8.3f}..{env.value_max:<10.3f}{flag}"
        )
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    """Format ratio (0.0 to 1.0) as percentage string, or '-' if None."""
    return "  -" if value is None else f"{100.0 * value:3.0f}"


def _fixed(value: float | None, places: int = 1) -> str:
    """Format float to fixed decimal places, or '-' if None."""
    return "-" if value is None else f"{value:.{places}f}"


def format_notes_report(
    project: Project,
    *,
    show_list: bool = False,
    notes_max: int = 0,
) -> str:
    """Format detailed text report of MIDI note distributions and metrics.

    Args:
        project: Project instance parsed with with_notes=True.
        show_list: Whether to append individual note listings.
        notes_max: Maximum notes to list per clip (0 for all).

    Returns:
        Multi-line formatted notes analysis string.

    Note:
        Expects a project from ``read_project(..., with_notes=True)``; without notes it
        says so and stays quiet otherwise. Only measured values are printed, never
        target values.
    """
    lines: list[str] = []
    add = lines.append
    rows = unique_note_clips(project)
    bpb = project.beats_per_bar or 4.0
    add("")
    add(f"NOTES PER MIDI CLIP ({len(rows)} clips with notes, meter {bpb:g} beats/bar)")
    if not rows:
        add("  none. Either an audio-only or empty project, or read_project() ran")
        add("  without with_notes=True.")
        return "\n".join(lines) + "\n"

    add("  Percentages are shares WITHIN the clip. Register in raw MIDI numbers")
    add("  (Live names the same note one octave lower).")
    add("  q = note length / distance to the next onset of the same line;")
    add("  intervals, q and recurrence use the highest note per onset.")
    add("  Every note content is written out ONCE (#n); each further occurrence")
    add("  is one line below it, with track and start beat.")
    add("")

    first: dict[str, int] = {}
    for track_name, clip in rows:
        stats = clip.note_stats
        if stats is None:
            continue
        if clip.fingerprint in first:
            add(
                f"      = #{first[clip.fingerprint]:<3d} {track_name[:16]:<16} "
                f"{(clip.name or '?')[:18]:<18} {clip.where:<5} start beat {clip.start:.0f}"
            )
            continue
        first[clip.fingerprint or ""] = len(first) + 1
        index = first[clip.fingerprint or ""]
        add(
            f"  #{index:<3d} {track_name[:16]:<16} {(clip.name or '?')[:18]:<18} "
            f"{clip.where:<5} {_fixed(stats.bars, 1):>5} bars ({stats.bars_source:<12}) "
            f"{stats.notes:5d} notes {_fixed(stats.notes_per_bar, 2):>5}/bar  "
            f"poly {_percent(stats.poly_share)}%"
        )
        grid = stats.grid_share
        clean = "   all on 1/4-1/16" if stats.grid_clean else ""
        add(
            f"      grid 1/4 {_percent(grid['1/4'])}  1/8 {_percent(grid['1/8'])}  "
            f"1/16 {_percent(grid['1/16'])}  1/32 {_percent(grid['1/32'])}  "
            f"1/8T {_percent(grid['1/8T'])}  1/16T {_percent(grid['1/16T'])}  "
            f"off {_percent(grid['off'])} %{clean}"
        )
        common = " ".join(f"{value:g}:{count}" for value, count in stats.dur_common)
        add(
            f"      length med {_fixed(stats.dur_median, 3)}  most common {common}   "
            f"q med {_fixed(stats.q_median, 2)} ({_percent(stats.q_legato_share)}% legato, "
            f"{_percent(stats.q_staccato_share)}% detached, {stats.q_pairs} pairs)"
        )
        add(
            f"      MIDI {stats.pitch_min}..{stats.pitch_max}  ambitus {stats.ambitus:2d}  "
            f"{stats.pitches_distinct:2d} pitches / {stats.pitch_classes_distinct} classes  "
            f"velocity {stats.vel_min:.0f}..{stats.vel_max:.0f} "
            f"(SD {_fixed(stats.vel_sd, 1)})  {stats.notes_disabled} muted"
        )
        shares = stats.interval_share
        if shares:
            add(
                f"      intervals 0 {_percent(shares['0'])}  1-2 {_percent(shares['1-2'])}  "
                f"3-4 {_percent(shares['3-4'])}  5 {_percent(shares['5'])}  "
                f"6-11 {_percent(shares['6-11'])}  >=12 {_percent(shares['>=12'])} %  "
                f"med {_fixed(stats.interval_median, 1)}  max {stats.interval_max}  "
                f"octave leaps {stats.octave_leaps}  "
                f"turns {_percent(stats.direction_changes)}%"
            )
        else:
            add("      intervals: fewer than 4 steps on the line, so no distribution.")
        if stats.recurrence:
            parts = "  ".join(
                f"{bar} bar{'s' if bar != '1' else ''} {_percent(value)}%"
                for bar, value in sorted(stats.recurrence.items(), key=lambda kv: int(kv[0]))
            )
            add(
                f"      recurrence {parts}  (share of notes returning n bars later, note-identical)"
            )

    add(_notes_summary(rows))
    if not show_list:
        add("  Full note lists come from Clip.notes; format_notes_report(show_list=True)")
        add("  prints them.")
        return "\n".join(lines) + "\n"
    add(_notes_listing(rows, first, notes_max))
    return "\n".join(lines) + "\n"


def _notes_summary(rows: list[tuple[str, Clip]]) -> str:
    """Format aggregate totals and grid distribution summary across clips."""
    lines = [""]
    total = 0
    muted = 0
    pool: collections.Counter[str] = collections.Counter()
    fingerprints: collections.Counter[str] = collections.Counter()
    clean = 0
    for _track_name, clip in rows:
        stats = clip.note_stats
        if stats is None:
            continue
        total += stats.notes
        muted += stats.notes_disabled
        clean += 1 if stats.grid_clean else 0
        for name, count in stats.grid.items():
            pool[name] += count
        fingerprints[clip.fingerprint or ""] += 1
    pooled_total = float(sum(pool.values())) or 1.0
    doubled = sum(count - 1 for count in fingerprints.values() if count > 1)
    grid = "  ".join(f"{name} {100.0 * pool[name] / pooled_total:.1f}%" for name in GRID_NAMES)
    lines.append(
        f"  Total: {total} notes in {len(rows)} clips, {muted} muted.  Pooled grid: {grid}"
    )
    lines.append(f"  All on 1/4-1/16: {clean} of {len(rows)} clips.")
    lines.append(
        f"  De-duplication: {doubled} clips carry a note content that already occurred "
        f"({len(fingerprints)} distinct\n  contents, most frequent {max(fingerprints.values())}x). "
        "Those are arrangement repetitions or layered\n  twin tracks; averaging over them "
        "averages one phrase several times."
    )
    lines.append("  CAUTION: the three lines above count every occurrence, so a clip copied")
    lines.append("  24 times counts 24 times. For a distribution, de-duplicate by")
    lines.append("  `fingerprint` first and then take the median over clips, which is how")
    lines.append("  the corpus numbers quoted in this module were arrived at.")
    return "\n".join(lines)


def _notes_listing(rows: list[tuple[str, Clip]], first: dict[str, int], notes_max: int) -> str:
    """Format full note listings per unique clip fingerprint."""
    lines = ["", "NOTE LISTS (time clip-local in beats, can be negative for a pick-up)"]
    once: list[tuple[str, Clip]] = []
    seen: set[str] = set()
    for track_name, clip in rows:
        key = clip.fingerprint or ""
        if key in seen:
            continue
        seen.add(key)
        once.append((track_name, clip))
    pending = sum(len(clip.notes or []) for _track_name, clip in once)
    lines.append(
        f"  {len(once)} distinct note contents, {pending} notes. The {len(rows) - len(once)} "
        "repeated occurrences\n  carry the same notes and are not printed again."
    )
    for track_name, clip in once:
        notes = clip.notes or []
        index = first.get(clip.fingerprint or "", 0)
        lines.append("")
        lines.append(f"  #{index}  {track_name} / {clip.name or '?'}  ({len(notes)} notes)")
        shown = notes if notes_max <= 0 else notes[:notes_max]
        for start in range(0, len(shown), 3):
            lines.append(
                "   "
                + " |".join(
                    f"{n.time:9.3f} {n.pitch:3d} {note_name(n.pitch):<4} "
                    f"d {n.duration:6.3f} v {n.velocity:3.0f}"
                    for n in shown[start : start + 3]
                )
            )
        if len(shown) < len(notes):
            lines.append(f"    ... {len(notes) - len(shown)} more (notes_max=0 shows all)")
    return "\n".join(lines)
