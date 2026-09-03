"""Channel B: the saved project file. Knows nothing about sockets.

The second of the two routes into Ableton (docs/architecture.md, 'two channels into
Ableton'), and a peer of the Live Object Model rather than a workaround for it. Channel A
sees Live's memory while Live runs; this package sees the saved bytes on disk. The two
are different states: an .als is only ever as fresh as the last Ctrl+S
(``docs/limits.md`` §1).

The split inside mirrors the risk.

:mod:`~ableton_maestro.als.read` is strictly read-only. It unpacks the gzip XML and
answers what the LOM cannot be asked about a project Live does not have open: what is in
a foreign project, where an envelope's breakpoints actually sit, how a sidechain is
wired, which plugin parameters were ever configured. Measured over 174 professional
projects: 52 (30 %) carry clip envelopes but 159 (91 %) carry track automation, so a
clip-only count misreports 110 of them as unautomated. It therefore reports the two
automation layers separately and never adds them together.

:mod:`~ableton_maestro.als.write` is the half that can destroy a day's work, so it
behaves accordingly: a timestamped, hash-verified backup before every write with no
opt-out, a refusal while Live is running or has just saved, and a re-read of the file
afterwards to prove what landed. It exists for the writes Channel A cannot reach: a
project Live does not have open, and track automation, which the LOM cannot write at all.

A sidechain source is not among those writes, though it is widely assumed to be.
Measured 2026-08-30 against Live 12.4.5: a device carries ``input_routing_type`` and
``input_routing_channel``, and a write to them reads back as ``applied``. The .als route
to a sidechain stays anyway, because a project Live has not loaded has no LOM to ask.

Neither module imports :mod:`ableton_maestro.client`, opens a socket, or needs a running
Live. A socket in this package is a layering bug, the same way a pitch in ``client.py``
is one.
"""

from __future__ import annotations

from ableton_maestro.als.read import (
    AutomationReport,
    Clip,
    Device,
    DrumPad,
    Envelope,
    EnvelopeRef,
    Macro,
    Note,
    NoteStats,
    Project,
    Sidechain,
    Track,
    automation_of,
    beats_per_bar,
    format_notes_report,
    format_report,
    load_xml,
    note_name,
    read_clip_notes,
    read_project,
    read_track,
    to_dict,
    unique_note_clips,
)
from ableton_maestro.als.write import (
    AlsRefused,
    AlsWriteError,
    Change,
    LiveCheck,
    SidechainSlot,
    WriteResult,
    list_sidechain_slots,
    restore_backup,
    set_attribute,
    set_sidechain_source,
)

__all__ = [
    "AlsRefused",
    "AlsWriteError",
    "AutomationReport",
    "Change",
    "Clip",
    "Device",
    "DrumPad",
    "Envelope",
    "EnvelopeRef",
    "LiveCheck",
    "Macro",
    "Note",
    "NoteStats",
    "Project",
    "Sidechain",
    "SidechainSlot",
    "Track",
    "WriteResult",
    "automation_of",
    "beats_per_bar",
    "format_notes_report",
    "format_report",
    "list_sidechain_slots",
    "load_xml",
    "note_name",
    "read_clip_notes",
    "read_project",
    "read_track",
    "restore_backup",
    "set_attribute",
    "set_sidechain_source",
    "to_dict",
    "unique_note_clips",
]
