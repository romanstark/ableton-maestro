"""Ableton Maestro Live Remote Script — generic Live Object Model bridge.

Runs inside Live's embedded Python runtime to provide a loopback TCP server (port 9878),
path resolution across the Live Object Model, and seventeen protocol handlers.

Core design rules:
- Generic LOM bridge: protocol handlers implement capabilities unexpressible via pure
  paths (notes, envelopes, browser hierarchy, events, and enum reflection).
- Exception containment: all exceptions are caught and converted to structured error codes.
- Safe path resolution: all indices are bounds-checked; method execution via path is blocked.
- Non-blocking ping: answered immediately on client thread without dispatching to main thread.
- Legacy note API refusal: blocks Clip.set_notes / remove_notes to prevent modal dialog locks.
- Read-back verification: lom_set verifies written state and distinguishes applied/clamped/not_observed.

Runtime constraints:
- Must run in Live's embedded interpreter (Python 2/3 compatible, no type annotations, no f-strings).
"""

from __future__ import absolute_import, print_function, unicode_literals

import codecs
import json
import re
import socket
import sys
import threading
import time
import traceback
from collections import deque

from _Framework.ControlSurface import ControlSurface

try:  # Python 2 fallback kept because Live's interpreter history is long
    import Queue as queue
except ImportError:
    import queue

try:
    import Live
except ImportError:  # pragma: no cover - only reachable outside Live
    Live = None


# --------------------------------------------------------------------------
# Identity and transport
# --------------------------------------------------------------------------

SCRIPT_NAME = "ableton-maestro"
SCRIPT_VERSION = "0.4.0"

#: What protocol 1 means for a caller — the shapes it may rely on:
#:
#: * ``ping`` answers ``{"pong", "script_version", "uptime"}`` and reads no Live
#:   object at all, so it stays answerable while the main thread is busy
#:   (protocol §5.1).
#: * ``lom_get`` on a LOM collection answers a **list** of encoded elements with
#:   a ``count``, each element carrying its own resolver path — never one opaque
#:   handle (protocol §5.3).
#: * ``lom_set`` answers ``read_back``: ``applied``, ``clamped`` or
#:   ``not_observed``. ``clamped`` is true only where Live really changed the
#:   value on the way in (protocol §5.4).
#: * ``lom_describe`` children carry ``count``, ``class`` and ``is_collection``,
#:   so "how many tracks are there" is one call and not a probe loop
#:   (protocol §5.6).
#: * ``browser_walk`` entries carry ``kind``, query matches carry ``match_rank``
#:   and come back ranked rather than in discovery order, and a walk that ran
#:   out of budget says which one stopped it in ``truncated_by``
#:   (protocol §5.10).
#:
#: Results are objects so that they can **grow** a field without the version
#: moving; only a changed or removed field is a break (protocol §4). Branch on
#: ``protocol_version`` — ``script_info`` reports it — and never on
#: ``script_version``.
PROTOCOL_VERSION = 1

#: Loopback only. The channel has no authentication — Live's Remote Script API
#: offers no auth primitive — so binding to 127.0.0.1 is not a default, it is
#: the security control. Anything that widens this to ``0.0.0.0`` hands every
#: machine on the network unauthenticated write access to the open set; if this
#: file is ever merged with code from elsewhere, check this line first
#: (``docs/protocol.md`` §1).
HOST = "127.0.0.1"

#: The port this script listens on, and the one the client connects to by
#: default. Live can host several Control Surfaces at once, so a second bridge
#: on the same machine is only a matter of giving it a different number — but
#: this one is compiled into the script and a change costs a Live restart
#: (docs/limits.md).
DEFAULT_PORT = 9878

#: Exactly the handlers of ``docs/protocol.md`` §5. Reported by ``script_info``
#: so the server can refuse to send what this build does not answer.
HANDLERS = (
    "ping",
    "script_info",
    "lom_get",
    "lom_set",
    "lom_call",
    "lom_describe",
    "lom_batch",
    "notes_get",
    "notes_set",
    "automation_read",
    "automation_write",
    "automation_clear",
    "browser_walk",
    "events_observe",
    "events_drain",
    "events_clear",
    "enum_names",
)

RECV_CHUNK = 8192

#: A client that never sends valid JSON would otherwise grow this buffer until
#: Live runs out of memory. There is no framing to resynchronise on, so the
#: only sane response is to drop the buffer and say so.
MAX_REQUEST_BYTES = 8 * 1024 * 1024

#: Why this script never touches Live's legacy note API — ``Clip.set_notes`` and
#: ``Clip.remove_notes``, the pair Ableton deprecated in favour of the Live 11
#: extended calls — in the words the caller gets.
#:
#: Measured 2026-08-29 against Live 12.4.5: calling ``clip.set_notes`` or
#: ``clip.remove_notes`` from a Remote Script makes Live open a MODAL dialog —
#: "Ein spezielles MIDI-Remote-Skript verwendet einen älteren Prozess zum
#: Modifizieren von MIDI-Noten... MPE, Probability, Velocity Deviation und
#: Release Velocity... Soll das Skript fortfahren?" with Continue and Decline
#: buttons. That dialog blocks Live's MAIN THREAD until a human clicks it, and
#: every queued request stalls behind it: in an automated server it is a total
#: outage, not a deprecation warning. The same API also silently discards MPE,
#: probability, velocity_deviation and release_velocity.
#:
#: So there is no fallback to it, not even "just in case": a fallback that can
#: freeze the host is not a safety net. Live 11 or later is a requirement, and a
#: build without the modern calls is told exactly that.
#:
#: The three method names below are the ones Live really has, and they are the
#: ones the version guard tests for. ``Clip.set_notes_extended`` is the name the
#: symmetry invites and it **does not exist on Clip** — read off a running Live
#: 12.4.5 via ``lom_describe`` on 2026-08-29, and absent from the LOM
#: documentation for every Live. Guarding on a method that does not exist would
#: tell a Live 12 user their Live is older than 11. ``add_new_notes`` is the
#: Live 11+ write call.
_MODERN_NOTE_API = "get_notes_extended, add_new_notes, remove_notes_extended"

_LEGACY_NOTE_API_REFUSED = (
    "{0} is unavailable, so this is a Live version older than 11. Ableton Maestro "
    "requires the Live 11+ note API (" + _MODERN_NOTE_API + ") and deliberately "
    "does not fall back to the legacy one: the legacy API (a) silently discards "
    "MPE, probability, velocity_deviation and release_velocity, and (b) makes Live "
    "open a modal dialog that blocks Live's main thread until a human clicks it, "
    "which halts this server completely (measured 2026-08-29 against Live 12.4.5). "
    "Upgrade to Live 11 or later."
)

#: The Live 11 note extension fields, in the order they are reported. Live 10
#: has none of them, and the handler drops them and says so rather than
#: inventing defaults (protocol §5.8).
_NOTE_EXTENSION_FIELDS = ("probability", "velocity_deviation", "release_velocity")


# --------------------------------------------------------------------------
# The method allowlist
# --------------------------------------------------------------------------
#
# Methods are never reachable through the path resolver (``docs/protocol.md``
# §6 rule 2). They go through ``lom_call``, and only if "<Class>.<method>" is
# listed here. An arbitrary method call inside Live's process can hang or crash
# an audio application that usually has the user's unsaved work in it.
#
# The allowlist lives in the script, not in the catalog, so that it cannot be
# widened from outside Live. The catalog's ``access: [call]`` says what the
# server will *offer*; this frozenset says what the script will *do*, and the
# script wins.
#
# Matching is against every name the object's type answers to: ``__name__``,
# ``__qualname__``, and each base class in the MRO. That is why the nested
# ``View`` classes appear both bare and qualified — Live's C-level types do not
# report ``__qualname__`` consistently across versions (not measured; both
# spellings are listed so either works).
#
# What is on the list is decided one entry at a time, and the reasoning sits
# next to the entry. The hazards that keep something OFF the list are these:
# a call that **latches** (a ``begin`` whose ``end`` may never arrive), a call
# whose target this script cannot see (a modal dialog, a GUI selection), and a
# call whose safety depends on a state rather than on its name.
#
# Deliberately NOT on the list, with reasons:
#
#   Song.begin_undo_step / Song.end_undo_step
#       Latching, and the most expensive kind: a ``begin`` without its ``end``
#       leaves Live accumulating one open undo step for the rest of the
#       session, and nothing here can notice. The catalog offers both rows
#       (``song.begin_undo_step``, ``song.end_undo_step``); the script refuses
#       them, and the script wins. A caller that wants one undoable unit of
#       work should send one ``lom_batch``.
#   Clip.set_notes / Clip.remove_notes
#       Live's legacy note API, deprecated by Ableton and worse than merely
#       lossy: it is not called anywhere in this script — see
#       ``_LEGACY_NOTE_API_REFUSED`` for the modal dialog it opens.
#       (``Clip.set_notes_extended`` is on no list because it does not exist;
#       the modern write call is ``Clip.add_new_notes``.)
#   Song.trigger_session_record, Track/ClipSlot fire-button state
#       Latching state. A begin without an end leaves Live recording or a
#       button held down, and there is no reliable way to notice from here.
#       ``Clip.set_fire_button_state`` *is* listed — one clip's button, one
#       object a caller already holds a path to.
#   DeviceParameter.begin_gesture / end_gesture
#       Same shape of hazard: an unmatched ``begin_gesture`` leaves the
#       parameter in a gesture that only the GUI can end.
#   Song.scrub_by
#       Scrubbing the transport with nothing that ends the gesture.
#       ``Clip.scrub`` is listed because ``Clip.stop_scrub`` always ends it.
#   Device.store_chosen_bank
#       Its safety depends on the device's bank count, which a name cannot
#       carry — and getting it wrong kills the Live process. The measurement is
#       at the bank getters in the list below.
#   (The Simpler and Sample editors were the open block here. They were read off
#   a running Live 12.4.5 on 2026-08-30 with a sample loaded in Slicing mode, and
#   are now on the list - six on SimplerDevice, five on Sample. The entries carry
#   what the reading settled, including that the slice editors are not on the
#   device at all.)
#   Application.press_current_dialog_button
#       Clicks whatever modal dialog happens to be open. Unknowable target.
#       It stays off the list even though a modal dialog is a MEASURED hazard
#       rather than a hypothetical one: on 2026-08-29 against Live 12.4.5,
#       Live's legacy note API opened one that blocked Live's main thread
#       until a human clicked it (``_LEGACY_NOTE_API_REFUSED``). The answer to
#       that is to stop opening the dialog, not to teach the script to click
#       away dialogs it cannot see. A blind click can accept "discard changes?"
#       or a plugin's licence prompt just as easily as the one that is in the
#       way. If a dialog ever does wedge the server, that is a monitoring
#       problem - ``ping`` still answers, and says so - and a human's job.
#   MixerDevice
#       Has no methods worth calling — volume, panning, sends, cue_volume,
#       crossfader and track_activator are all properties and are reachable by
#       path. Their display strings come from ``DeviceParameter.str_for_value``,
#       which *is* allowed.
#
# Destructive members (delete_track, delete_clip, delete_device, delete_scene)
# ARE listed: they are gated by the catalog's ``destructive`` flag and the
# server-side executor's ``confirm=True``, which is where that decision belongs.
METHOD_ALLOWLIST = frozenset([
    # -- Song: undo, transport, structure -------------------------------
    "Song.undo",
    "Song.redo",
    # Pure reads that happen to be methods rather than properties in the LOM
    # (catalog ``song.can_undo`` / ``song.can_redo``, both read-verified
    # 2026-08-29 against Live 12.4.5). They are the guard before calling undo
    # blindly, and calling them cannot change anything.
    "Song.can_undo",
    "Song.can_redo",
    "Song.start_playing",
    "Song.stop_playing",
    "Song.continue_playing",
    "Song.play_selection",
    "Song.stop_all_clips",
    "Song.jump_by",
    "Song.tap_tempo",
    "Song.capture_midi",
    "Song.re_enable_automation",
    "Song.create_midi_track",
    "Song.create_audio_track",
    "Song.create_return_track",
    "Song.duplicate_track",
    "Song.delete_track",
    "Song.delete_return_track",
    "Song.create_scene",
    "Song.duplicate_scene",
    "Song.delete_scene",
    "Song.set_or_delete_cue",
    "Song.jump_to_next_cue",
    "Song.jump_to_prev_cue",
    # The only way to reorder a device chain or move a device between tracks.
    # Structural but undoable, and not latching. Call verified 2026-08-30 against
    # Live 12.4.5: the order changed, and the row song.move_device is verified.
    # It takes objects, so the device and target chain travel as path references.
    "Song.move_device",

    # -- Locators -------------------------------------------------------
    "CuePoint.jump",

    # -- Tracks and chains ----------------------------------------------
    "Track.delete_device",
    "Track.duplicate_clip_slot",
    "Track.stop_all_clips",
    "Track.jump_in_running_session_clip",
    # THE Session-to-Arrangement step, and the one the server's `arrange` tool
    # is built on. Measured 2026-08-29 against Live 12.4.5: the method exists on
    # the Track object. Automation is writable only in Session clips and travels
    # with the clip when it is duplicated, so without this call there is no
    # route for automation into the Arrangement whatsoever.
    # Catalog row: arrangement_clip.duplicate_from_session.
    "Track.duplicate_clip_to_arrangement",
    # Its repair. An Arrangement clip on the wrong beat, or one that arrived
    # without its envelope, cannot be fixed in place - it has to go and be
    # duplicated again, and ClipSlot.delete_clip only reaches Session clips.
    # Destructive, and gated where destructive belongs: the catalog's flag and
    # the executor's confirm=True (catalog row arrangement_clip.delete).
    "Track.delete_clip",
    # The other way onto the timeline, for material that needs no automation
    # and therefore no Session detour. Catalog row arrangement_clip.create_midi
    # is verified against Live 12.4.5; on an older Live the method may be absent,
    # and lom_call then says so honestly instead of blaming policy.
    "Track.create_midi_clip",
    # Copying a device with its settings. There is no other way to do it, and
    # it changes nothing that was there before.
    "Track.duplicate_device",
    "Chain.delete_device",
    # Symmetry with Chain.delete_device. Racks are ordinary production work and
    # a chain is where their devices live. Expected from the LOM, not measured.
    "Chain.duplicate_device",
    # A drum pad's chain is a DrumChain, and whether that class reports `Chain`
    # anywhere in its MRO is unverified - so it is listed under its own name
    # too rather than relied upon (catalog drum_pad.chain_delete_device,
    # read-verified 2026-08-29).
    "DrumChain.delete_device",
    "DrumChain.duplicate_device",
    "DrumPad.delete_all_chains",

    # -- Clip slots ------------------------------------------------------
    "ClipSlot.create_clip",
    "ClipSlot.delete_clip",
    "ClipSlot.create_audio_clip",
    "ClipSlot.duplicate_clip_to",
    "ClipSlot.fire",
    "ClipSlot.stop",

    # -- Clips ------------------------------------------------------------
    "Clip.fire",
    "Clip.stop",
    "Clip.quantize",
    "Clip.quantize_pitch",
    "Clip.crop",
    "Clip.duplicate_loop",
    "Clip.duplicate_region",
    "Clip.select_all_notes",
    "Clip.deselect_all_notes",
    "Clip.move_playing_pos",
    "Clip.clear_all_envelopes",
    "Clip.clear_envelope",
    "Clip.create_automation_envelope",
    "Clip.automation_envelope",
    # Pure conversions between the clip's two time bases. They read and return
    # a number and touch nothing; warping an audio clip needs both.
    "Clip.beat_to_sample_time",
    "Clip.sample_to_beat_time",

    # -- Scenes -----------------------------------------------------------
    "Scene.fire",
    "Scene.fire_as_selected",

    # -- Devices and parameters -------------------------------------------
    # -- pure reads ------------------------------------------------------------
    # Every one of these only reports; none of them changes anything in the set.
    # Refusing a read costs a capability and buys no safety: this list exists to
    # stop a call that can hang Live or damage a set, and a getter can do
    # neither. All of them have catalog rows and were probed 2026-08-30 against
    # Live 12.4.5.
    "Application.get_major_version",
    "Application.get_minor_version",
    "Application.get_bugfix_version",
    "Application.get_version_string",
    "Application.get_document",
    "Application.has_option",
    "Application.get_build_id",
    "Application.get_variant",
    # The extended note readers. notes_get already uses get_notes_extended
    # internally; listing it here makes the same read reachable through lom_call
    # for a caller who wants a pitch/time window rather than the whole clip.
    "Clip.get_notes_extended",
    "Clip.get_all_notes_extended",
    "Clip.get_selected_notes_extended",
    "Clip.get_notes_by_id",
    # The plugin's own parameter names, as opposed to what its strip holds.
    # device.parameters reports the strip; measured 2026-09-01 against Live
    # 12.4.5, that is between 1 and 29 entries depending on the plugin and on
    # nothing else anybody has identified. This method is present on every
    # PluginDevice described so far, so unlike the bank getters below it demonstrably
    # exists. What it RETURNS is unmeasured, and so is its signature: probe it with
    # no arguments first, which makes Live report the real one.
    "PluginDevice.get_parameter_names",
    # -- NOT allowlisted: the bank getters, because they do not exist ---------
    # get_bank_count / get_bank_name / get_bank_parameters were listed here on the
    # assumption that they are plugin-only. Measured 2026-08-30 against Live
    # 12.4.5: they exist on NO device class. CompressorDevice, RackDevice and
    # PluginDevice -- a real VST3 -- each answered "has no method ... in this Live
    # version". The catalog rows that offered them are deleted rather than marked
    # broken: a method that does not exist is not a capability that is refused.
    # -- NOT allowlisted: Device.store_chosen_bank ----------------------------
    # Measured 2026-08-30 against Live 12.4.5: calling store_chosen_bank(0, 0)
    # through lom_call on a Reverb -- a device with ZERO banks -- killed Live with
    # EXCEPTION_ACCESS_VIOLATION. Not an error return, a process death, with
    # whatever was unsaved in it.
    #
    # The warning at the head of this list -- that an arbitrary method call can
    # hang or crash Live -- is not a caution. It is that measurement, with a date
    # and a Windows exception code behind it.
    #
    # store_chosen_bank is off the list because the allowlist knows a NAME and the
    # safety of this call depends on a STATE -- how many banks the device has,
    # which only that device instance can say. A name cannot carry that.
    # There is no getter to reach for first, either: get_bank_count and its two
    # siblings are not on this list because they exist on no device class in
    # 12.4.5 -- see the note above. Nothing here can ask how many banks a device
    # has, which is the whole reason store_chosen_bank cannot be made safe.
    # -- editing surface --------------------------------------------------------
    # Ordinary editing operations, not the hazards this list was written against:
    # none of them can hang Live, none opens a dialog, and every one is undoable
    # through Live's own undo (song.undo, verified separately). All have catalog
    # rows and were probed 2026-08-30 against Live 12.4.5.
    #
    # The note editors are Live's modern API. remove_notes_extended is what
    # notes_set calls internally; listing it makes the same operation reachable for
    # a caller who wants a pitch/time window rather than a whole-clip replace --
    # with the warning that add_new_notes ADDS. Sending it twice does not correct a
    # melody, it doubles it (measured: 63 notes in the clip, 23 written, 86
    # afterwards); only notes_set pairs the removal with the write. Live's LEGACY
    # set_notes/remove_notes stay excluded, and for a reason that is measured
    # rather than cautious: they open a modal dialog that halts Live's main thread
    # until a human clicks it.
    "Clip.add_new_notes",
    "Clip.apply_note_modifications",
    "Clip.remove_notes_extended",
    "Clip.remove_notes_by_id",
    "Clip.duplicate_notes_by_id",
    "Clip.replace_selected_notes",
    # Warp markers -- the answer to an open question in docs/limits.md.
    "Clip.add_warp_marker",
    "Clip.move_warp_marker",
    "Clip.remove_warp_marker",
    # Scrub and the fire button: transport-shaped, and stop_scrub always ends it.
    "Clip.scrub",
    "Clip.stop_scrub",
    "Clip.set_fire_button_state",
    # Clip view: cosmetic, and the only way to show a caller what was just written.
    "View.show_envelope",
    "View.hide_envelope",
    "View.select_envelope_parameter",
    "View.show_loop",
    # Scene capture from what is currently playing.
    "Song.capture_and_insert_scene",
    # Song.re_enable_automation does the whole set at once; this is the same
    # call for one parameter, and the finer instrument is the safer one.
    "DeviceParameter.re_enable_automation",

    # -- Racks: macros and variations ---------------------------------------
    #
    # All nine were read off a running Live 12.4.5 on 2026-08-29 (catalog rows
    # rack.*, status verified). None of them latches, none opens a dialog, and
    # macros and variations are ordinary production work on any rack.
    #
    # Listed under `RackDevice` only. Live's concrete classes are
    # InstrumentGroupDevice, DrumGroupDevice, AudioEffectGroupDevice and
    # MidiEffectGroupDevice, and this entry holds for them only if `RackDevice`
    # appears in their MRO - which `type_candidates` checks and nobody has
    # verified. If a live probe answers method_not_allowed on a real rack, the
    # fix is to add the four concrete spellings here, not to widen the matcher.
    "RackDevice.add_macro",
    "RackDevice.remove_macro",
    "RackDevice.rename_macro",
    "RackDevice.macro_map",
    "RackDevice.randomize_macros",
    "RackDevice.store_variation",
    "RackDevice.recall_selected_variation",
    "RackDevice.recall_last_used_variation",
    "RackDevice.delete_selected_variation",
    # str_for_value is the only route to physical units (dB, Hz) and is called
    # with the parameter's CURRENT value everywhere in this script. Passing
    # param.min or param.max instead has taken Live down: a crash inside a
    # plugin's own C code (measured with a 3rd-party sampler plugin) cannot be
    # caught by try/except, the process is simply gone. Callers that want the
    # endpoints are calibrating and accept that risk explicitly.
    "DeviceParameter.str_for_value",

    # -- Browser ------------------------------------------------------------
    # -- Simpler and its Sample -----------------------------------------------
    # Read off a running Live 12.4.5 on 2026-08-30, with a sample loaded and the
    # device in Slicing mode, which is what the previous note here asked for.
    #
    # The class is SimplerDevice, and the six below are the ones it really has.
    # Whether a warp call can run is state, not name: with a sample loaded,
    # can_warp_as was true while can_warp_double and can_warp_half were false, so
    # a caller reads the matching can_* property first and the call is refused by
    # Live otherwise.
    "SimplerDevice.crop",
    "SimplerDevice.reverse",
    "SimplerDevice.warp_as",
    "SimplerDevice.warp_double",
    "SimplerDevice.warp_half",
    "SimplerDevice.guess_playback_length",
    # The slice editors are NOT on SimplerDevice -- Live does not list them there.
    # They belong to the Sample, reached at <device>.sample, which is null until a
    # sample is loaded. The catalog rows that pointed at the device were wrong
    # about the object, not about the method.
    "Sample.insert_slice",
    "Sample.move_slice",
    "Sample.remove_slice",
    "Sample.clear_slices",
    "Sample.reset_slices",
    "Browser.load_item",
    "Browser.preview_item",
    "Browser.stop_preview",
    "Browser.relation_to_hotswap_target",

    # -- Views (nested classes: listed bare and qualified) -------------------
    "View.show_view",
    "Application.View.show_view",
    "View.hide_view",
    "Application.View.hide_view",
    "View.focus_view",
    "Application.View.focus_view",
    "View.is_view_visible",
    "Application.View.is_view_visible",
    "View.available_main_views",
    "Application.View.available_main_views",
    "View.scroll_view",
    "Application.View.scroll_view",
    "View.zoom_view",
    "Application.View.zoom_view",
    "View.toggle_browse",
    "Application.View.toggle_browse",
    "View.select_device",
    "Song.View.select_device",
    # Selecting a track's instrument is a GUI move and nothing else; it is what
    # a caller does before showing a device to a human (catalog
    # track.select_instrument, on song.tracks[{track}].view).
    "View.select_instrument",
    "Track.View.select_instrument",
])


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class LomError(Exception):
    """A structured failure, carrying one of the codes in protocol §4.

    Every handler raises this instead of returning a string. The dispatcher
    turns it into ``{"status": "error", "code": ..., "message": ...}`` plus any
    extra fields (``path`` above all) the caller needs to locate the problem.
    """

    def __init__(self, code, message, **extra):
        Exception.__init__(self, message)
        self.code = code
        self.message = message
        self.extra = extra

    def to_response(self):
        payload = {"status": "error", "code": self.code, "message": self.message}
        for key, value in self.extra.items():
            if value is not None:
                payload[key] = value
        return payload


# --------------------------------------------------------------------------
# A read-back that Live has not shown yet
# --------------------------------------------------------------------------

#: How long :meth:`AbletonMaestro._resolve_deferred` will wait for the second
#: read, in seconds. It is a single ``schedule_message(1, ...)`` tick - Live's
#: tick is of the order of 100 ms - so this is a generous ceiling and not an
#: expectation. It is clamped again against whatever is left of the handler's
#: own main-thread budget, which is itself below the client's (protocol §8), so
#: a second read can never push a write past the client's 20 s.
DEFERRED_READ_BACK_TIMEOUT = 2.0

#: The tick delay for the second read. ``0`` would run the follow-up in the same
#: batch of scheduled work and could see the same stale value; ``1`` puts it
#: after the writing task has finished, which is precisely the boundary the
#: measurement showed the write landing on.
DEFERRED_READ_BACK_DELAY = 1

_NOT_OBSERVED_NOTE = (
    "the property still reads its previous value, so this write has NOT been "
    "observed to take effect. Two things look identical from here: Live clamped "
    "the value to the one already stored, or Live has not applied the write yet. "
    "Some LOM properties are applied asynchronously - measured 2026-08-29 against "
    "Live 12.4.5, song.current_song_time applies after the current main-thread "
    "task ends, so a read inside that task sees the old value. Nothing was "
    "clamped as far as this script can prove, which is why 'clamped' is false. "
    "Read the path again to find out."
)


class DeferredReadBack(object):
    """A handler result that needs one more look at Live before it is true.

    ``lom_set`` runs *inside* a scheduled main-thread task, and some LOM writes
    land only once that task returns. A handler therefore cannot
    wait for its own write: scheduling a follow-up and blocking on it from the
    main thread deadlocks, because the follow-up cannot run until the blocking
    task returns.

    So the handler does not wait. It hands back one of these, and
    :meth:`AbletonMaestro._run_on_main_thread` - which runs on the *client*
    thread, where blocking is exactly what it already does - schedules the
    second read a tick later and waits for that instead.

    ``resolve`` runs on Live's main thread and returns the final result.
    ``unresolved(reason)`` returns the honest result for "there was no second
    read", and is used wherever a second read is impossible: a re-entrant call
    already on the main thread, a batch that owns the main thread for its whole
    run, a scheduler that refuses, or no time left in the budget.
    """

    def __init__(self, resolve, unresolved):
        self.resolve = resolve
        self.unresolved = unresolved


# --------------------------------------------------------------------------
# The path resolver (protocol §6)
# --------------------------------------------------------------------------

#: ``segment = name | name "[" int "]"``. No slices, no expressions, no minus
#: sign — a negative index is a parse failure rather than a Python wrap-around
#: that would silently address the wrong track.
_SEGMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[([0-9]+)\])?$")

#: The roots of protocol §6. ``song.view`` is reachable as an attribute of
#: ``song`` and needs no root of its own.
_ROOTS = ("song", "app")

#: Attributes that raise on some track kinds. The two error strings are Live's
#: own, measured ("Main and Return Tracks have no 'Arm' state!", "Main, Group
#: and Return Tracks have no arrangement clips"); the guard conditions below are
#: derived from them and have NOT been re-measured here.
#: Each entry is (attribute, flag, required_flag_value, why).
#: Return and main tracks also lack ``arrangement_clips`` but are not foldable,
#: so those still surface as ``live_error`` carrying Live's own text.
_TRACK_GUARDS = (
    ("arm", "can_be_armed", True,
     "group, return and main tracks have no arm state"),
    ("current_monitoring_state", "can_be_armed", True,
     "only armable tracks have a monitoring state"),
    ("arrangement_clips", "is_foldable", False,
     "group tracks have no arrangement clips"),
    ("fold_state", "is_foldable", True,
     "only foldable (group) tracks have a fold state"),
)

#: Never followed by ``lom_describe``: ``canonical_parent`` walks back up the
#: object graph and turns any recursive describe into a cycle.
_DESCRIBE_SKIP = frozenset(["canonical_parent"])

#: Most attributes ``enum_names`` will describe for one container.
_ATTRIBUTE_REPORT_CAP = 64

#: Longest list ``lom_get`` will encode before reporting ``truncated``.
_LIST_CAP = 512

#: Node budget for one ``lom_describe`` call, so a deep describe cannot sit on
#: Live's main thread past the client's 10 s read timeout (protocol §8).
_DESCRIBE_MAX_NODES = 400


def parse_path(path):
    """Split a path into ``[(name, index_or_None), ...]``, validating syntax.

    Raises :class:`LomError` with ``bad_path`` for anything the grammar of
    protocol §6 does not allow — including a leading underscore, which would
    otherwise let a caller reach into a Live object's private internals.
    """
    if not isinstance(path, str):
        raise LomError("bad_path",
                       "path must be a string, got {0}".format(type(path).__name__))
    text = path.strip()
    if not text:
        raise LomError("bad_path", "path is empty")

    segments = []
    for raw in text.split("."):
        match = _SEGMENT_RE.match(raw)
        if match is None:
            raise LomError(
                "bad_path",
                "segment {0!r} is not 'name' or 'name[int]' - no slices, no negative "
                "indices, no expressions".format(raw),
                path=path)
        name = match.group(1)
        if name.startswith("_"):
            raise LomError("bad_path",
                           "segment {0!r} addresses a private attribute".format(name),
                           path=path)
        index = match.group(2)
        segments.append((name, int(index) if index is not None else None))

    root = segments[0][0]
    if root not in _ROOTS:
        raise LomError(
            "bad_path",
            "root must be one of {0}, got {1!r}".format(", ".join(_ROOTS), root),
            path=path)
    if segments[0][1] is not None:
        raise LomError("bad_path", "the root segment takes no index", path=path)
    return segments


def _track_guard_reason(obj, name):
    """Return why ``name`` is unavailable on this track, or ``None``.

    Applied on every attribute read, so guards hold for ``lom_get``,
    ``lom_set``, ``lom_describe`` and ``lom_batch`` alike — "anywhere you touch
    tracks generically" (protocol §6 rule 4).
    """
    try:
        if not hasattr(obj, "can_be_armed"):
            return None  # not a track-like object; nothing to guard
    except Exception:
        return None
    for attribute, flag, wanted, why in _TRACK_GUARDS:
        if attribute != name:
            continue
        try:
            actual = bool(getattr(obj, flag))
        except Exception:
            return None  # cannot decide; let Live speak for itself
        if actual != wanted:
            return "{0} ({1}={2})".format(why, flag, actual)
    return None


def _get_attribute(obj, name, so_far):
    """One attribute step of the walk, with the errors of protocol §4."""
    # An empty clip slot answers None for .clip, and the next step down then
    # reports "NoneType has no attribute 'name'" -- the symptom, from a class the
    # caller never asked for. Say which step went missing and what makes one.
    if obj is None:
        empty_slot = so_far.endswith(".clip")
        raise LomError(
            "no_such_path",
            ("this clip slot is empty, so its .clip is None and {0!r} has nothing to read "
             "from. Make a clip first - clip_slot.create_clip for MIDI, create_audio_clip "
             "for audio.").format(name)
            if empty_slot else
            "{0} is None here, so {1!r} has nothing to read from".format(so_far, name),
            path=so_far)
    reason = _track_guard_reason(obj, name)
    if reason is not None:
        raise LomError("no_such_path",
                       "{0!r} is not available on this track: {1}".format(name, reason),
                       path=so_far)
    try:
        value = getattr(obj, name)
    except AttributeError:
        raise LomError("no_such_path",
                       "{0} has no attribute {1!r}".format(class_name(obj), name),
                       path=so_far)
    except RuntimeError as exc:
        raise LomError("live_error", str(exc), path=so_far)
    if callable(value):
        raise LomError(
            "method_not_allowed",
            "{0}.{1} is a method - methods are not reachable by path, use "
            "lom_call".format(class_name(obj), name),
            path=so_far)
    return value


def _get_index(collection, index, so_far, collection_path=None):
    """One index step, bounds-checked before use.

    An ``IndexError`` escaping a handler kills the client connection, so the
    check is never optional (docs/architecture.md, 'the Remote Script').

    ``so_far`` already carries the index being resolved, so it is the path that
    does not exist -- right for the error's ``path``, wrong for its prose.
    ``collection_path`` is the same path without the index and is what the
    message blames. Blaming ``so_far`` produces "index 0 is beyond the 0 entries
    of ...arrangement_clips[0]", which names the missing thing as the container it
    is missing from.
    """
    named = collection_path if collection_path is not None else so_far
    try:
        length = len(collection)
    except TypeError:
        raise LomError("bad_path",
                       "{0} is not indexable".format(class_name(collection)),
                       path=so_far)
    if index >= length:
        raise LomError("index_out_of_range",
                       "index {0} is beyond the {1} entries of {2}".format(
                           index, length, named),
                       path=so_far)
    try:
        return collection[index]
    except (IndexError, KeyError, TypeError, RuntimeError) as exc:
        raise LomError("live_error", str(exc), path=so_far)


def class_name(obj):
    """The unqualified class name, which is how the LOM names its types."""
    return type(obj).__name__


def type_candidates(obj):
    """Every name the allowlist may use for ``obj``'s class.

    ``__name__`` covers the ordinary case, ``__qualname__`` the nested ``View``
    classes, and the MRO makes an allowlist entry hold for subclasses too.
    """
    kind = type(obj)
    names = set()
    for attribute in ("__name__", "__qualname__"):
        value = getattr(kind, attribute, None)
        if isinstance(value, str):
            names.add(value)
    for base in getattr(kind, "__mro__", ()):
        value = getattr(base, "__name__", None)
        if isinstance(value, str):
            names.add(value)
    return names


def is_lom_object(value):
    """True for anything that must become a handle rather than be serialised."""
    return not isinstance(value, (bool, int, float, str, bytes, list, tuple, dict, type(None)))


def is_lom_collection(value):
    """True for a LOM collection — ``Vector``, ``BrowserItemVector`` and friends.

    The LOM's collections are **not** Python lists. ``song.tracks`` is a
    ``Vector`` and ``browser.drums.children`` is a ``BrowserItemVector``; both
    answer ``len()`` and ``[i]``, and neither passes ``isinstance(value, (list,
    tuple))``. That one wrong test is enough to make ``lom_describe`` report no
    count for any collection and ``browser_walk`` refuse to recurse, and both
    failures look like something else entirely — which is why the question has a
    function of its own rather than an ``isinstance`` at each call site.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes, dict)):
        return False
    if isinstance(value, (list, tuple)):
        return True
    kind = type(value)
    if not hasattr(kind, "__len__"):
        return False
    return hasattr(kind, "__getitem__") or hasattr(kind, "__iter__")


def collection_length(value):
    """``len(value)`` for a collection, or ``None`` when it cannot be had.

    ``None`` means *genuinely could not be determined* — some Live collections
    raise on ``len`` instead of answering — and never "did not try". A caller
    that sees ``null`` here knows the script asked and Live refused.
    """
    if value is None or isinstance(value, (str, bytes, dict)):
        return None
    try:
        return int(len(value))
    except Exception:
        return None


def sequence_items(value, limit=None):
    """A LOM collection's elements as a real Python list, defensively.

    ``list(vector)`` is the obvious spelling and it is not the reliable one.
    Measured 2026-08-29 against Live 12.4.5: a ``BrowserItemVector`` answers
    ``len()`` and ``children[0]`` perfectly well while ``list(children)`` fails,
    and a single failed ``list()`` empties a whole browser level, which makes
    ``browser_walk`` look as though it ignored ``depth``. So indexing comes
    first and iteration is the fallback.

    ``limit`` stops after that many elements, so encoding a capped list does not
    have to materialise a collection of thousands first.

    Returns ``[]`` when neither works: an empty walk beats an exception loose in
    Live's process (docs/architecture.md, 'the Remote Script').
    """
    length = collection_length(value)
    if length is not None:
        wanted = length if limit is None else min(length, limit)
        items = []
        for position in range(wanted):
            try:
                items.append(value[position])
            except Exception:
                break
        if items or wanted == 0:
            return items
    try:
        items = list(value)
    except Exception:
        return []
    if limit is not None and len(items) > limit:
        return items[:limit]
    return items


def same_lom_object(first, second):
    """True when two references name the same Live object.

    Object-valued properties cannot be compared with ``==``. Live hands out a
    fresh Python wrapper for every attribute read, so two reads of
    ``song.view.selected_track`` are two Python objects naming one track, and
    value equality would answer nonsense. ``_live_ptr`` is the underlying C++
    address and is the only identity the LOM exposes; ``is`` is tried first and
    ``==`` last, because ``_live_ptr`` is read from the LOM and is not promised
    on every class (not measured per class).
    """
    if first is None or second is None:
        return first is None and second is None
    if first is second:
        return True
    try:
        left = getattr(first, "_live_ptr", None)
        right = getattr(second, "_live_ptr", None)
    except Exception:
        left = right = None
    if left is not None and right is not None:
        return left == right
    try:
        return bool(first == second)
    except Exception:
        return False


def _safe_getattr(obj, name):
    """``getattr`` that answers ``None`` instead of raising.

    Live's modules are C extensions and some attributes raise on access rather
    than being absent. Nothing may escape a handler (docs/architecture.md), and
    an introspection sweep that dies on one odd attribute reports nothing at all
    about the several hundred that were fine.
    """
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _public_names(obj):
    """Sorted public attribute names of ``obj``, or an empty list."""
    try:
        return sorted(name for name in dir(obj) if not name.startswith("_"))
    except Exception:
        return []


def _int_members(candidate):
    """``{name: value}`` for the public attributes of ``candidate`` that are ints.

    Two members minimum: a single constant is not an enumeration. ``bool`` is
    excluded even though it subclasses int, because a pair of booleans is not an
    enum and attaching meanings to true/false would invent them.

    **It no longer demands that EVERY attribute be an int.** That rule was a
    guess about the shape of Live's enum containers, and it was wrong:
    ``Live.Song.Quantization`` resolved fine on 2026-08-31 and was rejected by
    it. What the container actually looks like is now reported by
    :meth:`_handle_enum_names` rather than judged here.
    """
    members = {}
    for name in _public_names(candidate):
        value = _safe_getattr(candidate, name)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        members[name] = value
    return members if len(members) >= 2 else {}


def _attribute_report(candidate):
    """Every public attribute of ``candidate``, with its type and a short value.

    The handler's answer when the int members do not tell the whole story. A
    probe that says "this is not what I expected" and nothing else forces the
    next guess to be blind - and a guess about this script costs a Live restart.
    """
    report = {}
    for name in _public_names(candidate)[:_ATTRIBUTE_REPORT_CAP]:
        value = _safe_getattr(candidate, name)
        entry = {"type": type(value).__name__}
        if isinstance(value, bool):
            entry["value"] = value
        elif isinstance(value, (int, float)):
            entry["value"] = value
        elif isinstance(value, str):
            entry["value"] = value[:80]
        else:
            try:
                entry["repr"] = repr(value)[:80]
            except Exception:
                entry["repr"] = "<unreprable>"
        report[name] = entry
    return report


def object_handle(value, path):
    """The handle form of protocol §7 — never the object itself."""
    handle = {"__lom__": class_name(value), "path": path}
    try:
        name = getattr(value, "name", None)
    except Exception:
        name = None
    if isinstance(name, str):
        handle["name"] = name
    return handle


def object_reference(value, path):
    """The ``before``/``after`` form of an object-valued property (§5.4, §7).

    Same shape as :func:`object_handle`, with one difference that carries the
    whole point: ``path`` is the path that names *the object*, not the property
    it was read from, and it is present only when this script honestly knows
    one. After a reference write that is the ``__path__`` the caller sent, and
    only when the read-back proved the same object. Otherwise the reference is
    reported by class and name, which is all that can be said about it.
    """
    if value is None:
        return None
    handle = {"__lom__": class_name(value)}
    if path is not None:
        handle["path"] = path
    try:
        name = getattr(value, "name", None)
    except Exception:
        name = None
    if isinstance(name, str):
        handle["name"] = name
    return handle


def encode_value(value, path=None, cap=_LIST_CAP):
    """Encode a resolved value for the wire. Returns ``(encoded, type, truncated)``.

    JSON-native values pass through. Everything else becomes a handle
    (protocol §7): serialising a Live object would produce a snapshot that
    looks like state and is not.
    """
    if value is None:
        return None, "null", False
    if isinstance(value, bool):
        return value, "bool", False
    if isinstance(value, int):
        return int(value), "int", False
    if isinstance(value, float):
        return float(value), "float", False
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace"), "string", False
    if isinstance(value, str):
        return value, "string", False
    if is_lom_collection(value):
        # song.tracks, device.parameters, clip_slots, browser children: the LOM's
        # collections are Vector objects, not Python lists, and an isinstance
        # against (list, tuple) catches none of them. Encoded as an object,
        # lom_get("song.tracks") answers with one opaque {"__lom__": "Vector"}
        # handle - no elements, no count - and the only way left to learn how
        # many tracks there are is to probe indices until index_out_of_range, one
        # round trip each (measured 2026-08-29 against Live 12.4.5). Encoded as a
        # list, each element carries its own resolver path, so a single read
        # hands back addresses that can be used directly.
        encoded = []
        truncated = False
        items = sequence_items(value, cap)
        total = collection_length(value)
        if total is not None and total > len(items):
            truncated = True
        for position, item in enumerate(items):
            if position >= cap:
                truncated = True
                break
            child_path = None
            if path is not None:
                child_path = "{0}[{1}]".format(path, position)
            item_value, _item_type, item_truncated = encode_value(item, child_path, cap)
            encoded.append(item_value)
            truncated = truncated or item_truncated
        return encoded, "list", truncated
    if isinstance(value, dict):
        return dict((str(k), encode_value(v, None, cap)[0]) for k, v in value.items()), \
            "dict", False
    return object_handle(value, path), "object", False


def parameter_display(param):
    """The physical reading of a DeviceParameter, or ``None``.

    Most parameter values are normalised, not physical: a filter cutoff of 0.5
    is not a frequency and a track volume of 0.85 is 0 dB (protocol §7). This
    is the only route to the real unit, and VST2 plugins never provide one — no
    amount of retrying changes that (measured).

    Only ever called with the parameter's *current* value; see the note on
    ``str_for_value`` in the allowlist header.
    """
    formatter = getattr(param, "str_for_value", None)
    if formatter is None:
        return None
    try:
        return str(formatter(param.value))
    except Exception:
        return None


def find_parameter(target, parent, last_name):
    """The DeviceParameter a path is talking about, or ``None``.

    Both spellings resolve: ``...mixer_device.volume`` (the parameter itself)
    and ``...mixer_device.volume.value`` (its float).
    """
    if hasattr(target, "str_for_value") and hasattr(target, "value"):
        return target
    if last_name == "value" and parent is not None and hasattr(parent, "str_for_value"):
        return parent
    return None


def values_equal(first, second):
    """Compare two read-back values, tolerating float representation noise.

    Exact equality would report ``clamped: true`` for a value Live stored
    faithfully and handed back one ULP away, which would make the single most
    important field in the protocol cry wolf.

    The tolerance has to be a FLOAT32 one, not a float64 one. Live stores
    parameter values as 32-bit floats and hands them back widened: ask for 0.85
    and the read-back is 0.8500000238418579, which is exactly float32(0.85).
    Float32 epsilon is about 1.2e-7, so a faithful round trip can differ by far
    more than double precision would explain -- measured 2026-08-30, a write
    landing 6e-9 from the request was reported clamped by the old 1e-9 threshold,
    which is the crying wolf this docstring was already warning about. 1e-6 sits
    an order of magnitude above float32 noise and orders below any real clamp.
    """
    if isinstance(first, bool) or isinstance(second, bool):
        return bool(first) == bool(second)
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        difference = abs(float(first) - float(second))
        scale = max(1.0, abs(float(first)), abs(float(second)))
        return difference <= 1e-6 * scale
    return first == second


def coerce_to(before, requested):
    """Coerce ``requested`` to the type Live currently holds in that slot.

    JSON has no int/float distinction worth trusting and no bool coercion at
    all, so a tempo sent as ``124`` must not become an int and ``arm`` sent as
    ``1`` must become ``True``.
    """
    if isinstance(before, bool):
        if isinstance(requested, bool):
            return requested
        if isinstance(requested, (int, float)):
            return bool(requested)
        if isinstance(requested, str):
            lowered = requested.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
        raise LomError("type_error",
                       "expected a boolean, got {0!r}".format(requested))
    if isinstance(before, int) and not isinstance(before, bool):
        if isinstance(requested, bool):
            return int(requested)
        try:
            as_float = float(requested)
        except (TypeError, ValueError):
            raise LomError("type_error",
                           "expected an integer, got {0!r}".format(requested))
        as_int = int(round(as_float))
        return as_int
    if isinstance(before, float):
        try:
            return float(requested)
        except (TypeError, ValueError):
            raise LomError("type_error",
                           "expected a number, got {0!r}".format(requested))
    if isinstance(before, str):
        if isinstance(requested, str):
            return requested
        if isinstance(requested, (int, float, bool)):
            return str(requested)
        raise LomError("type_error", "expected a string, got {0!r}".format(requested))
    # Unknown or currently-None slot: hand the value over unchanged and let
    # Live decide. Its refusal is more informative than a guess here.
    return requested


# --------------------------------------------------------------------------
# Automation helpers
# --------------------------------------------------------------------------

#: Every step of a written curve is one ``insert_step()`` call on Live's main
#: thread, so a careless resolution gets coarsened rather than freezing the UI.
MAX_AUTOMATION_STEPS = 4000
MIN_AUTOMATION_RESOLUTION = 1.0 / 128.0
DEFAULT_AUTOMATION_RESOLUTION = 1.0 / 16.0

#: At ``time = 0`` Live returns the parameter's DEFAULT value, not the curve
#: (measured across nine curves: Auto Filter Frequency reports 0.8997 at t=0
#: while the curve actually starts at 0.42). Reads and
#: verification samples therefore start one 64th of a beat in.
READ_EPSILON = 1.0 / 64.0

DEFAULT_AUTOMATION_SAMPLES = 64
MAX_AUTOMATION_SAMPLES = 512

#: Wall-clock budget for the ``insert_step`` loop of one ``automation_write``,
#: in seconds. Below the handler's main-thread budget (18.0 s, see
#: ``_MAIN_THREAD_TIMEOUTS``), which is below the client's 20 s write timeout
#: (protocol §8), so the loop gives up first and can say how far it got.
#:
#: That "how far" is the whole point. A curve is written one ``insert_step`` at
#: a time — there is no bulk call in the Live API of 11 or 12 — so a fine
#: resolution over a long clip is thousands of main-thread calls, and stopping
#: halfway leaves a real, audible, half-written envelope that the LOM cannot
#: roll back. Reported as a bare timeout, that is indistinguishable from a write
#: that never started: the exact silent-failure shape this whole project exists
#: to refuse (docs/architecture.md, "Read-back as a principle").
AUTOMATION_WRITE_TIME_BUDGET = 15.0

#: Protocol §5.9 names the modes ``linear``, ``hold``, ``exponential``,
#: ``ease_in`` and ``ease_out``. Internally there are four curves: ``ease_in``
#: is ``exponential``, and ``ease_out`` is its reciprocal, which curve tables
#: usually call ``logarithmic``. Every spelling in this table is accepted and
#: the canonical name is reported back, so nobody has to guess which one ran.
INTERPOLATION_ALIASES = {
    "linear": "linear",
    "lin": "linear",
    "exponential": "exponential",
    "exp": "exponential",
    "ease_in": "exponential",
    "logarithmic": "logarithmic",
    "log": "logarithmic",
    "ease_out": "logarithmic",
    "hold": "hold",
    "step": "hold",
    "constant": "hold",
    "none": "hold",
}


def normalize_interpolation(mode):
    """Canonical interpolation name, rejecting anything unknown."""
    key = "{0}".format(mode if mode is not None else "linear").strip().lower()
    key = key.replace("-", "_").replace(" ", "_")
    if key not in INTERPOLATION_ALIASES:
        raise LomError(
            "type_error",
            "unknown interpolation {0!r} - use one of: {1}".format(
                mode, ", ".join(sorted(set(INTERPOLATION_ALIASES.keys())))))
    return INTERPOLATION_ALIASES[key]


def interpolate(start_value, end_value, fraction, mode, exponent):
    """Value between two breakpoints at ``0.0 <= fraction <= 1.0``."""
    if fraction <= 0.0:
        return start_value
    if fraction >= 1.0:
        return end_value
    if mode == "hold":
        return start_value
    if mode == "linear":
        return start_value + (end_value - start_value) * fraction
    curve = exponent if exponent and exponent > 0.0 else 2.0
    if mode == "logarithmic":
        curve = 1.0 / curve
    return start_value + (end_value - start_value) * (fraction ** curve)


def clamp_parameter_value(param, value):
    """Coerce to float and clamp into the parameter's own range."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise LomError("type_error",
                       "parameter value must be a number, got {0!r}".format(value))
    try:
        minimum = float(param.min)
        maximum = float(param.max)
    except (AttributeError, TypeError, ValueError):
        return value
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def prepare_automation_points(param, points, default_mode, default_exponent):
    """Validate, clamp and sort the breakpoints of a curve.

    Accepts ``{"time": t, "value": v}`` objects — optionally with their own
    ``interpolation`` / ``exponent`` for the segment that starts at them — and
    plain ``[time, value]`` pairs.

    More than ``MAX_AUTOMATION_STEPS`` incoming points is a rejection, not a
    coarsening: the caller who pre-sampled 8000 points wants a different call
    (two breakpoints plus an interpolation mode), not a silently thinned curve.
    """
    if not points:
        raise LomError("type_error", "at least one automation point is required")
    if len(points) > MAX_AUTOMATION_STEPS:
        raise LomError("type_error",
                       "too many automation points ({0}, max {1}) - describe the shape "
                       "with breakpoints plus 'interpolation' instead".format(
                           len(points), MAX_AUTOMATION_STEPS))

    try:
        fallback_exponent = float(default_exponent)
    except (TypeError, ValueError):
        fallback_exponent = 2.0

    prepared = []
    for position, point in enumerate(points):
        if isinstance(point, dict):
            point_time = point.get("time", point.get("beat", None))
            point_value = point.get("value", None)
            point_mode = normalize_interpolation(point.get("interpolation", default_mode))
            try:
                point_exponent = float(point.get("exponent", fallback_exponent))
            except (TypeError, ValueError):
                point_exponent = fallback_exponent
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            point_time = point[0]
            point_value = point[1]
            point_mode = default_mode
            point_exponent = fallback_exponent
        else:
            raise LomError(
                "type_error",
                "automation point {0} must be an object with 'time' and 'value' "
                "(or a [time, value] pair)".format(position))

        if point_time is None or point_value is None:
            raise LomError("type_error",
                           "automation point {0} needs both 'time' and "
                           "'value'".format(position))
        try:
            point_time = float(point_time)
        except (TypeError, ValueError):
            raise LomError("type_error",
                           "automation point {0} has a non-numeric time".format(position))
        if point_time < 0.0:
            raise LomError("type_error",
                           "automation point {0} has a negative time".format(position))

        prepared.append({
            "time": point_time,
            "value": clamp_parameter_value(param, point_value),
            "interpolation": point_mode,
            "exponent": point_exponent,
        })

    prepared.sort(key=lambda entry: entry["time"])
    return prepared


def build_automation_steps(prepared, resolution, clip_length, log):
    """Turn breakpoints into ``(time, length, value)`` steps for ``insert_step``.

    Returns ``(steps, resolution_actually_used)``. The last value is held to
    the end of the clip, which is how envelopes behave after their final
    breakpoint. ``insert_step`` is the only envelope-writing call the Live API
    offers in 11 and 12, so a smooth ramp is a fine staircase.
    """
    try:
        step_grid = float(resolution)
    except (TypeError, ValueError):
        step_grid = DEFAULT_AUTOMATION_RESOLUTION
    if step_grid <= 0.0:
        step_grid = DEFAULT_AUTOMATION_RESOLUTION
    if step_grid < MIN_AUTOMATION_RESOLUTION:
        step_grid = MIN_AUTOMATION_RESOLUTION

    span = prepared[-1]["time"] - prepared[0]["time"]
    budget = MAX_AUTOMATION_STEPS - len(prepared) - 1
    if span > 0.0 and budget > 0 and (span / step_grid) > budget:
        step_grid = span / float(budget)
        log("automation_write: resolution coarsened to {0} beats to stay below "
            "{1} steps".format(step_grid, MAX_AUTOMATION_STEPS))

    steps = []
    for position in range(len(prepared) - 1):
        start = prepared[position]
        end = prepared[position + 1]
        segment = end["time"] - start["time"]
        if segment <= 0.0:
            continue  # two breakpoints on the same beat: treat it as a jump
        count = int(segment / step_grid)
        if count * step_grid < segment - 1e-9:
            count += 1
        if count < 1:
            count = 1
        step_length = segment / float(count)
        for step in range(count):
            fraction = float(step) / float(count)
            steps.append((
                start["time"] + step * step_length,
                step_length,
                interpolate(start["value"], end["value"], fraction,
                            start["interpolation"], start["exponent"]),
            ))

    last = prepared[-1]
    tail_length = clip_length - last["time"]
    if tail_length <= 0.0:
        tail_length = step_grid
    steps.append((last["time"], tail_length, last["value"]))
    return steps, step_grid


# --------------------------------------------------------------------------
# What a browser item *is*, and how well it answers a query
# --------------------------------------------------------------------------

#: Live's own preset extensions. ``.adg`` is a rack/group preset (a Drum Rack,
#: an Instrument Rack, an Audio Effect Rack), ``.adv`` a single device preset.
#: ``.amxd`` is deliberately absent: a Max for Live device is a device, and
#: ``is_device`` already says so.
_PRESET_SUFFIXES = frozenset(("adg", "adv"))

#: Audio file extensions Live's browser can hold. A hit here means the item
#: loads as a sample — into a Simpler, not as a device.
_SAMPLE_SUFFIXES = frozenset((
    "wav", "wave", "aif", "aiff", "aifc", "flac", "mp3", "m4a", "aac", "ogg",
    "wma", "caf", "au", "snd", "sd2", "w64", "rex", "rx2", "mp4", "alac",
))

#: The kinds ``browser_walk`` reports, and the four a caller may filter on.
#: ``other`` is "loadable, but this script cannot say what it is"; ``unknown``
#: is "not loadable and not a container either" — the browser roots themselves
#: read that way before the container fallback catches them.
BROWSER_KINDS = ("device", "preset", "sample", "folder", "other", "unknown")
BROWSER_KIND_FILTERS = ("device", "preset", "sample", "folder", "any")

#: A file extension at the very end of a name or of a uri's last segment.
#: Live's uris look like ``query:Drums#Drums:808%20Boom%20Kit.adg``, so the
#: separators below are all of ``/``, ``\``, ``#``, ``?`` and ``:``.
_SUFFIX_RE = re.compile(r"\.([A-Za-z0-9]{1,5})$")
_URI_SEGMENT_RE = re.compile(r"[/\\#?:]")


def name_suffix(text):
    """The lower-cased file extension at the end of ``text``, or ``""``.

    Works on a browser item's ``name`` and on its ``uri`` alike: both end in the
    file name when there is one, and neither has one when there is not.
    """
    if not text:
        return ""
    tail = _URI_SEGMENT_RE.split(str(text))[-1]
    match = _SUFFIX_RE.search(tail)
    return match.group(1).lower() if match else ""


def browser_kind(entry):
    """What a browser entry *is*, as one of :data:`BROWSER_KINDS`.

    Derived from ``is_folder`` / ``is_device`` / ``is_loadable`` and the file
    extension on the name or uri, in that order of authority:

    1. ``is_folder`` — a folder is a folder whatever it is called;
    2. a preset extension (``.adg``, ``.adv``) — **before** ``is_device``,
       because Live reports a rack preset as a device and the difference is the
       whole point of this function;
    3. an audio extension — a sample;
    4. ``is_device`` — a real device;
    5. ``is_loadable`` — loadable, unclassifiable: ``other``;
    6. anything with children — a container, so ``folder``. This is what
       catches the browser roots, which report ``is_folder=False`` and
       ``is_loadable=False`` (measured 2026-08-29 against Live 12.4.5) and
       would otherwise read as ``unknown``.

    Measured 2026-08-29 against Live 12.4.5, and the reason this exists:
    searching root "drums" for "808 Boom" answered with ``Kick 808 Boom Eb.wav``
    — a *sample*, which loads as an ``OriginalSimpler`` — and never with the
    ``808 Boom Kit.adg`` rack preset, which loads as a ``DrumGroupDevice`` and
    is what a human asking for "808 Boom" means. Nothing in the reply said
    which was which, so the caller could not have told either.

    Steps 2 and 3 are the derivation; what Live reports in ``is_device`` for a
    ``.adg`` on this installation is **unverified**, and this function is
    deliberately arranged so that it does not matter.
    """
    if entry.get("is_folder") is True:
        return "folder"
    suffix = name_suffix(entry.get("name")) or name_suffix(entry.get("uri"))
    if suffix in _PRESET_SUFFIXES:
        return "preset"
    if suffix in _SAMPLE_SUFFIXES:
        return "sample"
    if entry.get("is_device") is True:
        return "device"
    if entry.get("is_loadable") is True:
        return "other"
    if entry.get("child_count"):
        return "folder"
    return "unknown"


def match_rank(name, needle):
    """How well ``name`` answers ``needle``. **Lower is better.**

    ``0`` the name is the query; ``1`` it is the query plus a file extension;
    ``2`` it starts with the query; ``3`` the query starts at a word boundary
    inside it; ``4`` the query is in there somewhere; ``5`` it is not.

    Ranking is not a nicety here, it is what makes a query answerable. A plain
    substring search puts ``Kick 808 Boom Eb.wav`` and ``808 Boom Kit.adg`` on
    exactly equal footing and returns whichever the walk reaches first — the
    sample, measured 2026-08-29 against Live 12.4.5 — where a human asking for
    "808 Boom" means the rack preset. Under these ranks that query makes the
    preset a prefix match (2) and the sample a word-boundary match (3), so the
    preset comes first, and "808 Boom Kit" scores 1 against ``808 Boom
    Kit.adg``, an all-but-exact hit.

    ``needle`` arrives lower-cased and stripped.
    """
    lowered = (name or "").lower()
    if not needle:
        return 4
    if lowered == needle:
        return 0
    if lowered.rsplit(".", 1)[0] == needle:
        return 1
    if lowered.startswith(needle):
        return 2
    position = lowered.find(needle)
    if position < 0:
        return 5
    return 3 if not lowered[position - 1].isalnum() else 4


# --------------------------------------------------------------------------
# The browser walk budget
# --------------------------------------------------------------------------

class WalkBudget(object):
    """The shared node and time budget of one browser walk (protocol §5.10).

    The browser is a separate object graph of its own and a full walk is
    enormous: measured 2026-08-29 against Live 12.4.5, ``audio_effects`` alone
    has 60 children before a single folder is opened, ``instruments`` 32 and
    ``midi_effects`` 22 — and every one of those is a folder with its own tree.

    Two limits guard it and both are reported rather than swallowed: a node
    count, and a wall-clock deadline that keeps the handler inside its
    main-thread budget (``_MAIN_THREAD_TIMEOUTS``, which is itself inside the
    client's timeout — protocol §8). Which limit fired is kept, because
    "truncated because the tree is huge" and "truncated because Live was slow"
    are different problems.
    """

    def __init__(self, nodes, seconds):
        self.nodes = nodes
        self.seconds = seconds
        self.deadline = time.time() + seconds
        self.hit_node_limit = False
        self.hit_time_limit = False

    def spend(self):
        """Charge one visited node. ``False`` means the walk stops here."""
        if self.nodes <= 0:
            self.hit_node_limit = True
            return False
        if time.time() > self.deadline:
            self.hit_time_limit = True
            return False
        self.nodes -= 1
        return True

    def exhausted(self):
        return self.hit_node_limit or self.hit_time_limit

    def reason(self):
        if self.hit_node_limit:
            return "node_budget"
        if self.hit_time_limit:
            return "time_budget"
        return None


# --------------------------------------------------------------------------
# The control surface
# --------------------------------------------------------------------------

def create_instance(c_instance):
    """Entry point Live calls to build the control surface."""
    return AbletonMaestro(c_instance)


class AbletonMaestro(ControlSurface):
    """The socket server and the seventeen handlers of protocol §5.

    Lifecycle. The server runs on a daemon thread, each client gets its own
    daemon thread, and :meth:`disconnect` closes the listening socket, drops
    every registered Live listener and lets the threads fall out on their own.
    Daemon threads because Live owns the process: nothing here may keep it from
    quitting.

    Threading. Live's object model is not thread-safe. Everything that touches
    it is therefore scheduled onto Live's main thread via ``schedule_message``
    and answered through a queue with a per-handler timeout, deliberately
    shorter than the client's (protocol §8) so a stuck operation returns a
    structured error instead of a socket timeout.

    Three handlers answer on the client thread instead, and none of them touches
    a LOM object at request time. ``ping`` touches none at all (§5.1),
    ``events_drain`` reads nothing but this script's own ring buffer, and
    ``script_info`` answers the Live version from a value cached in ``__init__``
    on Live's own main thread — see ``_read_live_version``, which says why reading
    it per request would be both unsafe and self-defeating. So none of the three
    can block behind a busy Live, which is the entire reason they are there.
    """

    #: Main-thread budget per handler, in seconds. Every number here is
    #: **strictly below** the client's timeout for that handler's class in
    #: protocol §8 — 10 s for a read, 20 s for a write — so this side always
    #: loses the race and answers with a structured ``live_error`` instead of
    #: letting the client hit a socket timeout and close the connection. A
    #: client that times out first learns nothing except that something is
    #: wrong; this side can say which handler, and for how long it waited.
    #:
    #: The read handlers of §8 (``lom_get``, ``lom_describe``, ``notes_get``,
    #: ``automation_read``) therefore sit at 8.0 < 10.0. Everything in the 20 s
    #: class stays under 20.0: the slow ones (``lom_call``, ``notes_set``,
    #: ``automation_write``, ``automation_clear``) at 15–18, while ``lom_set``,
    #: ``events_observe`` and ``events_clear`` cost no more than a read and sit
    #: at 8.0. ``lom_batch`` is the one budget computed rather than tabled; see
    #: :meth:`_main_thread_timeout`.
    #:
    #: ``browser_walk`` is the awkward one. It is a read, so the client gives it
    #: the ordinary 10 s, and §8 declines to grant it more: a per-handler
    #: allowance generous enough for a full tree — 60 s, say — would mean
    #: tolerating a 45 s block on Live's main thread, during which nothing else
    #: in Live happens, and that is a worse outcome than a truncated walk. The
    #: budget therefore has to be under 10.0. It sits at 9.0: the last second
    #: available without breaking the invariant, because the browser needs every
    #: one of them (measured 2026-08-29 against Live 12.4.5, a query against the
    #: ``instruments`` and ``sounds`` roots spent its whole budget without
    #: examining a single item). The walk's own node and time budget
    #: (``WalkBudget``, ``_BROWSER_TIME_BUDGET``) fires before even that, so a
    #: big walk returns ``truncated: true`` with real results rather than an
    #: error with none.
    _MAIN_THREAD_TIMEOUTS = {
        "lom_get": 8.0,
        "lom_set": 8.0,
        "lom_call": 15.0,
        "lom_describe": 8.0,
        "notes_get": 8.0,
        "notes_set": 15.0,
        "enum_names": 8.0,
        "automation_read": 8.0,
        "automation_write": 18.0,
        "automation_clear": 15.0,
        "browser_walk": 9.0,
        "events_observe": 8.0,
        "events_clear": 8.0,
    }
    _DEFAULT_MAIN_THREAD_TIMEOUT = 8.0

    #: Bounded ring buffer for observed events. Bounded on purpose: listeners
    #: fire in an audio application, and a buffer that grows without limit
    #: while nobody drains it is a memory leak inside Live. Overflow is
    #: reported as ``dropped`` (protocol §5.11).
    _EVENT_BUFFER_SIZE = 2000

    #: Node budget for one browser walk. The browser is a separate object graph
    #: with its own recursion and a full walk is slow (protocol §5.10).
    _BROWSER_MAX_NODES = 5000
    _BROWSER_DEFAULT_DEPTH = 1
    _BROWSER_MAX_DEPTH = 12

    #: Wall-clock budget for one browser walk, in seconds. Below this handler's
    #: main-thread budget (9.0 s), which is below the client's 10 s read timeout
    #: (protocol §8), so the innermost limit fires first: the caller gets a
    #: truncated tree with an honest ``truncated_by`` instead of a timeout and
    #: nothing at all.
    #:
    #: 7.5 is the most the chain 7.5 < 9.0 < 10.0 allows, and the wide roots
    #: need it: measured 2026-08-29 against Live 12.4.5, ``instruments`` and
    #: ``sounds`` answered ``truncated=true`` having examined nothing at all
    #: under a 6.0 budget. Headroom is not the fix, though - a bigger budget
    #: spent depth-first still descends a single branch, which is why the search
    #: is breadth-first (:meth:`_browser_bfs`).
    _BROWSER_TIME_BUDGET = 7.5

    #: How many query matches are collected before they are ranked and cut down
    #: to the caller's ``limit``. Larger than the ``limit`` ceiling on purpose:
    #: ranking cannot promote an answer the walk never reached, and the right
    #: answer sitting behind a worse one is the failure this whole ranking exists
    #: to prevent. The node and time budgets remain the real ceiling on the
    #: work.
    _BROWSER_MATCH_POOL = 1000

    #: Ceiling on the breadth-first frontier. Breadth-first trades memory for
    #: reaching the shallow items first, and the browser is wide: measured
    #: 2026-08-29 against Live 12.4.5, ``audio_effects`` has 60 children,
    #: ``instruments`` 32 and ``midi_effects`` 22 before a single folder is
    #: opened. This is a safety valve inside an audio application, not a tuning
    #: knob - hitting it is reported as the node budget, because that is what it
    #: is: a limit on how much of the graph one walk may hold.
    _BROWSER_MAX_FRONTIER = 20000

    _BROWSER_ROOTS = ("instruments", "sounds", "drums", "audio_effects", "midi_effects",
                      "plugins", "max_for_live", "packs", "user_library", "clips",
                      "samples", "current_project")

    def __init__(self, c_instance):
        ControlSurface.__init__(self, c_instance)
        self._song_ref = self.song()

        #: Process-local facts, captured here so that ``ping`` can answer
        #: without touching a single Live object (protocol §5.1). Live builds
        #: the control surface on its own main thread, so this is where the
        #: main thread's identity can be had for free — and knowing it is what
        #: lets ``_run_on_main_thread`` tell "already there" from "could not get
        #: there", which are two very different answers.
        self._started_at = time.time()
        self._main_thread_id = threading.current_thread().ident

        #: The Live version, read exactly once, here, on Live's own main thread.
        #: ``script_info`` answers on the client thread, and reading the version
        #: from the LOM there would be an off-thread access that blocks whenever
        #: the main thread is busy - which is precisely when a caller asks
        #: whether the script is alive. Live's version cannot change while this
        #: process lives, so the cache costs nothing and takes the last LOM read
        #: out of the two handlers that have to answer while Live is busy.
        self._live_version_cached = self._read_live_version()

        self.server = None
        self.server_thread = None
        self.client_threads = []
        self.running = False

        # Observed events: the buffer, its lock, the drop counter and the
        # registered listeners keyed by (path, property).
        self._events = deque()
        self._events_lock = threading.Lock()
        self._events_dropped = 0
        self._listeners = {}

        self._handlers = {
            "ping": (self._handle_ping, False),
            "script_info": (self._handle_script_info, False),
            "lom_get": (self._handle_lom_get, True),
            "lom_set": (self._handle_lom_set, True),
            "lom_call": (self._handle_lom_call, True),
            "lom_describe": (self._handle_lom_describe, True),
            "lom_batch": (self._handle_lom_batch, True),
            "notes_get": (self._handle_notes_get, True),
            "notes_set": (self._handle_notes_set, True),
            "automation_read": (self._handle_automation_read, True),
            "automation_write": (self._handle_automation_write, True),
            "automation_clear": (self._handle_automation_clear, True),
            "browser_walk": (self._handle_browser_walk, True),
            "events_observe": (self._handle_events_observe, True),
            "events_drain": (self._handle_events_drain, False),
            "events_clear": (self._handle_events_clear, True),
            "enum_names": (self._handle_enum_names, True),
        }

        self.log_message("Ableton Maestro {0} initialising".format(SCRIPT_VERSION))
        self.start_server()
        self.show_message(
            "Ableton Maestro: listening on {0}:{1}".format(HOST, DEFAULT_PORT))

    # -- lifecycle ---------------------------------------------------------

    def disconnect(self):
        """Tear the server down when Live closes or the surface is removed.

        Listeners come off first: a callback still registered against a torn
        down script would fire into a dead closure inside Live's process.
        """
        self.log_message("Ableton Maestro disconnecting")
        self.running = False

        try:
            self._remove_all_listeners()
        except Exception as exc:
            self.log_message("Error removing listeners: {0}".format(exc))

        if self.server is not None:
            try:
                self.server.close()
            except Exception:
                pass
            self.server = None

        if self.server_thread is not None and self.server_thread.is_alive():
            # The accept loop wakes at most once per second, so this join is
            # bounded. Client threads are daemons and are not joined: one of
            # them may be blocked in recv() and would hold up Live's shutdown.
            self.server_thread.join(2.0)

        ControlSurface.disconnect(self)
        self.log_message("Ableton Maestro disconnected")

    def start_server(self):
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((HOST, DEFAULT_PORT))
            self.server.listen(5)
            self.running = True
            self.server_thread = threading.Thread(target=self._server_thread)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.log_message("Server listening on {0}:{1}".format(HOST, DEFAULT_PORT))
        except Exception as exc:
            self.log_message("Error starting server: {0}".format(exc))
            self.show_message("Ableton Maestro: could not start server - {0}".format(exc))

    def _server_thread(self):
        try:
            self.server.settimeout(1.0)
            while self.running:
                try:
                    client, address = self.server.accept()
                except socket.timeout:
                    continue
                except Exception as exc:
                    if self.running:
                        self.log_message("Accept error: {0}".format(exc))
                    time.sleep(0.5)
                    continue

                self.log_message("Connection from {0}".format(address))
                thread = threading.Thread(target=self._handle_client, args=(client,))
                thread.daemon = True
                thread.start()
                self.client_threads.append(thread)
                self.client_threads = [t for t in self.client_threads if t.is_alive()]
        except Exception as exc:
            self.log_message("Server thread error: {0}".format(exc))
        finally:
            self.log_message("Server thread stopped")

    # -- framing (protocol §2) --------------------------------------------

    def _handle_client(self, client):
        """The framing loop of protocol §2, exactly.

        There is no delimiter and no length prefix; the only message boundary
        is "the buffer parses as JSON now". Two details are easy to get wrong
        and expensive to debug, so both are spelled out here:

        * the UTF-8 decoder is **incremental**, because a multi-byte character
          split across a ``recv()`` boundary corrupts a buffer that decodes each
          chunk on its own;
        * the remainder after ``raw_decode`` is **kept**, because two requests
          can arrive in one ``recv()`` and dropping the tail desynchronises
          every later reply.
        """
        client.settimeout(None)
        decoder = codecs.getincrementaldecoder("utf-8")()
        json_decoder = json.JSONDecoder()
        buffer = ""

        try:
            while self.running:
                try:
                    chunk = client.recv(RECV_CHUNK)
                except Exception as exc:
                    self.log_message("recv failed: {0}".format(exc))
                    break
                if not chunk:
                    break

                try:
                    buffer += decoder.decode(chunk)
                except UnicodeDecodeError as exc:
                    self._send(client, {"status": "error", "code": "type_error",
                                        "message": "request was not valid UTF-8: "
                                                   "{0}".format(exc)})
                    break

                if len(buffer) > MAX_REQUEST_BYTES:
                    # No framing means no way to resynchronise on a delimiter.
                    # Dropping the buffer loudly beats growing it silently.
                    buffer = ""
                    decoder = codecs.getincrementaldecoder("utf-8")()
                    self._send(client, {
                        "status": "error", "code": "bad_path",
                        "message": "request exceeded {0} bytes without parsing as JSON; "
                                   "buffer discarded".format(MAX_REQUEST_BYTES)})
                    continue

                while True:
                    stripped = buffer.lstrip()
                    if not stripped:
                        buffer = ""
                        break
                    try:
                        request, end = json_decoder.raw_decode(stripped)
                    except ValueError:
                        buffer = stripped
                        break  # incomplete, read more
                    except Exception as exc:
                        # Not "incomplete" but "unparseable in a way ValueError
                        # does not cover" - a RecursionError from deeply nested
                        # JSON is the realistic one. There is no delimiter to
                        # resynchronise on, so the buffer goes; what must not
                        # happen is breaking out silently and leaving the client
                        # waiting for a reply that will never come.
                        buffer = ""
                        decoder = codecs.getincrementaldecoder("utf-8")()
                        self._send(client, {
                            "status": "error", "code": "internal",
                            "message": "could not parse the request buffer ({0}: "
                                       "{1}); buffer discarded".format(
                                           type(exc).__name__, exc)})
                        break
                    buffer = stripped[end:].lstrip()  # keep the remainder!
                    self._send(client, self._reply_to(request))
        except Exception as exc:
            self.log_message("Client handler error: {0}".format(exc))
            self.log_message(traceback.format_exc())
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _reply_to(self, request):
        """Exactly one reply object for one request, whatever happens.

        :meth:`dispatch` already promises never to raise; this is the belt to
        that pair of braces. On a strictly serial socket with no framing
        (protocol §2, §3) a missing reply is not a failed request, it is a
        stranded client: it waits out its whole timeout, closes the socket and
        never learns why.

        Measured 2026-08-29 against Live 12.4.5: a handler that runs inline on
        the client thread rather than through Live's main-thread queue has no
        timeout governing it at all, and a task that blocks there leaves the
        client with no reply whatsoever for the full 20 s of its own write
        timeout. :meth:`_run_on_main_thread` refuses to take that route; this
        method is the guarantee that no other one opens. Everything that leaves
        here is a dict with a ``status``.
        """
        try:
            reply = self.dispatch(request)
        except LomError as exc:
            reply = exc.to_response()
        except BaseException as exc:
            # BaseException on purpose: a reply the client can read is worth
            # more than a clean propagation out of a daemon thread that nothing
            # is watching.
            self.log_message("dispatch escaped: {0}".format(exc))
            self.log_message(traceback.format_exc())
            reply = {"status": "error", "code": "internal",
                     "message": "{0}: {1}".format(type(exc).__name__, exc)}
        if not isinstance(reply, dict):
            reply = {"status": "error", "code": "internal",
                     "message": "handler returned {0}, not a response object".format(
                         type(reply).__name__)}
        return reply

    def _send(self, client, payload):
        """Serialise and send one response, never raising past this method.

        ``json.dumps`` failing *after* a handler succeeded would leave the
        client waiting forever on a strictly serial socket, so an unserialisable
        value degrades to its ``repr`` and, failing that, to a structured
        ``internal`` error.
        """
        try:
            text = json.dumps(payload, default=_json_fallback)
        except Exception as exc:
            self.log_message("Response was not serialisable: {0}".format(exc))
            text = json.dumps({"status": "error", "code": "internal",
                               "message": "response could not be serialised: "
                                          "{0}".format(exc)})
        try:
            client.sendall(text.encode("utf-8"))
        except Exception as exc:
            self.log_message("send failed: {0}".format(exc))

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, request):
        """Route one request. Nothing escapes this method as an exception."""
        try:
            if not isinstance(request, dict):
                return {"status": "error", "code": "type_error",
                        "message": "request must be a JSON object, got "
                                   "{0}".format(type(request).__name__)}
            handler_name = request.get("type", "")
            params = request.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                return {"status": "error", "code": "type_error",
                        "message": "'params' must be an object, got "
                                   "{0}".format(type(params).__name__)}

            entry = self._handlers.get(handler_name)
            if entry is None:
                return {"status": "error", "code": "unknown_handler",
                        "message": "no handler {0!r}; this script answers: "
                                   "{1}".format(handler_name, ", ".join(HANDLERS))}

            handler, needs_main_thread = entry
            if needs_main_thread:
                result = self._run_on_main_thread(handler_name, handler, params)
            else:
                result = handler(params)
            if isinstance(result, DeferredReadBack):
                # Belt and braces. Only a main-thread handler can produce one
                # and _run_on_main_thread resolves those, so this is reachable
                # only if a handler is ever moved off the main thread. A
                # DeferredReadBack on the wire would be an unserialisable object
                # where a result belongs, and that is a dead connection.
                result = result.unresolved(
                    "{0} answered without a main thread to schedule one "
                    "on".format(handler_name))
            return {"status": "success", "result": result}
        except LomError as exc:
            return exc.to_response()
        except Exception as exc:
            self.log_message("Unhandled error in {0}: {1}".format(
                request.get("type", "?") if isinstance(request, dict) else "?", exc))
            self.log_message(traceback.format_exc())
            return {"status": "error", "code": "internal",
                    "message": "{0}: {1}".format(type(exc).__name__, exc)}

    def _run_on_main_thread(self, handler_name, handler, params):
        """Run a handler on Live's main thread and wait for its result.

        The LOM is not thread-safe, and even reads (``device.parameters``, clip
        envelopes) are unsafe off-thread. Live's API documents no thread-safety
        anywhere, which is why *everything* that touches it is scheduled rather
        than a chosen few (not measured per property, and not a distinction
        worth discovering at a user's expense).

        Every exit from here is a result or a :class:`LomError`, and
        :meth:`dispatch` turns the latter into a structured reply. That is
        load-bearing, and it is why there is no "if scheduling fails, just run
        it here" fallback: on the client thread the queue timeout below governs
        nothing, so a blocked LOM call blocks forever and nothing is ever sent —
        measured 2026-08-29 against Live 12.4.5, the client waits out its full
        20 s write timeout and learns nothing.

        So: being already on Live's main thread is recognised by thread identity
        rather than guessed from an exception, and a scheduling failure is an
        error the caller is told about rather than a silent, unbounded,
        off-thread LOM call.
        """
        if threading.current_thread().ident == self._main_thread_id:
            # Already on Live's main thread — a re-entrant call, or Live calling
            # in directly. Scheduling from here would deadlock: the task cannot
            # run until this thread returns, and this thread is waiting for it.
            result = handler(params)
            if isinstance(result, DeferredReadBack):
                return result.unresolved(
                    "this call is already running on Live's main thread, where "
                    "waiting for a later tick would deadlock")
            return result

        response_queue = queue.Queue()

        def task():
            try:
                response_queue.put(("ok", handler(params)))
            except LomError as exc:
                response_queue.put(("lom", exc))
            except Exception as exc:
                self.log_message("Error in {0}: {1}".format(handler_name, exc))
                self.log_message(traceback.format_exc())
                response_queue.put(("error", exc))

        try:
            self.schedule_message(0, task)
        except Exception as exc:
            # Deliberately NOT "run it here instead". Off the main thread the
            # LOM is unsafe and, worse, unbounded: an inline call has no timeout
            # and is the one way this method can strand a client for good.
            raise LomError(
                "internal",
                "could not schedule {0} onto Live's main thread ({1}: {2}); the "
                "operation was not attempted".format(
                    handler_name, type(exc).__name__, exc),
                handler=handler_name)

        timeout = self._main_thread_timeout(handler_name, params)
        deadline = time.time() + timeout
        try:
            kind, payload = response_queue.get(timeout=timeout)
        except queue.Empty:
            raise LomError(
                "live_error",
                "{0} did not finish within {1:.1f}s on Live's main thread; the "
                "operation may still be running. Live's main thread is busy - a "
                "modal dialog is the usual reason - and this script's budget is "
                "deliberately shorter than yours (protocol §8) so that you get "
                "this message rather than a socket timeout.".format(
                    handler_name, timeout),
                handler=handler_name, timeout_seconds=timeout)
        if kind == "ok":
            if isinstance(payload, DeferredReadBack):
                return self._resolve_deferred(handler_name, payload, deadline)
            return payload
        if kind == "lom":
            raise payload
        raise LomError("internal", "{0}: {1}".format(type(payload).__name__, payload))

    def _resolve_deferred(self, handler_name, deferred, deadline):
        """Take the one extra look at Live that a deferred read-back needs.

        Runs on the **client** thread, which is the whole point: the handler
        could not wait for its own write because it was running inside the
        scheduled main-thread task, and blocking there for a later tick would
        deadlock (:class:`DeferredReadBack`). Here, blocking is what this thread
        already does.

        Exactly one extra task, scheduled one tick out, bounded twice: by
        :data:`DEFERRED_READ_BACK_TIMEOUT` and by whatever is left of the
        handler's own main-thread budget, which is itself below the client's
        (protocol §8). So a second read can never push a write past the client's
        20 s, and every way this can fail produces an honest ``not_observed``
        rather than an error - the write itself was fine; only the confirmation
        is missing.
        """
        remaining = deadline - time.time()
        if remaining <= 0.2:
            return deferred.unresolved(
                "{0}'s main-thread budget was already spent".format(handler_name))

        second_queue = queue.Queue()

        def task():
            try:
                second_queue.put(("ok", deferred.resolve()))
            except LomError as exc:
                second_queue.put(("lom", exc))
            except Exception as exc:
                self.log_message("Error in the second read of {0}: {1}".format(
                    handler_name, exc))
                second_queue.put(("error", exc))

        try:
            self.schedule_message(DEFERRED_READ_BACK_DELAY, task)
        except Exception as exc:
            return deferred.unresolved(
                "the follow-up read could not be scheduled onto Live's main "
                "thread ({0}: {1})".format(type(exc).__name__, exc))

        budget = min(remaining, DEFERRED_READ_BACK_TIMEOUT)
        try:
            kind, payload = second_queue.get(timeout=budget)
        except queue.Empty:
            return deferred.unresolved(
                "the follow-up read did not run within {0:.1f}s - Live's main "
                "thread is busy".format(budget))
        if kind == "ok":
            return payload
        if kind == "lom":
            raise payload
        raise LomError("internal", "{0}: {1}".format(type(payload).__name__, payload))

    def _main_thread_timeout(self, handler_name, params):
        if handler_name != "lom_batch":
            return self._MAIN_THREAD_TIMEOUTS.get(
                handler_name, self._DEFAULT_MAIN_THREAD_TIMEOUT)
        # A batch is one main-thread task, so its budget scales with the op
        # count — but stays under the client's 20 s write timeout (protocol §8).
        ops = params.get("ops")
        count = len(ops) if isinstance(ops, list) else 0
        return min(18.0, 4.0 + 0.4 * count)

    # -- resolver plumbing --------------------------------------------------

    def _root_object(self, name):
        if name == "song":
            return self._song_ref
        if name == "app":
            application = None
            if Live is not None:
                try:
                    application = Live.Application.get_application()
                except Exception:
                    application = None
            if application is None:
                application = self.application()
            if application is None:
                raise LomError("live_error", "could not reach the Live application")
            return application
        raise LomError("bad_path", "unknown root {0!r}".format(name))

    def resolve(self, path):
        """Resolve a path to the object it names."""
        target, _parent, _last = self.resolve_with_parent(path)
        return target

    def resolve_with_parent(self, path):
        """Resolve a path, also returning ``(parent, last_segment_name)``.

        ``lom_set`` needs the parent to write through, and both ``lom_get`` and
        ``lom_set`` need it to find the DeviceParameter behind a ``.value``.
        """
        segments = parse_path(path)
        current = self._root_object(segments[0][0])
        parent = None
        last_name = segments[0][0]
        so_far = segments[0][0]

        for name, index in segments[1:]:
            so_far = "{0}.{1}".format(so_far, name)
            parent = current
            last_name = name
            current = _get_attribute(current, name, so_far)
            if index is not None:
                # The collection's path is what an out-of-range message has to
                # name; so_far becomes the indexed path on the next line and is
                # the wrong thing to blame for the index not existing.
                collection_path = so_far
                so_far = "{0}[{1}]".format(so_far, index)
                parent = current
                last_name = None
                current = _get_index(current, index, so_far, collection_path)
        return current, parent, last_name

    def resolve_clip(self, path):
        """Resolve a path to a Clip, accepting a ClipSlot for convenience.

        Returns ``(clip, clip_path)``. Accepting the slot costs one ``hasattr``
        and removes the most common addressing mistake in a protocol where the
        clip only exists when the slot is filled.
        """
        target = self.resolve(path)
        if hasattr(target, "has_clip") and not hasattr(target, "is_midi_clip"):
            if not target.has_clip:
                raise LomError("no_such_path", "clip slot is empty", path=path)
            return target.clip, "{0}.clip".format(path)
        if target is None:
            raise LomError("no_such_path",
                           "there is no clip here - the slot is empty, so its .clip is "
                           "None. Make one first (clip_slot.create_clip for MIDI, "
                           "create_audio_clip for audio), then address it again.",
                           path=path)
        if not hasattr(target, "is_midi_clip"):
            raise LomError("no_such_path",
                           "{0} is not a Clip or ClipSlot".format(class_name(target)),
                           path=path)
        return target, path

    def resolve_parameter(self, path):
        """Resolve a path to a DeviceParameter, rejecting anything else."""
        target = self.resolve(path)
        if not (hasattr(target, "value") and hasattr(target, "str_for_value")):
            raise LomError(
                "no_such_path",
                "{0} is not a DeviceParameter - point 'parameter' at something like "
                "song.tracks[0].devices[0].parameters[3] or "
                "song.tracks[0].mixer_device.volume".format(class_name(target)),
                path=path)
        return target

    def decode_argument(self, value):
        """Decode one ``lom_call`` argument, resolving object references.

        Half the allowlist takes a Live object (``Clip.clear_envelope(param)``,
        ``Browser.load_item(item)``, ``ClipSlot.duplicate_clip_to(slot)``), and
        JSON has no way to express one. ``{"__path__": "song.tracks[0]..."}``
        is that way; it goes through the same resolver, with the same guards.
        """
        if isinstance(value, dict) and "__path__" in value:
            return self.resolve(value["__path__"])
        if isinstance(value, list):
            return [self.decode_argument(item) for item in value]
        return value

    # -- 5.1 ping ----------------------------------------------------------

    def _handle_ping(self, params):
        """Liveness check (protocol §5.1). Touches no LOM object, on purpose.

        Answering on the client thread is not enough by itself. Reading one Live
        property is what breaks a liveness check: measured 2026-08-29 against
        Live 12.4.5, while Live's main thread holds the LOM lock — a modal
        dialog is the everyday case — a read of ``song.current_song_time``
        blocks, so the ping times out in exactly the situation a liveness check
        exists for. A ping that only works when Live is idle answers a question
        nobody asks.

        So everything reported here is process-local: this script's own version
        and how long this process has been up. No attribute of any Live object,
        no ``self._song_ref``, no ``self.application()``, and no main-thread
        hop. **Nothing may be added to this handler that does not hold to that
        rule** — the moment it reads one Live property it becomes a liveness
        check that fails whenever liveness is in doubt.

        ``script_info`` is the handler for anything that needs Live itself.
        """
        return {
            "pong": True,
            "script_version": SCRIPT_VERSION,
            "uptime": round(time.time() - self._started_at, 3),
        }

    # -- 5.2 script_info ---------------------------------------------------

    def _handle_script_info(self, params):
        """Capabilities handshake (protocol §5.2).

        Reports what this build actually has — its version, the protocol it
        speaks, the handlers it answers to and the size of the method allowlist
        — so the server can refuse to send a request this script would not
        understand, instead of discovering it one timeout at a time.
        """
        return {
            "name": SCRIPT_NAME,
            "script_version": SCRIPT_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "handlers": list(HANDLERS),
            # Cached in __init__ on Live's main thread - never read here. See
            # _read_live_version.
            "live_version": self._live_version_cached,
            "host": HOST,
            "port": DEFAULT_PORT,
            "allowlist_size": len(METHOD_ALLOWLIST),
        }

    def _read_live_version(self):
        """The Live version string, or ``None``. Reads the LOM - main thread only.

        Called exactly once, from ``__init__``, which Live runs on its own main
        thread. Every later answer comes from ``self._live_version_cached``.

        Reading it per request would fail the way a LOM read in ``ping`` fails:
        ``script_info`` is answered on the client thread on purpose - a handshake
        that queues behind a busy Live is a handshake that fails at startup - and
        a LOM read from that thread blocks behind Live's lock (measured
        2026-08-29 against Live 12.4.5) and is unsafe besides. Nothing in this
        method may be called from a client thread.
        """
        try:
            application = self._root_object("app")
        except LomError:
            return None
        getter = getattr(application, "get_version_string", None)
        if getter is not None:
            try:
                return str(getter())
            except Exception:
                pass
        parts = []
        for name in ("get_major_version", "get_minor_version", "get_bugfix_version"):
            method = getattr(application, name, None)
            if method is None:
                return None
            try:
                parts.append(str(method()))
            except Exception:
                return None
        return ".".join(parts)

    # -- 5.3 lom_get -------------------------------------------------------

    def _handle_lom_get(self, params):
        path = params.get("path")
        target, parent, last_name = self.resolve_with_parent(path)
        value, kind, truncated = encode_value(target, path)
        result = {"path": path, "value": value, "type": kind}
        if kind == "list":
            # The full length of the collection, not len(value): `value` stops
            # at _LIST_CAP and `truncated` below says so. A count that shrank
            # with the cap would be the wrong number for the one question this
            # field exists to answer.
            result["count"] = collection_length(target)
            result["class"] = class_name(target)
        if truncated:
            result["truncated"] = True
            result["list_cap"] = _LIST_CAP
        param = find_parameter(target, parent, last_name)
        if param is not None:
            display = parameter_display(param)
            if display is not None:
                result["display"] = display
        return result

    # -- 5.4 lom_set -------------------------------------------------------

    def _handle_lom_set(self, params):
        """Write a property and read it straight back (protocol §5.4).

        This is the single most important decision in the protocol. Live refuses
        an out-of-range value out loud and stores nothing (measured 2026-08-30
        against Live 12.4.5), which is not the usual reason given for reading back.
        What it does do silently is everything else: it snaps a quantised value to the
        nearest step, applies some writes late enough that the first read is
        stale, ignores an unknown property name, and reports success for a write
        that did nothing - the same failure in several costumes, and the reason a
        bridge that only forwards writes cannot be trusted. So the answer never says 
        only "success": it says what was asked for, what was there, what is there now, 
        whether Live clamped it and whether anything changed at all.

        What the read-back proves is the *stored value*, never the audible
        effect (protocol §10). A parameter can be exactly right and do nothing
        because the device is off.

        **Object references.** ``value`` may also be ``{"__path__": "<path>"}``
        — the same form ``lom_call`` arguments use (protocol §5.5), resolved
        through the same resolver with the same guards. Several real LOM
        properties take a Live object rather than a value:
        ``song.view.selected_track``, ``song.view.selected_scene``,
        ``song.view.detail_clip``, ``browser.hotswap_target`` and
        ``clip.groove``. Without that form none of them can be written at all:
        a bare value gives ``not_settable: "Track is a Live object, not a
        value"`` (measured 2026-08-29 against Live 12.4.5).

        For such a property the read-back still happens and still matters, but
        it proves something different: **the reference, not a value**. ``after``
        carries the path of the object the property now holds, and ``changed``
        compares object identity rather than ``==``. See
        :meth:`_set_object_reference`.

        **The read-back is not always synchronous, and a stale reading must not
        be reported as a clamp.** Measured 2026-08-29 against Live 12.4.5 on
        ``song.current_song_time``, transport stopped, starting value 4.0::

            set 8.0  ->  after=4.0  clamped=true  changed=false  (4.0 was BEFORE)
            set 1.0  ->  after=8.0  clamped=true  changed=false  (8.0 was the
                                                                  PREVIOUS request)

        The write lands - a later read shows 8.0 - but Live applies it once the
        current main-thread task finishes, so a read-back taken inside that same
        task sees the value from before. Calling that a clamp would be a false
        statement in the one field this protocol exists to make true: nothing
        was clamped, the answer was simply taken too early.

        So three outcomes are told apart, and ``read_back`` names which one
        happened:

        ``applied``
            ``after`` equals what was requested. The value is stored.
        ``clamped``
            ``after`` is neither the requested value nor the previous one, so
            **Live** put that number there. A real clamp, and ``clamped`` is
            true only here.
        ``not_observed``
            ``after`` still equals ``before``. Either Live clamped to the value
            that was already stored, or the write has not been applied yet -
            and from inside the writing task those are indistinguishable. One
            bounded second read is taken a main-thread tick later
            (:class:`DeferredReadBack`); if it shows the requested value the
            answer becomes ``applied``, and if it still does not, both readings
            are reported and ``clamped`` stays **false**. Guessing "clamped"
            here would be a wrong answer; "not observed" is a true one.

        Which properties are asynchronous is not something the LOM documents.
        ``song.current_song_time`` is the measured one, and it should be read as
        an example of a class rather than as a special case: transport and
        playback position, anything Live recomputes on its own clock, and
        properties whose setter Live defers to the end of the current task all
        behave this way. Any property can be in that class; ``read_back`` is how
        a caller finds out without having to know in advance.
        """
        path = params.get("path")
        if "value" not in params:
            raise LomError("type_error", "lom_set needs a 'value'", path=path)
        requested = params.get("value")
        is_reference = isinstance(requested, dict) and "__path__" in requested

        target, parent, last_name = self.resolve_with_parent(path)
        if parent is None or last_name is None:
            raise LomError(
                "not_settable",
                "cannot write through {0!r} - a list element or a root is not a "
                "settable property".format(path),
                path=path)
        if is_lom_collection(target):
            # song.tracks, device.parameters, track.clip_slots and friends are
            # read-only collections in the LOM. Refusing here rather than
            # letting Live raise keeps the error code precise and means a
            # generic bridge never tries to overwrite a collection.
            #
            # The test has to be is_lom_collection: an isinstance against
            # (list, tuple) catches none of them - they are Vector objects - so
            # such a guard would name these collections and then never catch
            # one.
            raise LomError(
                "not_settable",
                "{0} is a LOM collection and cannot be assigned; address an element "
                "and set one of its properties instead".format(path),
                path=path)
        if is_reference:
            return self._set_object_reference(path, parent, last_name, target,
                                              requested)
        if is_lom_object(target):
            hint = (" - to write a reference, send the value as "
                    "{\"__path__\": \"<path of the object>\"} (protocol §5.5)")
            if hasattr(target, "value") and hasattr(target, "str_for_value"):
                hint = " - write {0}.value instead".format(path)
            raise LomError("not_settable",
                           "{0} is a Live object, not a value{1}".format(
                               class_name(target), hint),
                           path=path)

        before, before_kind, _t = encode_value(target, path)
        coerced = coerce_to(target, requested)

        try:
            setattr(parent, last_name, coerced)
        except AttributeError as exc:
            raise LomError("not_settable",
                           "{0}.{1} is read-only in the LOM ({2})".format(
                               class_name(parent), last_name, exc),
                           path=path)
        except (TypeError, ValueError) as exc:
            raise LomError("type_error", str(exc), path=path)
        except RuntimeError as exc:
            raise LomError("live_error", str(exc), path=path)

        def read_back():
            """Read the property again. Returns ``(raw, encoded)``."""
            try:
                raw = getattr(parent, last_name)
            except Exception as exc:
                raise LomError("live_error",
                               "write accepted but read-back failed: {0}".format(exc),
                               path=path)
            encoded, _kind, _truncated = encode_value(raw, path)
            return raw, encoded

        def build(after_raw, after, state, extra):
            """The §5.4 result for one reading, with its outcome named."""
            result = {
                "path": path,
                "requested": requested,
                "before": before,
                "after": after,
                # True ONLY for a value Live itself put there. The third
                # outcome - nothing observed - is not a clamp and must not
                # claim to be one.
                "clamped": state == "clamped",
                "changed": not values_equal(after, before),
                "read_back": state,
                "type": before_kind,
            }
            if not values_equal(coerced, requested):
                result["coerced"] = coerced
            for key in sorted(extra.keys()):
                result[key] = extra[key]

            param = find_parameter(after_raw, parent, last_name)
            if param is None:
                param = find_parameter(target, parent, last_name)
            if param is not None:
                display = parameter_display(param)
                if display is not None:
                    result["display"] = display
                # A quantized parameter takes discrete steps: set 0.35, get
                # 0.25. Without this field the clamp looks arbitrary (measured
                # across several plugins).
                try:
                    if bool(param.is_quantized):
                        result["is_quantized"] = True
                except Exception:
                    pass
            return result

        def classify(after):
            if values_equal(after, coerced):
                return "applied"
            if not values_equal(after, before):
                return "clamped"
            return "not_observed"

        after_raw, after = read_back()
        state = classify(after)
        if state != "not_observed":
            return build(after_raw, after, state, {"read_back_attempts": 1})

        # after == before != requested. The write may have been clamped to the
        # value already there, or it may simply not have been applied yet -
        # measured on song.current_song_time, which applies once this task ends.
        # One more look, a main-thread tick later, decides it in the common
        # case. It cannot happen from here: this code is running *inside* that
        # task, so it hands the second read to the client thread instead.
        def resolve():
            second_raw, second = read_back()
            second_state = classify(second)
            extra = {"read_back_attempts": 2, "first_read": after}
            if second_state == "not_observed":
                extra["note"] = _NOT_OBSERVED_NOTE
            return build(second_raw, second, second_state, extra)

        def unresolved(reason):
            return build(after_raw, after, "not_observed",
                         {"read_back_attempts": 1,
                          "note": _NOT_OBSERVED_NOTE + " No second read was "
                                  "taken: " + reason})

        return DeferredReadBack(resolve, unresolved)

    def _set_object_reference(self, path, parent, last_name, before_object, requested):
        """Write a Live object reference and read the reference back (§5.4).

        JSON cannot express a Live object, so ``lom_set`` takes the same
        ``{"__path__": "<path>"}`` form that ``lom_call`` arguments take
        (protocol §5.5). It goes through the same resolver with the same guards,
        so a ``__path__`` here reaches nothing a ``lom_get`` could not reach: it
        is bounds-checked, it rejects private attributes, it resolves objects
        only and it never calls anything on the way.

        The read-back is what ``lom_set`` is for and it still happens — but for
        an object property **it proves the reference, not a value**. ``after``
        carries the path of the object the property now holds, ``changed``
        compares object identity, and ``clamped`` says whether Live actually
        stored the object that was asked for or quietly kept another one. Value
        equality could not answer any of the three: Live hands out a fresh
        Python wrapper for every attribute read, so two references to one track
        are never the same Python object (see :func:`same_lom_object`).

        Measured 2026-08-29 against Live 12.4.5: before this,
        ``song.view.selected_track`` and its four siblings
        (``selected_scene``, ``detail_clip``, ``browser.hotswap_target``,
        ``clip.groove``) were refused with ``not_settable: "Track is a Live
        object, not a value"``.
        """
        value_path = requested.get("__path__")
        new_object = self.resolve(value_path)
        if not is_lom_object(new_object):
            raise LomError(
                "type_error",
                "{0!r} resolves to a {1}, which is a value and not a Live object - "
                "send it as the value itself, not as a __path__ reference".format(
                    value_path, class_name(new_object)),
                path=path)

        try:
            setattr(parent, last_name, new_object)
        except AttributeError as exc:
            raise LomError("not_settable",
                           "{0}.{1} is read-only in the LOM ({2})".format(
                               class_name(parent), last_name, exc),
                           path=path)
        except (TypeError, ValueError) as exc:
            raise LomError(
                "type_error",
                "{0} does not accept a {1} ({2})".format(
                    path, class_name(new_object), exc),
                path=path)
        except RuntimeError as exc:
            raise LomError("live_error", str(exc), path=path)

        def read_back():
            try:
                return getattr(parent, last_name)
            except Exception as exc:
                raise LomError("live_error",
                               "write accepted but read-back failed: {0}".format(exc),
                               path=path)

        def build(after_object, extra):
            """The §5.4 reference result, with its outcome named.

            The same three outcomes as a value write, decided on object
            identity rather than on ``==``: ``applied`` when the property
            holds the object that was asked for, ``clamped`` when it holds some
            *other* object - which only Live can have put there - and
            ``not_observed`` when it still holds the one it held before, which
            is either a refusal or a write that has not landed yet.
            """
            stored = same_lom_object(after_object, new_object)
            changed = not same_lom_object(before_object, after_object)
            if stored:
                state = "applied"
            elif changed:
                state = "clamped"
            else:
                state = "not_observed"
            result = {
                "path": path,
                "requested": {"__path__": value_path},
                "before": object_reference(before_object, None),
                # The identity path is reported only when the read-back proved it.
                "after": object_reference(after_object,
                                          value_path if stored else None),
                "clamped": state == "clamped",
                "changed": changed,
                "read_back": state,
                "type": "object",
                "value_kind": "reference",
                "reference_stored": stored,
            }
            if state == "clamped":
                result["note"] = ("the write was accepted but the property now holds "
                                  "a different object than the one requested; the "
                                  "reference was not stored")
            for key in sorted(extra.keys()):
                result[key] = extra[key]
            return result

        after_object = read_back()
        first = build(after_object, {"read_back_attempts": 1})
        if first["read_back"] != "not_observed":
            return first

        def resolve():
            second = build(read_back(), {"read_back_attempts": 2})
            if second["read_back"] == "not_observed":
                second["note"] = _NOT_OBSERVED_NOTE
            return second

        def unresolved(reason):
            first["note"] = (_NOT_OBSERVED_NOTE + " No second read was taken: "
                             + reason)
            return first

        return DeferredReadBack(resolve, unresolved)

    # -- 5.5 lom_call ------------------------------------------------------

    def _handle_lom_call(self, params):
        path = params.get("path")
        method_name = params.get("method")
        args = params.get("args", [])
        if not isinstance(method_name, str) or not method_name:
            raise LomError("type_error", "lom_call needs a 'method' name", path=path)
        if method_name.startswith("_"):
            raise LomError("method_not_allowed",
                           "private members are not callable", path=path)
        if args is None:
            args = []
        if not isinstance(args, list):
            raise LomError("type_error",
                           "'args' must be a list; the LOM does not use keyword "
                           "arguments", path=path)

        target = self.resolve(path)
        candidates = type_candidates(target)
        allowed = [name for name in candidates
                   if "{0}.{1}".format(name, method_name) in METHOD_ALLOWLIST]
        # Existence first, THEN the allowlist. The other order answers "not on
        # this script's allowlist ... cannot be widened from outside Live" for a
        # method that does not exist on the object at all, which reads as a policy
        # this script chose and invites a caller to ask for it to be lifted. There
        # is nothing to lift: Song has no group_tracks, no export, no save --
        # Live's Remote Script API does not expose them to any script. Measured
        # 2026-08-30: an invented method name answered word for word what
        # Song.group_tracks answered, so the message could not be told apart.
        method = getattr(target, method_name, None)
        if method is None:
            raise LomError("no_such_path",
                           "{0} has no method {1!r} in this Live version. This is not "
                           "an allowlist decision -- the object does not have it, and "
                           "no script can reach what Live does not expose.".format(
                               class_name(target), method_name),
                           path=path)

        if not allowed:
            raise LomError(
                "method_not_allowed",
                "{0}.{1} exists on the object but is not on this script's allowlist. "
                "The allowlist lives in the script so it cannot be widened from "
                "outside Live (protocol §6).".format(class_name(target), method_name),
                path=path)
        if not callable(method):
            raise LomError("method_not_allowed",
                           "{0}.{1} is not callable".format(
                               class_name(target), method_name),
                           path=path)

        decoded = [self.decode_argument(item) for item in args]
        try:
            returned = method(*decoded)
        except (TypeError, ValueError) as exc:
            raise LomError("type_error", str(exc), path=path)
        except (RuntimeError, AttributeError, IndexError, KeyError) as exc:
            raise LomError("live_error", "{0}: {1}".format(type(exc).__name__, exc),
                           path=path)

        encoded, kind, truncated = encode_value(returned, None)
        result = {"path": path, "method": method_name, "result": encoded,
                  "result_type": kind}
        if truncated:
            result["truncated"] = True
        return result

    # -- 5.6 lom_describe --------------------------------------------------

    def _handle_lom_describe(self, params):
        """Introspect a live object (protocol §5.6).

        This is how the *dynamic surface* is reached — the parameters a loaded
        plugin happens to expose, which cannot be known when the catalog is
        written. A VST that reports only ``Device On`` has nothing taken into
        Live's parameter strip via *Configure*; that limit is in Live, not here
        (measured).

        ``settable`` is ``true``/``false`` only when Python can tell. Live's
        objects expose C-level descriptors that accept ``__set__`` and raise at
        call time, so writability is genuinely not readable from here and is
        reported as ``null`` rather than guessed. The catalog is the authority.

        Every child that is a LOM collection carries a real ``count``, which is
        what makes "how many tracks are there" a single round trip instead of a
        probe loop up the indices until ``index_out_of_range``. A ``count`` of
        ``null`` means Live refused to say — it is never the default.
        """
        path = params.get("path")
        depth = params.get("depth", 1)
        try:
            depth = int(depth)
        except (TypeError, ValueError):
            raise LomError("type_error", "'depth' must be an integer", path=path)
        depth = max(1, min(depth, 4))

        target = self.resolve(path)
        budget = [_DESCRIBE_MAX_NODES]
        described = self._describe(target, path, depth, budget)
        described["budget_left"] = budget[0]
        if budget[0] <= 0:
            described["truncated"] = True
        return described

    def _describe(self, obj, path, depth, budget):
        result = {
            "path": path,
            "class": class_name(obj),
            "properties": [],
            "children": [],
            "methods": [],
            "allowed_methods": [],
        }
        try:
            name = getattr(obj, "name", None)
            if isinstance(name, str):
                result["name"] = name
        except Exception:
            pass

        candidates = type_candidates(obj)
        try:
            attributes = sorted(dir(obj))
        except Exception as exc:
            raise LomError("live_error", "dir() failed: {0}".format(exc), path=path)

        for attribute in attributes:
            if attribute.startswith("_") or attribute in _DESCRIBE_SKIP:
                continue
            if budget[0] <= 0:
                break
            budget[0] -= 1

            reason = _track_guard_reason(obj, attribute)
            if reason is not None:
                result["properties"].append(
                    {"name": attribute, "type": None, "settable": False,
                     "unavailable": reason})
                continue

            try:
                value = getattr(obj, attribute)
            except Exception as exc:
                result["properties"].append(
                    {"name": attribute, "type": None, "settable": None,
                     "error": "{0}: {1}".format(type(exc).__name__, exc)})
                continue

            if callable(value):
                result["methods"].append(attribute)
                for candidate in candidates:
                    if "{0}.{1}".format(candidate, attribute) in METHOD_ALLOWLIST:
                        result["allowed_methods"].append(attribute)
                        break
                continue

            child_path = "{0}.{1}".format(path, attribute)
            if is_lom_collection(value):
                # "How many tracks are there" has to be answerable from one
                # describe, so a child collection reports a real count. The test
                # has to be is_lom_collection: the LOM's collections are Vector
                # objects and pass no isinstance against (list, tuple), so a
                # describe that asks the wrong question hands back every child
                # without a count, and the only way left to count anything is a
                # probe loop up the indices until index_out_of_range - one round
                # trip per track (measured 2026-08-29 against Live 12.4.5).
                #
                # count is null ONLY when Live genuinely will not say (some
                # collections raise on len), and ``count_unknown`` then says so
                # rather than leaving a bare null to be guessed at.
                count = collection_length(value)
                entry = {"name": attribute, "type": "list",
                         "class": class_name(value), "is_collection": True,
                         "count": count, "path": child_path}
                if count is None:
                    entry["count_unknown"] = ("this collection does not answer len() "
                                              "in this Live version")
                if depth > 1 and count:
                    entry["items"] = self._describe_list(value, child_path, depth, budget)
                result["children"].append(entry)
                continue
            if is_lom_object(value):
                entry = {"name": attribute, "type": class_name(value),
                         "path": child_path, "is_collection": False}
                try:
                    child_name = getattr(value, "name", None)
                    if isinstance(child_name, str):
                        entry["name_value"] = child_name
                except Exception:
                    pass
                if depth > 1 and budget[0] > 0:
                    entry["detail"] = self._describe(value, child_path, depth - 1, budget)
                result["children"].append(entry)
                continue

            encoded, kind, _truncated = encode_value(value, child_path)
            result["properties"].append({
                "name": attribute,
                "type": kind,
                "settable": self._settable(obj, attribute),
                "value": encoded,
            })

        result["methods"].sort()
        result["allowed_methods"].sort()
        return result

    def _describe_list(self, values, path, depth, budget):
        """The elements of one child collection, as far as the budget allows.

        Goes through :func:`sequence_items` rather than iterating directly: a
        LOM collection is a C type and not every build offers the iterator
        protocol, while ``len()`` plus ``[i]`` always answers.
        """
        items = []
        for position, item in enumerate(sequence_items(values)):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            item_path = "{0}[{1}]".format(path, position)
            if is_lom_object(item):
                entry = {"index": position, "path": item_path,
                         "class": class_name(item)}
                try:
                    item_name = getattr(item, "name", None)
                    if isinstance(item_name, str):
                        entry["name"] = item_name
                except Exception:
                    pass
                items.append(entry)
            else:
                encoded, kind, _t = encode_value(item, item_path)
                items.append({"index": position, "value": encoded, "type": kind})
        return items

    def _settable(self, obj, attribute):
        """``True``/``False`` when Python can tell, otherwise ``None``.

        Only a real ``property`` object exposes ``fset``. Live's generated types
        use C-level get/set descriptors that always look writable and raise at
        assignment time, so anything else is honestly unknown.
        """
        descriptor = getattr(type(obj), attribute, None)
        if isinstance(descriptor, property):
            return descriptor.fset is not None
        return None

    # -- 5.7 lom_batch -----------------------------------------------------

    def _handle_lom_batch(self, params):
        """Run several operations in one round trip (protocol §5.7).

        Batching is what makes a strictly serial protocol usable: a 20-track
        mixer pass is one round trip instead of twenty.

        ``atomic`` is advisory and defaults to ``false``. The LOM has no
        transaction, so a failing op cannot roll back the ops before it. With
        ``atomic: true`` the script stops at the first error and the rest report
        ``skipped`` — nothing is undone either way. Do not present a batch as
        transactional.
        """
        ops = params.get("ops")
        if not isinstance(ops, list):
            raise LomError("type_error", "lom_batch needs 'ops' as a list")
        atomic = bool(params.get("atomic", False))

        handlers = {
            "get": self._handle_lom_get,
            "set": self._handle_lom_set,
            "call": self._handle_lom_call,
        }

        results = []
        ok_count = 0
        error_count = 0
        stopped = False

        for position, op in enumerate(ops):
            if stopped:
                results.append({"status": "error", "code": "skipped",
                                "message": "skipped: an earlier op failed and "
                                           "atomic was requested"})
                error_count += 1
                continue
            if not isinstance(op, dict):
                results.append({"status": "error", "code": "type_error",
                                "message": "op {0} must be an object".format(position)})
                error_count += 1
                if atomic:
                    stopped = True
                continue

            handler = handlers.get(op.get("op"))
            if handler is None:
                results.append({
                    "status": "error", "code": "unknown_handler",
                    "message": "op {0}: 'op' must be get, set or call, got "
                               "{1!r}".format(position, op.get("op"))})
                error_count += 1
                if atomic:
                    stopped = True
                continue

            try:
                entry = handler(op)
                if isinstance(entry, DeferredReadBack):
                    # A batch owns Live's main thread for its whole run, so it
                    # cannot yield for a later tick the way a single lom_set
                    # can: the follow-up would not run until the
                    # batch returned, and waiting for it here would deadlock.
                    # The op says so instead of guessing "clamped".
                    entry = entry.unresolved(
                        "this set ran inside a lom_batch, which holds Live's "
                        "main thread until every op is done; send this write on "
                        "its own if you need the second read")
                entry["status"] = "success"
                results.append(entry)
                ok_count += 1
            except LomError as exc:
                results.append(exc.to_response())
                error_count += 1
                if atomic:
                    stopped = True
            except Exception as exc:
                results.append({"status": "error", "code": "internal",
                                "message": "{0}: {1}".format(type(exc).__name__, exc)})
                error_count += 1
                if atomic:
                    stopped = True

        return {"results": results, "ok_count": ok_count, "error_count": error_count,
                "atomic": atomic, "rolled_back": False}

    # -- 5.8 notes_get / notes_set -----------------------------------------

    def _notes_from_clip(self, clip, from_time=None, time_span=None):
        """Read a clip's notes, or one time window of them. Returns ``(notes, api)``.

        ``from_time`` and ``time_span`` are clip-local beats and default to the
        whole clip. ``time_span`` is a LENGTH, not an end beat -- the extended
        API takes it that way and this passes it straight through.

        The window exists because there was no way to ask a narrow question. A
        session on 2026-08-30 needed the note density of 59 drum clips and could
        only get it by reading every note of each: one clip alone came
        back as 384 notes and 57k characters. Live has taken the window since
        Live 9; nothing above it offered one.

        The extended API is the only one used, and ``api_used`` is therefore
        always ``"extended"`` — it stays in the reply because a caller reading
        an old log should be able to see which API answered.

        There is no fallback to Live's legacy ``clip.get_notes``. It is not
        merely lossy, it is dangerous: see ``_LEGACY_NOTE_API_REFUSED``.

        Argument order, which is **not** the same in the two APIs and is easy to
        swap by accident (read from the LOM, and the call below matches it):

            get_notes_extended(from_pitch, pitch_span, from_time, time_span)
            get_notes(from_time, from_pitch, time_span, pitch_span)

        The Live 11+ extension fields (``probability``, ``velocity_deviation``,
        ``release_velocity``) are included only when Live actually provides
        them, never invented as defaults — protocol §5.8. The MPE fields ride
        along on the same rule.
        """
        getter = getattr(clip, "get_notes_extended", None)
        if getter is None:
            raise LomError("live_error",
                           _LEGACY_NOTE_API_REFUSED.format("Clip.get_notes_extended"))

        notes = []
        span = self._note_span(clip)
        start = 0.0 if from_time is None else float(from_time)
        length = span if time_span is None else float(time_span)
        if length < 0:
            raise LomError("type_error",
                           "time_span is a length in beats and cannot be negative "
                           "(got {0}); it is not an end beat".format(length))
        try:
            raw = getter(0, 128, start, length)
        except (TypeError, ValueError) as exc:
            raise LomError("type_error", str(exc))
        except RuntimeError as exc:
            raise LomError("live_error", str(exc))

        for note in sequence_items(raw):
            entry = {
                "pitch": int(getattr(note, "pitch", 0)),
                "start_time": float(getattr(note, "start_time", 0.0)),
                "duration": float(getattr(note, "duration", 0.0)),
                "velocity": float(getattr(note, "velocity", 0)),
                "mute": bool(getattr(note, "mute", False)),
            }
            for optional, caster in (("probability", float),
                                     ("velocity_deviation", float),
                                     ("release_velocity", float),
                                     ("note_id", int)):
                if hasattr(note, optional):
                    try:
                        entry[optional] = caster(getattr(note, optional))
                    except Exception:
                        pass
            for optional in ("pitch_bend_range", "pressure", "timbre", "slide"):
                if hasattr(note, optional):
                    try:
                        entry[optional] = float(getattr(note, optional))
                    except Exception:
                        pass
            notes.append(entry)
        return notes, "extended"

    def _note_span(self, clip):
        """Time span that certainly covers every note in the clip.

        ``clip.length`` is the loop, not the content: notes can sit past it
        after a loop change. The end marker and a beat of headroom cover that.
        """
        span = 0.0
        for attribute in ("length", "end_marker", "end_time"):
            try:
                value = float(getattr(clip, attribute))
            except (AttributeError, TypeError, ValueError):
                continue
            if value > span:
                span = value
        return span + 1.0

    def _handle_notes_get(self, params):
        """Notes of one MIDI clip, optionally windowed, optionally counted only.

        ``count_only`` answers "how many" without carrying the notes back over
        the socket. There is no note_count anywhere in the LOM, so counting used
        to mean transferring every note in order to call len() on the result.
        """
        path = params.get("path")
        clip, clip_path = self.resolve_clip(path)
        if not bool(getattr(clip, "is_midi_clip", False)):
            raise LomError("no_such_path", "clip is not a MIDI clip", path=path)
        from_time = params.get("from_time")
        time_span = params.get("time_span")
        notes, api = self._notes_from_clip(clip, from_time, time_span)
        reply = {"path": clip_path, "count": len(notes), "api": api}
        if from_time is not None or time_span is not None:
            reply["window"] = {"from_time": from_time, "time_span": time_span}
        if not bool(params.get("count_only")):
            reply["notes"] = notes
        return reply

    def _handle_notes_set(self, params):
        """Replace or append a clip's notes in one handler call (protocol §5.8).

        ``mode`` defaults to ``replace``, and replace is
        ``remove_notes_extended`` followed by ``add_new_notes`` inside this one
        call. Adding is the only write the LOM offers, so a second write
        silently doubles a melody instead of correcting it — measured: 63 notes
        in the clip, 23 written, 86 afterwards. Making replace the default and
        doing the removal here is what turns "write these notes" into a real
        read-modify-write; ``append`` stays available and has to be asked for by
        name.

        Atomic here means "one client call", not "one Live transaction": if the
        write fails after the removal, the notes are gone. The LOM has no
        rollback and this handler does not pretend otherwise.

        **The write call. Measured 2026-08-29 against Live 12.4.5** by reading
        the clip's own method list through ``lom_describe``: ``Clip`` has
        ``add_new_notes``, ``apply_note_modifications``, ``get_notes_extended``,
        ``remove_notes_extended``, ``remove_notes_by_id`` and Live's legacy
        ``set_notes`` / ``remove_notes`` — and it has **no**
        ``set_notes_extended``, the name the symmetry with
        ``get_notes_extended`` invites. Guarding on a method that does not exist
        would tell a Live 12 user their Live is older than 11, so the version
        guard tests for ``add_new_notes`` and ``remove_notes_extended``, which is
        the real test for "Live 11 or later".

        There is no fallback to Live's legacy note API. Live answers
        ``clip.set_notes`` / ``clip.remove_notes`` from a Remote Script with a
        modal dialog that blocks its own main thread until a human clicks a
        button, and every queued request stalls behind it (measured 2026-08-29
        against Live 12.4.5). ``_LEGACY_NOTE_API_REFUSED`` says so to the
        caller.

        ``note_id`` may ride along on an incoming note — every note
        ``notes_get`` returns carries one — and is **ignored**: ``add_new_notes``
        allocates fresh ids, and there is no way to ask it for a particular one.
        The result says so in ``ignored_fields`` rather than letting a caller
        believe the ids round-tripped. Editing notes in place, by id, is what
        ``Clip.apply_note_modifications`` is for; it is deliberately not wired
        up here (see :meth:`_add_new_notes`).
        """
        path = params.get("path")
        notes = params.get("notes")
        mode = params.get("mode", "replace")
        if not isinstance(notes, list):
            raise LomError("type_error", "notes_set needs 'notes' as a list", path=path)
        if mode not in ("replace", "append"):
            raise LomError("type_error",
                           "'mode' must be 'replace' or 'append', got "
                           "{0!r}".format(mode), path=path)

        clip, clip_path = self.resolve_clip(path)
        if not bool(getattr(clip, "is_midi_clip", False)):
            raise LomError("no_such_path", "clip is not a MIDI clip", path=path)

        adder = getattr(clip, "add_new_notes", None)
        remover = getattr(clip, "remove_notes_extended", None)
        if adder is None:
            raise LomError("live_error",
                           _LEGACY_NOTE_API_REFUSED.format("Clip.add_new_notes"),
                           path=path)
        if mode == "replace" and remover is None:
            raise LomError("live_error",
                           _LEGACY_NOTE_API_REFUSED.format("Clip.remove_notes_extended"),
                           path=path)

        existing, read_api = self._notes_from_clip(clip)
        before_count = len(existing)
        validated = [self._validate_note(note, position)
                     for position, note in enumerate(notes)]
        ignored = set()
        for note in notes:
            if isinstance(note, dict) and note.get("note_id") is not None:
                ignored.add("note_id")

        span = self._note_span(clip)
        for note in validated:
            end = note["start_time"] + note["duration"] + 1.0
            if end > span:
                span = end

        if mode == "replace":
            # Argument order differs between the two APIs and is not
            # symmetrical - it is easy to swap by accident, and the call below
            # is the extended one (read from the LOM, 2026-08-29):
            #   remove_notes_extended(from_pitch, pitch_span, from_time, time_span)
            #   remove_notes(from_time, from_pitch, time_span, pitch_span)
            # So this clears pitches 0..127 from beat 0 to `span`, which covers
            # the clip's own length, its end marker and anything the caller is
            # about to write past either. Written with the legacy order it
            # would clear pitch 0 only, from beat 128 - and the write
            # afterwards would look like a doubling, not a failure.
            try:
                remover(0, 128, 0.0, span)
            except (TypeError, ValueError) as exc:
                raise LomError("type_error", str(exc), path=path)
            except RuntimeError as exc:
                raise LomError("live_error", str(exc), path=path)

        # What the clip holds *now*, after the removal — the reference point for
        # "did a refused argument shape leave anything behind". In append mode
        # nothing was removed, so it is the count read a moment ago.
        baseline = self._count_notes(clip) if mode == "replace" else before_count
        shape, dropped_fields, attempts = self._add_new_notes(
            clip, adder, validated, path, baseline)

        after, _after_api = self._notes_from_clip(clip)
        result = {
            "path": clip_path,
            "mode": mode,
            "before_count": before_count,
            "after_count": len(after),
            "written": len(validated),
            "api": "add_new_notes",
            "read_api": read_api,
            "shape": shape,
        }
        if attempts:
            # Which spellings Live refused on the way to the one that worked.
            # Nothing else in this process knows, and the next reader of a log
            # should not have to guess.
            result["shapes_attempted"] = attempts
        if ignored:
            result["ignored_fields"] = sorted(ignored)
            result["ignored_note"] = ("add_new_notes allocates its own note ids; "
                                      "an incoming 'note_id' cannot be honoured and "
                                      "was ignored")
        if dropped_fields:
            # Say so rather than pretend the extension fields landed.
            result["dropped_fields"] = sorted(dropped_fields)
            result["note"] = ("this Live build does not accept the listed note "
                              "extension fields; they were not written")
        return result

    def _add_new_notes(self, clip, adder, validated, path, baseline):
        """Hand ``validated`` to ``Clip.add_new_notes``. Returns the shape used.

        Returns ``(shape, dropped_fields, attempts)`` — the label of the
        argument shape Live accepted, the extension fields that had to be
        dropped to build it, and the labels of every shape refused before it.

        **Why there is more than one shape.** The LOM documents
        ``add_new_notes(specification)`` as taking a
        ``Live.Clip.MidiNoteSpecification`` "or a sequence of them", and that
        sentence is the whole specification of the argument: it does not say
        which sequence types convert, and Live's Boost.Python bindings are not
        uniform about it across versions. Guessing one spelling and calling the
        failure "Live's fault" is exactly the silent-failure shape this repo
        exists to avoid, so the shapes are tried in order and the ones that
        failed are reported either way:

        1. ``add_new_notes(tuple_of_MidiNoteSpecification)`` — the documented
           sequence form, one call for the whole list. Preferred: one call
           cannot land half a melody.
        2. ``add_new_notes(spec)`` once per note — the documented single form.
        3. ``add_new_notes(tuple_of_dicts)`` and 4. one dict per note — a
           defensive fallback for a binding that takes the field names but not
           the specification class. **Unverified**: no Live is known to accept
           it, and it costs one refused call to find out.
        5. ``add_new_notes(tuple_of_5_tuples)`` — ``(pitch, start_time,
           duration, velocity, mute)``, the tuple the legacy ``set_notes`` took.
           Also **unverified** here, and last because it cannot carry the
           extension fields at all.

        If every shape is refused the error names **all** of them with Live's
        own text for each, because "add_new_notes did not work" is not an
        actionable sentence and the next person to look at this needs to know
        which spelling Live objected to and how.

        **A shape is only retried while nothing has landed.** Between attempts
        the clip is counted; if a per-note shape got some notes in before
        failing, retrying another spelling would duplicate them, so the handler
        stops and reports the clip as partially written — the same honesty
        ``automation_write`` owes a half-drawn envelope. The LOM has no
        rollback here either.

        ``apply_note_modifications(notes)`` is the third write call Live offers
        and is **not** used: it edits notes that already exist, matched by
        ``note_id``, so it can change a velocity but never add or remove a note.
        Wiring it up means a third ``mode`` and a wire-shape decision, and it is
        untestable without a running Live. It is the obvious next step, not part
        of this handler.
        """
        if not validated:
            # Nothing to add. Calling add_new_notes with an empty sequence would
            # be a needless way to fail, and in replace mode "no notes" is a
            # legitimate request: the removal above already emptied the clip.
            return None, set(), []

        specs, dropped, spec_unavailable = self._build_note_specs(validated)
        plain = [dict(note) for note in validated]
        legacy_tuples = [(note["pitch"], note["start_time"], note["duration"],
                          note["velocity"], note["mute"]) for note in validated]

        shapes = []
        if specs is not None:
            shapes.append(("MidiNoteSpecification sequence", tuple(specs), False))
            shapes.append(("MidiNoteSpecification, one call per note", specs, True))
        shapes.append(("dict sequence", tuple(plain), False))
        shapes.append(("dict, one call per note", plain, True))
        shapes.append(("tuple sequence (pitch, start_time, duration, velocity, mute)",
                       tuple(legacy_tuples), False))

        attempts = []
        if spec_unavailable:
            attempts.append("MidiNoteSpecification: " + spec_unavailable)

        for label, payload, per_note in shapes:
            try:
                if per_note:
                    for item in payload:
                        adder(item)
                else:
                    adder(payload)
            except Exception as exc:
                failure = "{0}: {1}: {2}".format(label, type(exc).__name__, exc)
                current = self._count_notes(clip)
                if current < 0 or baseline < 0:
                    # The clip would not say how many notes it holds, so a
                    # partial write cannot be ruled out. Trying the next shape
                    # could double what is already in there; stopping cannot.
                    raise LomError(
                        "live_error",
                        "add_new_notes failed with the {0} shape and the clip would "
                        "not report its note count afterwards, so it is unknown "
                        "whether anything landed. No further argument shape was "
                        "tried, because retrying one could double the clip. Read it "
                        "with notes_get before writing again.".format(label),
                        path=path, partial_unknown=True,
                        shapes_attempted=attempts + [failure])
                landed = current - baseline
                if landed > 0:
                    raise LomError(
                        "live_error",
                        "add_new_notes failed with the {0} shape after {1} of {2} "
                        "notes had already landed: {3}: {4}. THE CLIP IS PARTIALLY "
                        "WRITTEN - the LOM has no rollback and nothing was undone, "
                        "and retrying another argument shape now would double what "
                        "is in there. Read the clip with notes_get and rewrite it "
                        "with mode='replace'.".format(
                            label, landed, len(validated), type(exc).__name__, exc),
                        path=path, notes_written=landed,
                        notes_total=len(validated), partial=True,
                        shapes_attempted=attempts + [failure])
                attempts.append(failure)
                if isinstance(exc, RuntimeError):
                    # Live raised for a reason of its own, not because the
                    # argument would not convert. Another spelling of the same
                    # notes will not help, and trying one hides Live's answer.
                    raise LomError(
                        "live_error",
                        "Clip.add_new_notes refused the write and nothing was "
                        "written: {0}. Shapes attempted: {1}".format(
                            exc, "; ".join(attempts)),
                        path=path, shapes_attempted=attempts)
                continue
            return label, dropped, attempts

        raise LomError(
            "live_error",
            "Clip.add_new_notes accepted none of the {0} argument shapes this "
            "script knows, and nothing was written. Attempted, in order: "
            "{1}".format(len(shapes), "; ".join(attempts)),
            path=path, shapes_attempted=attempts)

    def _count_notes(self, clip):
        """How many notes the clip holds right now, or ``-1`` if it will not say.

        Counts the vector without decoding a single note, because this runs
        once before the write and once after every refused argument shape, and
        the only question it answers is "did anything land". A read that itself
        fails must not mask the write error being reported, hence the sentinel
        rather than a raise — and ``-1`` means *unknown*, which the caller
        treats as "a partial write cannot be ruled out", never as zero.
        """
        getter = getattr(clip, "get_notes_extended", None)
        if getter is None:
            return -1
        try:
            count = collection_length(getter(0, 128, 0.0, self._note_span(clip)))
        except Exception:
            return -1
        return -1 if count is None else count

    def _validate_note(self, note, position):
        """Coerce and bounds-check one incoming note object.

        Only the fields this script writes are carried through. Anything else
        an incoming note happens to hold — ``note_id`` above all, which every
        note ``notes_get`` returns carries — is dropped here, so a
        read-modify-write round trip cannot break the write path. ``notes_set``
        reports the drop in ``ignored_fields``.
        """
        if not isinstance(note, dict):
            raise LomError("type_error",
                           "note {0} must be an object".format(position))
        try:
            pitch = int(note.get("pitch", 60))
            start_time = float(note.get("start_time", 0.0))
            duration = float(note.get("duration", 0.25))
            velocity = float(note.get("velocity", 100))
        except (TypeError, ValueError) as exc:
            raise LomError("type_error", "note {0}: {1}".format(position, exc))
        if pitch < 0 or pitch > 127:
            raise LomError("type_error",
                           "note {0}: pitch {1} is outside 0..127".format(position, pitch))
        if start_time < 0.0:
            raise LomError("type_error",
                           "note {0}: start_time must not be negative".format(position))
        if duration <= 0.0:
            raise LomError("type_error",
                           "note {0}: duration must be greater than zero".format(position))
        if velocity < 0.0 or velocity > 127.0:
            raise LomError("type_error",
                           "note {0}: velocity {1} is outside 0..127".format(
                               position, velocity))

        entry = {
            "pitch": pitch,
            "start_time": start_time,
            "duration": duration,
            "velocity": velocity,
            "mute": bool(note.get("mute", False)),
        }
        for field in _NOTE_EXTENSION_FIELDS:
            if field in note and note[field] is not None:
                try:
                    entry[field] = float(note[field])
                except (TypeError, ValueError):
                    raise LomError("type_error",
                                   "note {0}: {1} must be a number".format(position, field))
        return entry

    def _build_note_specs(self, validated):
        """Build ``MidiNoteSpecification`` objects. **Never raises.**

        Returns ``(specs, dropped, unavailable)``: the specification objects,
        the extension fields Live refused, and ``None``. On failure ``specs`` is
        ``None`` and ``unavailable`` is a sentence saying why there are none —
        :meth:`_add_new_notes` then falls through to the plain shapes and names
        this reason in its error if those fail too. Raising here instead would
        let one unusable constructor spelling kill the whole write with no other
        spelling tried.

        Three constructor spellings per note, because Live 11 introduced the
        extension fields and Live's Boost.Python classes are not uniform about
        keyword arguments across versions:

        1. keywords including the extension fields;
        2. keywords with the five core fields only — the extras are reported as
           ``dropped``, never invented as defaults (protocol §5.8);
        3. positional core fields, for a binding that takes no keywords at all.

        Spelling 1 is the one Live 12.4.5 is expected to take. 2 and 3 are read
        from the LOM documentation and **unverified**: this file has never run
        against a Live older than 12.4.5.
        """
        spec_class = getattr(getattr(Live, "Clip", None), "MidiNoteSpecification", None)
        if spec_class is None:
            return None, set(), ("Live.Clip.MidiNoteSpecification is unavailable in "
                                 "this Live version")
        dropped = set()
        specs = []
        for position, note in enumerate(validated):
            core = {
                "pitch": note["pitch"],
                "start_time": note["start_time"],
                "duration": note["duration"],
                "velocity": note["velocity"],
                "mute": note["mute"],
            }
            extras = {}
            for field in _NOTE_EXTENSION_FIELDS:
                if field in note:
                    extras[field] = note[field]

            spec = None
            if extras:
                kwargs = dict(core)
                kwargs.update(extras)
                try:
                    spec = spec_class(**kwargs)
                except TypeError:
                    spec = None
            if spec is None:
                try:
                    spec = spec_class(**core)
                except TypeError:
                    spec = None
                else:
                    for field in extras:
                        dropped.add(field)
            if spec is None:
                try:
                    spec = spec_class(note["pitch"], note["start_time"],
                                      note["duration"], note["velocity"],
                                      note["mute"])
                except TypeError as exc:
                    return None, set(), (
                        "MidiNoteSpecification rejected note {0} as keyword "
                        "arguments and as positional ones: {1}".format(position, exc))
                for field in extras:
                    dropped.add(field)
            specs.append(spec)
        return specs, dropped, None

    # -- 5.9 automation_read / write / clear -------------------------------

    def _envelope(self, clip, param, create=False):
        """The clip's envelope for a parameter, optionally creating it.

        ``clip.automation_envelope()`` returns ``None`` for Arrangement clips
        and for parameters belonging to a different track — quoted from the LOM
        and measured. The build order is therefore Session clip, notes,
        automation, and only then duplicate into the Arrangement; envelopes are
        part of the clip and travel with the copy.
        """
        envelope = None
        try:
            envelope = clip.automation_envelope(param)
        except (AttributeError, RuntimeError, TypeError):
            envelope = None
        if envelope is not None or not create:
            return envelope

        creator = getattr(clip, "create_automation_envelope", None)
        if creator is None:
            raise LomError("live_error",
                           "Clip.create_automation_envelope is unavailable in this "
                           "Live version")
        try:
            envelope = creator(param)
        except (RuntimeError, TypeError) as exc:
            raise LomError("live_error", str(exc))
        if envelope is None:
            raise LomError(
                "live_error",
                "could not obtain an automation envelope for {0!r}. Clip automation "
                "exists only in Session clips, and only for parameters on the clip's "
                "own track.".format(getattr(param, "name", "?")))
        if not hasattr(envelope, "insert_step"):
            raise LomError("live_error",
                           "Envelope.insert_step is unavailable in this Live version")
        return envelope

    def _sample_envelope(self, envelope, start, end, count):
        points = []
        step = (end - start) / float(count - 1) if count > 1 else 0.0
        for index in range(count):
            moment = start + index * step
            try:
                points.append([round(moment, 6), float(envelope.value_at_time(moment))])
            except Exception:
                points.append([round(moment, 6), None])
        return points

    def _handle_automation_read(self, params):
        """Sample an envelope (protocol §5.9).

        Live cannot enumerate an envelope's breakpoints, so this samples
        ``value_at_time`` and answers the three questions that matter: does a
        curve exist, what is its range, and does it move at all. What it cannot
        answer is where the breakpoints sit — for that the ``.als`` file is the
        only source (docs/architecture.md, 'two channels into Ableton', channel B).

        Sampling starts at ``READ_EPSILON`` rather than 0 unless the caller asks
        for a start explicitly: at ``time = 0`` Live returns the parameter's
        default value, not the curve (measured).
        """
        path = params.get("path")
        clip, clip_path = self.resolve_clip(path)
        param = self.resolve_parameter(params.get("parameter"))

        try:
            clip_length = float(clip.length)
        except (AttributeError, TypeError, ValueError):
            clip_length = 0.0

        epsilon_applied = False
        if params.get("start") is None:
            start = READ_EPSILON
            epsilon_applied = True
        else:
            try:
                start = float(params.get("start"))
            except (TypeError, ValueError):
                raise LomError("type_error", "'start' must be a number", path=path)
        try:
            end = clip_length if params.get("end") is None else float(params.get("end"))
        except (TypeError, ValueError):
            raise LomError("type_error", "'end' must be a number", path=path)
        if end <= start:
            end = start + 1.0

        try:
            count = int(params.get("points", DEFAULT_AUTOMATION_SAMPLES))
        except (TypeError, ValueError):
            raise LomError("type_error", "'points' must be an integer", path=path)
        count = max(2, min(count, MAX_AUTOMATION_SAMPLES))

        envelope = self._envelope(clip, param, create=False)
        result = {
            "path": clip_path,
            "parameter": params.get("parameter"),
            "parameter_name": str(getattr(param, "name", "")),
            "clip_length": clip_length,
            "start": start,
            "end": end,
            "epsilon_applied": epsilon_applied,
            "has_envelope": envelope is not None,
        }
        if envelope is None:
            result["points"] = []
            result["count"] = 0
            result["min"] = None
            result["max"] = None
            result["moves"] = False
            return result

        points = self._sample_envelope(envelope, start, end, count)
        result["points"] = points
        result["count"] = len(points)
        # Every sample is returned, because the caller asked for this window. But
        # a sample at or below the guard band is the parameter's DEFAULT, not the
        # curve, and the summary fields must not be computed over it: measured
        # 2026-08-30 against Live 12.4.5, a ramp written 0.6 -> 0.1 read back with
        # start=0 answered max 0.85 and span 0.748 against a real span of 0.5,
        # because the default sat above the whole curve. first was the default
        # too, on a curve that starts at 0. The prose note said beat 0 means
        # nothing while these fields quietly used it.
        trusted = [value for time, value in points
                   if value is not None and time > READ_EPSILON]
        ignored = sum(1 for time, value in points
                      if value is not None and time <= READ_EPSILON)
        result["ignored_at_zero"] = ignored
        if ignored:
            result["ignored_note"] = (
                "{0} sample(s) at or below beat {1} are the parameter default rather "
                "than the curve and are excluded from min/max/span/first/last/moves; "
                "they are still listed in points because you asked for that "
                "window".format(ignored, READ_EPSILON))
        values = trusted
        if values:
            low = min(values)
            high = max(values)
            result["min"] = low
            result["max"] = high
            result["span"] = high - low
            result["first"] = values[0]
            result["last"] = values[-1]
            # A curve that does not move is usually not a curve.
            result["moves"] = bool((high - low) > 1e-9)
        else:
            result["min"] = None
            result["max"] = None
            result["moves"] = False
        return result

    def _handle_automation_write(self, params):
        """Write a curve into a Session clip's envelope (protocol §5.9).

        Interpolation happens here, in the script, so two breakpoints plus a
        mode describe a whole ramp — one request instead of a few hundred
        pre-sampled points across a strictly serial socket.

        Everything is computed before Live is touched, so a rejected request
        can never leave a half-cleared envelope behind.

        What it cannot prevent is a failure *during* the write. There is no bulk
        envelope call in the Live API of 11 or 12: a curve is written one
        ``insert_step`` at a time, and a curve that stops halfway is a real,
        audible, half-written envelope which the LOM cannot roll back. So the
        loop carries its own deadline (``AUTOMATION_WRITE_TIME_BUDGET``, under
        this handler's main-thread budget, under the client's — protocol §8) and
        every way out of it reports ``steps_written`` of ``steps_total``,
        ``partial: true`` and the beat range that did land. A partial curve that
        reported a bare timeout would be indistinguishable from no curve at all.
        """
        path = params.get("path")
        clip, clip_path = self.resolve_clip(path)
        parameter_path = params.get("parameter")
        param = self.resolve_parameter(parameter_path)
        points = params.get("points")
        if not isinstance(points, list):
            raise LomError("type_error",
                           "automation_write needs 'points' as a list", path=path)

        mode = normalize_interpolation(params.get("interpolation", "linear"))
        prepared = prepare_automation_points(param, points, mode,
                                             params.get("exponent", 2.0))
        try:
            clip_length = float(clip.length)
        except (AttributeError, TypeError, ValueError):
            clip_length = 0.0
        steps, resolution = build_automation_steps(
            prepared, params.get("resolution"), clip_length, self.log_message)

        # Probe at t > 0: at t = 0 Live reports the parameter default, not the
        # curve, so a before/after pair taken there would compare two defaults.
        probe = max(READ_EPSILON,
                    (prepared[0]["time"] + prepared[-1]["time"]) / 2.0)
        existing = self._envelope(clip, param, create=False)
        before_sample = None
        if existing is not None:
            try:
                before_sample = float(existing.value_at_time(probe))
            except Exception:
                before_sample = None

        clear_first = bool(params.get("clear_first", True))
        cleared = False
        if clear_first:
            cleared = self._clear_envelope(clip, param)

        envelope = self._envelope(clip, param, create=True)
        total_steps = len(steps)
        written_steps = 0
        deadline = time.time() + AUTOMATION_WRITE_TIME_BUDGET
        for step_time, step_length, step_value in steps:
            if written_steps and time.time() > deadline:
                raise LomError(
                    "live_error",
                    "automation_write ran out of time after {0} of {1} steps "
                    "({2:.1f}s budget). THE ENVELOPE IS PARTIALLY WRITTEN: beats "
                    "{3} to {4} carry the new curve and the rest of the range does "
                    "not. The LOM has no rollback, so nothing was undone - clear "
                    "the envelope and retry with a coarser 'resolution', or write a "
                    "shorter range.".format(
                        written_steps, total_steps, AUTOMATION_WRITE_TIME_BUDGET,
                        steps[0][0], steps[written_steps - 1][0]),
                    path=path, steps_written=written_steps,
                    steps_total=total_steps, partial=True)
            try:
                envelope.insert_step(step_time, step_length, step_value)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise LomError(
                    "live_error",
                    "insert_step failed at t={0}, after {1} of {2} steps: {3}. THE "
                    "ENVELOPE IS PARTIALLY WRITTEN - the LOM has no rollback and "
                    "nothing was undone. Clear the envelope before retrying, or the "
                    "next write lands on top of half a curve.".format(
                        step_time, written_steps, total_steps, exc),
                    path=path, steps_written=written_steps,
                    steps_total=total_steps, partial=True)
            written_steps += 1

        after_sample = None
        try:
            after_sample = float(envelope.value_at_time(probe))
        except Exception:
            after_sample = None

        values = [entry["value"] for entry in prepared]
        return {
            "path": clip_path,
            "parameter": parameter_path,
            "parameter_name": str(getattr(param, "name", "")),
            "written": len(prepared),
            "steps": len(steps),
            "resolution": resolution,
            "interpolation": mode,
            "cleared": cleared,
            "clip_length": clip_length,
            "probe_time": probe,
            "before_sample": before_sample,
            "after_sample": after_sample,
            "time_range": [prepared[0]["time"], prepared[-1]["time"]],
            "value_range": [min(values), max(values)],
        }

    def _clear_envelope(self, clip, param):
        """Clear one parameter's envelope. ``True`` when something ran."""
        clearer = getattr(clip, "clear_envelope", None)
        if clearer is not None:
            try:
                clearer(param)
                return True
            except Exception as exc:
                self.log_message("clear_envelope failed for {0!r}: {1}".format(
                    getattr(param, "name", "?"), exc))

        # Fallback for Live 12.2+, where the envelope deletes its own events.
        envelope = self._envelope(clip, param, create=False)
        deleter = getattr(envelope, "delete_events_in_range", None) if envelope else None
        if deleter is not None:
            try:
                end = float(clip.length)
            except (AttributeError, TypeError, ValueError):
                end = 0.0
            deleter(0.0, end + 1.0)
            return True
        return False

    def _handle_automation_clear(self, params):
        """Clear one envelope, or every envelope of the clip.

        With no ``parameter`` — or with ``all: true`` — the whole clip is
        cleared via ``clear_all_envelopes``. There is no ``envelope.clear()``
        in the LOM.
        """
        path = params.get("path")
        clip, clip_path = self.resolve_clip(path)
        parameter_path = params.get("parameter")
        wants_all = bool(params.get("all", False)) or parameter_path is None

        if wants_all:
            clearer = getattr(clip, "clear_all_envelopes", None)
            if clearer is None:
                raise LomError("live_error",
                               "Clip.clear_all_envelopes is unavailable in this Live "
                               "version", path=path)
            clearer()
            return {"path": clip_path, "cleared": True, "cleared_all": True,
                    "has_envelopes": bool(getattr(clip, "has_envelopes", False))}

        param = self.resolve_parameter(parameter_path)
        if not self._clear_envelope(clip, param):
            raise LomError(
                "live_error",
                "could not clear the envelope of {0!r} - neither Clip.clear_envelope "
                "nor Envelope.delete_events_in_range is available in this Live "
                "version".format(getattr(param, "name", "?")),
                path=path)
        return {
            "path": clip_path,
            "parameter": parameter_path,
            "parameter_name": str(getattr(param, "name", "")),
            "cleared": True,
            "cleared_all": False,
            "has_envelopes": bool(getattr(clip, "has_envelopes", False)),
        }

    # -- 5.10 browser_walk -------------------------------------------------

    def _browser(self):
        application = self._root_object("app")
        browser = getattr(application, "browser", None)
        if browser is None:
            raise LomError("live_error", "the Live browser is not available")
        return browser

    def _browser_roots(self, browser):
        """The browser's top-level categories that this installation has."""
        roots = []
        for name in self._BROWSER_ROOTS:
            try:
                item = getattr(browser, name, None)
            except Exception:
                continue
            if item is not None:
                roots.append((name, item, "app.browser.{0}".format(name)))
        return roots

    def _browser_entry(self, item, path):
        """One browser item, in the shape §5.10 promises for *every* entry.

        ``name``, ``path``, ``uri``, ``is_folder``, ``is_device`` and
        ``is_loadable`` are always present — ``null`` where this Live version
        does not answer — so a caller never has to tell "key missing" from
        "false".

        ``path`` is a real resolver path (``app.browser.instruments.children[6]``)
        and that is the whole point of the walk: a search result feeds straight
        back into ``lom_call(app.browser, load_item, [{"__path__": ...}])``
        (protocol §5.5) without the client reconstructing anything.

        ``child_count`` is ``null``, not 0, when the collection will not answer
        ``len()``. The difference matters: handing out 0 for both makes the walk
        treat "unknown" as "empty" and stop there.

        ``kind`` is always present: one of :data:`BROWSER_KINDS`, derived by
        :func:`browser_kind` from the flags above and the file extension. It is
        on *every* entry, in every mode, so that a caller can filter client-side
        as well — the handler's own ``kind`` filter only narrows a ``query``.
        """
        entry = {"path": path, "name": None, "uri": None, "is_folder": None,
                 "is_device": None, "is_loadable": None}
        for attribute, caster in (("name", str), ("uri", str), ("is_folder", bool),
                                  ("is_device", bool), ("is_loadable", bool),
                                  ("is_selected", bool)):
            try:
                value = getattr(item, attribute)
            except Exception:
                continue
            try:
                entry[attribute] = caster(value)
            except Exception:
                pass
        try:
            entry["child_count"] = collection_length(getattr(item, "children", None))
        except Exception:
            entry["child_count"] = None
        entry["kind"] = browser_kind(entry)
        return entry

    def _browser_child_items(self, item):
        """A browser item's children as a real Python list, or ``[]``.

        Every level of every browser walk goes through here, so the fix for
        ``list(BrowserItemVector)`` lives in exactly one place
        (:func:`sequence_items`).
        """
        try:
            children = getattr(item, "children", None)
        except Exception:
            return []
        if children is None:
            return []
        return sequence_items(children)

    def _browser_count(self, entries):
        """Total items in a nested walk result, so ``count`` means what it says."""
        total = 0
        for entry in entries:
            total += 1
            children = entry.get("children")
            if children:
                total += self._browser_count(children)
        return total

    def _browser_bfs(self, roots, depth, budget):
        """Yield ``(root_name, item, path, level)`` **breadth-first**.

        ``level`` counts from 1 at a root, and ``depth`` is how many levels may
        be examined - so ``depth=1`` yields the roots and nothing else.

        Breadth-first is about *budgets* here, not about ordering. Measured
        2026-08-29 against Live 12.4.5, walking the same roots depth-first under
        the same budgets::

            root=drums,       query="808 Boom",     depth=6  -> considered=6, 6 hits
            root=instruments, query="808 Boom Kit", depth=8  -> considered=0,
                                                                truncated=true
            root=sounds,      same                           -> considered=0,
                                                                truncated=true
            root=packs,       same                           -> considered=1, FOUND

        ``instruments`` has 32 children and every one of them is a folder with
        its own tree. Depth-first, the entire budget goes into the first branch
        and the walk never comes back up, so nothing at any level gets examined -
        including the shallow items that would have answered the query. Level by
        level, the same budget buys every candidate down to whatever depth it
        reaches, and what it did not reach is reported rather than implied.

        The budget is charged **per examined node**, once, here - so every
        caller of this generator shares one accounting and ``considered`` can be
        a real number. The frontier has a ceiling of its own
        (``_BROWSER_MAX_FRONTIER``): a wide graph held level by level is a
        memory cost inside Live's own process, and an unbounded one is not
        acceptable there.
        """
        frontier = deque()
        for root_name, item, path in roots:
            frontier.append((root_name, item, path, 1))
        while frontier:
            root_name, item, path, level = frontier.popleft()
            if not budget.spend():
                return
            yield root_name, item, path, level
            if level >= depth:
                continue
            if len(frontier) >= self._BROWSER_MAX_FRONTIER:
                budget.hit_node_limit = True
                continue
            for position, child in enumerate(self._browser_child_items(item)):
                frontier.append((root_name, child,
                                 "{0}.children[{1}]".format(path, position),
                                 level + 1))

    def _handle_browser_walk(self, params):
        """Walk, search or resolve in the browser's own object graph (§5.10).

        Four shapes in one handler because they all share one walk:

        * ``uri`` resolves a single item and reports where it sits;
        * ``path`` navigates ``Folder/Subfolder`` by name below one root;
        * ``query`` returns every item whose **name** contains the substring,
          case-insensitively, as a flat match list, best match first;
        * otherwise the roots are walked ``depth`` levels deep as a tree.

        Every entry carries a ``path`` the resolver understands, so a result
        feeds straight back into ``lom_call`` with ``{"__path__": ...}`` — which
        is how a device gets loaded. Every entry also carries ``kind``, one of
        :data:`BROWSER_KINDS`, so the caller can tell a rack preset from a
        sample before it loads one.

        **Measured 2026-08-29 against Live 12.4.5:** an unranked, unlabelled
        search of root "drums" for "808 Boom" answers with ``Kick 808 Boom
        Eb.wav`` and never with ``808 Boom Kit.adg``. Loading the first gives an
        ``OriginalSimpler``; the second is the ``DrumGroupDevice`` a human means
        by "the 808 Boom kit". Both halves of that have to be answered:

        * an entry has to say what it *is*, or neither the caller nor a human
          reading the log can tell the sample from the preset. Every entry
          reports ``kind``, and ``kind`` as a **parameter** — ``device`` |
          ``preset`` | ``sample`` | ``folder`` | ``any`` (the default) — narrows
          a ``query`` to one of them;
        * matches have to be ranked. In discovery order a substring hit in the
          middle of a sample's name outranks the prefix hit on the preset;
          ranked (:func:`match_rank`), the preset comes first, and every match
          carries its ``match_rank``.

        The ``kind`` parameter narrows **``query`` only**. Filtering a tree by
        kind would delete the folders on the way to the thing being looked for
        and answer with nothing at all; a caller walking a tree filters the
        reported ``kind`` itself, which is why it is on every entry.

        Measured 2026-08-29 against Live 12.4.5: a walk that builds its levels
        with ``list(item.children)`` answers ``{"root": "drums", "depth": 3}``
        with exactly one item — the root itself — and ``depth`` has no effect
        whatsoever. A ``BrowserItemVector`` answers ``len()`` and
        ``children[0]`` perfectly well while ``list()`` fails on it, so one
        failed call empties an entire level, and the recursion, gated on the
        resulting ``child_count`` of 0, never happens. Every level goes through
        :func:`sequence_items` instead.

        The same measurement pinned down the shape of the graph, through the
        generic resolver, which always worked:

        * ``app.browser.drums`` is a ``BrowserItem`` named "Drums" with
          ``is_folder=False`` and ``is_loadable=False`` — the roots are items,
          not folders, and their content hangs off ``.children``;
        * ``app.browser.drums.children[0]`` is "Drum Hits", ``is_folder=True``;
        * ``instruments`` has 32 children ([6] "Drum Rack", [21] "Instrument
          Rack", [25] "Sampler", [26] "Simpler", [31] "Wavetable"),
          ``audio_effects`` 60 ([3] "Audio Effect Rack", [28] "EQ Eight", [37]
          "Hybrid Reverb", [41] "Looper"), ``midi_effects`` 22 ([8] "MIDI Effect
          Rack") — before a single folder is opened.

        Which is why both limits are real and both are reported. The whole walk
        shares one :class:`WalkBudget`: a node count and a wall-clock deadline
        that expires inside this handler's main-thread budget, which is itself
        inside the client's timeout (protocol §8). ``truncated`` says whether
        either fired and ``truncated_by`` says which, because "the tree is huge"
        and "Live was slow" are different problems.

        **Measured 2026-08-29 against Live 12.4.5:** searched depth-first, a
        query against the wide roots reaches nothing at all::

            root=drums,       query="808 Boom",     depth=6  -> considered=6, 6 hits
            root=instruments, query="808 Boom Kit", depth=8  -> considered=0,
                                                                truncated=true
            root=sounds,      same                           -> considered=0,
                                                                truncated=true
            root=packs,       same                           -> considered=1, FOUND

        ``considered=0`` with ``truncated=true`` is a walk that spent its whole
        budget without examining a single item: on a root with 32 folders under
        it, a depth-first search pours the budget into the first branch and never
        comes back up. Level by level (:meth:`_browser_bfs`) the same budget buys
        every shallow candidate first, which is where the answer usually is, and
        the uri lookup is breadth-first for the same reason.

        The reporting has to match, or the caller cannot see any of this:
        ``considered`` counts items **examined**, ``matched`` counts the hits and
        ``depth_reached`` says how deep the budget carried the walk. A
        ``considered`` that counted matches would say nothing at all about how
        far a fruitless walk got, which is the one thing its caller needs.

        The budgets are stacked to fit: 7.5 s for the walk inside 9.0 s for the
        handler inside the client's 10 s read timeout, the last second available
        without breaking that chain. Headroom is not the mechanism, though - a
        bigger budget spent depth-first still descends one branch.
        """
        browser = self._browser()
        budget = WalkBudget(self._BROWSER_MAX_NODES, self._BROWSER_TIME_BUDGET)

        root_name = params.get("root")
        uri = params.get("uri")
        query = params.get("query")
        sub_path = params.get("path")

        raw_depth = params.get("depth")
        try:
            depth = int(self._BROWSER_DEFAULT_DEPTH if raw_depth is None else raw_depth)
        except (TypeError, ValueError):
            raise LomError("type_error", "'depth' must be an integer")
        depth = max(1, min(depth, self._BROWSER_MAX_DEPTH))

        kind = params.get("kind")
        kind = "any" if kind is None else str(kind).strip().lower()
        if kind not in BROWSER_KIND_FILTERS:
            raise LomError("type_error",
                           "'kind' must be one of {0}, got {1!r}".format(
                               ", ".join(BROWSER_KIND_FILTERS), params.get("kind")))

        roots = self._browser_roots(browser)
        if root_name:
            roots = [entry for entry in roots if entry[0] == root_name]
            if not roots:
                raise LomError(
                    "no_such_path",
                    "unknown browser root {0!r}; this installation has: {1}".format(
                        root_name,
                        ", ".join(name for name, _i, _p in self._browser_roots(browser))))

        if uri:
            found = self._browser_find_uri(roots, str(uri),
                                           self._BROWSER_MAX_DEPTH, budget)
            if found is not None:
                found_root, entry, entry_path = found
                return {"mode": "uri", "uri": uri, "found": True, "root": found_root,
                        "item": entry, "item_path": entry_path,
                        "budget_left": budget.nodes,
                        "truncated": False, "truncated_by": None}
            # "Not found" and "gave up looking" are different answers, and a
            # caller that cannot tell them apart will conclude the item does not
            # exist. Hence truncated on a miss.
            return {"mode": "uri", "uri": uri, "found": False, "item": None,
                    "item_path": None,
                    "budget_left": budget.nodes,
                    "truncated": budget.exhausted(),
                    "truncated_by": budget.reason()}

        if sub_path:
            if len(roots) != 1:
                raise LomError("type_error",
                               "'path' needs a single 'root' to start from")
            name, item, path = roots[0]
            item, path = self._browser_navigate(item, path, sub_path)
            # Here ``depth`` counts the levels BELOW the named folder: the point
            # of navigating to one is to see what is in it.
            children = self._browser_children(item, path, depth, budget)
            return {"mode": "path", "root": name, "path": sub_path, "depth": depth,
                    "item": self._browser_entry(item, path),
                    "children": children,
                    "count": self._browser_count(children),
                    "budget_left": budget.nodes,
                    "truncated": budget.exhausted(),
                    "truncated_by": budget.reason()}

        if query:
            needle = str(query).strip().lower()
            try:
                limit = int(params.get("limit", 200))
            except (TypeError, ValueError):
                raise LomError("type_error", "'limit' must be an integer")
            limit = max(1, min(limit, 1000))
            # A search stopped at the default depth of 1 would only ever see the
            # roots, so the default here is the full depth; an explicit ``depth``
            # is honoured and is the way to keep a search shallow and quick.
            search_depth = depth if raw_depth is not None else self._BROWSER_MAX_DEPTH
            # Collect past ``limit`` and rank afterwards. Ranking can only rank
            # what was collected, so stopping at ``limit`` in discovery order
            # would throw away the best answer whenever it sits behind a worse
            # one - which is the failure ranking exists to prevent. The pool is
            # what the node and time budgets already guard.
            pool = max(limit, self._BROWSER_MATCH_POOL)
            candidates, considered, depth_reached = self._browser_search(
                roots, needle, pool, search_depth, budget, kind)
            # Stable: equal ranks keep discovery order, and a shorter name is
            # the more specific answer at the same rank.
            candidates.sort(key=lambda entry: (entry["match_rank"],
                                               len(entry.get("name") or "")))
            matches = candidates[:limit]
            # Three separate reasons a query answer can be short of the truth,
            # and they are told apart because they call for different next
            # steps: ask for a bigger ``limit``, narrow the query, or narrow the
            # ``root``/``depth``.
            hit_limit = len(candidates) > limit
            pool_full = len(candidates) >= pool
            if hit_limit:
                truncated_by = "limit"
            elif pool_full:
                truncated_by = "match_pool"
            else:
                truncated_by = budget.reason()
            return {"mode": "query", "query": query, "matches": matches,
                    "count": len(matches), "limit": limit, "depth": search_depth,
                    "kind": kind,
                    # How many items the walk LOOKED AT - never how many it
                    # matched. A search that finds nothing has to be able to say
                    # whether it examined two items or two thousand: that number
                    # is the only evidence a caller has about a truncated
                    # walk.
                    "considered": considered,
                    "matched": len(candidates),
                    # How many levels below a root the budget actually carried
                    # the walk. Breadth-first, this is a real statement about
                    # coverage: everything above it was examined.
                    "depth_reached": depth_reached,
                    "budget_left": budget.nodes,
                    "truncated": bool(truncated_by),
                    "truncated_by": truncated_by}

        items = []
        for name, item, path in roots:
            if not budget.spend():
                break
            entry = self._browser_entry(item, path)
            entry["root"] = name
            if depth > 1:
                children = self._browser_children(item, path, depth - 1, budget)
                if children:
                    entry["children"] = children
            items.append(entry)
        return {"mode": "tree", "depth": depth, "items": items,
                "count": self._browser_count(items),
                "roots": [name for name, _i, _p in self._browser_roots(browser)],
                "budget_left": budget.nodes,
                "truncated": budget.exhausted(),
                "truncated_by": budget.reason()}

    def _browser_children(self, item, path, depth, budget):
        """The children of one browser item, recursively, ``depth`` levels down.

        ``depth`` counts levels **below** ``item``: 0 returns nothing, 1 the
        children, 2 the children and their children.

        The recursion is **not** gated on ``child_count``. A collection that
        will not answer ``len()`` reports 0, and treating that as empty makes a
        whole subtree vanish with nothing to say so — which is how ``depth``
        comes to look as though it were ignored (measured 2026-08-29 against
        Live 12.4.5).
        """
        children = []
        if depth <= 0:
            return children
        for position, child in enumerate(self._browser_child_items(item)):
            if not budget.spend():
                break
            child_path = "{0}.children[{1}]".format(path, position)
            entry = self._browser_entry(child, child_path)
            if depth > 1:
                grandchildren = self._browser_children(child, child_path,
                                                       depth - 1, budget)
                if grandchildren:
                    entry["children"] = grandchildren
            children.append(entry)
        return children

    def _browser_navigate(self, item, path, sub_path):
        """Follow a ``Folder/Subfolder`` path below a root, by name."""
        for part in str(sub_path).split("/"):
            part = part.strip()
            if not part:
                continue
            found = None
            for position, child in enumerate(self._browser_child_items(item)):
                try:
                    child_name = str(child.name)
                except Exception:
                    continue
                if child_name.lower() == part.lower():
                    found = (child, "{0}.children[{1}]".format(path, position))
                    break
            if found is None:
                raise LomError("no_such_path",
                               "browser path segment {0!r} not found under "
                               "{1}".format(part, path), path=path)
            item, path = found
        return item, path

    def _browser_find_uri(self, roots, uri, depth, budget):
        """The one item with this ``uri``, as ``(root_name, entry, path)``.

        ``None`` when it was not reached — which is not the same as "it does not
        exist", and the caller reports ``truncated`` for exactly that reason.

        Breadth-first over every root at once. Depth-first per root would make a
        uri sitting two levels down under the *second* root unreachable whenever
        the first root's tree is big enough to eat the budget, and a uri lookup
        is the one call that is supposed to be cheap.
        """
        for root_name, item, path, _level in self._browser_bfs(roots, depth, budget):
            try:
                if str(getattr(item, "uri", None)) == uri:
                    return root_name, self._browser_entry(item, path), path
            except Exception:
                pass
        return None

    def _browser_search(self, roots, needle, pool, depth, budget, kind="any"):
        """Every item whose name contains ``needle``, case-insensitively.

        Returns ``(matches, considered, depth_reached)``.

        ``needle`` arrives already lowered and stripped. Matching is on the name
        only: a uri match is what the ``uri`` mode is for, and folders match on
        their own name like anything else, so "reverb" finds the folder as well
        as the devices inside it.

        Four properties of this search, none of them optional:

        * ``kind`` drops entries that are not the kind the caller asked for
          **before** they take a slot in the pool — filtering afterwards would
          throw away exactly the budget the search needs. It never stops the
          walk, though: a folder is not a preset, and the presets are inside the
          folders.
        * every match carries ``match_rank`` (:func:`match_rank`), so the caller
          in :meth:`_handle_browser_walk` can sort the pool before cutting it
          down to ``limit``. ``5`` — no name match — is never appended; it is
          the value the ranking function returns for a name the search would not
          have looked at.

        * the walk is **breadth-first** (:meth:`_browser_bfs`), so the budget is
          spent on candidates across the whole graph rather than on descending
          one branch of one root;
        * ``considered`` is the number of items **examined**, as protocol §5.10
          promises — never the number of matches. A search that finds nothing
          would otherwise report ``considered=0`` whether it had looked at two
          items or two thousand, and that number is the only evidence a caller
          has for how far a truncated walk got.

        ``pool`` is the collection ceiling, not the caller's ``limit``. See
        ``_BROWSER_MATCH_POOL``.
        """
        matches = []
        considered = 0
        depth_reached = 0
        for _root_name, item, path, level in self._browser_bfs(roots, depth, budget):
            considered += 1
            if level > depth_reached:
                depth_reached = level
            try:
                name = str(getattr(item, "name", ""))
            except Exception:
                name = ""
            rank = match_rank(name, needle)
            if rank >= 5:
                continue
            entry = self._browser_entry(item, path)
            if kind != "any" and entry.get("kind") != kind:
                continue
            entry["match_rank"] = rank
            matches.append(entry)
            if len(matches) >= pool:
                break
        return matches, considered, depth_reached

    # -- 5.11 events_observe / drain / clear -------------------------------

    def _handle_events_observe(self, params):
        """Register a Live listener that feeds the ring buffer (protocol §5.11).

        This is how the server notices that a *human* changed something.
        Listeners run in Live's process, in an application with a real-time
        audio thread; the cost of many registrations is unmeasured
        (docs/limits.md §5, 'Cost of introspection and of listeners'), so the
        count is reported back and the buffer is bounded.
        """
        path = params.get("path")
        prop = params.get("property")
        if not isinstance(prop, str) or not prop or prop.startswith("_"):
            raise LomError("type_error",
                           "events_observe needs a 'property' name", path=path)

        target = self.resolve(path)
        adder = getattr(target, "add_{0}_listener".format(prop), None)
        remover = getattr(target, "remove_{0}_listener".format(prop), None)
        if adder is None or remover is None:
            raise LomError(
                "no_such_path",
                "{0} has no add_{1}_listener / remove_{1}_listener - not every LOM "
                "property is observable".format(class_name(target), prop),
                path=path)

        key = (path, prop)
        if key in self._listeners:
            return {"path": path, "property": prop, "observing": True,
                    "already_registered": True,
                    "listener_count": len(self._listeners)}

        def callback():
            # Runs on Live's main thread. It must never raise and must stay
            # cheap: an exception here fires inside Live's own notification
            # machinery, and slow work here is slow work next to the audio
            # thread.
            try:
                self._record_event(path, prop, target)
            except Exception:
                pass

        try:
            adder(callback)
        except (RuntimeError, TypeError) as exc:
            raise LomError("live_error", str(exc), path=path)

        self._listeners[key] = {"object": target, "callback": callback,
                                "remover": remover}
        return {"path": path, "property": prop, "observing": True,
                "already_registered": False,
                "listener_count": len(self._listeners)}

    def _record_event(self, path, prop, target):
        event = {"time": time.time(), "path": path, "property": prop}
        try:
            value = getattr(target, prop)
        except Exception:
            value = None
            event["value_unavailable"] = True
        if isinstance(value, (bool, int, float, str)):
            event["value"] = value
        elif is_lom_collection(value):
            # A LOM collection is a Vector, so an isinstance against
            # (list, tuple) would let an event on song.tracks fall through to
            # the branch below and report nothing but the word "Vector". For a
            # collection the count IS the signal - "a track was added" is the
            # whole message.
            event["count"] = collection_length(value)
            event["value_kind"] = "list"
        elif value is not None:
            event["value_kind"] = class_name(value)

        with self._events_lock:
            if len(self._events) >= self._EVENT_BUFFER_SIZE:
                self._events.popleft()
                self._events_dropped += 1
            self._events.append(event)

    def _handle_events_drain(self, params):
        """Empty the ring buffer. Answers on the client thread.

        Reads nothing but this script's own buffer, so it needs neither the
        main thread nor Live to be idle — which is the point: a stuck Live is
        exactly when the recent event history is worth having.
        """
        try:
            maximum = int(params.get("max", self._EVENT_BUFFER_SIZE))
        except (TypeError, ValueError):
            raise LomError("type_error", "'max' must be an integer")
        maximum = max(1, min(maximum, self._EVENT_BUFFER_SIZE))

        drained = []
        with self._events_lock:
            while self._events and len(drained) < maximum:
                drained.append(self._events.popleft())
            dropped = self._events_dropped
            self._events_dropped = 0
            remaining = len(self._events)

        return {"events": drained, "count": len(drained), "dropped": dropped,
                "remaining": remaining, "buffer_size": self._EVENT_BUFFER_SIZE,
                "listener_count": len(self._listeners)}

    # -- 5.12 enum_names ---------------------------------------------------

    def _handle_enum_names(self, params):
        """Member names of a Live enum type, or the enum types this build has.

        A PROBE, not a runtime capability. Live decodes its own device
        parameters and decodes nothing for a song or track property: measured
        2026-08-31, ``song.clip_trigger_quantization`` answers a bare ``4`` out
        of fourteen legal integers, and the catalog holds those fourteen legal
        values with not one of their meanings. The names live right here, in
        this interpreter, because this module imports ``Live`` -- so a meaning
        taken from ``Live.Song.Quantization`` is MEASURED, while the same
        meaning copied out of the LOM reference is only documented elsewhere.
        The catalog cares about that difference.

        Without ``type`` this lists the enum-like containers it can find, which
        is how the mapping from a catalog row to a type gets settled rather than
        guessed. With ``type`` -- a dotted name under ``Live``, such as
        ``Song.Quantization`` -- it returns ``{name: value}``.

        Reads module attributes only. It calls nothing and touches no LOM
        object, so there is nothing here that can block Live or mutate a set.
        """
        wanted = params.get("type")
        if wanted is None:
            return {"types": self._enum_types()}
        if not isinstance(wanted, str) or not wanted.strip():
            raise LomError("type_error",
                           "enum_names: 'type' must be a dotted name under Live, "
                           "for example 'Song.Quantization'")
        target = self._resolve_under_live(wanted)
        members = _int_members(target)
        reply = {"type": wanted, "members": members,
                 "class": type(target).__name__}
        if not members:
            # Say what it IS rather than only that it is not what was expected.
            reply["attributes"] = _attribute_report(target)
        return reply

    def _enum_types(self):
        """Enum-like containers, found by walking the Live modules that are loaded.

        Discovered through ``sys.modules`` rather than ``dir(Live)``. The first
        version used ``dir`` and returned an empty list on 2026-08-31 while
        ``getattr(Live, "Song")`` worked perfectly -- Live is a C extension
        package and its submodules are reachable without being listed. Reading
        the import table is observation; a hardcoded list of submodule names
        would have been a guess.
        """
        found = []
        module_names = sorted(
            name for name in list(sys.modules)
            if name == "Live" or name.startswith("Live.")
        )
        for module_name in module_names:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            for name in _public_names(module):
                members = _int_members(_safe_getattr(module, name))
                if members:
                    found.append({
                        "type": "{0}.{1}".format(module_name[len("Live."):] or module_name, name),
                        "count": len(members),
                        "sample": sorted(members.items(), key=lambda pair: pair[1])[:4],
                    })
        return found

    def _resolve_under_live(self, dotted):
        """Resolve a dotted name under ``Live``, or raise ``no_such_path``."""
        target = Live
        for part in dotted.split("."):
            if not part or part.startswith("_"):
                raise LomError("bad_path",
                               "enum_names: {0!r} is not a public dotted name under "
                               "Live".format(dotted))
            target = _safe_getattr(target, part)
            if target is None:
                raise LomError("no_such_path",
                               "Live has no {0!r} in this version. Call enum_names with "
                               "no 'type' to see what it does have.".format(dotted))
        return target

    def _handle_events_clear(self, params):
        """Remove listeners and empty the buffer.

        With ``path`` and ``property`` only that one listener goes; without
        them, all of them. Runs on the main thread because removing a Live
        listener is a LOM call like any other.
        """
        path = params.get("path")
        prop = params.get("property")
        if path is not None and prop is not None:
            removed = self._remove_listener((path, prop))
            return {"removed": removed, "listener_count": len(self._listeners),
                    "buffer_cleared": False}

        removed = self._remove_all_listeners()
        with self._events_lock:
            self._events.clear()
            self._events_dropped = 0
        return {"removed": removed, "listener_count": len(self._listeners),
                "buffer_cleared": True}

    def _remove_listener(self, key):
        entry = self._listeners.pop(key, None)
        if entry is None:
            return 0
        try:
            entry["remover"](entry["callback"])
        except Exception as exc:
            self.log_message("Could not remove listener {0}: {1}".format(key, exc))
        return 1

    def _remove_all_listeners(self):
        removed = 0
        for key in list(self._listeners.keys()):
            removed += self._remove_listener(key)
        return removed


def _json_fallback(value):
    """Last resort for ``json.dumps``: a repr beats a dead socket.

    The protocol is strictly serial and has no framing, so a response that
    fails to serialise would leave the client waiting on a socket that will
    never speak again.
    """
    try:
        return {"__unserialisable__": type(value).__name__, "repr": repr(value)[:400]}
    except Exception:
        return {"__unserialisable__": "unknown"}
