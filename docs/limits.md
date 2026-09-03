# API Limits and Constraints: Ableton Maestro

This document details known limitations across Ableton Live's Object Model, the graphical user interface, and the Maestro socket protocol.

Categories of limitations:
- **LOM:** Not exposed or permitted by Ableton Live's Python API.
- **Live GUI:** Available in Live, but gated behind GUI interactions (e.g., plugin parameter configuration).
- **Protocol:** Limitations inherent to our socket transport or serialization format.
- **Physical / Contextual:** Fundamental differences between data inspection and human auditory judgment.

---

## 1. Hard limits: no workaround exists

### Audio rendering and exporting (LOM)
Ableton Live's Object Model does not expose functions for rendering, bouncing, or exporting audio. Audio export must be performed directly in Live's interface via **File -> Export Audio/Video**.

### Saving the project (LOM)
The LOM does not provide an API method to save the open set. Project files on disk (`.als`) represent the state from the last manual save (Ctrl+S / Cmd+S). When reading from disk via Channel B (`als/read.py`), data reflects the saved state rather than unsaved in-memory changes.

### Auditory judgment (Physical)
The MCP server can read parameter values and verify stored numbers (e.g. confirming a cutoff is set to 800 Hz), but cannot evaluate musical aesthetic quality, vocal tuning, or balance.

---

## 2. LOM limits: Live could lift these, we cannot

### Device reordering
Device positions in chains can be adjusted programmatically using `Song.move_device(device, target, index)`.

### Track reordering (LOM)
There is no equivalent for tracks. Neither `Track` nor `Song` offers a method for moving one, so a track's position is fixed once it is created. Reordering means creating a track at the target position, loading the instrument again, rebuilding the clips and notes, restoring the mixer settings and deleting the original. Anything not explicitly recreated is lost, and every cached track index shifts.

### Track grouping (LOM)
`group_track` and `is_grouped` are read-only properties in the LOM. Tracks must be grouped manually in Live (Ctrl+G / Cmd+G). Grouping inserts a new container track, which shifts the numerical index of subsequent tracks.

### Sidechain routing configuration
Sidechain routing is configured directly on the receiving device rather than at the track level:
- `device.input_routing_type`: Selects the source track from available routing types.
- `device.input_routing_channel`: Selects the routing tap point (`Pre FX`, `Post FX`, `Post Mixer`).
- The `S/C On` parameter arms it. Parameters are addressed by integer index, so the path is `device.parameters[N]`; a quoted name is not a legal path. Use `describe` with `with_parameters=True` to find N.

Routing types must be assigned by object reference (using `{"__path__": "<path>"}`) rather than by string name.

### Track automation vs. clip envelopes (LOM)
Arrangement track automation envelopes are not directly writable via the live LOM API (`clip.automation_envelope()` returns `None` for arrangement clips). Automation must be written into Session clips first and then duplicated to the Arrangement timeline via `Track.duplicate_clip_to_arrangement()`.

When both layers target the same parameter over the same stretch, the **clip envelope is what plays**. Measured 2026-09-01 against Live 12.4.5 on a 51-track arrangement: sampling a filter cutoff during playback returned the clip envelope's value wherever one existed (573 Hz, then 402, 335 and 199 Hz across a later section) and the track automation's value only where none did (10.0 kHz). Read this before deleting either layer: the only one reachable from here is the clip envelope, which is the audible one. Removing it does not tidy a redundant curve; it uncovers the track automation underneath, which in that measured case would have opened a drone from around 200 Hz to 10 kHz.

---

## 3. Live's GUI gates: the user can lift these, we cannot

### Third-party VST parameter exposure (Live)
How much of a plugin the Object Model sees is decided by the plugin. Measured 2026-09-01 against Live 12.4.5, each loaded from the browser with no GUI interaction:

| Plugin | Format | `type` | Parameters on load |
|---|---|---|---|
| Vendor A, synth | VST3 | instrument | 1 |
| Vendor A, electric piano | VST3 | instrument | 1 |
| Vendor A, tape effect | VST3 | audio effect | 16 |
| Vendor B, synth | VST3 | instrument | 29 |
| Vendor C, lo-fi effect | VST2 | audio effect | 17 |

It is not the format, not the vendor (Vendor A appears at both 1 and 16) and not instrument versus effect. No rule has been established, so do not predict it: read `parameter_count`, which `describe` reports and also diagnoses. One parameter means this instance exposes nothing yet, not that plugins never do.

What a plugin *could* expose is readable in full: `lom_call(device, 'get_parameter_names')` returns the plugin's own list whatever the strip holds. Expect noise in it (program-change, MPE and MIDI-CC helper entries can run into the hundreds) with the real controls at the head.

**Configure** mode adds a parameter to the strip: press Configure, click the control in the plugin's own GUI, leave the mode. Live allocates 128 slots per instance and stores the mapping in the set rather than in the plugin, so a second instance of the same plugin starts empty.

That mapping is **writable in the file**, which makes this the rare GUI gate that can be lifted. Measured 2026-09-01 against Live 12.4.5, on an instance whose strip held nothing: writing `ParameterName`, `ParameterId` (the index into the declared list) and `VisualIndex` into two empty slots of the saved set produced two working parameters on reopen, read back with their real values, writable through `lom_set` with `read_back: applied`, one of them reporting a unit. Nothing is inserted: Live leaves all 128 slots in the file, so this fills three fields of an element that is already there. The slot's placeholder value does not need writing; Live asks the plugin. See `als/write.py`, `configure_plugin_parameters`, which needs Live closed and takes a backup.

### VST2 display units (Live)
`str_for_value()` does not return formatted physical units for VST2 plugins. Parameter values remain normalized floats between 0.0 and 1.0.

### Normalized value vs. physical display value (Live)
In Ableton Live, internal parameter values (`value`) often use a normalized scale (0.0 to 1.0) or custom integer ranges, while the user interface displays physical units (`display`, such as Hz or dB) via non-linear mapping curves:

| Device | Parameter | Internal `value` | Range `min`..`max` | Display `display` |
|---|---|---|---|---|
| Utility | Bass Freq | 0.3802 | 0.0 .. 1.0 | 120 Hz |
| Utility | Output | -0.171875 | -1.0 .. 1.0 | -6.02 dB |
| Max device | LPF | 8794.30 | 20 .. 21000 | 288 Hz |

All three measured 2026-08-31 against Live 12.4.5. Note that `min` and `max` bound `value`, not `display`. On Live's own devices the value is usually normalized, and on a Max device it is whatever range the device defines, so the LPF above is out by a factor of thirty if read as hertz.

Set values in the internal range and read `display` back to confirm what was stored.

---

## 4. Protocol limits: ours, and therefore fixable

### Non-transactional batch operations
`lom_batch` executes operations sequentially. The LOM does not support atomic database-style rollbacks. When `atomic: true` is passed, execution stops at the first error, but previously executed steps remain in effect.

### Write timeouts and retry policy
Write operations that encounter socket timeouts are not retried automatically, because a timed-out write may have succeeded in Live. Repeating the write could result in duplicate actions (such as duplicate notes). Clients should verify state using a read operation following a timeout.

### Round-trip cost
One round trip costs about 450 ms (measured 2026-08-29 against Live 12.4.5). That is the reason `lom_batch` exists: a twenty-track mixer pass is twenty round trips without it and one with it.

### Strictly serial transport
The socket protocol processes one request at a time per connection without multiplexing. Multi-operation tasks should use `lom_batch` to execute within a single round trip.

---

## 5. Open questions: unverified, and how to settle them

### Warp markers
- Reading `clip.warp_markers` is supported.
- `Clip.move_warp_marker(marker_beat_time, beat_time_distance)` and `Clip.remove_warp_marker(beat_time)` are available in the Remote Script allowlist.
- `Clip.add_warp_marker` requires constructing an internal `WarpMarker` object in Python, which is not directly instantiable over JSON.

### Concurrent Control Surfaces
Ableton Live supports multiple simultaneous Control Surfaces. Maestro uses port 9878 by default to avoid collisions with other socket-based scripts.

### Cost of introspection and of listeners
Listeners registered through `events_observe` run inside Live's own process, which carries a real-time audio thread. The cost of many simultaneous registrations has not been measured, and neither has the cost of a deep `lom_describe` over a large set.

Properties that fire on every buffer are the ones to watch: `clip.playing_position` and `clip.is_playing` are observable, and a listener on either across many clips means a callback storm rather than an occasional event. Because the cost is unmeasured rather than known to be safe, `events_observe` reports the registration count back to the caller and bounds its event buffer instead of assuming either is free.

To settle it: register listeners in increasing numbers on `clip.playing_position` across a set during playback and watch Live's CPU meter and audio dropout counter. Until that is done, treat a high listener count as an unknown risk and prefer polling a property when a poll will do.

---

## 6. What a success result actually proves

A success response confirms that the command was executed by the Remote Script. For `lom_set`, it confirms that the value is stored in memory.

It does not guarantee:
- That the audio output is audibly altered (e.g. if the device is disabled or the track is muted).
- That the project has been saved to disk.
- That the change will not be overwritten by subsequent user actions or undo commands.

---

## 7. Properties that do not exist, and are therefore not in the catalog

The following properties do not exist in the Live Object Model and return `no_such_path`:

| Property | Note |
|---|---|
| `master.arm`, `return.arm` | Main and Return tracks do not support arming or monitoring modes. |
| `clip.follow_action_*` | Follow actions are managed via GUI controls and are not exposed in the LOM. |
| `device.presets` | Device presets are loaded through the Browser rather than directly on the Device object. |
| `device.view.is_showing_chain_devices` | Applicable only to Rack device views. |
| `sampler.sample`, `sampler.playback_mode` | These properties belong to `Simpler` devices; `MultiSampler` (Sampler) exposes only the common `Device` interface. |

---

## 8. Preconditions are part of the question

Object Model attributes depend on the concrete object class and state:
- `choke_group` is specific to `DrumChain` (not standard `Chain`).
- Warping properties apply only to `AudioClip` instances (not `MidiClip`).
- Drum pads are present only on `DrumGroupDevice` instances.

### Track index renumbering
Live references tracks by zero-based positional index. Inserting or deleting a track shifts the indices of subsequent tracks. Always re-survey session track lists (`get_session`) after adding or removing tracks.
