# The Path Catalog

The catalog defines Ableton Live's fixed Object Model surface. It is maintained as YAML files under `src/ableton_maestro/catalog/*.yaml` and loaded at runtime by `registry.py`.

Adding support for a Live property or allowlisted method is done by adding a row to this catalog, without modifying the Remote Script inside Ableton Live.

---

## Where the rows come from

Because Ableton does not publish a machine-readable specification of the Live Object Model (LOM), catalog entries were initially created from the official LOM documentation and subsequently verified against active Live instances using introspection and probing scripts.

- `scripts/sync_catalog.py` compares Live's runtime objects (via `lom_describe`) against catalog rows to discover uncatalogued properties and flag non-existent paths.
- `scripts/probe_paths.py` executes read and write operations against an active Live set to confirm behavior.

### Handling Non-Existent Properties

If a property does not exist in the Object Model (`no_such_path`), its row is removed from the catalog rather than marked as `broken`. The catalog represents an accurate inventory of real paths. Any non-existent properties that developers might expect to find are documented in [limits.md §7](limits.md#7-properties-that-do-not-exist--and-are-therefore-not-in-the-catalog).

The `broken` status is reserved for properties that exist in Live's API but produce errors or unexpected refusals when accessed (such as `song.master_track.mute`).

---

## Why a directory and not one file

The catalog is split into multiple files by domain:

| File | Domain |
|---|---|
| `10-song.yaml` | Song, Application, transport, scenes, and cue points |
| `20-track.yaml` | Track, MixerDevice, returns, and master |
| `30-clip.yaml` | ClipSlot, Clip, MIDI notes, and clip envelopes |
| `40-device.yaml` | Device, DeviceParameter, Racks, chains, and drum pads |
| `50-browser.yaml` | Browser navigation, items, and Groove Pool |

IDs must be unique across all files. The loader raises an error if duplicate IDs are detected.

---

## Row schema

```yaml
- id: track.volume
  path: song.tracks[{track}].mixer_device.volume
  access: [get, observe, automate]
  kind: float
  range: [0.0, 1.0]
  unit: normalized
  display: db
  quantized: false
  verify: read_back
  destructive: false
  status: verified
  doc: >
    Track volume parameter. Values are normalized between 0.0 and 1.0 (0.85 = 0 dB). Use display for human-readable decibel output.
    Read verified 2026-08-29 against Live 12.4.5; a write was NOT attempted on this row.
  params:
    - {name: track, kind: int, required: true}
```

### Field Definitions

| Field | Required | Description |
|---|:---:|---|
| `id` | Yes | Unique identifier in the format `domain.name`. |
| `path` | Yes | LOM path template with placeholders (e.g. `song.tracks[{track}]`). |
| `access` | Yes | List of supported operations: `get`, `set`, `call`, `observe`, `automate`. |
| `kind` | No | Data type: `bool`, `int`, `float`, `str`, `enum`, `list`, `object`. Defaults to `object`. |
| `range` | No | List `[min, max]` for numeric validation before sending across socket. |
| `enum` | No | Allowed values for enum types. |
| `unit` | No | Unit type: `normalized`, `db`, `hz`, `semitones`, `beats`, `seconds`, `percent`, `none`. |
| `display` | No | Expected display unit format from Live (via `str_for_value()`). |
| `quantized` | No | Boolean indicating whether parameter uses discrete steps rather than a continuous range. |
| `method` | No | Method name to execute (used only with `access: [call]`). |
| `verify` | No | Verification hook name (defaults to `read_back`). |
| `destructive` | No | Boolean. If true, executor requires `confirm=True` to run. |
| `status` | Yes | Validation state: `verified`, `broken`, or `untested`. |
| `doc` | Yes | Documentation string explaining parameter behavior, scale, and constraints. |
| `means` | No | Mapping of specific values to their semantic meaning. |
| `params` | No | Parameter definitions for path placeholders. |
| `args` | No | Argument definitions for method calls (`access: [call]` only). |

---

### `params` and `args`

`params` and `args` serve distinct roles in the schema:

| Feature | `params` | `args` |
|---|---|---|
| **Role** | Substitutes `{placeholder}` in `path` | Passed as arguments to a method |
| **Appears in** | Path template string | Wire payload for method execution |
| **Allowed on** | Any row with `{placeholder}` | Rows with `access: [call]` only |

Example method row:

```yaml
- id: track.delete_device
  path: song.tracks[{track}]
  access: [call]
  method: delete_device
  destructive: true
  status: verified
  doc: >
    Removes a device from a track by chain index. Deleting shifts the index of every device after it.
    Call verified 2026-08-30 against Live 12.4.5 on a scratch track.
  params:
    - {name: track, kind: int, required: true}
  args:
    - {name: index, kind: int, required: true, doc: "Zero-based position in the device chain."}
```

---

### `doc` Guidelines

Documentation in `doc` is made available to AI clients through the `ableton://catalog` resource. Clear documentation helps avoid incorrect assumptions about parameters.

Effective documentation should note:
1. Physical unit interpretations (e.g., whether `0.5` represents normalized gain or frequency).
2. Known constraints, asynchronous behavior, or snapping.
3. Read-only limitations that are not obvious from the property name.
4. Version differences between Live 11 and Live 12.

---

### `means`: Value decoding

Some Live properties return numeric enums or sentinel values without descriptive labels. The `means` map provides human-readable explanations for specific return values:

```yaml
- id: track.playing_slot_index
  means:
    "-1": "No clip is currently playing in this track."
    "-2": "Track is currently stopping playback."
```

When a property value matches an entry in `means`, the server attaches the explanation to the result. It reaches raw paths too: `lom_get`, `lom_set` and `lom_batch` look the row up with `registry.rows_for_path()`.

---

## Status and verification discipline

Every new row starts with `status: untested`.

- `status: verified`: Confirmed working against an active Ableton Live instance.
- `status: broken`: The property exists in the API, but Live refuses access or raises an unhandled error.
- `status: untested`: Documented in the catalog, but not yet verified with a live probe.

### Three outcomes, not two

When probing a path with `scripts/probe_paths.py`, the result falls into one of several categories based on Live's response:

| Response | Meaning | Action |
|---|---|---|
| Valid value returned | Property functions as expected | Set `status: verified` and record details in `doc`. |
| `no_such_path` | Property does not exist on object | Remove the row from the catalog and document in `limits.md §7`. |
| `live_error` | Property exists, but Live raises an error | Set `status: broken` with the error message in `doc`. |
| `index_out_of_range` | Target object was missing from test set | Keep `status: untested` until tested with an appropriate set. |
| `not_settable` | Property is read-only | Remove `set` from `access` list. |

---

## What the catalog deliberately does not cover

The catalog covers static properties in Ableton's core Object Model.

Dynamic properties (such as third-party VST/AU plugin parameters) vary by user session and are only populated when configured in Live's device strip. These dynamic parameters are discovered at runtime via `lom_describe` and `lom/introspect.py`.

---

## Adding a new row

1. Identify the target LOM property (use `python scripts/sync_catalog.py --report` to check for uncatalogued properties in an open set).
2. Add the entry to the relevant YAML file under `src/ableton_maestro/catalog/` with `status: untested` and an informative `doc` string.
3. Run `pytest` to validate catalog syntax and schema integrity.
4. Run `python scripts/probe_paths.py --id <new.row.id> --go` against an open test set to verify behavior.
5. Update `status` to `verified` once confirmed.
