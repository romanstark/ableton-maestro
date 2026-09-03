# Architecture: Ableton Maestro

Ableton Maestro bridges AI assistants to Ableton Live's Object Model (LOM). Instead of writing dedicated handlers inside Live for every individual property, the architecture describes the LOM surface as declarative data (a YAML catalog) and executes path resolution, validation, and verification through a shared engine.

Key architectural concepts:
- **Generic LOM Bridge:** The Remote Script inside Live focuses on generic object navigation and execution. Handler families cover property access, method calls, and specialized objects across Live.
- **Data-Driven Catalog:** YAML definitions paired with a path builder and executor. Adding support for a new Live property requires only a new catalog entry, not new code inside Live.
- **High-Level MCP Tools with Escape Hatches:** Common production tasks use intent-based tools (`get_session`, `write_clip_notes`, `set_mix`), while low-level access remains available via generic LOM tools (`lom_get`, `lom_set`, `lom_call`).
- **Read-Back Verification:** Parameter changes are verified by reading values back from Live to confirm whether values were applied, quantized, or clamped.

See also: [protocol.md](protocol.md) for the wire specification, [catalog.md](catalog.md) for the catalog schema, and [limits.md](limits.md) for known API boundaries.

---

## The restart tax: Why "as data" matters here

Live loads Remote Scripts only at startup and reuses cached bytecode in `__pycache__` if present. Modifying code inside Live requires restarting Live completely. In contrast, updating catalog files on the Python server side requires no restart of the DAW.

```
Per-command bridge: N handlers inside Live -> every fix requires restarting Live
Data-driven bridge: Generic handler set   -> a catalog change needs a server restart
```

As a general rule, any capability that can be represented as a path within the object model should be defined in the catalog rather than as a bespoke Remote Script handler. Custom handlers inside Live are reserved for operations that generic path navigation cannot handle, such as MIDI note arrays, automation breakpoint interpolation, browser traversal, and event listeners.

---

## Two channels into Ableton

Ableton Live is accessed through two complementary channels:

```
                    +-----------------------------------------+
   MCP client  ---> |            ableton-maestro              |
                    |                                         |
                    |  Channel A: LOM      Channel B: .als    |
                    +-------+--------------------+------------+
                            |                    |
                   TCP 127.0.0.1:9878       file on disk
                            |                    |
                            v                    v
                   live-remote-script      Project.als (gzip XML)
                   (in Live's process)     only while Live is closed
                            |                    |
                            v                    v
                      Ableton Live          saved set on disk
                      (live memory)
```

| Feature | Channel A (LOM) | Channel B (`.als` file) |
|---|---|---|
| **Availability** | While Live is running | While Live is closed (file is locked otherwise) |
| **State** | In-memory session state | Saved project on disk |
| **Primary Use** | Real-time session control and automation | Reading/writing project files without opening Live |

Channel B complements the live bridge by handling data that the LOM does not expose in the open set, such as track automation in arrangements or reading unopened projects.

---

## Layers

The codebase is strictly separated into distinct layers:

```
catalog/*.yaml           Declarative catalog defining paths, types, access modes, and validation.
   |
spec.py                  PathSpec, ParamSpec, ArgSpec, path resolution, and value validation.
   |
registry.py              Loads catalog files, aggregates specs, and provides lookup indices.
   |
executor.py              Handles validation, guards, dispatch, error mapping, and verification.
   |
client.py                TCP client handling JSON encoding, timeouts, and connection locking.
   |
live-remote-script/      Python Remote Script running inside Ableton Live's process.
```

Supporting modules:
- `lom/paths.py`: Client-side parsing and validation of LOM path grammar.
- `lom/introspect.py`: Runtime inspection of dynamic device parameters and session snapshots.
- `music/`: Pure Python musical theory, chord voicings, note transformations, and groove humanization.
- `als/`: Parsers and writers for `.als` project files.
- `automation.py`: Curve generation, interpolation algorithms, and envelope sampling.
- `server.py`: FastMCP server exposing tools and resources to AI clients.

Design rule: Transport, path resolution, and musical logic remain isolated. `client.py` has no dependency on musical theory, and `music/` has no dependency on network sockets.

---

## The catalog

The catalog in `src/ableton_maestro/catalog/*.yaml` serves as the single source of truth for Live's fixed object model surface.

Example entry:

```yaml
- id: track.volume_value
  path: song.tracks[{track}].mixer_device.volume.value
  access: [get, set]
  kind: float
  range: [0.0, 1.0]
  unit: normalized
  display: db
  verify: read_back
  status: verified
  doc: >
    Track volume. The raw value is normalized between 0.0 and 1.0 (where 0.85 corresponds to 0 dB).
    Use display for human-readable decibel output.
  params:
    - {name: track, kind: int, required: true}
```

Catalog files are partitioned by LOM domain (`10-song.yaml`, `20-track.yaml`, `30-clip.yaml`, `40-device.yaml`, `50-browser.yaml`) to keep them organized and easy to maintain.

### Fixed and dynamic surface

The catalog covers static properties defined by Live's core object model. In contrast, third-party VST and AU plugins expose parameters dynamically at runtime only after being configured in Live's device chain. Dynamic parameters are discovered at runtime using `lom_describe` and `lom/introspect.py` rather than hardcoded in the catalog.

---

## Spec and builder (`spec.py`)

`spec.py` defines the data models for catalog specifications:
- `ParamSpec`: Placeholders in paths (e.g., `{track}` index or `{root}` category name).
- `ArgSpec`: Positional arguments for method calls (`access: [call]`).
- `PathSpec`: Complete path description including allowed operations, type definitions, and verification hooks.

`build_path(spec, **kwargs)` interpolates placeholders and validates against LOM grammar. `validate_value(spec, value)` validates data types, boundaries, and enum options before sending data across the socket.

---

## Executor (`executor.py`)

The executor is the central dispatch layer. It enforces safety guards, validates arguments, sends commands over the client socket, maps responses to `Result` objects, and runs verification routines.

```python
def execute(client, registry, spec_id, *, confirm=False, **args) -> Result:
    spec = registry.get(spec_id)
    if spec.destructive and not confirm:
        return Result.blocked_(spec, "destructive operation; requires confirm=True")
    if requested_access not in spec.access:
        return Result.blocked_(spec, "requested operation not permitted by catalog")
    prepared = _prepare(spec, op, args)
    result = _dispatch(client, prepared)
    if spec.verify:
        result.verified = VERIFIERS[spec.verify](client, spec, args, result)
    return result
```

Multi-step operations can be grouped using `execute_batch()`, which sends them in a single `lom_batch` request to reduce round-trip latency.

### Read-back as a principle

When setting parameters via `lom_set`, the response includes `before`, `after`, `clamped`, `changed`, and a `read_back` verdict:

| `read_back` | Description |
|---|---|
| `applied` | The value in Live matches the requested value. |
| `clamped` | Live adjusted or quantized the value to a different valid step. |
| `not_observed` | The value has not changed yet (some properties update asynchronously). |

This ensures that clients receive accurate feedback on whether a value was accepted as-is or modified by Live.

---

## The Remote Script: Handler families

The Remote Script running inside Live provides ten handler families, seventeen handlers in total once `ping` and `script_info` are counted:

### Generic Handlers (5)
- `lom_get`: Reads a property value or list of child handles.
- `lom_set`: Writes a property value and reads back the result.
- `lom_call`: Calls an allowlisted method on a resolved object.
- `lom_describe`: Introspects properties, children, and methods of a live object.
- `lom_batch`: Executes multiple operations in a single main-thread invocation.

### Specialized Handlers (5)
- `notes_get` / `notes_set`: Reading, replacing, and appending MIDI note arrays.
- `automation_read` / `automation_write` / `automation_clear`: Sampling, interpolating, and clearing clip envelopes.
- `browser_walk`: Breadth-first searching and ranking of Live's browser graph.
- `events_observe` / `events_drain` / `events_clear`: Event subscription via ring buffers.
- `enum_names`: Reads the member names of a Live enum type, so an integer property's values can be decoded. Used to fill the catalog rather than at runtime.

---

## The MCP surface (`server.py`)

The FastMCP server exposes three access tiers:

1. **Intent Tools:** High-level musical operations (`get_session`, `create_track`, `load_device`, `write_clip_notes`, `set_mix`, `write_automation`, `arrange`, `play`).
2. **LOM Escape Hatches:** Low-level operations (`lom_get`, `lom_set`, `lom_call`, `lom_batch`, `lom_describe`) for direct access to any catalogued path.
3. **MCP Resources:**
   - `ableton://catalog`: Full catalog index with summary documentation.
   - `ableton://catalog/{selector}`: Detailed entry documentation by area or status.
   - `ableton://session`: Live snapshot of tracks, clips, and mixer state.
   - `ableton://limits`: Documentation of known constraints.

### Tool argument schema definitions (`toolargs.py`)

Tools declare explicit structured input types via `toolargs.py` rather than generic container types (e.g. `list[dict[str, Any]]`). Explicit types ensure the emitted JSON Schema exposes exact field names, data types, and constraints to MCP clients:

- **Shape models:** `NoteIn` and `BatchOp` define field specifications and numeric type coercion for incoming parameters.
- **Inlined schemas:** MCP client implementations often fail to resolve external `$ref` and `$defs` schema references, leading to empty or unconstrained argument objects. `NoteArg` and `BatchOpArg` wrap models with inlined schemas via `WithJsonSchema` so that schema specifications are flattened without runtime indirection.
- **Contract verification:** `tests/test_tool_schema.py` enforces that all tool containers declare concrete shapes and that schema references remain flattened without `$ref` pointers.

---

## Verification (`verify` hooks)

Catalog entries can declare a verification hook to check post-conditions:
- `read_back`: Verifies that the property matches the requested value.
- `envelope_present` / `envelope_absent`: Checks for automation envelope existence.
- `has_clip` / `no_clip`: Checks clip slot occupancy.

---

## Directory structure

```
src/ableton_maestro/
  client.py         Socket transport, timeout classes, and retry policy. No framing:
                    the reader reads until the buffer parses (protocol section 2).
  spec.py           PathSpec, ParamSpec, ArgSpec, and validation rules.
  registry.py       Catalog loading, indexing, and lookup functions.
  executor.py       Dispatch engine, safety guards, and verification hooks.
  models.py         Data models, enums, and result containers.
  toolargs.py       Argument models for the nested tool arguments, so the JSON
                    schema a client reads carries their field names and types.
  automation.py     Curve generation, interpolation, and envelope sampling.
  catalog/*.yaml    Declarative catalog files grouped by domain.
  lom/
    paths.py        LOM path grammar parser and validation helpers.
    introspect.py   Runtime object inspection and session snapshot builders.
  music/
    notes.py        MIDI note manipulation and formatting.
    theory.py       Musical scales, chord voicings, and transposition.
    humanize.py     Timing swing and velocity variations.
  als/
    read.py         Parser for saved .als project files.
    write.py        Direct modifier for .als project files.
  server.py         FastMCP server implementation.

live-remote-script/
  __init__.py       Remote Script running inside Ableton Live.

scripts/
  install_script.py Installs the Remote Script into Ableton User Library.
  sync_catalog.py   Reconciles catalog entries against a running Live instance.
  probe_paths.py    Validates specific paths and updates catalog status.
```
