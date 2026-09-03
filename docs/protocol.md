# Wire Protocol: Ableton Maestro

This document specifies the socket protocol between the external Python client (`src/ableton_maestro/client.py`) and the Remote Script running inside Ableton Live (`live-remote-script/__init__.py`).

Companion documents: [architecture.md](architecture.md) for system design, [catalog.md](catalog.md) for the path catalog schema, and [limits.md](limits.md) for API constraints.

---

## 1. Transport

| Setting | Value | Notes |
|---|---|---|
| Socket | TCP | Bound to `127.0.0.1` (loopback only) |
| Port | `9878` | Configurable |
| Encoding | UTF-8 JSON | One JSON object per message |
| Framing | None | Stream-based JSON parsing (see §2) |
| Concurrency | Strictly serial | One outstanding request per socket (see §3) |
| Authentication | None | Secured by loopback interface binding |

Binding strictly to `127.0.0.1` ensures that communication remains internal to the host machine.

---

## 2. There is no framing

Messages do not use delimiters or length prefixes. Message boundaries are determined by parsing JSON objects directly from the incoming byte stream.

Both the client and server implement the following decoding loop:

```python
buffer = ""
while True:
    buffer += socket.recv(4096).decode("utf-8")
    try:
        obj, end = json.JSONDecoder().raw_decode(buffer)
    except ValueError:
        continue  # Incomplete message, read more bytes
    buffer = buffer[end:].lstrip()  # Retain unparsed buffer remainder
    process_message(obj)
```

Implementation considerations:
- Use an incremental UTF-8 decoder to avoid issues with multi-byte characters split across `recv()` calls.
- Retain unparsed data after `raw_decode` because multiple responses can arrive in a single `recv()` call.

---

## 3. Strictly serial

The connection handles exactly one request at a time. Sending a second request before the first response has been received will desynchronize the stream.

The client manages this using a mutex lock across send-and-receive cycles. When multiple operations need to be executed efficiently, use [`lom_batch`](#57-lom_batch) to package them into a single round trip.

If a timeout occurs, the client closes and reopens the socket to ensure no stale data remains in the buffer. Automatic retries upon reconnect are limited to read-only handlers (see §8).

---

## 4. Message shape

### Request Format
```json
{
  "type": "lom_get",
  "params": {
    "path": "song.tempo"
  }
}
```

### Response Format (Success)
```json
{
  "status": "success",
  "result": {
    "path": "song.tempo",
    "value": 124.0,
    "type": "float"
  }
}
```

### Response Format (Error)
```json
{
  "status": "error",
  "code": "no_such_path",
  "message": "Track has no attribute 'volume'",
  "path": "song.tracks[0].volume"
}
```

`params` is always a JSON object. `result` is always a JSON object rather than a raw primitive or array, allowing fields to be added without breaking existing parsers.

### Error Codes

| `code` | Description |
|---|---|
| `unknown_handler` | Handler type not recognized |
| `bad_path` | Path syntax is invalid |
| `no_such_path` | Attribute or index does not exist on the target object |
| `index_out_of_range` | Collection index exceeds current length |
| `not_settable` | Property exists but is read-only in the LOM |
| `method_not_allowed` | Method is not included in the allowlist |
| `type_error` | Value cannot be converted to target property type |
| `live_error` | Ableton Live raised an internal error |
| `internal` | Script execution error |

---

## 5. Handlers

The Remote Script implements 17 handlers across generic and specialized families.

### 5.1 `ping`
- **Request:** `{"type": "ping", "params": {}}`
- **Response:** `{"pong": true, "script_version": "<SCRIPT_VERSION>", "uptime": 123.45}`

The version in these examples is a placeholder on purpose. `SCRIPT_VERSION` in
`live-remote-script/__init__.py` is the only place it is stated; a literal here would
be wrong one release later, and it has been.

The ping handler operates independently of Live's main thread and LOM locks, allowing liveness checks even when Live is executing modal operations.

### 5.2 `script_info`
- **Request:** `{"type": "script_info", "params": {}}`
- **Response:**
  ```json
  {
    "name": "ableton-maestro",
    "script_version": "<SCRIPT_VERSION>",
    "protocol_version": 1,
    "handlers": ["lom_get", "lom_set", "..."],
    "live_version": "12.4.3",
    "port": 9878
  }
  ```

`protocol_version` indicates the wire format specification. `live_version` is cached at startup.

### 5.3 `lom_get`
- **Request:** `{"type": "lom_get", "params": {"path": "song.tracks"}}`
- **Response:**
  ```json
  {
    "path": "song.tracks",
    "type": "list",
    "class": "Vector",
    "count": 4,
    "value": [
      {"__lom__": "Track", "path": "song.tracks[0]", "name": "Drums"},
      {"__lom__": "Track", "path": "song.tracks[1]", "name": "Bass"}
    ]
  }
  ```

LOM collections (Vectors) are returned as lists of element handles with individual paths and a total `count`.

### 5.4 `lom_set`
- **Request:** `{"type": "lom_set", "params": {"path": "song.tempo", "value": 128.0}}`
- **Response:**
  ```json
  {
    "path": "song.tempo",
    "requested": 128.0,
    "before": 120.0,
    "after": 128.0,
    "read_back": "applied",
    "clamped": false,
    "changed": true,
    "read_back_attempts": 1
  }
  ```

`lom_set` reads back the property after writing to verify the change:
- `applied`: Target value matches requested value.
- `clamped`: Live quantized or clamped the value to a valid step.
- `not_observed`: The value has not updated yet (asynchronous properties).

### 5.5 `lom_call`
- **Request:**
  ```json
  {
    "type": "lom_call",
    "params": {
      "path": "song.tracks[0].clip_slots[0].clip",
      "method": "clear_envelope",
      "args": [{"__path__": "song.tracks[0].devices[0].parameters[3]"}]
    }
  }
  ```

Arguments using `{"__path__": "<path>"}` resolve to existing Live objects before method execution.

### 5.6 `lom_describe`
- **Request:** `{"type": "lom_describe", "params": {"path": "song.tracks[0].devices[0]"}}`
- **Response:**
  ```json
  {
    "path": "song.tracks[0].devices[0]",
    "class": "PluginDevice",
    "name": "Wavetable",
    "properties": [{"name": "is_active", "type": "bool", "settable": true, "value": true}],
    "children": [{"name": "parameters", "type": "list", "count": 32}],
    "methods": ["store_chosen_bank"]
  }
  ```

Inspects the properties, child collections, and available methods on any live object.

### 5.7 `lom_batch`
- **Request:**
  ```json
  {
    "type": "lom_batch",
    "params": {
      "ops": [
        {"op": "set", "path": "song.tracks[0].mixer_device.volume.value", "value": 0.85},
        {"op": "set", "path": "song.tracks[1].mixer_device.volume.value", "value": 0.75}
      ],
      "atomic": false
    }
  }
  ```
- **Response:** `{"results": [...], "ok_count": 2, "error_count": 0, "atomic": false, "rolled_back": false}`

Each entry in `results` is that operation's own unwrapped result with a `status` field added, flat rather than nested under `result`. A `get` therefore returns `{"path": ..., "value": ..., "type": ..., "status": "success"}`, and a failed operation carries `status: "error"` with the section 4 code alongside it.

Executes multiple operations in a single round trip. `atomic` is advisory, not a transaction: the LOM has no rollback, so operations that already ran stay applied. With `atomic: true` the batch stops at the first error and the remaining results carry the code `skipped`. `rolled_back` is always `false`.

### 5.8 `notes_get` / `notes_set`
- **`notes_get`:**
  - `params: {path, from_time?, time_span?, count_only?}`
  - Returns MIDI note specifications within the specified time window.
- **`notes_set`:**
  - `params: {path, notes: [...], mode: "replace" | "append"}`
  - Default mode is `"replace"`. It removes every note in the clip (pitches 0 to 127, from beat 0 to the end of the clip or of the incoming notes, whichever is later) and then inserts the new list. There is no windowed replacement, and the LOM has no rollback: if the insert fails after the removal, the notes are gone.

Note structure:
```json
{
  "pitch": 60,
  "start_time": 0.0,
  "duration": 0.5,
  "velocity": 100,
  "mute": false,
  "probability": 1.0,
  "velocity_deviation": 0.0,
  "release_velocity": 64
}
```

`notes_get` adds one key the structure above does not list: **`note_id`**, Live 11+'s own handle for an existing note. Every note from a read carries one (measured 2026-09-02 against Live 12.4.5: three written notes came back carrying `note_id` 1, 2 and 3). It is identity rather than content and is never written back: the `write_clip_notes` tool drops it on input and reports it as `input_keys_ignored`, so a list read from a clip can be edited and written back unchanged. Live may also return the MPE per-note keys `pitch_bend_range`, `pressure`, `timbre` and `slide`; these are treated the same way. Any key outside both sets is refused rather than dropped.

### 5.9 `automation_read` / `automation_write` / `automation_clear`
- **`automation_read`:**
  - `params: {path, parameter, start?, end?, points?}`
  - Samples parameter curve values across the clip length.
- **`automation_write`:**
  - `params: {path, parameter, points: [...], resolution?, interpolation?}`
  - Generates and writes breakpoints. `interpolation` accepts `linear`, `hold`, `exponential`, `ease_in` and `ease_out`; `ease_in` is exponential and `ease_out` is reported back as `logarithmic`. Anything else answers `type_error`.
- **`automation_clear`:**
  - `params: {path, parameter}`
  - Removes envelope data for the specified parameter.

### 5.10 `browser_walk`
- **Request:** `{"type": "browser_walk", "params": {"root": "drums", "query": "909", "limit": 10}}`
- **Response:** List of matching browser items sorted by relevance.

Performs breadth-first searches of Live's internal library. Every entry carries a `kind`: `device`, `preset`, `sample`, `folder`, `other` or `unknown`. The optional `kind` filter accepts only the first four plus `any`, so filtering cannot reach an `other` or `unknown` entry.

### 5.11 `events_observe` / `events_drain` / `events_clear`
- **`events_observe`:** Registers a listener for property changes, storing notifications in a ring buffer.
- **`events_drain`:** Retrieves and flushes queued events.
- **`events_clear`:** Unregisters active event listeners.

### 5.12 `enum_names`
- **Request:** `{"type": "enum_names", "params": {"type": "Song.Quantization"}}`
- **Response:** Map of symbolic names to integer values for LOM enumerations.

---

## 6. The resolver, and why calls need an allowlist

Paths are evaluated starting from root objects (`song`, `app`):

```
song.tracks[3].mixer_device.volume
 |    |         |            |
root  list      attribute    attribute
```

Path rules:
- Grammar format: `segment ("." segment)*` where `segment = name | name "[" int "]"`.
- Indices must be non-negative integers.
- Property access is permitted dynamically through path traversal.
- Method calls are permitted only via `lom_call` and must be present in the Remote Script's allowlist.
- Group tracks are guarded during track iterations because they lack arming state and arrangement clips.

---

## 7. Values

Primitive JSON values are passed directly. LOM objects are represented as handle dictionaries:

```json
{"__lom__": "Track", "path": "song.tracks[3]", "name": "Bass"}
```

Parameter value scales:
- Most continuous Live parameters are normalized between 0.0 and 1.0.
- Decibel and frequency readings from `str_for_value()` are provided in the `display` field.
- Quantized parameters (`is_quantized: true`) accept discrete steps listed in `value_items`.

---

## 8. Timeouts and retries

| Category | Default Timeout | Connection Loss Retry |
|---|---|---|
| Read operations (`ping`, `script_info`, `lom_get`, `lom_describe`, `notes_get`, `automation_read`, `browser_walk`, `events_drain`, `enum_names`) | 10 seconds | Retries once on reconnect |
| Everything else (`lom_set`, `lom_call`, `lom_batch`, `notes_set`, `automation_write`, `automation_clear`, `events_observe`, `events_clear`) | 20 seconds | Never retried automatically |

All seventeen handlers are in one row or the other. The classification lives in `READ_ONLY_HANDLERS` in `client.py`, and a handler left out of it is treated as a write: it gets the longer timeout and no retry.

To avoid duplicate actions (such as appending notes twice), write operations that time out are not retried automatically. The calling client can perform a read operation to verify whether the change landed.

---

## 9. Changing the script costs a restart

Ableton Live compiles and caches Remote Scripts on startup. Modifications to `live-remote-script/__init__.py` require:
1. Copying the script to `<User Library>\Remote Scripts\AbletonMaestro\__init__.py`.
2. Deleting any `__pycache__` folder in that directory.
3. Restarting Ableton Live.

---

## 10. What a successful response proves

A response with `"status": "success"` confirms that the Remote Script executed without raising an exception. For `lom_set`, it also confirms that the new value is stored in memory.

It does not indicate:
- Whether the result is audible (e.g., if a track is muted or an effect is bypassed).
- Whether the change has been saved to the `.als` project file on disk.
- Whether the change will persist across an undo action.
