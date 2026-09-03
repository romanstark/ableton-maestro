# The Remote Script

`__init__.py` in this folder is the whole of Ableton Maestro that runs inside Ableton Live's own Python interpreter: a socket server on `127.0.0.1:9878` and the seventeen handlers of [`docs/protocol.md`](../docs/protocol.md) §5.

Everything else in the repo runs outside Live and can be changed at will. This file cannot — see [Changing it costs a restart](#changing-it-costs-a-restart).

---

## Install

Copy this folder into the **User Library**, under `Remote Scripts`:

```
<User Library>\Remote Scripts\AbletonMaestro\__init__.py
```

Then in Live: **Preferences → Link, Tempo & MIDI → Control Surface** and pick `AbletonMaestro` in a free slot.

`scripts/install_script.py` does the copy and finds the User Library for you. If you do it by hand, the three rules below are the ones that cost time.

### 1. The User Library path is written down — never guess it

Read `Preferences\Library.cfg`. It is frequently **not** `Documents`, and there is no fallback worth trying.

`<ProjectPath>` in that file is where to start, but on its own it is **not reliably the User Library**. Measured 2026-08-29: `Library.cfg` gave `<ProjectPath Value="D:/Documents/Ableton"/>` alongside `<ProjectName Value="User Library"/>`, and Live was loading its Remote Scripts one level deeper, from `<ProjectPath>\<ProjectName>`. A second `Remote Scripts` folder sat directly under `<ProjectPath>` and Live never read it. So check both, and take the one holding Live's own `Ableton Folder Info` marker folder — which is exactly what `scripts/install_script.py` does rather than picking on a rule.

From the wrong folder Live loads **nothing and says nothing**: no entry in the Control Surface dropdown, no error in the log, no clue. If the entry is missing after a restart, the path is wrong — not the script.

Not `%APPDATA%\Ableton\Live x.x.x\Preferences\User Remote Scripts\`. That is the location for Live 10 and older.

The User Library is also version-independent, which matters: Live updates itself and its preferences folder moves with it (`Live 12.4.2` → `Live 12.4.3`, measured). A script in the User Library survives that; one in the preferences folder does not.

### 2. The folder name is what Live shows in the dropdown

`Remote Scripts\AbletonMaestro\` appears as **AbletonMaestro**. Rename the folder and the dropdown entry changes with it — which also means the selection in Preferences is lost and has to be set again.

The folder name in *this repo* (`live-remote-script/`) is free and only has to be readable. The one in `Remote Scripts` is user-visible.

### 3. Delete `__pycache__`, then restart Live completely

Live loads Remote Scripts **only at startup**, and it loads the compiled `__pycache__` if one is there. So after every edit:

```powershell
Remove-Item -Recurse -Force "<User Library>\Remote Scripts\AbletonMaestro\__pycache__"
# then quit Live entirely and start it again
```

Without the deletion Live silently runs the previous version. Without the full restart it runs nothing new at all. Neither is reported anywhere.

### Alongside another Remote Script

This script listens on **9878**, and Live can hold several Control Surfaces at once, so it should be installable next to another socket-based Remote Script that uses a different port.

**Unverified:** that Live is happy with two Control Surfaces each running a socket server of its own. It should be — but nobody has measured it. If it turns out not to hold, the fallback is to switch between them instead, at the cost of one Live restart per switch.

---

## Check that it is running

```powershell
Get-Process "Ableton Live*"                    # is Live up at all?
Test-NetConnection 127.0.0.1 -Port 9878        # is the script listening?
```

Then, from the repo:

```bash
python -m ableton_maestro.client ping
python -m ableton_maestro.client info
```

The `info` subcommand sends the `script_info` request, which reports the script version, the protocol version, the handler list, the Live version, the host and port, and the size of the method allowlist. The request is deliberately **not** called `get_script_info`: that name is in use by other Ableton Remote Script bridges, with their own response shape, and this script makes no claim to be compatible with them.

Live's own log (`Log.txt` in the current preferences folder) carries every `log_message` from the script — including the coarsened-resolution warning from `automation_write` and any listener that could not be removed.

---

## Changing it costs a restart

That is the single fact this whole design is built around (`docs/architecture.md`, *The restart tax*). A handler in here costs every user a Live restart to change; a row in the server-side catalog costs nothing.

So: **a capability that can be expressed as a path must not become a handler.** Adding one is an architecture break and needs a written justification. The sixteen that exist are the five generic ones (`lom_get`, `lom_set`, `lom_call`, `lom_describe`, `lom_batch`), the nine that make up the four families a path genuinely cannot express (notes, envelopes, the browser graph, listeners), and two infrastructure handlers (`ping`, `script_info`).

---

## What the script enforces

These are not conventions, they are code. See `docs/protocol.md` §6.

| Rule | Why |
|---|---|
| Binds `127.0.0.1` only | The channel has no authentication and Live's API offers no auth primitive. Loopback is the security control, not a default — the first line to check after any change to the socket setup |
| Properties read and write freely by path | That is the generic surface |
| Methods are **never** reachable by path | An arbitrary method call inside an audio application with the user's unsaved work in it can hang or crash it |
| `lom_call` checks a **frozenset in this file** | 140 entries over 17 Live class names, covering 129 distinct method names. It lives in the script so it cannot be widened from outside Live. The catalog says what the server will *offer*; this says what the script will *do*, and the script wins |
| Every index is bounds-checked before use | An `IndexError` escaping a handler kills the client connection |
| No slices, no negative indices, no private names | A negative index would silently address the wrong track instead of failing |
| Group tracks are guarded | They have no arm state and no arrangement clips; unguarded iteration raises at the first group (measured) |
| Nothing escapes a handler as an exception | Every failure is one of the structured codes in protocol §4 |
| `lom_set` always reads back | Live refuses an out-of-range value out loud, but snaps a quantised one silently, can apply a write late enough that the first read is stale, and reports success for a write that did nothing. See below |

### `lom_set` never answers only "success"

```json
{"path": "song.tracks[3].mixer_device.volume.value",
 "requested": 1.4, "before": 0.85, "after": 1.0,
 "read_back": "clamped", "clamped": true, "changed": true, "display": "6.0 dB"}
```

`read_back` names which of three things happened — `applied`, `clamped`, or `not_observed` when `after` still equals `before` and a clamp to the stored value cannot be told from a write Live has not applied yet (protocol §5.4). `clamped` is `read_back == "clamped"`; `changed` is `after != before`. Both comparisons use a relative tolerance so float representation noise does not cry wolf. A quantized parameter also reports `is_quantized`, because otherwise a clamp from 0.35 to 0.25 looks arbitrary.

What this proves is the **stored value**, never the audible effect (protocol §10). A parameter can sit at exactly the right value and do nothing because the device is off.

---

## Calling methods with Live objects as arguments

Half the allowlist takes a Live object — `Clip.clear_envelope(param)`, `Browser.load_item(item)`, `ClipSlot.duplicate_clip_to(slot)` — and JSON has no way to write one. Wrap a path instead:

```json
{"type": "lom_call",
 "params": {"path": "app.browser", "method": "load_item",
            "args": [{"__path__": "app.browser.instruments.children[3]"}]}}
```

The reference goes through the same resolver with the same guards. Every entry `browser_walk` returns carries a `path` in exactly that form, so a search result feeds straight back into a load.

---

## Threading

Live's object model is not thread-safe, and even reads (`device.parameters`, clip envelopes) are unsafe off-thread. So every handler that touches the LOM is scheduled onto Live's main thread via `schedule_message` and answered through a queue with a per-handler timeout that is deliberately **shorter** than the client's (protocol §8) — this side gives up first and can say why, instead of the client hitting a bare socket timeout.

Three handlers answer on the client thread on purpose:

- `ping` and `script_info`, so they still report while Live's main thread is busy. That is their job.
- `events_drain`, because it reads nothing but this script's own ring buffer — and a stuck Live is exactly when the recent event history is worth having.

---

## Known limits of this file

Stated here rather than discovered later:

- **`settable` in `lom_describe` is often `null`.** Live's objects expose C-level descriptors that accept assignment and raise at call time, so writability genuinely cannot be read from Python. `true`/`false` appears only for real `property` objects. The catalog is the authority.
- **Automation exists only in Session clips.** `clip.automation_envelope()` returns `None` for Arrangement clips — and for a parameter belonging to a different track (quoted from the LOM, measured). Write in Session, then duplicate into the Arrangement — never the reverse.
- **At `time = 0` Live returns the parameter default, not the curve** (measured across nine curves). `automation_read` therefore starts at 1/64 beat unless a `start` is given, and reports `epsilon_applied` so nobody has to remember.
- **`notes_set` replace is atomic per *call*, not per *transaction*.** If the write fails after the removal, the notes are gone. The LOM has no rollback and neither does `lom_batch` — `atomic` there only means "stop at the first error".
- **`str_for_value` is only ever called with a parameter's current value.** Passing `min` or `max` has taken Live down: a crash inside a plugin's own C code cannot be caught by `try`/`except` (measured with a 3rd-party sampler plugin).
- **Track automation is not writable through the LOM at all**, so it is not readable back either. That is Channel B's job (`docs/architecture.md`, *Two channels into Ableton*).

---

## Editing this file

It is the one file in the repo exempt from the project's Python conventions, because Live imports it, not us:

- the standard library, plus what Live's own interpreter provides (`_Framework.ControlSurface`, `Live`). No third-party dependencies;
- no type annotations, no f-strings — the interpreter is Live's, and its history is long;
- the comments are kept long on purpose. This is the only code that runs inside Live's own interpreter, and what they record — Live's C-level behaviour, the main-thread rules, LOM internals — is written down nowhere else and cannot be rediscovered without an install and a restart. Several of them are the only surviving account of a measurement that cost a Live session to make, and they are what the next Live version will be debugged against. Cut rhetoric where you find it; leave the mechanics;
- everything else still applies: English only, docstrings that explain *why*, and honesty markers. A behaviour that was not measured says so in the comment. "expected from the LOM, not measured" is a complete and acceptable answer; a measurement that was never made is not.
