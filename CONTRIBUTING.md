# Contributing to Ableton Maestro

Thank you for your interest in contributing to Ableton Maestro. This project maps Ableton Live's Object Model (LOM) as data and provides an MCP server for AI-assisted music production.

Before getting started, please review our core architectural references:
- [docs/architecture.md](docs/architecture.md): System architecture and design goals.
- [docs/protocol.md](docs/protocol.md): Wire protocol specification for socket communication.
- [docs/catalog.md](docs/catalog.md): Catalog schema and path validation rules.
- [docs/limits.md](docs/limits.md): Known API boundaries and constraints.

---

## 1. Development Setup

### Prerequisites
- Python 3.11 or higher
- Ableton Live 11 or 12
- Git

### Initial Setup

```bash
git clone https://github.com/romanstark/ableton-maestro.git
cd ableton-maestro
python -m venv .venv
```

Activate the virtual environment:
- Windows (PowerShell): `.venv\Scripts\activate`
- macOS / Linux: `source .venv/bin/activate`

Install dependencies in editable mode:
```bash
pip install -e ".[dev]"
```

Run test suite and linter:
```bash
pytest
ruff check src scripts tests
```

The test suite does not require an active Ableton Live instance. Continuous Integration (CI) runs tests on Linux and Windows without access to a DAW.

---

## 2. Core Workflow: Verifying Catalog Rows

The catalog in `src/ableton_maestro/catalog/*.yaml` describes the fixed Live Object Model surface. Each entry represents an addressable path in Live.

Each catalog row has a `status` field, which defaults to `untested`:
- `verified`: Confirmed working against an active Ableton Live instance.
- `broken`: Exists on the object, but Live refuses the call or raises an error.
- `untested`: Defined in the catalog but not yet validated live.

Verifying catalog entries against running Live installations is one of the most helpful contributions you can make.

### Available Helper Scripts

| Script | Purpose |
|---|---|
| `scripts/probe_paths.py` | Tests specific catalog rows against an open Live set and updates their status. |
| `scripts/sync_catalog.py` | Compares Live's runtime objects against the catalog to find missing or obsolete paths. |

### How to Probe a Path

1. **Open a test set in Ableton Live.** Use a disposable project rather than an active production set.
2. Ensure the Remote Script is installed and selected in Live preferences (`python scripts/install_script.py`).
3. Run a dry run to inspect the target paths:
   ```bash
   python scripts/probe_paths.py --id clip.warping --track 0 --slot 0
   ```
4. Execute the live probe:
   ```bash
   python scripts/probe_paths.py --id clip.warping --track 0 --slot 0 --go
   ```
5. Update the row in `src/ableton_maestro/catalog/`:
   - If read and write succeed: set `status: verified` and document what was tested in `doc`.
   - If Live returns an error (`live_error`): check whether the test value was within valid range. If within range and refused by Live, set `status: broken` with the error explanation.
   - If the property does not exist (`no_such_path`): remove the row from the catalog and document the finding in [docs/limits.md](docs/limits.md).
   - If a target was missing (`index_out_of_range`): keep `status: untested` until tested with an appropriate set.

### Retracting a limit

A limit in [docs/limits.md](docs/limits.md) is a measurement, and it is retracted the same way it was made: ask Live again on a named version and write down the answer. Two rules apply.

Leave the entry standing and name the assumption. Somebody planned around that limit and needs to find out it was lifted, so `docs/limits.md` keeps an entry for device reordering with `Song.move_device` disproving it in place. Write about the claim, never about the page: a reader never saw the earlier version, and the revision history belongs in the commit message.

Never remove a measured fact. Correct the wrong part and leave the evidence around it standing, including the date and the Live version.

---

## 3. Pull Request Guidelines

1. Fork the repository and create a feature branch from `main`.
2. Ensure all tests pass (`pytest`) and linting is clean (`ruff check src scripts tests`).
3. In your pull request description, specify:
   - What changed and why.
   - The OS and Ableton Live version used for testing.
   - Any new or updated catalog paths.

---

## 4. Coding Conventions

- **Language:** English for all code, comments, docstrings, and documentation.
- **Python Code Style:** Python 3.11+, complete type annotations, 100 character line length, compliant with `ruff`.
- **Remote Script Restrictions:** `live-remote-script/` runs directly inside Ableton's Python interpreter. It must use only the standard library, avoid external dependencies, and avoid syntax unsupported by older Python interpreters.
- **Separation of Layers:**
  - `client.py`: Socket transport and wire serialisation.
  - `music/`: Pure Python musical theory, notes, and humanisation without socket dependencies.
  - `executor.py`: Connects catalog specifications to client dispatch.
  - `server.py`: MCP tool definitions and high-level endpoints.
- **Remote Script Changes:** Adding handlers inside Live requires a full restart of Ableton Live. Prefer expressing capabilities as catalog rows whenever possible.
- **Safety Checks:** Method calls must be explicitly listed in the Remote Script's allowlist. Destructive operations (such as deleting tracks or devices) require `confirm=True`.

---

## 5. Contributor License Agreement (CLA)

Ableton Maestro is dual-licensed (AGPL-3.0 for open source, with separate commercial licensing available). Contributors agree to a lightweight [CLA](CLA.md) before their first pull request is merged, ensuring you retain copyright while allowing dual-licensing maintenance.

### Third-Party Code

Please submit original work. If a patch incorporates code from other open-source projects, clearly identify the source and its license in the pull request description so appropriate notices can be preserved.

---

## 6. Commercial Inquiries

For commercial licensing options (e.g., proprietary embedding without AGPL requirements), please contact Roman Stark (mail@romanstark.de).
