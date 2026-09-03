---
name: Bug report about: Something doesn't work as expected labels: bug
---

**Before anything else — the three questions that solve most reports**

Remote Scripts load only when Live starts, and Live silently prefers a compiled `__pycache__` over your source. Almost every "it stopped working" is one of these:

- [ ] The Remote Script was **copied to the User Library** after the last change (`<User Library>\Remote Scripts\AbletonMaestro\__init__.py` — the path from `Preferences\Library.cfg`, `<ProjectPath>`, not a guess)
- [ ] `__pycache__` next to it was **deleted**
- [ ] Live was **restarted completely** afterwards
- [ ] `AbletonMaestro` is selected under Preferences → Link, Tempo & MIDI → *Control Surface*

If you are not sure, do all four and try again — then report either way, because "the instructions were followed and it still failed" is useful information.

**What happened** A clear description of the bug.

**To reproduce** The tool call or command you ran (e.g. `lom_set("song.tracks[0].mixer_device.volume", 0.9)`), and the state of the set it ran against.

**Expected** What you expected to happen.

**What came back** The full response, including `status`, `code` and `message` — and for a write, the `before` / `after` / `clamped` fields. Please paste it rather than summarising it; the difference between "accepted", "read back" and "no change" is exactly what is being diagnosed.

**Environment**
- Ableton Live version (e.g. Live 12.4.3):
- OS (and version):
- Python version:
- ableton-maestro version / commit:
- Remote Script reinstalled + Live restarted after the last change: yes / no
- Is another Ableton Remote Script also installed and running (port 9877 is a common one)?

**Log** Anything relevant from Live's `Log.txt`, and from the server's own output.
