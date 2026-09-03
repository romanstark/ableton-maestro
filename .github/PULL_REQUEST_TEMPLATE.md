## What & why

Briefly: what does this change, and why?

## How verified

- [ ] `pytest` green
- [ ] `ruff check src tests` clean
- [ ] If it touches the catalog: probed against real Live, and the `status` set to what was actually observed (`verified` / `broken`), with the surprise — clamping, quantization, an unexpected unit — written into `doc:`

**If anything was measured against Live, say where:**

- Live version:
- OS:
- Remote Script reinstalled and Live restarted after the change: yes / no

## Checklist

- [ ] Kept the layers apart (transport / catalog+executor / music / server)
- [ ] New capability is a **catalog row**, not a new handler in the Remote Script (a new handler costs every user a Live restart — if this PR adds one, the reasoning is in the description and in the docs)
- [ ] Method calls stay on the script's allowlist
- [ ] Destructive operations still require `confirm=True`
- [ ] Honesty markers used: **measured** / **read from the source** / **estimated** / **unverified**. No behaviour asserted that was not observed
- [ ] Tool results distinguish "accepted" from "read back" from "audible"
- [ ] Outside code, if any, is named with its license (see [THIRD-PARTY.md](../THIRD-PARTY.md))
- [ ] First-time contributors: I agree to the [CLA](../CLA.md)
