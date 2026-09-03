---
name: Feature request / catalog entry about: Propose a feature, or a LOM path that should be in the catalog labels: enhancement
---

**What would you like?** The feature, or the Live Object Model path(s) it needs (e.g. `song.tracks[{track}].clip_slots[{slot}].clip.warp_mode`).

**Why** The musical or workflow motivation. What are you trying to do in Live?

**Is it a catalog row or a handler?** Most capabilities are a **catalog row** — a path, its access, its type and range. A new **handler** means new code inside Live's process, which costs every user a Live restart, so it needs a reason a path cannot cover (notes, automation envelopes, the browser and event listeners are the four that earned it; `enum_names` is a fifth, and protocol §5.12 calls it a probe rather than a capability for that reason). Say which you think this is; being wrong is fine.

**Have you probed it?** If you have run it against real Live, that turns this from a request into a measurement. Please include:

- Live version and OS:
- What `lom_get` returned (value and type):
- Whether `lom_set` held — the `before` / `after` / `clamped` fields:
- Anything surprising: silent clamping, a quantized parameter, a display unit that is not what the name suggests

**Notes** Anything else — a link to the LOM documentation for the property, a screenshot of what it looks like in Live's GUI, or the reason you expect it to be difficult.
