# Third-Party Notices

Ableton Maestro is licensed under the **AGPL-3.0-or-later** license (see [LICENSE](LICENSE)). It interacts with and builds on the third-party components listed below.

---

## Runtime Dependencies

### Model Context Protocol Python SDK (`mcp`)
- **License:** MIT (permissive, compatible with AGPL-3.0)
- **Project:** https://github.com/modelcontextprotocol/python-sdk
- **Usage:** Provides the MCP server foundation (`mcp.server.fastmcp`), communicating over stdio. Installed as a dependency; not vendored.

### PyYAML (`pyyaml`)
- **License:** MIT (permissive, compatible with AGPL-3.0)
- **Project:** https://github.com/yaml/pyyaml
- **Usage:** Loads YAML path catalog files from `src/ableton_maestro/catalog/*.yaml` at runtime (see [docs/catalog.md](docs/catalog.md)). Installed as a dependency; not vendored.

---

## Development Dependencies

Development tools include **pytest** (MIT) and **ruff** (MIT) for testing and linting. These are used only in local development and CI environments.

---

## Ableton Live and Live Object Model

Ableton Live is a commercial product developed by **Ableton AG**. Ableton Maestro is an independent open-source project and is **not affiliated with, sponsored by, or endorsed by Ableton AG**.

- Trademarks such as "Ableton", "Ableton Live", "Live", "Max for Live", and "Push" are property of Ableton AG and are used here solely for descriptive purposes to indicate software compatibility.
- The Remote Script imports `_Framework.ControlSurface` and the `Live` module directly from your local Ableton Live installation at runtime. No proprietary Ableton code is redistributed in this repository.
- The Live Object Model definitions in the catalog represent interface descriptions and parameter names observed through runtime introspection.

Ableton Live itself is licensed separately by Ableton AG under their standard software license terms.
