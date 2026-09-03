"""Loader and lookup registry for catalog path specifications.

Loads YAML definitions from `src/ableton_maestro/catalog/*.yaml` into in-memory
`PathSpec` models and provides indexed lookups by ID, LOM path, status, and area.

Design principles:
- Split YAML files: Catalog entries are organized into separate files by LOM area
  (song, track, clip, device, browser) and merged in filename order.
- Unique ID validation: Enforces uniqueness of spec IDs across all catalog files.
- Bidirectional lookup: Supports resolving concrete runtime paths back to their
  matching catalog entries via path normalization.
"""

from __future__ import annotations

import re
from dataclasses import fields
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import yaml

from ableton_maestro.models import Access, Kind, PathStatus, Unit
from ableton_maestro.spec import ArgSpec, ParamSpec, PathSpec

DEFAULT_CATALOG_DIR = Path(__file__).parent / "catalog"

#: Allowed keys in a catalog YAML entry, derived from PathSpec fields.
ROW_KEYS: frozenset[str] = frozenset(f.name for f in fields(PathSpec))

#: Row keys converted to specific data types during catalog loading.
_CONVERTED = frozenset({"access", "kind", "unit", "status", "params", "args"})

#: Boolean flag keys in catalog rows.
_FLAGS = ("quantized", "destructive")

_E = TypeVar("_E", bound=Enum)

# ------------------------------------------------------------------ path lookup

_TEMPLATE_INDEX = re.compile(r"\[\{[^}]+\}\]")
_CONCRETE_INDEX = re.compile(r"\[\d+\]")
_TEMPLATE_SEGMENT = re.compile(r"\{[^}]+\}")


def normalise_path(path: str) -> str:
    """Normalize a concrete path by replacing numeric indices with wildcard asterisks.

    >>> normalise_path("song.tracks[5].clip_slots[2].clip.name")
    'song.tracks[*].clip_slots[*].clip.name'
    """
    return _CONCRETE_INDEX.sub("[*]", path)


def _normalise_template(template: str) -> str:
    """Normalize template path by replacing placeholders with asterisks.

    >>> _normalise_template("song.tracks[{track}].playing_slot_index")
    'song.tracks[*].playing_slot_index'
    >>> _normalise_template("app.browser.{root}.children[{index}]")
    'app.browser.*.children[*]'
    """
    return _TEMPLATE_SEGMENT.sub("*", _TEMPLATE_INDEX.sub("[*]", template))


def _segments_match(pattern: str, path: str) -> bool:
    """Compare normalized path segments where '*' matches any single segment.

    >>> _segments_match("app.browser.*.children[*]", "app.browser.sounds.children[*]")
    True
    >>> _segments_match("app.browser.*.children[*]", "app.browser.sounds.children[*].name")
    False
    """
    wanted = pattern.split(".")
    got = path.split(".")
    if len(wanted) != len(got):
        return False
    return all(w == "*" or w == g for w, g in zip(wanted, got, strict=True))


class CatalogError(ValueError):
    """Raised when a catalog file contains invalid YAML, duplicates, or errors."""


class Registry:
    """In-memory index of PathSpec entries keyed by ID and normalized LOM path."""

    def __init__(self, specs: list[PathSpec]) -> None:
        self._by_id: dict[str, PathSpec] = {}
        for spec in specs:
            if spec.id in self._by_id:
                raise CatalogError(f"duplicate catalog id {spec.id!r}")
            self._by_id[spec.id] = spec
        self._by_path: dict[str, tuple[PathSpec, ...]] | None = None
        self._by_segments: tuple[tuple[str, PathSpec], ...] | None = None

    # ------------------------------------------------------------- lookup by path
    def _build_path_index(self) -> None:
        """Build path-to-spec index for property rows (methods excluded)."""
        by_path: dict[str, list[PathSpec]] = {}
        by_segments: list[tuple[str, PathSpec]] = []
        for spec in self._by_id.values():
            if spec.method is not None:
                continue
            normalised = _normalise_template(spec.path)
            if "*" in normalised.split("."):
                by_segments.append((normalised, spec))
            else:
                by_path.setdefault(normalised, []).append(spec)
        self._by_path = {key: tuple(value) for key, value in by_path.items()}
        self._by_segments = tuple(by_segments)

    def rows_for_path(self, path: str) -> tuple[PathSpec, ...]:
        """Return all property PathSpecs matching the concrete path in catalog order.

        >>> registry = default_registry()
        >>> [s.id for s in registry.rows_for_path("song.tracks[5].playing_slot_index")]
        ['track.playing_slot_index']
        >>> [s.id for s in registry.rows_for_path("song.tracks[0].devices")]
        ['track.devices', 'device.list']

        >>> path = "song.tracks[0].devices[1].parameters[7].value"
        >>> [s.id for s in registry.rows_for_path(path)]
        ['param.value_raw', 'rack.macro_value', 'eq8.frequency_hint_value']

        >>> [s.id for s in registry.rows_for_path("app.browser.sounds.children[3].name")]
        ['browser.item_name']

        >>> registry.rows_for_path("song.tracks[0].devices[1].parameters[7].plugin_field")
        ()
        """
        if self._by_path is None or self._by_segments is None:
            self._build_path_index()
        assert self._by_path is not None and self._by_segments is not None
        normalised = normalise_path(path)
        found = self._by_path.get(normalised)
        if found is not None:
            return found
        return tuple(
            spec for pattern, spec in self._by_segments if _segments_match(pattern, normalised)
        )

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> Registry:
        """Load catalog entries from a directory of YAML files or a single file.

        Args:
            path: Directory of *.yaml files, or a single file. Defaults to the packaged catalog.

        Returns:
            A populated Registry instance.

        Raises:
            CatalogError: If catalog files cannot be found, parsed, or contain duplicate IDs.
        """
        root = Path(path) if path is not None else DEFAULT_CATALOG_DIR
        files = cls._catalog_files(root)

        specs: list[PathSpec] = []
        source_of: dict[str, Path] = {}
        for file in files:
            for entry in cls._rows(file):
                spec = cls._spec_from_dict(entry, file)
                first = source_of.get(spec.id)
                if first is not None:
                    raise CatalogError(
                        f"duplicate catalog id {spec.id!r}: defined in {first.name} "
                        f"and again in {file.name}"
                    )
                source_of[spec.id] = file
                specs.append(spec)
        return cls(specs)

    @staticmethod
    def _catalog_files(root: Path) -> list[Path]:
        """Return sorted list of catalog YAML files in root directory."""
        if root.is_file():
            return [root]
        if not root.is_dir():
            raise CatalogError(f"catalog not found: {root}")
        files = sorted(root.glob("*.yaml"), key=lambda p: p.name)
        if not files:
            raise CatalogError(f"no *.yaml catalog files in {root}")
        return files

    @staticmethod
    def _rows(file: Path) -> list[dict[str, Any]]:
        """Parse rows list from a single YAML catalog file."""
        try:
            raw = yaml.safe_load(file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise CatalogError(f"{file.name}: not valid YAML: {exc}") from exc
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise CatalogError(
                f"{file.name}: top level must be a list of catalog rows, got {type(raw).__name__}"
            )
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise CatalogError(f"{file.name}: row {i} is not a mapping: {entry!r}")
        return raw

    @classmethod
    def _spec_from_dict(cls, entry: dict[str, Any], file: Path) -> PathSpec:
        """Construct a PathSpec from a raw dictionary entry."""
        spec_id = entry.get("id")
        if not isinstance(spec_id, str) or not spec_id:
            raise CatalogError(f"{file.name}: a row is missing a string 'id': {entry!r}")

        unknown = sorted(k for k in entry if k not in ROW_KEYS)
        if unknown:
            raise CatalogError(
                f"{file.name}: {spec_id}: unknown catalog key(s) {unknown}; a row may use: "
                f"{sorted(ROW_KEYS)}. A misspelt key would otherwise be dropped in silence."
            )
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise CatalogError(f"{file.name}: {spec_id}: missing 'path'")
        for flag in _FLAGS:
            if flag in entry and not isinstance(entry[flag], bool):
                raise CatalogError(
                    f"{file.name}: {spec_id}: {flag}={entry[flag]!r} must be true or false"
                )

        kwargs: dict[str, Any] = {k: v for k, v in entry.items() if k not in _CONVERTED}
        if "access" in entry:
            if not isinstance(entry["access"], list):
                raise CatalogError(f"{file.name}: {spec_id}: 'access' must be a list")
            kwargs["access"] = [
                _enum(Access, a, field="access", spec_id=spec_id, file=file)
                for a in entry["access"]
            ]
        for field_name, enum_cls in (("kind", Kind), ("unit", Unit), ("status", PathStatus)):
            if field_name in entry:
                kwargs[field_name] = _enum(
                    enum_cls, entry[field_name], field=field_name, spec_id=spec_id, file=file
                )
        if "params" in entry:
            kwargs["params"] = cls._params_from_dict(entry, spec_id, file)
        if "args" in entry:
            kwargs["args"] = cls._args_from_dict(entry, spec_id, file)

        try:
            return PathSpec(**kwargs)
        except (TypeError, ValueError) as exc:
            text = str(exc)
            named = "" if text.startswith(f"{spec_id}:") else f"{spec_id}: "
            raise CatalogError(f"{file.name}: {named}{text}") from exc

    @staticmethod
    def _params_from_dict(entry: dict[str, Any], spec_id: str, file: Path) -> list[ParamSpec]:
        """Construct list of ParamSpecs from row's 'params' key."""
        raw = entry.get("params") or []
        if not isinstance(raw, list):
            raise CatalogError(f"{file.name}: {spec_id}: 'params' must be a list")
        params: list[ParamSpec] = []
        for pd in raw:
            if not isinstance(pd, dict):
                raise CatalogError(f"{file.name}: {spec_id}: param entry is not a mapping: {pd!r}")
            try:
                params.append(ParamSpec(**pd))
            except TypeError as exc:
                raise CatalogError(
                    f"{file.name}: {spec_id}: bad param entry {pd!r}: {exc}"
                ) from exc
        return params

    @staticmethod
    def _args_from_dict(entry: dict[str, Any], spec_id: str, file: Path) -> list[ArgSpec]:
        """Construct list of ArgSpecs from row's 'args' key."""
        raw = entry.get("args") or []
        if not isinstance(raw, list):
            raise CatalogError(f"{file.name}: {spec_id}: 'args' must be a list")
        args: list[ArgSpec] = []
        for ad in raw:
            if not isinstance(ad, dict):
                raise CatalogError(f"{file.name}: {spec_id}: arg entry is not a mapping: {ad!r}")
            try:
                args.append(ArgSpec(**ad))
            except TypeError as exc:
                raise CatalogError(f"{file.name}: {spec_id}: bad arg entry {ad!r}: {exc}") from exc
        return args

    # ------------------------------------------------------------------ lookups
    def get(self, spec_id: str) -> PathSpec:
        """Return PathSpec for spec_id, raising KeyError if not found."""
        return self._by_id[spec_id]

    def by_area(self, area: str) -> list[PathSpec]:
        """Return all PathSpecs in the specified area (e.g. 'track', 'clip', 'song')."""
        return [s for s in self._by_id.values() if area_of(s.id) == area]

    def by_status(self, status: PathStatus | str) -> list[PathSpec]:
        """Return all PathSpecs matching the specified PathStatus."""
        wanted = PathStatus(status)
        return [s for s in self._by_id.values() if s.status == wanted]

    def all(self) -> list[PathSpec]:
        """Return all catalog PathSpecs in definition order."""
        return list(self._by_id.values())

    def status_counts(self) -> dict[str, int]:
        """Return dictionary mapping each PathStatus value to its count in the catalog."""
        counts: dict[str, int] = {s.value: 0 for s in PathStatus}
        for spec in self._by_id.values():
            counts[spec.status.value] += 1
        return counts

    def search(self, query: str) -> list[PathSpec]:
        """Search specs matching query against ID, LOM path, or documentation."""
        needle = query.strip().lower()
        if not needle:
            return []
        return [
            s
            for s in self._by_id.values()
            if needle in s.id.lower() or needle in s.path.lower() or needle in s.doc.lower()
        ]


def area_of(spec_id: str) -> str:
    """Return the catalog area for spec_id (the segment before the first dot)."""
    return spec_id.split(".", 1)[0]


def _enum(cls: type[_E], raw: object, *, field: str, spec_id: str, file: Path) -> _E:
    """Convert raw catalog value to an Enum instance."""
    try:
        return cls(raw)
    except ValueError as exc:
        allowed = ", ".join(str(m.value) for m in cls)
        raise CatalogError(
            f"{file.name}: {spec_id}: {field}={raw!r} is not valid; expected one of: {allowed}"
        ) from exc


@lru_cache(maxsize=1)
def default_registry() -> Registry:
    """Return cached process-wide Registry instance loaded from default catalog files."""
    return Registry.load()
