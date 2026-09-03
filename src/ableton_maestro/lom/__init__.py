"""The LOM layer: the path grammar, and the dynamic surface behind it.

Two modules with two different characters.

:mod:`~ableton_maestro.lom.paths` is the fixed half: the grammar of
``docs/protocol.md`` §6, mirrored client-side. Pure string work, with no socket, no
client import and no knowledge of Live. It lets a malformed path fail in a process where
a traceback is free, rather than in the Live process, where an escaping ``IndexError``
leaves a dead socket.

:mod:`~ableton_maestro.lom.introspect` is the dynamic half: what a loaded device actually
exposes, which is knowable only at runtime and therefore cannot live in the catalog files
(docs/architecture.md, 'the catalog'). It may call the client through a structural
protocol rather than an import, and it carries the measured diagnoses that keep a
confusing answer from looking like an empty one, chiefly the plugin that reports nothing
but ``Device On``.
"""

from __future__ import annotations

from ableton_maestro.lom.introspect import (
    BATCH_OP_LIMIT,
    CONFIGURED_THRESHOLD,
    DEVICE_ON,
    MEASURED_UNIT_TOKENS,
    VST_PARAMETER_SLOTS,
    ClipView,
    DescribeCache,
    DeviceView,
    Diagnosis,
    IntrospectionError,
    LomClient,
    ParameterMatch,
    ParameterNotFoundError,
    ParameterView,
    SessionSnapshot,
    TrackView,
    configuration_advice,
    describe_device,
    diagnose,
    find_parameter,
    rank_parameters,
    require_parameter,
    snapshot,
    unit_of,
)
from ableton_maestro.lom.paths import (
    ROOTS,
    PathSyntaxError,
    Segment,
    build,
    clip,
    clip_slot,
    describe_path,
    device,
    is_valid,
    join,
    master,
    mixer,
    parameter,
    parent,
    parse,
    return_track,
    scene,
    send,
    track,
    unparse,
    validate,
)

__all__ = [
    "BATCH_OP_LIMIT",
    "CONFIGURED_THRESHOLD",
    "DEVICE_ON",
    "MEASURED_UNIT_TOKENS",
    "ROOTS",
    "VST_PARAMETER_SLOTS",
    "ClipView",
    "DescribeCache",
    "DeviceView",
    "Diagnosis",
    "IntrospectionError",
    "LomClient",
    "ParameterMatch",
    "ParameterNotFoundError",
    "ParameterView",
    "PathSyntaxError",
    "Segment",
    "SessionSnapshot",
    "TrackView",
    "build",
    "clip",
    "clip_slot",
    "configuration_advice",
    "describe_device",
    "describe_path",
    "device",
    "diagnose",
    "find_parameter",
    "is_valid",
    "join",
    "master",
    "mixer",
    "parameter",
    "parent",
    "parse",
    "rank_parameters",
    "require_parameter",
    "return_track",
    "scene",
    "send",
    "snapshot",
    "track",
    "unit_of",
    "unparse",
    "validate",
]
