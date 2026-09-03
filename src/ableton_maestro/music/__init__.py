"""Musical logic: notes, theory, humanisation. Knows nothing about sockets.

This package is the layer that may never touch the transport. Nothing here imports
``ableton_maestro.client`` or opens a connection; it takes plain data structures (MIDI
numbers, beats, chord labels) and returns plain data structures. That separation is an
architectural rule, not a preference: ``client.py`` knows sockets and no music,
``music/`` knows music and no sockets. A pitch in ``client.py`` is a layering bug.

The public surface of :mod:`ableton_maestro.music.theory` is re-exported flat, so callers
can write ``from ableton_maestro.music import voicing``. Anything not listed in
:data:`__all__` is an implementation detail of its module.

:mod:`~ableton_maestro.music.notes` and :mod:`~ableton_maestro.music.humanize` are
exported as modules rather than flattened::

    from ableton_maestro.music import notes, humanize
    notes.transpose(...)      humanize.to_ableton(...)

The two share four names: ``TICKS_PER_QUARTER``, ``VELOCITY_RANGE``, ``beats_to_ticks``
and ``ticks_to_beats``. Flattening both would let one silently shadow the other, and a
shadowed conversion between ticks and beats is the 480x error this package guards against
everywhere else. Keeping them namespaced forces the caller to say which one they mean.
"""

from __future__ import annotations

from ableton_maestro.music import humanize, notes
from ableton_maestro.music.theory import (
    CHORD_QUALITIES,
    FILTER_HZ_MAX,
    FILTER_HZ_MIN,
    FLAT_NAMES,
    MEASURED_005_BLOCKS,
    MEASURED_DEGREE_POOL,
    MEASURED_FILTER_POINTS,
    MEASURED_PROGRESSIONS,
    MEASURED_VOICINGS,
    MODES,
    NAME_TO_PC,
    NOTE_NAMES,
    PROGRESSION_ABSENT,
    VOICING_BASS_ABSOLUTE_LOW,
    VOICING_BASS_RANGE,
    VOICING_BOTTOM_COUNTS,
    VOICING_BOTTOM_GOOD,
    VOICING_DEFAULT_SHAPE,
    VOICING_INVERSION_PCT,
    VOICING_MAX_VOICES,
    VOICING_SHAPES,
    VOICING_TOP_DEGREE_PCT,
    VOICING_TOP_MAX_LEAP,
    VOICING_TOP_RANGE,
    MeasuredProgression,
    MeasuredVoicing,
    VoicingShape,
    bar_beats,
    build_chord,
    check_against_chords,
    check_in_key,
    check_sections,
    check_voicing,
    chord_symbol,
    degree_chord,
    fitting_shape,
    harmony_line,
    hz_to_norm,
    identify_chord,
    norm_to_hz,
    note_name,
    parse_degree,
    pc_name,
    pitch_classes,
    progression,
    progression_pool,
    quality_degrees,
    scale_pitches,
    self_check,
    shape_fits,
    shape_text,
    transpose_diatonic,
    voice_progression,
    voicing,
    voicing_facts,
    voicing_shape_of,
)

__all__ = [
    "CHORD_QUALITIES",
    "FILTER_HZ_MAX",
    "FILTER_HZ_MIN",
    "FLAT_NAMES",
    "MEASURED_005_BLOCKS",
    "MEASURED_DEGREE_POOL",
    "MEASURED_FILTER_POINTS",
    "MEASURED_PROGRESSIONS",
    "MEASURED_VOICINGS",
    "MODES",
    "NAME_TO_PC",
    "NOTE_NAMES",
    "PROGRESSION_ABSENT",
    "VOICING_BASS_ABSOLUTE_LOW",
    "VOICING_BASS_RANGE",
    "VOICING_BOTTOM_COUNTS",
    "VOICING_BOTTOM_GOOD",
    "VOICING_DEFAULT_SHAPE",
    "VOICING_INVERSION_PCT",
    "VOICING_MAX_VOICES",
    "VOICING_SHAPES",
    "VOICING_TOP_DEGREE_PCT",
    "VOICING_TOP_MAX_LEAP",
    "VOICING_TOP_RANGE",
    "MeasuredProgression",
    "MeasuredVoicing",
    "VoicingShape",
    "bar_beats",
    "build_chord",
    "check_against_chords",
    "check_in_key",
    "check_sections",
    "check_voicing",
    "chord_symbol",
    "degree_chord",
    "fitting_shape",
    "harmony_line",
    "humanize",  # submodule, namespaced on purpose (see module docstring)
    "hz_to_norm",
    "identify_chord",
    "norm_to_hz",
    "note_name",
    "notes",  # submodule, namespaced on purpose (see module docstring)
    "parse_degree",
    "pc_name",
    "pitch_classes",
    "progression",
    "progression_pool",
    "quality_degrees",
    "scale_pitches",
    "self_check",
    "shape_fits",
    "shape_text",
    "transpose_diatonic",
    "voice_progression",
    "voicing",
    "voicing_facts",
    "voicing_shape_of",
]
