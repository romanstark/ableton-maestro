"""Test Channel B, the ``.als`` reader and writer.

Covers :mod:`ableton_maestro.als.read` and :mod:`ableton_maestro.als.write`.

No Ableton, no real project file, no network. There is no corpus in this
repository and there never will be, because someone else's sets are not ours to
ship. Every fixture here is a constructed ``.als``: gzip-compressed XML shaped
like Live's own, small enough to read in one screen and complete enough to reach
the code paths that matter. Every write goes to ``tmp_path``.

The fixtures reproduce the measured list from ``als/read.py``'s module docstring
and ``docs/limits.md`` §2:

* ``.als`` is gzip, not ZIP.
* Automation lives in two places and only one of them is script-writable. The
  fixture :func:`track_only_als` is the case that matters, because a reader that
  counts clip envelopes alone reports 110 of 174 corpus projects as unautomated.
* The lead-in point at ``Time = -63072000`` is not part of the curve.
* Live writes every time point twice.
* ``FloatEvent`` / ``BoolEvent`` / ``EnumEvent`` all occur in track automation.
* Drum pad notes are stored mirrored, as ``128 - value``.
* The pitch hangs on the ``KeyTrack``, not on the note event.
* ``MainTrack`` is Live 12's name for ``MasterTrack``.
* ``session_clips`` also contains the Arrangement clips.

On the writing side the safety net is under test: a backup before the write
(proven by comparing the backup's bytes against the original's), a read-back
from disk afterwards, and a refusal when the set looks like Live has it open. The
process table is faked, exactly as the socket is faked elsewhere. A test must
never depend on whether the machine running it happens to have Live open.
"""

from __future__ import annotations

import gzip
import json
import os
import stat
import subprocess
import time
import types
from pathlib import Path

import pytest

from ableton_maestro.als import write
from ableton_maestro.als.read import (
    Envelope,
    Project,
    automation_of,
    format_report,
    load_xml,
    note_name,
    read_clip_notes,
    read_project,
    read_track,
    to_dict,
    unique_note_clips,
)
from ableton_maestro.als.write import (
    AlsRefused,
    AlsWriteError,
    list_sidechain_slots,
    restore_backup,
    set_attribute,
    set_sidechain_source,
)
from ableton_maestro.client import AbletonError

# ============================================================ fixture material
#
# Live's lead-in point. It is not part of the curve and usually carries the
# parameter's DEFAULT value, which is why the fixtures give it an absurd one:
# if it ever leaked into a curve, value_max would say 999.
LEAD_IN = "-63072000"
LEAD_IN_VALUE = "999"


def _time_signature(numerator: int = 4, denominator: int = 4) -> str:
    return f"""
        <TimeSignature><TimeSignatures>
          <RemoteableTimeSignature Id="0">
            <Numerator Value="{numerator}" />
            <Denominator Value="{denominator}" />
            <Time Value="0" />
          </RemoteableTimeSignature>
        </TimeSignatures></TimeSignature>"""


# One clip envelope. Two support points, each written twice, plus the lead-in.
CLIP_ENVELOPE = f"""
        <Envelopes>
          <ClipEnvelope Id="0">
            <EnvelopeTarget><PointeeId Value="20001" /></EnvelopeTarget>
            <Automation><Events>
              <FloatEvent Id="1" Time="{LEAD_IN}" Value="{LEAD_IN_VALUE}" />
              <FloatEvent Id="2" Time="0" Value="0.1" />
              <FloatEvent Id="3" Time="0" Value="0.1" />
              <FloatEvent Id="4" Time="4" Value="0.9" />
              <FloatEvent Id="5" Time="4" Value="0.9" />
            </Events></Automation>
          </ClipEnvelope>
        </Envelopes>"""

# The pitch is NOT on the event: it hangs on the surrounding KeyTrack. One note
# is muted (IsEnabled="false"), one starts before the clip's 1|1 (a pick-up).
NOTES = """
        <Notes><KeyTracks>
          <KeyTrack Id="0">
            <Notes>
              <MidiNoteEvent Time="-0.5" Duration="0.5" Velocity="100" OffVelocity="64"
                             IsEnabled="true" NoteId="1" />
              <MidiNoteEvent Time="1" Duration="0.5" Velocity="90" OffVelocity="64"
                             IsEnabled="false" NoteId="2" />
            </Notes>
            <MidiKey Value="60" />
          </KeyTrack>
          <KeyTrack Id="1">
            <Notes>
              <MidiNoteEvent Time="2" Duration="1" Velocity="80" OffVelocity="64"
                             IsEnabled="true" NoteId="3" />
            </Notes>
            <MidiKey Value="67" />
          </KeyTrack>
        </KeyTracks></Notes>"""


def _midi_clip(name: str, start: float, end: float, envelope: str = "") -> str:
    """Build one ``MidiClip``, shaped the way Live writes it."""
    return f"""
      <MidiClip Id="0" Time="{start}">
        <CurrentStart Value="{start}" />
        <CurrentEnd Value="{end}" />
        <Loop><LoopStart Value="0" /><LoopEnd Value="4" /></Loop>
        <Name Value="{name}" />
{_time_signature()}
{NOTES}
{envelope}
      </MidiClip>"""


# Three track-automation envelopes on one track: a float knob, a bool switch and
# a stepped enum. Each carries the lead-in point and writes its support points
# twice, which is what Live does.
TRACK_AUTOMATION = f"""
        <AutomationEnvelopes><Envelopes>
          <AutomationEnvelope Id="1">
            <EnvelopeTarget><PointeeId Value="10024" /></EnvelopeTarget>
            <Automation><Events>
              <FloatEvent Id="1" Time="{LEAD_IN}" Value="{LEAD_IN_VALUE}" />
              <FloatEvent Id="2" Time="0"  Value="200" />
              <FloatEvent Id="3" Time="0"  Value="200" />
              <FloatEvent Id="4" Time="4"  Value="1200" />
              <FloatEvent Id="5" Time="4"  Value="1200" />
              <FloatEvent Id="6" Time="8"  Value="400" />
              <FloatEvent Id="7" Time="8"  Value="400" />
              <FloatEvent Id="8" Time="12" Value="800" />
              <FloatEvent Id="9" Time="12" Value="800" />
            </Events></Automation>
          </AutomationEnvelope>
          <AutomationEnvelope Id="2">
            <EnvelopeTarget><PointeeId Value="10025" /></EnvelopeTarget>
            <Automation><Events>
              <BoolEvent Id="1" Time="{LEAD_IN}" Value="true" />
              <BoolEvent Id="2" Time="0" Value="true" />
              <BoolEvent Id="3" Time="8" Value="false" />
            </Events></Automation>
          </AutomationEnvelope>
          <AutomationEnvelope Id="3">
            <EnvelopeTarget><PointeeId Value="10026" /></EnvelopeTarget>
            <Automation><Events>
              <EnumEvent Id="1" Time="{LEAD_IN}" Value="0" />
              <EnumEvent Id="2" Time="0" Value="2" />
              <EnumEvent Id="3" Time="16" Value="5" />
            </Events></Automation>
          </AutomationEnvelope>
        </Envelopes></AutomationEnvelopes>"""

# An Auto Filter whose Cutoff and LfoOn carry the AutomationTarget ids the
# envelopes above point at. Id 10026 is deliberately absent, so that an
# unresolved PointeeId stays a bare id instead of being guessed at.
AUTO_FILTER = """
              <AutoFilter Id="11">
                <On><Manual Value="true" /></On>
                <Cutoff>
                  <AutomationTarget Id="10024"><LockEnvelope Value="0" /></AutomationTarget>
                  <Manual Value="800" />
                </Cutoff>
                <LfoOn>
                  <AutomationTarget Id="10025" />
                  <Manual Value="false" />
                </LfoOn>
              </AutoFilter>"""

# A drum rack in the .als spelling (``Branches``/``DrumBranch``) and one in the
# .adg spelling (``BranchPresets``/``DrumBranchPreset``). Both must yield pads:
# searching for only one of the two names reports a full kit as empty.
DRUM_RACK = """
              <DrumGroupDevice Id="12">
                <On><Manual Value="true" /></On>
                <Branches>
                  <DrumBranch Id="0">
                    <Name><EffectiveName Value="Kick" /></Name>
                    <ZoneSettings><ReceivingNote Value="92" /></ZoneSettings>
                    <DeviceChain><Devices>
                      <OriginalSimpler Id="13">
                        <On><Manual Value="true" /></On>
                        <Player><SampleRef><FileRef>
                          <Path Value="C:\\Samples\\Kick Analog 4.aif" />
                        </FileRef></SampleRef></Player>
                      </OriginalSimpler>
                    </Devices></DeviceChain>
                  </DrumBranch>
                  <DrumBranch Id="1">
                    <Name><EffectiveName Value="Snare" /></Name>
                    <ZoneSettings><ReceivingNote Value="90" /></ZoneSettings>
                    <DeviceChain><Devices /></DeviceChain>
                  </DrumBranch>
                </Branches>
              </DrumGroupDevice>
              <DrumGroupDevice Id="14">
                <UserName Value="Preset Kit" />
                <On><Manual Value="true" /></On>
                <BranchPresets>
                  <DrumBranchPreset Id="0">
                    <Name><EffectiveName Value="Clap" /></Name>
                    <ZoneSettings><ReceivingNote Value="89" /></ZoneSettings>
                  </DrumBranchPreset>
                </BranchPresets>
              </DrumGroupDevice>"""


def _wrap(tracks: str) -> str:
    """Wrap track XML in the ``Ableton``/``LiveSet`` prologue Live writes."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Ableton MajorVersion="5" MinorVersion="12.0_12203" SchemaChangeCount="3"
         Creator="Ableton Live 12.4.5" Revision="fixture">
  <LiveSet>
    <Tracks>
{tracks}
    </Tracks>
  </LiveSet>
</Ableton>
"""


# ``MainTrack`` is Live 12's name for ``MasterTrack``. Without it every Live 12
# project loses its master track, its device chain and the project meter: 49 of
# the 174 corpus projects.
MAIN_TRACK = f"""
      <MainTrack Id="20">
        <Name><EffectiveName Value="Main" /></Name>
        <DeviceChain>
          <Mixer>
            <Tempo><Manual Value="126" /></Tempo>
            <Volume><Manual Value="1" /></Volume>
          </Mixer>
          <DeviceChain><Devices /></DeviceChain>
        </DeviceChain>
{_time_signature()}
      </MainTrack>"""

LEAD_TRACK = f"""
      <MidiTrack Id="8">
        <Name><EffectiveName Value="Lead" /><UserName Value="" /></Name>
        <TrackGroupId Value="-1" />
{TRACK_AUTOMATION}
        <DeviceChain>
          <Mixer>
            <Volume><Manual Value="0.8" /></Volume>
            <Pan><Manual Value="-0.25" /></Pan>
            <Speaker><Manual Value="true" /></Speaker>
            <Sends>
              <TrackSendHolder Id="0"><Send><Manual Value="0.3" /></Send></TrackSendHolder>
              <TrackSendHolder Id="1"><Send><Manual Value="0" /></Send></TrackSendHolder>
            </Sends>
          </Mixer>
          <MainSequencer>
            <ClipTimeable><ArrangerAutomation><Events>
{_midi_clip("Lead Arr", 16, 24, CLIP_ENVELOPE)}
            </Events></ArrangerAutomation></ClipTimeable>
            <ClipSlotList>
              <ClipSlot Id="0"><ClipSlot><Value>
{_midi_clip("Lead A", 0, 4, CLIP_ENVELOPE)}
              </Value></ClipSlot></ClipSlot>
            </ClipSlotList>
          </MainSequencer>
          <DeviceChain>
            <Devices>
{AUTO_FILTER}
            </Devices>
          </DeviceChain>
        </DeviceChain>
      </MidiTrack>"""

DRUM_TRACK = f"""
      <AudioTrack Id="9">
        <Name><EffectiveName Value="Drums" /></Name>
        <TrackGroupId Value="-1" />
        <DeviceChain>
          <Mixer><Volume><Manual Value="1" /></Volume><Pan><Manual Value="0" /></Pan></Mixer>
          <DeviceChain>
            <Devices>
{DRUM_RACK}
            </Devices>
          </DeviceChain>
        </DeviceChain>
      </AudioTrack>"""

PROJECT_XML = _wrap(LEAD_TRACK + DRUM_TRACK + MAIN_TRACK)

# The fixture the whole layer exists for: track automation and nothing else.
# No ClipEnvelope anywhere. A tool that counts only clip envelopes calls this
# project unautomated, and would have said that about 110 of 174 real ones.
TRACK_AUTOMATION_ONLY_XML = _wrap(
    f"""
      <MidiTrack Id="8">
        <Name><EffectiveName Value="Lead" /></Name>
        <TrackGroupId Value="-1" />
{TRACK_AUTOMATION}
        <DeviceChain>
          <Mixer><Volume><Manual Value="1" /></Volume><Pan><Manual Value="0" /></Pan></Mixer>
          <MainSequencer>
            <ClipSlotList>
              <ClipSlot Id="0"><ClipSlot><Value>
{_midi_clip("Lead A", 0, 4)}
              </Value></ClipSlot></ClipSlot>
            </ClipSlotList>
          </MainSequencer>
          <DeviceChain><Devices>
{AUTO_FILTER}
          </Devices></DeviceChain>
        </DeviceChain>
      </MidiTrack>"""
    + MAIN_TRACK
)

# Two tracks and one compressor whose sidechain is wired to nothing. `Id` is the
# XML attribute, never the position in the track list.
SIDECHAIN_XML = _wrap(
    """
      <AudioTrack Id="10">
        <Name><EffectiveName Value="Bass" /><UserName Value="" /></Name>
        <DeviceChain>
          <Mixer>
            <Volume><Manual Value="1" /></Volume>
            <Speaker><Manual Value="true" /></Speaker>
          </Mixer>
          <DeviceChain>
            <Devices>
              <Compressor2 Id="3">
                <On><Manual Value="true" /></On>
                <Threshold><Manual Value="1" /></Threshold>
                <SideChain>
                  <OnOff><Manual Value="false" /></OnOff>
                  <RoutedInput>
                    <Routable>
                      <Target Value="AudioIn/None" />
                      <UpperDisplayString Value="Kick 1" />
                      <LowerDisplayString Value="" />
                    </Routable>
                    <Volume><Manual Value="1" /></Volume>
                  </RoutedInput>
                  <DryWet><Manual Value="1" /></DryWet>
                </SideChain>
              </Compressor2>
            </Devices>
          </DeviceChain>
        </DeviceChain>
      </AudioTrack>
      <AudioTrack Id="15">
        <Name><EffectiveName Value="SC" /></Name>
        <DeviceChain>
          <Mixer>
            <Volume><Manual Value="1" /></Volume>
            <Speaker><Manual Value="false" /></Speaker>
          </Mixer>
          <DeviceChain><Devices /></DeviceChain>
        </DeviceChain>
      </AudioTrack>"""
)

# Three sidechain slots that between them cover the three ways the file misleads:
# the Live 12 layout, the Live 9 layout one level deeper, and a display name that
# outlived its routing.
SIDECHAIN_VARIANTS_XML = _wrap(
    """
      <AudioTrack Id="1">
        <Name><EffectiveName Value="Flat" /></Name>
        <DeviceChain>
          <Mixer><Volume><Manual Value="1" /></Volume></Mixer>
          <DeviceChain><Devices>
            <Compressor2 Id="2">
              <On><Manual Value="true" /></On>
              <SideChain>
                <OnOff><Manual Value="true" /></OnOff>
                <RoutedInput>
                  <Target Value="AudioIn/Track.7/PostFxOut" />
                  <UpperDisplayString Value="SC" />
                  <LowerDisplayString Value="Post FX" />
                  <Volume><Manual Value="1" /></Volume>
                </RoutedInput>
                <DryWet><Manual Value="1" /></DryWet>
              </SideChain>
            </Compressor2>
          </Devices></DeviceChain>
        </DeviceChain>
      </AudioTrack>
      <AudioTrack Id="3">
        <Name><EffectiveName Value="Nested" /></Name>
        <DeviceChain>
          <Mixer><Volume><Manual Value="1" /></Volume></Mixer>
          <DeviceChain><Devices>
            <AudioEffectGroupDevice Id="4">
              <On><Manual Value="true" /></On>
              <UserName Value="Glue Rack" />
              <Branches>
                <AudioEffectBranch Id="0">
                  <Name><EffectiveName Value="Chain A" /></Name>
                  <DeviceChain><AudioToAudioDeviceChain><Devices>
                    <Compressor2 Id="5">
                      <On><Manual Value="true" /></On>
                      <SideChain>
                        <OnOff><Manual Value="true" /></OnOff>
                        <RoutedInput>
                          <Routable>
                            <Target Value="AudioIn/Track.7/PreFxOut" />
                            <UpperDisplayString Value="SC" />
                            <LowerDisplayString Value="Pre FX" />
                          </Routable>
                          <Volume><Manual Value="1" /></Volume>
                        </RoutedInput>
                      </SideChain>
                    </Compressor2>
                  </Devices></AudioToAudioDeviceChain></DeviceChain>
                </AudioEffectBranch>
              </Branches>
            </AudioEffectGroupDevice>
          </Devices></DeviceChain>
        </DeviceChain>
      </AudioTrack>
      <AudioTrack Id="6">
        <Name><EffectiveName Value="Liar" /></Name>
        <DeviceChain>
          <Mixer><Volume><Manual Value="1" /></Volume></Mixer>
          <DeviceChain><Devices>
            <Compressor2 Id="8">
              <On><Manual Value="true" /></On>
              <SideChain>
                <OnOff><Manual Value="true" /></OnOff>
                <RoutedInput>
                  <Routable>
                    <Target Value="AudioIn/None" />
                    <UpperDisplayString Value="Kick 1" />
                  </Routable>
                </RoutedInput>
              </SideChain>
            </Compressor2>
          </Devices></DeviceChain>
        </DeviceChain>
      </AudioTrack>"""
)

# A rack file (.adg): no tracks at all, and its devices are NOT under `Device`
# but under BranchPresets/AudioEffectBranchPreset/DevicePresets/...
RACK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Ableton MajorVersion="5" MinorVersion="12.0_12203" Creator="Ableton Live 12.4.5" Revision="fx">
  <GroupDevicePreset>
    <Device>
      <AudioEffectGroupDevice Id="0">
        <UserName Value="Vocal Chain" />
        <On><Manual Value="true" /></On>
        <MacroDisplayNames.0 Value="Air" />
        <MacroControls.0><Manual Value="64" /></MacroControls.0>
        <MacroDefaults.0 Value="0" />
        <MacroAnnotations.0 Value="high shelf" />
        <MacroDisplayNames.1 Value="Macro 2" />
        <MacroControls.1><Manual Value="0" /></MacroControls.1>
      </AudioEffectGroupDevice>
    </Device>
    <BranchPresets>
      <AudioEffectBranchPreset Id="0">
        <Name Value="Main" />
        <DevicePresets>
          <AbletonDevicePreset Id="0">
            <Device>
              <Eq8 Id="1">
                <UserName Value="Top End" />
                <On><Manual Value="true" /></On>
              </Eq8>
            </Device>
          </AbletonDevicePreset>
        </DevicePresets>
      </AudioEffectBranchPreset>
    </BranchPresets>
  </GroupDevicePreset>
</Ableton>
"""


# ==================================================================== fixtures


def _als(path: Path, xml: str) -> Path:
    """Write ``xml`` as a gzip-compressed ``.als`` and return the path."""
    path.write_bytes(gzip.compress(xml.encode("utf-8")))
    return path


def age(path: Path, seconds: float = 3600.0) -> None:
    """Backdate the file's mtime.

    ``als/write.py`` refuses a set written seconds ago, on the reasoning that the
    most likely author of a very fresh save is Live itself. A fixture written by
    the test is always that fresh, so anything that means to reach the write has
    to age the file first, and one test asserts the refusal instead.
    """
    when = time.time() - seconds
    os.utime(path, (when, when))


@pytest.fixture
def project_als(tmp_path: Path) -> Path:
    return _als(tmp_path / "Fixture.als", PROJECT_XML)


@pytest.fixture
def project(project_als: Path) -> Project:
    return read_project(project_als, with_notes=True)


@pytest.fixture
def track_only_als(tmp_path: Path) -> Path:
    return _als(tmp_path / "TrackAutomationOnly.als", TRACK_AUTOMATION_ONLY_XML)


@pytest.fixture
def sidechain_als(tmp_path: Path) -> Path:
    return _als(tmp_path / "Sidechain.als", SIDECHAIN_XML)


@pytest.fixture
def variants_als(tmp_path: Path) -> Path:
    return _als(tmp_path / "Variants.als", SIDECHAIN_VARIANTS_XML)


def _fake_process_table(monkeypatch: pytest.MonkeyPatch, listing: str) -> None:
    """Answer ``als/write.py``'s process-table question with a canned listing.

    The module shells out to ``tasklist``/``ps`` to ask whether a process called
    "Ableton Live" is alive. Whether one *is* alive on the machine running the
    tests must never decide the outcome, so the whole ``subprocess`` reference
    inside the module is swapped for a shim. This is the Channel B equivalent of
    faking the socket.
    """
    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=listing, stderr="")

    monkeypatch.setattr(
        write,
        "subprocess",
        types.SimpleNamespace(run=run, SubprocessError=subprocess.SubprocessError),
    )


@pytest.fixture
def no_live(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_process_table(monkeypatch, '"chrome.exe","1","Console","1","10 K"\n')


@pytest.fixture
def live_running(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_process_table(
        monkeypatch, '"Ableton Live 12 Suite.exe","4711","Console","1","2 000 K"\n'
    )


# ================================================================ gzip and XML


def test_an_als_is_gzip_not_zip(project_als: Path) -> None:
    """The single most common way to fail at reading an Ableton project."""
    assert project_als.read_bytes()[:2] == b"\x1f\x8b"
    root = load_xml(project_als)
    assert root.tag == "Ableton"
    assert root.get("Creator") == "Ableton Live 12.4.5"


def test_the_gzip_round_trip_survives_read_project(project: Project) -> None:
    assert project.file == "Fixture.als"
    assert project.live_version == "12.0_12203"
    assert project.creator == "Ableton Live 12.4.5"
    assert project.tempo == 126.0
    assert project.is_rack is False


def test_reading_something_that_is_not_gzip_fails_loudly(tmp_path: Path) -> None:
    plain = tmp_path / "NotReally.als"
    plain.write_text("<Ableton/>", encoding="utf-8")
    with pytest.raises(gzip.BadGzipFile):
        load_xml(plain)
    with pytest.raises(AlsWriteError, match="not a readable gzip file"):
        list_sidechain_slots(plain)


def test_gzip_that_is_not_xml_fails_loudly(tmp_path: Path) -> None:
    broken = tmp_path / "Broken.als"
    broken.write_bytes(gzip.compress(b"this is not xml"))
    with pytest.raises(AlsWriteError, match="not parseable XML"):
        list_sidechain_slots(broken)


# ====================================================================== tracks


def test_main_track_is_read_as_a_track(project: Project) -> None:
    """``MainTrack`` is Live 12's spelling of ``MasterTrack``.

    Without that name a Live 12 set loses its master track and its device chain.
    That is 49 of the 174 corpus projects.
    """
    assert [track.name for track in project.tracks] == ["Lead", "Drums", "Main"]
    assert project.track("Main").type == "MainTrack"
    assert project.track(2) is project.track("Main")


def test_a_missing_track_name_raises(project: Project) -> None:
    with pytest.raises(KeyError, match="no track named"):
        project.track("Nope")


def test_mixer_values_are_the_files_own_units(project: Project) -> None:
    """Volume in the file is a linear factor (1.0 = 0 dB).

    Over the remote script the same fader arrives normalised (0.85 = 0 dB).
    Confusing the two reads six tracks at 0 dB as six tracks at +6 dB. The reader
    passes the file's number through untouched and never converts.
    """
    lead = project.track("Lead")
    assert lead.volume == 0.8
    assert lead.panning == -0.25
    assert lead.sends == [0.3, 0.0]
    assert project.track("Main").volume == 1.0


def test_devices_are_found_including_the_ones_nested_in_racks(project: Project) -> None:
    """A reader that walks only the top chain misses where the craft sits."""
    names = [device.name for device in project.track("Drums").devices]
    assert "DrumGroupDevice" in names
    assert "Kick > OriginalSimpler" in names, names
    nested = next(d for d in project.track("Drums").devices if d.name.startswith("Kick >"))
    assert nested.depth == 1


# ================================================== the two automation layers


def test_the_two_layers_are_reported_separately(project: Project) -> None:
    """Clip envelopes and track automation are never merged or added up."""
    report = automation_of(project)
    assert {ref.envelope.layer for ref in report.clip_envelopes} == {"clip"}
    assert {ref.envelope.layer for ref in report.track_automation} == {"track"}
    # A clip envelope knows its clip. Track automation belongs to the timeline.
    assert all(ref.clip for ref in report.clip_envelopes)
    assert all(ref.clip is None for ref in report.track_automation)
    assert all(ref.track == "Lead" for ref in report.track_automation)


def test_track_automation_alone_is_not_unautomated(track_only_als: Path) -> None:
    """The measured failure this whole layer exists for.

    Over 174 professional projects, 52 (30 %) have clip envelopes but 159 (91 %)
    have track automation. A reader that counts only clip envelopes reports
    110 of them as unautomated. This fixture is one of those 110: no
    ``ClipEnvelope`` anywhere, three ``AutomationEnvelope``s on the track.
    """
    report = automation_of(read_project(track_only_als))

    assert report.clip_envelopes == []
    assert report.has_clip_envelopes is False
    assert len(report.track_automation) == 3
    assert report.has_track_automation is True

    # The assertion that matters: not unautomated.
    assert report.automated is True


def test_clip_envelopes_alone_also_count_as_automated(project: Project) -> None:
    """The property is an OR over both layers, in both directions."""
    report = automation_of(project)
    assert report.has_clip_envelopes and report.has_track_automation
    assert report.automated is True


def test_a_project_with_neither_layer_is_unautomated(tmp_path: Path) -> None:
    bare = _als(tmp_path / "Bare.als", _wrap(MAIN_TRACK))
    report = automation_of(read_project(bare))
    assert report.clip_envelopes == []
    assert report.track_automation == []
    assert report.automated is False


def test_an_arrangement_clips_envelope_is_reported_once(project: Project) -> None:
    """``session_clips`` already contains the Arrangement clips.

    Walking ``session_clips + arrangement_clips`` therefore lists every
    Arrangement clip's envelope twice: the same double count
    :func:`unique_note_clips` exists to prevent. The fixture has exactly two
    clips, one envelope each.
    """
    report = automation_of(project)
    assert len(report.clip_envelopes) == 2
    assert sorted(ref.clip or "" for ref in report.clip_envelopes) == ["Lead A", "Lead Arr"]


# ========================================================= envelope arithmetic


def _float_envelope(project: Project) -> Envelope:
    return next(
        env for env in project.track("Lead").track_automation if env.event_kind == "float"
    )


def test_the_lead_in_point_is_not_part_of_the_curve(project: Project) -> None:
    """``Time = -63072000`` is Live's lead-in and carries the default value.

    Left in, it falsifies both the time span and the value range.
    """
    envelope = _float_envelope(project)
    assert envelope.time_from == 0.0
    assert envelope.time_to == 12.0
    assert envelope.value_max != float(LEAD_IN_VALUE)
    assert envelope.value_min >= 0.0


def test_duplicated_time_points_are_counted_once(project: Project) -> None:
    """Live writes every time point twice: 9 events here are 4 support points.

    ``events`` keeps the raw count so the doubling stays visible rather than
    being quietly normalised away.
    """
    envelope = _float_envelope(project)
    assert envelope.events == 9
    assert envelope.points == 4


def test_the_value_range_ignores_the_edge_points(project: Project) -> None:
    """The value range comes from inside the curve.

    First and last point often carry the default.
    """
    envelope = _float_envelope(project)
    assert (envelope.value_min, envelope.value_max) == (400.0, 1200.0)
    assert envelope.largest_gap_beats == 4.0
    assert envelope.span_beats == 12.0
    assert envelope.gap_share == pytest.approx(4.0 / 12.0)


def test_all_three_event_kinds_parse(project: Project) -> None:
    """Track automation carries floats, bools and enums.

    Clip envelopes carry floats.
    """
    kinds = {env.event_kind: env for env in project.track("Lead").track_automation}
    assert set(kinds) == {"float", "bool", "enum"}

    switch = kinds["bool"]
    assert switch.points == 2
    # "true"/"false" become 1.0/0.0 so a bool curve has a comparable range.
    assert (switch.value_min, switch.value_max) == (0.0, 1.0)

    stepped = kinds["enum"]
    assert stepped.points == 2
    assert (stepped.value_min, stepped.value_max) == (2.0, 5.0)


def test_a_pointee_id_is_resolved_where_it_can_be_and_left_alone_where_it_cannot(
    project: Project,
) -> None:
    """``PointeeId`` is a set-internal id, not a parameter index.

    It is matched against the track's ``AutomationTarget`` ids. An id that
    matches nothing stays a bare id rather than being guessed at, and the id is
    reported either way.
    """
    by_id = {env.pointee_id: env for env in project.track("Lead").track_automation}
    assert by_id["10024"].target == "AutoFilter / Cutoff"
    assert by_id["10025"].target == "AutoFilter / LfoOn"
    assert by_id["10026"].target is None
    assert all(env.pointee_id is not None for env in by_id.values())


def test_target_resolution_can_be_switched_off(track_only_als: Path) -> None:
    """``resolve_targets=False`` skips the per-track parent walk.

    Corpus sweeps use it.
    """
    unresolved = read_project(track_only_als, resolve_targets=False)
    envelopes = unresolved.track("Lead").track_automation
    assert [env.target for env in envelopes] == [None, None, None]
    assert [env.pointee_id for env in envelopes] == ["10024", "10025", "10026"]


# =================================================================== drum pads


def test_drum_pad_notes_are_stored_mirrored(project: Project) -> None:
    """Every branch stores ``128 - note``.

    Read raw, the kick lands where no kit plays.
    """
    rack = next(d for d in project.track("Drums").devices if d.tag == "DrumGroupDevice")
    assert [(pad.receiving_note, pad.note) for pad in rack.pads] == [(92, 36), (90, 38)]
    assert note_name(36) == "C2"
    assert [pad.name for pad in rack.pads] == ["Kick", "Snare"]
    # Pads come back sorted by the decoded note, not by the mirrored file value.
    assert [pad.note for pad in rack.pads] == sorted(pad.note for pad in rack.pads)


def test_a_drum_pads_sample_name_is_recovered(project: Project) -> None:
    rack = next(d for d in project.track("Drums").devices if d.tag == "DrumGroupDevice")
    assert rack.pads[0].sample == "Kick Analog 4.aif"
    assert rack.pads[1].sample is None


def test_both_drum_branch_spellings_are_found(project: Project) -> None:
    """``DrumBranch`` in an ``.als``, ``DrumBranchPreset`` in an ``.adg``.

    Searching for only one of the two finds zero pads in the other file type and
    reports a full kit as empty.
    """
    preset_rack = next(d for d in project.track("Drums").devices if d.name == "Preset Kit")
    assert [(pad.receiving_note, pad.note) for pad in preset_rack.pads] == [(89, 39)]


def test_a_device_that_is_not_a_rack_has_no_pads(project: Project) -> None:
    assert project.track("Lead").devices[0].pads == []


# ======================================================================= notes


def test_the_pitch_comes_from_the_key_track_not_the_event(project: Project) -> None:
    """``MidiNoteEvent`` carries no pitch: it hangs on the surrounding ``KeyTrack``."""
    notes = read_clip_notes(project.path, "Lead", 1)
    assert [(note.time, note.pitch) for note in notes] == [(-0.5, 60), (2.0, 67)]


def test_note_times_are_clip_local_and_may_be_negative(project: Project) -> None:
    """A note before the clip's 1|1 is a pick-up, not corruption."""
    clip = project.track("Lead").session_clips[1]
    assert clip.name == "Lead A"
    assert min(note.time for note in clip.notes or []) == -0.5


def test_muted_notes_are_counted_but_not_returned(project: Project) -> None:
    """``note_count`` counts every event; the note list leaves the muted ones out."""
    clip = project.track("Lead").session_clips[1]
    assert clip.note_count == 3
    assert len(clip.notes or []) == 2
    assert clip.note_stats is not None
    assert clip.note_stats.notes == 2
    assert clip.note_stats.notes_disabled == 1


def test_no_notes_are_read_without_the_flag(project_als: Path) -> None:
    """Note tracking memory overhead test for note sweep scale validation."""
    survey = read_project(project_als)
    clip = survey.track("Lead").session_clips[0]
    assert clip.notes is None
    assert clip.note_stats is None
    assert clip.fingerprint is None
    assert clip.in_arranger is None
    assert survey.beats_per_bar is None
    # The pitch-class condensation is cheap and is read either way.
    assert clip.note_count == 3
    assert clip.pitch_classes == {0: 2, 7: 1}


def test_session_clips_also_contains_the_arrangement_clips(project: Project) -> None:
    """The documented trap: ``arrangement=False`` walks the whole track subtree."""
    lead = project.track("Lead")
    assert [clip.name for clip in lead.session_clips] == ["Lead Arr", "Lead A"]
    assert [clip.name for clip in lead.arrangement_clips] == ["Lead Arr"]
    assert [clip.in_arranger for clip in lead.session_clips] == [True, False]
    assert [clip.where for clip in lead.session_clips] == ["Arr", "Ses"]
    assert all(clip.in_freeze is False for clip in lead.session_clips)


def test_identical_note_content_shares_a_fingerprint(project: Project) -> None:
    """Arrangement repetitions and layered twin tracks collapse into one."""
    lead = project.track("Lead")
    assert lead.session_clips[0].fingerprint == lead.session_clips[1].fingerprint
    rows = unique_note_clips(project)
    # Two clips, same notes, different start beats: both kept, because the key
    # includes the position; the fingerprint is what says they are the same music.
    assert [name for name, _clip in rows] == ["Lead", "Lead"]
    assert len({clip.fingerprint for _name, clip in rows}) == 1


def test_read_track_and_read_clip_notes_take_a_name_or_an_index(project_als: Path) -> None:
    by_name = read_track(project_als, "Lead")
    by_index = read_track(project_als, 0)
    assert by_name.name == by_index.name == "Lead"
    session = read_clip_notes(project_als, "Lead", 0)
    arrangement = read_clip_notes(project_als, "Lead", 0, arrangement=True)
    assert [note.pitch for note in session] == [60, 67]
    assert [note.pitch for note in arrangement] == [60, 67]


def test_beats_per_bar_follows_the_meter_not_the_denominator(tmp_path: Path) -> None:
    """Live counts note times in quarter notes: 6/8 is 3.0 beats per bar, not 6."""
    six_eight = _wrap(
        f"""
      <MainTrack Id="20">
        <Name><EffectiveName Value="Main" /></Name>
        <DeviceChain>
          <Mixer><Tempo><Manual Value="90" /></Tempo></Mixer>
          <DeviceChain><Devices /></DeviceChain>
        </DeviceChain>
{_time_signature(6, 8)}
      </MainTrack>"""
    )
    project = read_project(_als(tmp_path / "SixEight.als", six_eight), with_notes=True)
    assert project.beats_per_bar == 3.0


# ============================================================ sidechain layout


def test_the_target_is_found_in_both_live_layouts(variants_als: Path) -> None:
    """Live 12 puts ``Target`` under ``RoutedInput``, Live 9 under ``Routable``.

    A hard-coded path makes a fully wired older project report "no sidechain", so
    the search is relative and both layouts have to resolve.
    """
    project = read_project(variants_als)
    flat = project.track("Flat").devices[0].sidechain
    assert flat is not None
    assert flat.target == "AudioIn/Track.7/PostFxOut"
    assert flat.configured is True

    nested = next(
        device
        for device in project.track("Nested").devices
        if device.sidechain is not None and device.sidechain.target
    )
    assert nested.name == "Chain A > Compressor2"
    assert nested.sidechain is not None
    assert nested.sidechain.target == "AudioIn/Track.7/PreFxOut"
    assert nested.sidechain.configured is True


def test_the_display_name_lies_and_only_the_target_counts(variants_als: Path) -> None:
    """Verify ``UpperDisplayString`` retains track name after routing removal.

    Evaluating the display name reported 14 sidechains where there were 4.
    """
    liar = read_project(variants_als).track("Liar").devices[0].sidechain
    assert liar is not None
    assert liar.on is True
    assert liar.source == "Kick 1"
    assert liar.target == "AudioIn/None"
    assert liar.configured is False


def test_a_wired_sidechain_shows_up_in_the_report(variants_als: Path) -> None:
    text = format_report(read_project(variants_als))
    assert "SIDECHAIN" in text
    assert "Flat" in text
    assert "TRACK AUTOMATION" in text
    assert "CLIP ENVELOPES" in text


def test_the_report_names_both_layers_even_when_one_is_empty(track_only_als: Path) -> None:
    text = format_report(read_project(track_only_als))
    assert "TRACK AUTOMATION in the arrangement (3)" in text
    assert "CLIP ENVELOPES (0)" in text


# ======================================================================= racks


def test_a_rack_file_has_no_tracks_and_is_reported_as_one(tmp_path: Path) -> None:
    """An ``.adg`` keeps its devices under ``BranchPresets``, not under ``Device``."""
    rack = read_project(_als(tmp_path / "Vocal Chain.adg", RACK_XML))
    assert rack.is_rack is True
    assert len(rack.tracks) == 1
    track = rack.tracks[0]
    assert track.name == "Vocal Chain (rack)"
    assert track.type == "RackFile"
    assert [device.name for device in track.devices] == ["Vocal Chain", "Main > Top End"]


def test_only_named_macros_are_reported(tmp_path: Path) -> None:
    """"Macro 2" is Live's placeholder name and says nothing about the rack."""
    rack = read_project(_als(tmp_path / "Vocal Chain.adg", RACK_XML))
    assert [(macro.index, macro.name, macro.value) for macro in rack.macros] == [
        (0, "Air", "64")
    ]
    assert rack.macros[0].note == "high shelf"


# ==================================================================== to_dict


def test_to_dict_survives_a_json_round_trip(project: Project) -> None:
    """The reader's output has to reach an MCP client, which means JSON."""
    payload = to_dict(project)
    restored = json.loads(json.dumps(payload))
    assert restored["file"] == "Fixture.als"
    assert [track["name"] for track in restored["tracks"]] == ["Lead", "Drums", "Main"]
    assert restored["tracks"][0]["track_automation"][0]["layer"] == "track"


# ================================================================ write: setup


def test_listing_sidechain_slots_reads_without_writing(
    sidechain_als: Path, no_live: None
) -> None:
    before = sidechain_als.read_bytes()
    slots = list_sidechain_slots(sidechain_als)
    assert sidechain_als.read_bytes() == before
    assert len(slots) == 1
    slot = slots[0]
    assert (slot.track, slot.track_id, slot.device_tag) == ("Bass", "10", "Compressor2")
    assert slot.target == "AudioIn/None"
    assert slot.source_display == "Kick 1"  # the display name that outlived its routing
    assert slot.enabled is False
    assert slot.routed is False


# ============================================== write: the backup comes first


def test_a_write_makes_a_backup_before_it_touches_the_file(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """The backup is a copy of the file as it was, verified by hash before the write.

    Comparing the backup's bytes against the bytes captured before the call is
    the whole proof: if the backup had been taken afterwards it would carry the
    change, and the undo would be worthless.
    """
    original = sidechain_als.read_bytes()
    backups = tmp_path / "backups"
    age(sidechain_als)

    result = set_sidechain_source(
        sidechain_als, target_track="Bass", source_track="SC", backup_dir=backups
    )

    assert result.backup.is_file()
    assert result.backup.parent == backups
    assert result.backup.read_bytes() == original
    assert result.backup.suffix == ".als"
    assert "maestro" in result.backup.name
    assert sidechain_als.read_bytes() != original  # and the set really did change


def test_the_backup_restores_the_set_byte_for_byte(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    original = sidechain_als.read_bytes()
    age(sidechain_als)
    result = set_sidechain_source(
        sidechain_als, target_track="Bass", source_track="SC", backup_dir=tmp_path / "b"
    )
    restore_backup(result.backup, sidechain_als)
    assert sidechain_als.read_bytes() == original


def test_restore_refuses_something_that_is_not_a_live_set(tmp_path: Path) -> None:
    """A corrupted backup must not be usable to overwrite a working file."""
    junk = tmp_path / "junk.als"
    junk.write_text("definitely not gzip", encoding="utf-8")
    target = _als(tmp_path / "Target.als", SIDECHAIN_XML)
    intact = target.read_bytes()
    with pytest.raises(AlsWriteError, match="not a readable gzip file"):
        restore_backup(junk, target)
    assert target.read_bytes() == intact
    with pytest.raises(AlsWriteError, match="backup not found"):
        restore_backup(tmp_path / "nothing-here.als", target)


# =========================================================== write: the write


def test_wiring_a_sidechain_changes_four_attributes_and_verifies_them(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """The write the LOM cannot do, proved by re-reading the bytes from disk."""
    age(sidechain_als)
    result = set_sidechain_source(
        sidechain_als, target_track="Bass", source_track="SC", backup_dir=tmp_path / "b"
    )

    assert result.verified is True
    assert result.verify_failures == ()
    assert {change.after for change in result.changes} == {
        "true",
        "AudioIn/Track.15/PostFxOut",
        "SC",
        "Post FX",
    }
    # The target carries the source track's Id XML attribute (15), never its
    # position in the track list (1).
    target = next(change for change in result.changes if change.what.endswith("Target"))
    assert target.before == "AudioIn/None"
    assert target.after == "AudioIn/Track.15/PostFxOut"
    assert target.created is False
    assert result.size_before > 0
    assert result.live_check.live_running is False


def test_the_wiring_is_visible_to_the_reader_afterwards(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """Both halves of Channel B have to agree about the same bytes."""
    age(sidechain_als)
    set_sidechain_source(
        sidechain_als, target_track="Bass", source_track="SC", backup_dir=tmp_path / "b"
    )
    slot = list_sidechain_slots(sidechain_als)[0]
    assert slot.routed is True
    assert slot.enabled is True
    assert slot.source_display == "SC"

    sidechain = read_project(sidechain_als).track("Bass").devices[0].sidechain
    assert sidechain is not None
    assert sidechain.configured is True
    assert sidechain.target == "AudioIn/Track.15/PostFxOut"
    assert sidechain.source == "SC"


def test_the_result_says_what_was_not_proven(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """A success here means the bytes say so, not that Live reads them that way."""
    age(sidechain_als)
    result = set_sidechain_source(
        sidechain_als, target_track="Bass", source_track="SC", backup_dir=tmp_path / "b"
    )
    joined = " ".join(result.notes)
    assert "not proof Live reads it that way" in joined
    # The source track in this fixture is muted, which is a measured fact worth
    # reporting and not a reason to refuse: the tap sits before the mixer.
    assert "muted" in joined


def test_the_written_container_is_shaped_like_lives_own(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """MTIME 0, no embedded filename, XFL 0, OS 0x0a, and CRLF line endings.

    ``gzip.open`` would stamp the current time and the output filename into the
    header instead, so the container is built by hand.
    """
    age(sidechain_als)
    set_sidechain_source(
        sidechain_als, target_track="Bass", source_track="SC", backup_dir=tmp_path / "b"
    )
    raw = sidechain_als.read_bytes()
    assert raw[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x0a"

    body = gzip.decompress(raw)
    assert body.startswith(b'<?xml version="1.0" encoding="UTF-8"?>\r\n')
    assert body.endswith(b"\r\n")
    assert body.count(b"\n") == body.count(b"\r\n")  # no bare LF survives


def test_pre_fx_is_offered_and_flagged_as_the_rare_choice(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    age(sidechain_als)
    result = set_sidechain_source(
        sidechain_als,
        target_track="Bass",
        source_track="SC",
        tap="pre",
        backup_dir=tmp_path / "b",
    )
    lower = next(change for change in result.changes if change.what.endswith("LowerDisplayString"))
    assert lower.after == "Pre FX"
    target = next(change for change in result.changes if change.what.endswith("Target"))
    assert target.after.endswith("/PreFxOut")
    assert any("423 of 425" in note for note in result.notes)


# ================================================= write: refusing to proceed


def test_a_write_refuses_while_live_is_running_and_leaves_no_trace(
    sidechain_als: Path, tmp_path: Path, live_running: None
) -> None:
    """The refusal that matters: Live keeps the set in memory and overwrites on save.

    Nothing is written and no backup is made: a refusal must not litter the
    project folder.
    """
    original = sidechain_als.read_bytes()
    backups = tmp_path / "backups"
    age(sidechain_als)

    with pytest.raises(AlsRefused, match="Ableton Live is running"):
        set_sidechain_source(
            sidechain_als, target_track="Bass", source_track="SC", backup_dir=backups
        )

    assert sidechain_als.read_bytes() == original
    assert not backups.exists()


def test_allow_live_running_overrides_that_refusal(
    sidechain_als: Path, tmp_path: Path, live_running: None
) -> None:
    """The check cannot tell *which* set Live has open, so it can be overridden."""
    age(sidechain_als)
    result = set_sidechain_source(
        sidechain_als,
        target_track="Bass",
        source_track="SC",
        allow_live_running=True,
        backup_dir=tmp_path / "b",
    )
    assert result.verified is True
    assert result.live_check.live_running is True


def test_a_set_saved_seconds_ago_is_refused(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """A file written moments ago was most likely just saved by Live.

    The fixture is written by the test, so it is always in that state, which is
    why every writing test above has to age it first.
    """
    backups = tmp_path / "backups"
    with pytest.raises(AlsRefused, match="was written"):
        set_sidechain_source(
            sidechain_als, target_track="Bass", source_track="SC", backup_dir=backups
        )
    assert not backups.exists()


def test_a_file_another_process_holds_open_is_refused_without_an_override(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """The one refusal with no override: the file cannot be opened for writing at all.

    Simulated with a read-only file, which is the same ``OSError`` from the same
    probe. (This says nothing about Live: measured 2026-08-29, Live holds no lock
    on a set it has open: it reads the file and closes the handle.)
    """
    age(sidechain_als)
    original = sidechain_als.read_bytes()
    sidechain_als.chmod(stat.S_IREAD)
    try:
        try:
            with open(sidechain_als, "r+b"):
                writable = True
        except OSError:
            writable = False
        if writable:
            pytest.skip("this filesystem/user can write a read-only file (running as root?)")

        with pytest.raises(AlsRefused, match="no override"):
            set_sidechain_source(
                sidechain_als,
                target_track="Bass",
                source_track="SC",
                allow_live_running=True,
                backup_dir=tmp_path / "backups",
            )
    finally:
        sidechain_als.chmod(stat.S_IREAD | stat.S_IWRITE)
    assert sidechain_als.read_bytes() == original


def test_an_unknown_process_table_is_unknown_and_not_a_no(
    sidechain_als: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the process table cannot be read, that is reported as ``None``.

    "Unknown" is not "no", and the result says which of the two it was.
    """

    def explode(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("no such command")

    monkeypatch.setattr(
        write,
        "subprocess",
        types.SimpleNamespace(run=explode, SubprocessError=subprocess.SubprocessError),
    )
    age(sidechain_als)
    result = set_sidechain_source(
        sidechain_als, target_track="Bass", source_track="SC", backup_dir=tmp_path / "b"
    )
    assert result.live_check.live_running is None
    assert result.live_check.method == "unavailable"


# =========================================== write: refusing a bad instruction


def test_two_tracks_with_the_same_name_are_refused_rather_than_guessed(
    tmp_path: Path, no_live: None
) -> None:
    """Duplicate names are legal in Live and common (two ``SC`` tracks).

    Taking the first match would write to the wrong device and report success.
    """
    doubled = SIDECHAIN_XML.replace('Value="Bass"', 'Value="SC"', 1)
    als = _als(tmp_path / "Doubled.als", doubled)
    age(als)
    with pytest.raises(AlsWriteError, match="2 tracks are named"):
        set_sidechain_source(als, target_track="SC", source_track="SC")


def test_an_unknown_track_name_lists_what_the_set_actually_has(
    sidechain_als: Path, no_live: None
) -> None:
    age(sidechain_als)
    with pytest.raises(AlsWriteError, match="no target track named 'Guitar'") as caught:
        set_sidechain_source(sidechain_als, target_track="Guitar", source_track="SC")
    assert "Bass" in str(caught.value)


def test_a_track_cannot_duck_itself(sidechain_als: Path, no_live: None) -> None:
    age(sidechain_als)
    with pytest.raises(AlsWriteError, match="feedback, not ducking"):
        set_sidechain_source(sidechain_als, target_track="Bass", source_track="Bass")


def test_a_device_index_out_of_range_names_what_is_there(
    sidechain_als: Path, no_live: None
) -> None:
    age(sidechain_als)
    with pytest.raises(AlsWriteError, match="device 4 does not exist") as caught:
        set_sidechain_source(sidechain_als, target_track="Bass", source_track="SC", device=4)
    assert "Compressor2" in str(caught.value)


def test_a_track_with_no_sidechain_capable_device_says_so(
    sidechain_als: Path, no_live: None
) -> None:
    age(sidechain_als)
    with pytest.raises(AlsWriteError, match="no device on 'SC' has a sidechain input"):
        set_sidechain_source(sidechain_als, target_track="SC", source_track="Bass")


def test_an_unknown_tap_is_refused_before_anything_is_read(
    sidechain_als: Path, no_live: None
) -> None:
    with pytest.raises(AlsWriteError, match="tap must be one of"):
        set_sidechain_source(
            sidechain_als,
            target_track="Bass",
            source_track="SC",
            tap="sideways",  # type: ignore[arg-type]
        )


def test_a_missing_file_is_refused(tmp_path: Path, no_live: None) -> None:
    with pytest.raises(AlsWriteError, match="no such file"):
        set_sidechain_source(
            tmp_path / "absent.als", target_track="Bass", source_track="SC"
        )


# ================================================== write: the generic setter


def test_set_attribute_writes_one_value_and_reads_it_back(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """The units here are the file's, not the LOM's normalised ones."""
    age(sidechain_als)
    result = set_attribute(
        sidechain_als,
        ".//Compressor2/Threshold/Manual",
        0.3,
        backup_dir=tmp_path / "b",
    )
    assert result.verified is True
    assert result.changes[0].before == "1"
    assert result.changes[0].after == "0.3"
    root = load_xml(sidechain_als)
    assert root.find(".//Compressor2/Threshold/Manual").get("Value") == "0.3"


def test_set_attribute_refuses_an_expression_that_matches_more_than_one_element(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """The guard between a typo and several thousand silently rewritten parameters.

    Measured on one 5.3 MB set: ``.//Manual`` matches 4496 elements.
    """
    age(sidechain_als)
    original = sidechain_als.read_bytes()
    backups = tmp_path / "backups"
    with pytest.raises(AlsWriteError, match="matches [0-9]+ elements"):
        set_attribute(sidechain_als, ".//Manual", 0.5, backup_dir=backups)
    assert sidechain_als.read_bytes() == original
    assert not backups.exists()


def test_set_attribute_takes_an_index_out_of_an_ambiguous_match(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    age(sidechain_als)
    result = set_attribute(
        sidechain_als, ".//Manual", True, index=0, backup_dir=tmp_path / "b"
    )
    assert result.verified is True
    assert result.changes[0].after == "true"  # a bool is written Live's way
    with pytest.raises(AlsWriteError, match="index 99 is out of range"):
        set_attribute(sidechain_als, ".//Manual", 1, index=99, allow_live_running=True)


def test_set_attribute_refuses_an_expression_that_matches_nothing(
    sidechain_als: Path, no_live: None
) -> None:
    age(sidechain_als)
    with pytest.raises(AlsWriteError, match="matches nothing"):
        set_attribute(sidechain_als, ".//NoSuchDevice/Manual", 1)


def test_set_attribute_will_not_invent_an_attribute_without_being_told(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    age(sidechain_als)
    with pytest.raises(AlsWriteError, match="pass create=True"):
        set_attribute(
            sidechain_als,
            ".//Compressor2/Threshold/Manual",
            0.3,
            attribute="Nonsense",
            backup_dir=tmp_path / "b",
        )


def test_set_attribute_refuses_control_characters(
    sidechain_als: Path, no_live: None
) -> None:
    age(sidechain_als)
    with pytest.raises(AlsWriteError, match="control characters"):
        set_attribute(sidechain_als, ".//Compressor2/Threshold/Manual", "a\nb")


# ============================================ write: the read-back is genuine


def test_verification_reads_the_bytes_on_disk_and_not_the_tree_in_memory(
    sidechain_als: Path, tmp_path: Path, no_live: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fault injection: write bytes that parse but carry none of the edits.

    ``docs/protocol.md`` §5.4 applied to a file: a write that reports only
    success is indistinguishable from a write that did nothing. If verification
    checked the tree it had just edited it would always pass, so this test hands
    it a file that disagrees and requires it to notice.
    """
    original = sidechain_als.read_bytes()
    real_write = write._write_atomic

    def stale(path: Path, _data: bytes) -> None:
        real_write(path, original)

    age(sidechain_als)
    monkeypatch.setattr(write, "_write_atomic", stale)
    result = set_sidechain_source(
        sidechain_als, target_track="Bass", source_track="SC", backup_dir=tmp_path / "b"
    )

    assert result.verified is False
    assert len(result.verify_failures) == 4
    assert all("expected" in failure for failure in result.verify_failures)


def test_a_write_that_does_not_read_back_as_a_set_restores_the_backup(
    sidechain_als: Path, tmp_path: Path, no_live: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fault injection: corrupt the first write and require the backup to come back.

    That case is an exception rather than a result: there is nothing for the
    caller to inspect, only a set to put right.
    """
    original = sidechain_als.read_bytes()
    real_write = write._write_atomic
    calls: list[Path] = []

    def sabotage(path: Path, data: bytes) -> None:
        calls.append(path)
        real_write(path, b"not a gzip stream at all" if len(calls) == 1 else data)

    age(sidechain_als)
    monkeypatch.setattr(write, "_write_atomic", sabotage)
    with pytest.raises(AlsWriteError, match="has been restored") as caught:
        set_sidechain_source(
            sidechain_als, target_track="Bass", source_track="SC", backup_dir=tmp_path / "b"
        )

    assert "backup" in str(caught.value).lower()
    assert sidechain_als.read_bytes() == original


def test_the_write_leaves_no_temporary_file_behind(
    sidechain_als: Path, tmp_path: Path, no_live: None
) -> None:
    """The new bytes go to a sibling temp file and are moved into place atomically.

    A crash mid-write therefore cannot leave half an ``.als`` behind, and a
    completed one must not leave the scaffolding.
    """
    age(sidechain_als)
    set_sidechain_source(
        sidechain_als, target_track="Bass", source_track="SC", backup_dir=tmp_path / "b"
    )
    leftovers = [path.name for path in tmp_path.iterdir() if ".tmp" in path.name]
    assert leftovers == []


# --------------------------------------------------------------------------------
# The one write that must never happen: into the set Live is holding
# --------------------------------------------------------------------------------


class _LiveHolding:
    """A client that answers ``song.file_path`` and nothing else."""

    def __init__(self, loaded: str | None) -> None:
        self._loaded = loaded

    def get(self, path: str) -> dict[str, object]:
        assert path == "song.file_path", path
        if self._loaded is None:
            raise AbletonError("Live most likely closed the connection")
        return {"value": self._loaded}


def test_als_write_refuses_the_set_live_has_open_even_when_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``allow_live_running=True`` must not reach the one dangerous case.

    The file cannot answer which set Live holds: measured 2026-08-29 against Live
    12.4.5, an exclusive open succeeds on the very set Live has loaded, because
    Live reads the ``.als`` into memory and closes the handle. Live can answer it
  exactly (``song.file_path``) and it is asked before the file layer falls
    back to "is a Live process alive".

    What makes this worth a test: a write into the open set is not corruption. It
    lands on disk, Live saves over it, and the change is simply gone with no error
    anywhere. So the refusal has to be unconditional, and the override that
    existed for the coarse check must not survive into this one.
    """
    from ableton_maestro import server

    target = tmp_path / "held.als"
    target.write_bytes(b"not really a set")
    before = target.read_bytes()

    monkeypatch.setattr(server, "_client_instance", lambda: _LiveHolding(str(target)))
    for override in (False, True):
        result = server.als_write(
            str(target),
            "attribute",
            confirm=True,
            expression=".//Tempo/Manual",
            value="120",
            allow_live_running=override,
        )
        assert result["ok"] is False, override
        assert result.get("blocked") is True, override
        assert "close the SET" in result["error"]
        assert target.read_bytes() == before, "the file was touched"


def test_als_write_stops_asking_the_file_when_live_says_another_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A running Live holding a different set is not a reason to refuse.

    This is the half that saves work: the requirement is that the SET is closed,
    not that the application is. Quitting Live to edit an unrelated project costs
    a full plugin rescan and buys nothing.
    """
    from ableton_maestro import server

    target = tmp_path / "other.als"
    target.write_bytes(b"not really a set")

    monkeypatch.setattr(
        server, "_client_instance", lambda: _LiveHolding(str(tmp_path / "something else.als"))
    )
    result = server.als_write(
        str(target), "attribute", confirm=True, expression=".//Tempo/Manual", value="120"
    )
    # It gets past the guard and fails on the file being no Live set at all,
    # which is the proof that the Live-is-running refusal was lifted.
    assert result["ok"] is False
    assert result.get("blocked") is not True
    assert "close the SET" not in str(result.get("error", ""))


def test_als_write_keeps_the_conservative_refusal_when_live_cannot_be_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No answer is not the same as "no". Unreachable Live keeps the old guard."""
    from ableton_maestro import server

    target = tmp_path / "unknown.als"
    target.write_bytes(b"not really a set")

    monkeypatch.setattr(server, "_client_instance", lambda: _LiveHolding(None))
    result = server.als_write(
        str(target), "attribute", confirm=True, expression=".//Tempo/Manual", value="120"
    )
    assert result["ok"] is False
