# Ableton Maestro

> An AI copilot for Ableton Live. Speak in natural language to program drums, generate chord progressions, tweak synthesizers, shape automation, and mix your tracks, right inside your open Live set.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/) [![Ableton Live 11+](https://img.shields.io/badge/Ableton%20Live-11%20%7C%2012-orange.svg)](https://www.ableton.com/) [![No telemetry](https://img.shields.io/badge/telemetry-none-brightgreen.svg)](#your-music-stays-yours)

---

## Stay in the creative flow

Making music in a DAW is inspiring, but clicking around to draw notes, adjust mixer faders, and tweak envelopes over and over can pull you out of the zone.

**Ableton Maestro** acts like an assistant producer sitting next to you in the studio:
- **Keep your hands on the music:** Live stays open right in front of you. You listen, evaluate, and make the creative decisions.
- **Natural studio dialogue:** Ask for beats, basslines, filter sweeps, sidechain setups, or mix adjustments in plain language.
- **Real parameters, real clips:** Nothing is faked or pre-rendered. Everything is written directly into your active project as editable MIDI notes, native devices, and standard Ableton automation envelopes.

---

## How it feels in the studio

Here is what collaborating with Maestro looks like in a session:

### 1. Beatmaking & Groove
> **You:** *"Build me a driving 126 BPM techno groove. Punchy 909 kick, a crisp clap on 2 and 4, and rolling 16th closed hats with a bit of velocity groove and swing."*
>
> **Assistant:** Created track **Techno Drums** with a 909 Drum Rack and a 4-bar looped clip at 126 BPM:
> - **Kick:** Four-on-the-floor on C1 with accented downbeats.
> - **Clap:** Sitting on beats 2 and 4 on D#1.
> - **Hi-Hats:** 16th notes on F#1 with alternating velocity dynamics (92 / 76 / 86 / 76) and a 57% groove swing.
>
> Clip is ready and playing.

### 2. Sound Design & Automation
> **You:** *"The hats feel too loud and harsh. Drop them by 2.5 dB, add an Auto Filter to the lead synth, and draw a 4-bar low-pass sweep that opens up towards the drop."*
>
> **Assistant:** 
> - Reduced the Hi-Hat chain volume by **-2.5 dB**.
> - Added an **Auto Filter** (Low-Pass) to track **Lead Synth**.
> - Created a 4-bar automation envelope on the filter cutoff frequency, sweeping from **600 Hz** smoothly up to **18 kHz**.

### 3. Harmony & Chords
> **You:** *"Add a lush Neo-Soul chord progression in D minor on an Electric piano track. Keep the chords laid back with realistic human velocity."*
>
> **Assistant:** 
> - Created a MIDI track **Electric Piano** with a Rhodes preset.
> - Wrote a 4-bar progression (*Dm9 - G13 - Cmaj9 - A7#9*) with slight timing humanization and soft velocities between 65 and 85.

---

## What you can do

| Workflow | Example prompt | What Maestro does |
|---|---|---|
| **🥁 Drums & Beats** | *"Create a 4-bar UK Garage drum beat with swung hats and ghost snares"* | Creates drum tracks, loads kits, places MIDI notes with realistic velocities and swing |
| **🎹 Chords & Melodies** | *"Write an 8-bar melodic bassline in F minor that follows the root notes"* | Generates chord progressions, basslines, melodies, and arpeggios |
| **🎛️ Sound Design** | *"Open the Wavetable filter cutoff to 65% and increase resonance slightly"* | Adjusts native instruments, synthesizers, and audio effect parameters |
| **📈 Automation & Envelopes** | *"Automate a reverb swell over the last 2 bars before bar 33"* | Draws precise parameter curves, filter sweeps, volume swells, and modulation |
| **🎚️ Mixing & Levels** | *"Turn the bass down 3 dB, pan the rhythm guitar 25% left, and add a chorus"* | Sets track volumes, panning, sends, returns, and insert effects |
| **🎼 Arrangement & Structure** | *"Duplicate the verse clip to bar 17 in the Arrangement and drop a locator called Drop"* | Moves clips to the arrangement timeline, sets cue points/locators, loops sections |
| **🎚️ Editing & Groove** | *"Transpose the synth lead up a minor third and quantize to 1/16 notes at 70%"* | Transposes pitches, quantizes timing, modifies note lengths, and adjusts velocities |

---

## Why Maestro is reliable

Most AI DAW integrations send commands blindly and hope for the best. 

Maestro is built on a **two-way verification engine**:
- **Every change is checked:** When Maestro adjusts a parameter, writes a note, or moves a fader, it reads the value back from Ableton Live in real time to verify that the change actually took effect.
- **Accurate feedback:** If a parameter is at its maximum or Live constraints prevent an action, Maestro reports this immediately instead of giving false confirmations.
- **Empirical LOM catalog:** Built on a catalog of 1,164 rows across 5 files, with 1,128 verified live against Ableton Live, spanning tracks, clips, native devices, and the browser.

---

## Quick start

### 1. Set up the server
Clone the repository and set up the Python environment:

```bash
git clone https://github.com/romanstark/ableton-maestro.git
cd ableton-maestro
python -m venv .venv
```

Activate the environment:
- **Windows (PowerShell):** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

Install dependencies:
```bash
pip install -e ".[dev]"
```

### 2. Install the Ableton Remote Script
Run the automated installer:

```bash
python scripts/install_script.py
```

This copies the remote script to your Ableton User Library. Then **restart Ableton Live** and enable it:
> **Live → Preferences / Settings → Link, Tempo & MIDI → Control Surface → `AbletonMaestro`**

Verify the connection:
```bash
python -m ableton_maestro.client ping
```

### 3. Connect your AI assistant
Add the server to your MCP client configuration (e.g., Claude Desktop, Antigravity IDE, Cursor):

```json
{
  "mcpServers": {
    "ableton-maestro": {
      "command": "/absolute/path/to/ableton-maestro/.venv/Scripts/python.exe",
      "args": ["-m", "ableton_maestro.server"]
    }
  }
}
```

Open a project in Ableton Live and start by asking: *"What tracks are in this set?"*

---

## Your music stays yours

- **100% Local:** All communication between your AI assistant and Ableton Live happens over a local, internal loopback connection on your computer.
- **No telemetry or cloud tracking:** Maestro collects zero analytics, has no database, and does not upload your MIDI, audio, project files, or prompts to any external server.
- **Minimal dependencies:** Pure local code with no hidden web scrapers or cloud telemetry.

---

## What stays in your hands

Ableton Live's API is extensive, but some things are intentionally reserved for you in the DAW interface:

| Task | Why | How to do it |
|---|---|---|
| **Export / Audio Bounce** | Not exposed by Live's scripting API | Use **File → Export Audio/Video** in Live |
| **Save Project** | Not exposed by Live's API | Press **Ctrl+S** / **Cmd+S** as usual |
| **Group Tracks** | Read-only in Live's API | Press **Ctrl+G** / **Cmd+G** in Live |
| **Unconfigured Third-Party VSTs** | Live only surfaces plugin parameters configured in the device strip | Click **Configure** on the VST and click the parameters you want exposed |
| **Critical Listening** | AI can shape parameters, but only you have ears | Listen on your monitors/headphones and guide the music |

---

## Also using Steinberg Dorico?

If you also work with music notation, check out **[Dorico Maestro](https://github.com/romanstark/dorico-maestro)**, an MCP server built with the same architecture for Steinberg Dorico. Use the same AI assistant to bridge your workflow between session sketching in Ableton and engraving parts in Dorico.

---

## Documentation & Developer Resources

For technical details, architecture specs, and contributing:

- [docs/architecture.md](docs/architecture.md): Internal architecture and Live Object Model integration
- [docs/protocol.md](docs/protocol.md): Wire protocol and communication specification
- [docs/catalog.md](docs/catalog.md): LOM catalog schema and verification rules
- [docs/limits.md](docs/limits.md): Technical constraints and API boundary measurements
- [CONTRIBUTING.md](CONTRIBUTING.md): Contribution guidelines and developer setup
- [LICENSE](LICENSE): AGPL-3.0 License
