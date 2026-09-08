# Audio BPM Light Sync

A lightweight Blender add-on for creating **audio-reactive light animations** and **BPM-synchronized lighting** directly inside Blender.

The add-on analyzes WAV audio without external Python packages and converts audio energy into animated light intensity. It also includes frequency-biased responses for individual instrument-like elements such as kick, snare, hi-hat, guitar, bass, piano, and synth.

> **Blender Version:** 3.2+  
> **Audio Format:** WAV  
> **Dependencies:** None

---

## ✨ Features

### 🎵 Audio Reactive Modes

Choose how your lights respond to the audio:

- **Volume** — reacts to overall audio energy
- **Beat** — reacts primarily to detected beats
- **Hybrid** — combines volume and beat response

### 🎸 Instrument Response

The add-on includes frequency-based responses for different parts of a musical arrangement:

| Category | Response |
|---|---|
| 🥁 Drums | Kick / Bass Drum |
| 🥁 Drums | Snare |
| 🥁 Drums | Hi-Hat |
| 🥁 Drums | Toms |
| 🥁 Drums | Cymbals |
| 🎸 Guitar | Rhythm |
| 🎸 Guitar | Melody |
| 🎸 | Bass Guitar |
| 🎹 | Keyboard / Piano |
| 🎛️ | Synth (Optional) |

These responses allow different lights to react to different frequency characteristics of the music.

> **Note:** Instrument responses are frequency-based approximations and are not true audio source/stem separation.

---

## 🔊 How It Works

The analyzer processes the audio using lightweight signal analysis:

1. Loads a WAV audio file.
2. Converts stereo/multichannel audio to mono.
3. Resamples the audio to **16 kHz**.
4. Calculates overall RMS audio energy.
5. Detects beat peaks using adaptive energy thresholds.
6. Uses **Goertzel frequency analysis** to estimate energy around representative frequencies.
7. Normalizes and smooths the response curves.
8. Generates Blender keyframes for the selected lights.

The frequency analysis is intentionally lightweight and does not require NumPy, SciPy, or machine-learning models.

---

## 📦 Installation

1. Download the `.py` file from this repository.
2. Open **Blender**.
3. Go to:

   **Edit → Preferences → Add-ons**

4. Click **Install...**
5. Select:

   `audio_bpm_light_sync_instrument_response.py`

6. Enable the add-on.
7. Open the 3D Viewport sidebar with:

   `N`

8. Open the:

   **Audio Light**

   tab.

---

## 🚀 Basic Workflow

### 1. Select Your Lights

Create or select the lights you want to animate.

You can also use:

**Quick Setup → Create Light Rig**

to automatically create a simple three-light visualizer setup.

---

### 2. Select Audio

Under **Audio Input**, click:

**Select WAV**

Then select your audio file.

Currently, the analyzer directly supports:

```text
.wav
