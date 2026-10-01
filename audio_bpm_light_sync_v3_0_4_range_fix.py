bl_info = {
    "name": "Audio BPM Light Sync — Clean",
    "author": "Mortalsinner",
    "version": (3, 0, 2),
    "blender": (3, 2, 0),
    "location": "View3D > Sidebar > Audio Light",
    "description": "Clean audio-to-light synchronizer with robust BPM/frequency analysis, section isolation, and per-section light assignment.",
    "category": "Animation",
}

import bpy
import math
import os
import wave
import struct

try:
    import audioop
except ImportError:
    audioop = None

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================
# GLOBAL ANALYSIS CACHE
# ============================================================

_ANALYSIS_CACHE = {}


# ============================================================
# AUDIO I/O
# ============================================================

def read_wav_mono(filepath, target_rate=22050):
    if not filepath:
        raise ValueError("Select a WAV file first.")
    if not os.path.isfile(filepath):
        raise ValueError("Audio file does not exist.")
    if os.path.splitext(filepath)[1].lower() != ".wav":
        raise ValueError("For this clean version, use a WAV file.")

    with wave.open(filepath, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if width not in (1, 2, 4):
        raise ValueError("Unsupported WAV sample width.")

    if channels > 1:
        if audioop is None:
            raise ValueError("Stereo WAV conversion requires audioop in this Blender Python build.")
        raw = audioop.tomono(raw, width, 0.5, 0.5)
        channels = 1

    if rate != target_rate:
        if audioop is None:
            raise ValueError("Audio resampling requires audioop in this Blender Python build.")
        raw, _ = audioop.ratecv(raw, width, channels, rate, target_rate, None)
        rate = target_rate

    if width == 1:
        values = struct.unpack("<{}B".format(len(raw)), raw)
        samples = [(v - 128.0) / 128.0 for v in values]
    elif width == 2:
        count = len(raw) // 2
        values = struct.unpack("<{}h".format(count), raw)
        samples = [v / 32768.0 for v in values]
    else:
        count = len(raw) // 4
        values = struct.unpack("<{}i".format(count), raw)
        samples = [v / 2147483648.0 for v in values]

    return samples, rate


def normalize_array(values):
    if len(values) == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def smooth_array(values, amount):
    amount = max(0, int(amount))
    if amount <= 0 or len(values) < 2:
        return values
    kernel_size = amount * 2 + 1
    kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
    padded = np.pad(values, (amount, amount), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

# These are intentionally simple musical ranges, not claims of true source
# separation. A mixed song cannot be perfectly separated into vocals/instruments
# by frequency filtering alone.
BASS_RANGE = (30.0, 250.0)
VOCAL_MELODY_RANGE = (250.0, 5000.0)


def make_analysis_frames(samples, sample_rate, fps):
    """Return per-frame volume/bass/vocal response and dominant frequency."""
    if np is None:
        raise RuntimeError(
            "This clean analyzer requires Blender's bundled NumPy. "
            "Use a standard Blender build that includes NumPy."
        )

    audio = np.asarray(samples, dtype=np.float64)
    hop = max(128, int(round(sample_rate / float(max(fps, 1.0)))))
    window = max(2048, int(round(sample_rate * 0.09288)))  # ~2048 @ 22.05k
    fft_size = 1
    while fft_size < window:
        fft_size *= 2
    fft_size = max(4096, fft_size)

    # Precompute Hann window.
    win = np.hanning(window)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    bass_mask = (freqs >= BASS_RANGE[0]) & (freqs < BASS_RANGE[1])
    vocal_mask = (freqs >= VOCAL_MELODY_RANGE[0]) & (freqs < min(VOCAL_MELODY_RANGE[1], sample_rate * 0.5 - 1.0))

    total_frames = max(1, int(math.ceil(len(audio) / float(hop))))
    volume = np.zeros(total_frames, dtype=np.float64)
    bass = np.zeros(total_frames, dtype=np.float64)
    vocal = np.zeros(total_frames, dtype=np.float64)
    dominant = np.zeros(total_frames, dtype=np.float64)

    prev_mag = None
    onset = np.zeros(total_frames, dtype=np.float64)

    for i in range(total_frames):
        center = i * hop
        start = max(0, center - window // 2)
        end = min(len(audio), start + window)
        chunk = audio[start:end]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))

        volume[i] = math.sqrt(float(np.mean(chunk * chunk)))
        spectrum = np.abs(np.fft.rfft(chunk * win, n=fft_size))
        power = spectrum * spectrum

        bass[i] = float(np.sqrt(np.mean(power[bass_mask]))) if np.any(bass_mask) else 0.0
        vocal[i] = float(np.sqrt(np.mean(power[vocal_mask]))) if np.any(vocal_mask) else 0.0

        # Ignore the extreme sub/ultrasonic regions when finding a dominant
        # musical frequency. This produces a much more useful frequency readout.
        musical = (freqs >= 30.0) & (freqs <= min(12000.0, sample_rate * 0.5 - 1.0))
        if np.any(musical):
            idx = np.argmax(spectrum[musical])
            musical_freqs = freqs[musical]
            dominant[i] = float(musical_freqs[idx])

        # Spectral flux is a much better beat/onset signal than raw RMS alone.
        mag = np.log1p(spectrum)
        if prev_mag is not None:
            diff = mag - prev_mag
            onset[i] = float(np.sum(np.maximum(diff, 0.0)))
        prev_mag = mag

    volume = normalize_array(volume)
    bass = normalize_array(bass)
    vocal = normalize_array(vocal)
    onset = normalize_array(onset)

    # Light smoothing for stable visual response. Keep onset comparatively sharp.
    volume = smooth_array(volume, 2)
    bass = smooth_array(bass, 2)
    vocal = smooth_array(vocal, 2)

    return {
        "fps": float(fps),
        "sample_rate": sample_rate,
        "hop": hop,
        "volume": volume,
        "bass": bass,
        "vocal": vocal,
        "onset": onset,
        "dominant_hz": dominant,
        "duration": len(audio) / float(sample_rate),
        "sample_count": len(audio),
    }


def detect_bpm(analysis):
    """Estimate tempo using onset autocorrelation + pulse scoring.

    This is intentionally a tempo estimator, not a guarantee of 95–100% accuracy.
    Actual accuracy depends on the recording, meter, tempo changes, and ambiguity
    between half/double tempo.
    """
    onset = np.asarray(analysis["onset"], dtype=np.float64)
    fps = float(analysis["fps"])
    if len(onset) < 16:
        return 0.0, 0.0

    onset = onset - np.mean(onset)
    std = float(np.std(onset))
    if std < 1e-9:
        return 0.0, 0.0
    onset /= std

    min_bpm = 50.0
    max_bpm = 220.0
    min_lag = max(1, int(round(60.0 * fps / max_bpm)))
    max_lag = min(len(onset) - 1, int(round(60.0 * fps / min_bpm)))

    # FFT autocorrelation is substantially faster than O(n * lag) loops.
    n = 1
    while n < len(onset) * 2:
        n *= 2
    spectrum = np.fft.rfft(onset, n=n)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=n)[:len(onset)]
    autocorr /= np.maximum(1.0, np.arange(len(autocorr), 0, -1))

    best_bpm = 0.0
    best_score = -1e30
    candidates = []

    for lag in range(min_lag, max_lag + 1):
        if lag >= len(autocorr):
            break
        base = float(autocorr[lag])
        half = float(autocorr[max(1, lag // 2)])
        double_lag = float(autocorr[min(len(autocorr) - 1, lag * 2)])
        score = base + 0.35 * half + 0.15 * double_lag
        candidates.append((score, lag))

    candidates.sort(reverse=True)
    if not candidates:
        return 0.0, 0.0

    # Refine the best few candidates by measuring how strongly the onset
    # envelope lines up with a beat grid. This reduces common half/double errors.
    for _, lag in candidates[:12]:
        phase_scores = []
        max_offset = min(lag, 32)
        for offset in range(max_offset):
            positions = np.arange(offset, len(onset), lag)
            if len(positions) < 4:
                continue
            phase_scores.append(float(np.mean(onset[positions])))
        pulse = max(phase_scores) if phase_scores else 0.0
        score = float(autocorr[lag]) + 0.75 * pulse
        # Parabolic interpolation around the autocorrelation peak gives
        # sub-frame lag precision and reduces BPM quantization at 24/30 FPS.
        refined_lag = float(lag)
        if 1 <= lag < len(autocorr) - 1:
            y0 = float(autocorr[lag - 1])
            y1 = float(autocorr[lag])
            y2 = float(autocorr[lag + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-9:
                delta = 0.5 * (y0 - y2) / denom
                refined_lag += max(-0.5, min(0.5, delta))
        bpm = 60.0 * fps / refined_lag
        if score > best_score:
            best_score = score
            best_bpm = bpm

    # Report a normalized confidence-like diagnostic, not a probability.
    peak = max(float(c[0]) for c in candidates[:min(20, len(candidates))])
    confidence = 0.0 if peak <= 0 else max(0.0, min(1.0, best_score / (abs(peak) + 1e-9)))
    return best_bpm, confidence


def frame_response(analysis, response, frame_index, gain=1.0, threshold=0.0):
    idx = max(0, min(int(frame_index), len(analysis["volume"]) - 1))
    if response == "BASS":
        value = float(analysis["bass"][idx])
    elif response == "VOCAL_MELODY":
        value = float(analysis["vocal"][idx])
    else:
        value = float(analysis["volume"][idx])

    if value < threshold:
        value = 0.0
    else:
        # Remap the area above threshold back to 0–1.
        if threshold > 0.0:
            value = (value - threshold) / max(1e-6, 1.0 - threshold)

    return max(0.0, min(1.0, value * max(0.0, gain)))


# ============================================================
# LIGHT HELPERS
# ============================================================

def light_objects_from_collection(collection):
    if collection is None:
        return []
    return [obj for obj in collection.all_objects if obj.type == "LIGHT"]


def clear_generated_fcurves(obj, start=None, end=None):
    if not obj.animation_data or not obj.animation_data.action:
        return 0
    action = obj.animation_data.action
    removed = 0
    for fc in list(action.fcurves):
        if fc.data_path != "energy":
            continue
        if not fc.group or fc.group.name != "Audio Light Sync":
            continue
        if start is None or end is None:
            action.fcurves.remove(fc)
            removed += 1
            continue
        for kp in list(fc.keyframe_points):
            if start <= kp.co.x <= end:
                fc.keyframe_points.remove(kp)
        fc.update()
        if not fc.keyframe_points:
            action.fcurves.remove(fc)
            removed += 1
    return removed


def keyframe_energy(light_data, frame, value, interpolation="LINEAR"):
    """Insert an energy keyframe with controllable interpolation."""
    light_data.energy = max(0.0, float(value))
    light_data.keyframe_insert(data_path="energy", frame=int(frame), group="Audio Light Sync")
    if light_data.animation_data and light_data.animation_data.action:
        for fc in light_data.animation_data.action.fcurves:
            if fc.data_path == "energy":
                if fc.group:
                    fc.group.name = "Audio Light Sync"
                for kp in fc.keyframe_points:
                    kp.interpolation = interpolation


# ============================================================
# SECTION / MARKER HELPERS
# ============================================================

def marker_name(base, side):
    return "[ALS] {} {}".format(base, side)


def find_scene_marker(scene, name):
    for marker in scene.timeline_markers:
        if marker.name == name:
            return marker
    return None


def sync_section_markers(scene, section):
    for name in (marker_name(section.name, "START"), marker_name(section.name, "END")):
        marker = find_scene_marker(scene, name)
        if marker:
            scene.timeline_markers.remove(marker)
    scene.timeline_markers.new(marker_name(section.name, "START"), frame=int(section.start_frame))
    scene.timeline_markers.new(marker_name(section.name, "END"), frame=int(section.end_frame))


def remove_section_markers(scene, section_name):
    for marker in list(scene.timeline_markers):
        if marker.name in (marker_name(section_name, "START"), marker_name(section_name, "END")):
            scene.timeline_markers.remove(marker)


def parse_section_marker(name):
    text = str(name).strip()
    if text.startswith("[ALS]"):
        text = text[5:].strip()
    upper = text.upper()
    if upper.endswith(" START"):
        return text[:-6].strip(), "START"
    if upper.endswith(" END"):
        return text[:-4].strip(), "END"
    return None, None


def import_timeline_sections(scene, props):
    pairs = {}
    for marker in scene.timeline_markers:
        base, side = parse_section_marker(marker.name)
        if not base:
            continue
        pairs.setdefault(base, {})[side] = marker.frame

    added = 0
    for name, pair in pairs.items():
        if "START" not in pair or "END" not in pair:
            continue
        existing = None
        for item in props.sections:
            if item.name == name:
                existing = item
                break
        if existing is None:
            item = props.sections.add()
            item.name = name
            existing = item
            added += 1
        existing.start_frame = min(int(pair["START"]), int(pair["END"]))
        existing.end_frame = max(int(pair["START"]), int(pair["END"]))

    if props.sections:
        props.active_section = min(props.active_section, len(props.sections) - 1)
    return added


def section_for_frame(props, frame):
    for i, section in enumerate(props.sections):
        if section.start_frame <= frame <= section.end_frame:
            return i
    return -1


# ============================================================
# PROPERTIES
# ============================================================

class ALS_Section(bpy.types.PropertyGroup):
    expanded: bpy.props.BoolProperty(name="Expanded", default=False)
    name: bpy.props.StringProperty(name="Section", default="Section")
    start_frame: bpy.props.IntProperty(name="Start", default=1, min=0)
    end_frame: bpy.props.IntProperty(name="End", default=250, min=1)

    response: bpy.props.EnumProperty(
        name="Response",
        items=[
            ("BASS", "Bass (Low Frequencies)", "Low-frequency energy, roughly 30–250 Hz"),
            ("VOCAL_MELODY", "Vocals & Melodies (Mid–High)", "Mid-to-high frequency energy, roughly 250–5000 Hz"),
            ("VOLUME", "Volume (Dynamic Range)", "Overall RMS loudness / dynamic energy"),
        ],
        default="VOLUME",
    )

    reaction_style: bpy.props.EnumProperty(
        name="Reaction Style",
        items=[
            ("SMOOTH", "Smooth", "Smoothed audio response with linear transitions"),
            ("FAST", "Fast", "No smoothing; follows frame-by-frame audio changes"),
            ("STROBE", "Strobe", "Instant ON/OFF response using constant interpolation"),
            ("PUNCH", "Punch", "Sharp hit followed by a short decay"),
        ],
        default="FAST",
    )

    punch_decay: bpy.props.IntProperty(
        name="Punch Decay",
        description="Frames used to return from a punch back to base energy",
        default=3,
        min=1,
        max=30,
    )

    target_mode: bpy.props.EnumProperty(
        name="Light Target",
        items=[
            ("INDIVIDUAL", "Individual Light", "Drive one light"),
            ("COLLECTION", "Collection", "Drive all lights in a collection"),
        ],
        default="INDIVIDUAL",
    )

    light: bpy.props.PointerProperty(type=bpy.types.Object)
    collection: bpy.props.PointerProperty(type=bpy.types.Collection)

    base_energy: bpy.props.FloatProperty(name="Base Energy", default=100.0, min=0.0)
    max_energy: bpy.props.FloatProperty(name="Max Energy", default=2500.0, min=0.0)
    gain: bpy.props.FloatProperty(name="Gain", default=1.0, min=0.0, max=5.0)
    threshold: bpy.props.FloatProperty(name="Threshold", default=0.15, min=0.0, max=1.0)
    smoothing: bpy.props.IntProperty(name="Smoothing", default=2, min=0, max=20)
    keyframe_step: bpy.props.IntProperty(name="Keyframe Step", default=1, min=1, max=20)


class ALS_Properties(bpy.types.PropertyGroup):
    audio_file: bpy.props.StringProperty(name="Audio File", subtype="FILE_PATH")
    audio_loaded: bpy.props.BoolProperty(default=False)
    status: bpy.props.StringProperty(default="Select a WAV file.")
    detected_bpm: bpy.props.FloatProperty(name="Detected BPM", default=0.0)
    bpm_confidence: bpy.props.FloatProperty(name="BPM Diagnostic", default=0.0, min=0.0, max=1.0)
    duration: bpy.props.FloatProperty(name="Duration", default=0.0)
    analyzed_frames: bpy.props.IntProperty(default=0)
    dominant_frequency: bpy.props.FloatProperty(name="Dominant Frequency", default=0.0)

    audio_start_frame: bpy.props.IntProperty(name="Audio Start", default=1, min=0)
    audio_end_frame: bpy.props.IntProperty(name="Audio End", default=250, min=1)
    waveform_channel: bpy.props.IntProperty(name="Waveform Channel", default=12, min=1, max=128)
    waveform_volume: bpy.props.FloatProperty(name="Waveform Volume", default=1.0, min=0.0, max=10.0)

    active_section: bpy.props.IntProperty(default=0, min=0)
    sections: bpy.props.CollectionProperty(type=ALS_Section)

    auto_import_markers: bpy.props.BoolProperty(name="Use Timeline Markers", default=True)
    clear_section_only: bpy.props.BoolProperty(name="Clear Only Generated Section", default=True)

    analysis_fps: bpy.props.FloatProperty(name="Analysis FPS", default=30.0, min=1.0, max=240.0)
    show_audio_analysis: bpy.props.BoolProperty(name="Audio Analysis", default=True)
    show_waveform: bpy.props.BoolProperty(name="Waveform", default=False)
    show_sections: bpy.props.BoolProperty(name="Audio Sections", default=True)
    show_workflow: bpy.props.BoolProperty(name="Workflow", default=False)


# ============================================================
# OPERATORS
# ============================================================

class ALS_OT_load_audio(bpy.types.Operator):
    bl_idname = "audio_light_sync.load_audio"
    bl_label = "Select WAV"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.audio_light_sync
        props.audio_file = self.filepath
        props.audio_loaded = False
        props.status = os.path.basename(self.filepath) if self.filepath else "No audio selected."
        return {"FINISHED"}


class ALS_OT_analyze(bpy.types.Operator):
    bl_idname = "audio_light_sync.analyze"
    bl_label = "Analyze Audio"
    bl_description = "Analyze BPM, dynamic range, bass energy, vocal/melody energy, and dominant frequency"

    def execute(self, context):
        props = context.scene.audio_light_sync
        scene = context.scene
        try:
            samples, rate = read_wav_mono(props.audio_file)
            fps = scene.render.fps / max(scene.render.fps_base, 1e-9)
            props.analysis_fps = fps
            analysis = make_analysis_frames(samples, rate, fps)
            bpm, confidence = detect_bpm(analysis)

            _ANALYSIS_CACHE[props.audio_file] = analysis
            props.audio_loaded = True
            props.duration = analysis["duration"]
            props.detected_bpm = bpm
            props.bpm_confidence = confidence
            props.analyzed_frames = len(analysis["volume"])

            nonzero = analysis["dominant_hz"][analysis["dominant_hz"] > 0]
            props.dominant_frequency = float(np.median(nonzero)) if len(nonzero) else 0.0
            props.audio_end_frame = props.audio_start_frame + max(0, props.analyzed_frames - 1)
            props.status = "Analysis complete."
            self.report({"INFO"}, "Audio analyzed: {:.2f} BPM".format(bpm))
        except Exception as exc:
            props.audio_loaded = False
            props.status = "Analysis error: {}".format(exc)
            self.report({"ERROR"}, str(exc))
        return {"FINISHED"}


class ALS_OT_add_section(bpy.types.Operator):
    bl_idname = "audio_light_sync.add_section"
    bl_label = "Add Section"

    def execute(self, context):
        scene = context.scene
        props = scene.audio_light_sync
        item = props.sections.add()
        item.name = "Section {}".format(len(props.sections))
        item.start_frame = scene.frame_current
        item.end_frame = max(scene.frame_current + 120, scene.frame_current + 1)
        item.base_energy = 100.0
        item.max_energy = 2500.0
        item.response = "VOLUME"
        item.target_mode = "INDIVIDUAL"
        props.active_section = len(props.sections) - 1
        sync_section_markers(scene, item)
        return {"FINISHED"}


class ALS_OT_remove_section(bpy.types.Operator):
    bl_idname = "audio_light_sync.remove_section"
    bl_label = "Remove Section"
    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        props = context.scene.audio_light_sync
        if not props.sections:
            return {"CANCELLED"}
        index = max(0, min(self.index, len(props.sections) - 1))
        name = props.sections[index].name
        remove_section_markers(context.scene, name)
        props.sections.remove(index)
        props.active_section = max(0, min(props.active_section, len(props.sections) - 1)) if props.sections else 0
        return {"FINISHED"}


class ALS_OT_import_markers(bpy.types.Operator):
    bl_idname = "audio_light_sync.import_markers"
    bl_label = "Import Timeline Markers"

    def execute(self, context):
        count = import_timeline_sections(context.scene, context.scene.audio_light_sync)
        self.report({"INFO"}, "Imported {} section(s).".format(count))
        return {"FINISHED"}


class ALS_OT_sync_section(bpy.types.Operator):
    bl_idname = "audio_light_sync.sync_section"
    bl_label = "Sync Markers"
    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        props = context.scene.audio_light_sync
        if not props.sections:
            return {"CANCELLED"}
        idx = max(0, min(self.index, len(props.sections) - 1))
        item = props.sections[idx]
        item.end_frame = max(item.end_frame, item.start_frame + 1)
        sync_section_markers(context.scene, item)
        return {"FINISHED"}


class ALS_OT_generate_section(bpy.types.Operator):
    bl_idname = "audio_light_sync.generate_section"
    bl_label = "Generate Section"
    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        scene = context.scene
        props = scene.audio_light_sync
        if not props.audio_file:
            self.report({"ERROR"}, "Select and analyze a WAV first.")
            return {"CANCELLED"}

        if props.audio_file not in _ANALYSIS_CACHE:
            self.report({"ERROR"}, "Analyze Audio first.")
            return {"CANCELLED"}

        if not props.sections:
            self.report({"ERROR"}, "Add/import a section first.")
            return {"CANCELLED"}

        idx = self.index
        if idx < 0:
            idx = section_for_frame(props, scene.frame_current)
        if idx < 0 or idx >= len(props.sections):
            self.report({"ERROR"}, "Put the playhead inside a section or select one.")
            return {"CANCELLED"}

        section = props.sections[idx]
        analysis = _ANALYSIS_CACHE[props.audio_file]
        fps = analysis["fps"]

        if section.end_frame <= section.start_frame:
            self.report({"ERROR"}, "Section end must be after section start.")
            return {"CANCELLED"}

        if section.target_mode == "INDIVIDUAL":
            lights = [section.light] if section.light and section.light.type == "LIGHT" else []
        else:
            lights = light_objects_from_collection(section.collection)

        lights = [obj for obj in lights if obj and obj.type == "LIGHT"]
        if not lights:
            self.report({"ERROR"}, "Assign an individual light or a collection containing lights.")
            return {"CANCELLED"}

        # Clear only this section, preserving other generated sections.
        if props.clear_section_only:
            for obj in lights:
                clear_generated_fcurves(obj, section.start_frame, section.end_frame)
        else:
            for obj in lights:
                clear_generated_fcurves(obj)

        # Convert Blender timeline frames directly to analysis-frame indices.
        # The analysis is generated at the scene FPS, so one analysis sample
        # corresponds to one Blender frame. Multiplying the frame offset by FPS
        # here would incorrectly turn e.g. frame 220 into audio index 5280 at 24 FPS.
        audio_start = int(props.audio_start_frame)
        analyzed_count = len(analysis["volume"])
        audio_end = audio_start + analyzed_count - 1

        if section.start_frame < audio_start or section.end_frame > audio_end:
            self.report(
                {"ERROR"},
                "Section outside analyzed range: {}-{} (audio: {}-{}).".format(
                    section.start_frame, section.end_frame, audio_start, audio_end
                )
            )
            return {"CANCELLED"}

        start_audio_index = section.start_frame - audio_start
        end_audio_index = section.end_frame - audio_start

        # Pre-smooth the selected response only. This avoids changing other sections.
        if section.response == "BASS":
            values = analysis["bass"][start_audio_index:end_audio_index + 1].copy()
        elif section.response == "VOCAL_MELODY":
            values = analysis["vocal"][start_audio_index:end_audio_index + 1].copy()
        else:
            values = analysis["volume"][start_audio_index:end_audio_index + 1].copy()

        # Reaction Style controls how quickly the light follows the analyzed signal.
        # Blender interpolation is applied to the generated F-Curve; Constant gives
        # the abrupt/strobe behavior while Linear avoids the extra smoothing of Bezier.
        if section.reaction_style == "SMOOTH":
            values = smooth_array(values, section.smoothing)
            interpolation = "LINEAR"
        elif section.reaction_style == "FAST":
            values = smooth_array(values, 0)
            interpolation = "LINEAR"
        else:
            values = smooth_array(values, 0)
            interpolation = "CONSTANT" if section.reaction_style == "STROBE" else "LINEAR"

        # Collection targets receive the same section response.
        step = max(1, int(section.keyframe_step))
        for light_index, obj in enumerate(lights):
            data = obj.data

            # Start from base energy so every generated section has a known state.
            keyframe_energy(data, section.start_frame, section.base_energy, interpolation)

            for frame in range(section.start_frame, section.end_frame + 1, step):
                rel = frame - section.start_frame
                idx_value = min(len(values) - 1, int(round(rel * fps / max(fps, 1.0))))
                raw = float(values[idx_value])
                if raw < section.threshold:
                    response = 0.0
                else:
                    response = (raw - section.threshold) / max(1e-6, 1.0 - section.threshold)
                response = max(0.0, min(1.0, response * section.gain))

                if section.reaction_style == "STROBE":
                    # Binary response: above threshold = full light, otherwise base.
                    energy = section.max_energy if response > 0.0 else section.base_energy
                    keyframe_energy(data, frame, energy, "CONSTANT")

                elif section.reaction_style == "PUNCH":
                    # Only meaningful hits create a sharp peak and a short decay.
                    if response > 0.0:
                        energy = section.base_energy + (section.max_energy - section.base_energy) * response
                        keyframe_energy(data, frame, energy, "LINEAR")
                        decay_frame = min(section.end_frame, frame + int(section.punch_decay))
                        keyframe_energy(data, decay_frame, section.base_energy, "LINEAR")

                else:
                    energy = section.base_energy + (section.max_energy - section.base_energy) * response
                    keyframe_energy(data, frame, energy, interpolation)

            # Guarantee exact endpoint keyframes.
            keyframe_energy(data, section.end_frame, section.base_energy, interpolation)

        props.active_section = idx
        self.report({"INFO"}, "Generated '{}' from frame {}–{} using {}.".format(
            section.name, section.start_frame, section.end_frame, section.response
        ))
        return {"FINISHED"}


class ALS_OT_generate_playhead(bpy.types.Operator):
    bl_idname = "audio_light_sync.generate_playhead"
    bl_label = "Generate Section at Playhead"

    def execute(self, context):
        props = context.scene.audio_light_sync
        idx = section_for_frame(props, context.scene.frame_current)
        if idx < 0:
            self.report({"ERROR"}, "Playhead is not inside a section.")
            return {"CANCELLED"}
        return bpy.ops.audio_light_sync.generate_section(index=idx)


class ALS_OT_add_waveform(bpy.types.Operator):
    bl_idname = "audio_light_sync.add_waveform"
    bl_label = "Add / Refresh Waveform"

    def execute(self, context):
        props = context.scene.audio_light_sync
        scene = context.scene
        if not props.audio_file or not os.path.isfile(props.audio_file):
            self.report({"ERROR"}, "Select a valid WAV first.")
            return {"CANCELLED"}

        seq = scene.sequence_editor_create()
        strips = getattr(seq, "strips", None)
        if strips is None:
            strips = getattr(seq, "sequences", None)
        if strips is None:
            self.report({"ERROR"}, "Blender sequence editor API unavailable.")
            return {"CANCELLED"}

        # Remove our previous waveform strips only.
        for strip in list(strips):
            if strip.name.startswith("[ALS] Waveform"):
                try:
                    strips.remove(strip)
                except Exception:
                    pass

        channel = int(props.waveform_channel)
        # Find a free channel from the requested one upward.
        used = {int(s.channel) for s in strips if int(s.channel) == channel}
        while channel in used and channel < 128:
            channel += 1
            used = {int(s.channel) for s in strips if int(s.channel) == channel}

        try:
            # Blender's SequenceEditor.new_sound() expects a FILEPATH string,
            # not a bpy.types.Sound datablock. Passing the loaded Sound object
            # causes: "argument 2, \"filepath\" - Function filepath expected string type, not Sound".
            # Keep the filepath as the source so this works with current Blender
            # sequence APIs.
            filepath = os.path.abspath(bpy.path.abspath(props.audio_file))

            if hasattr(strips, "new_sound"):
                strip = strips.new_sound(
                    "[ALS] Waveform",
                    filepath,
                    channel,
                    int(props.audio_start_frame),
                )
            else:
                strip = seq.sequences.new_sound(
                    "[ALS] Waveform",
                    filepath,
                    channel,
                    int(props.audio_start_frame),
                )

            strip.show_waveform = True
            strip.volume = float(props.waveform_volume)
        except Exception as exc:
            self.report({"ERROR"}, "Could not add waveform: {}".format(exc))
            return {"CANCELLED"}

        self.report({"INFO"}, "Waveform added to Sequencer channel {}.".format(channel))
        return {"FINISHED"}


class ALS_OT_clear(bpy.types.Operator):
    bl_idname = "audio_light_sync.clear"
    bl_label = "Clear Generated Animation"

    def execute(self, context):
        count = 0
        for obj in bpy.data.objects:
            if obj.type == "LIGHT":
                count += clear_generated_fcurves(obj)
        self.report({"INFO"}, "Cleared {} generated light curve(s).".format(count))
        return {"FINISHED"}


# ============================================================
# UI
# ============================================================

class ALS_PT_panel(bpy.types.Panel):
    bl_label = "Audio Light Sync"
    bl_idname = "ALS_PT_clean_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Audio Light"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.audio_light_sync

        # AUDIO ANALYSIS
        box = layout.box()
        row = box.row(align=True)
        row.prop(props, "show_audio_analysis", text="", icon="TRIA_DOWN" if props.show_audio_analysis else "TRIA_RIGHT", emboss=False)
        row.label(text="1. AUDIO ANALYSIS", icon="SOUND")
        if props.show_audio_analysis:
            box.operator("audio_light_sync.load_audio", icon="FILE_FOLDER", text="Select WAV")
            if props.audio_file:
                box.label(text=os.path.basename(props.audio_file), icon="FILE_SOUND")
            box.prop(props, "audio_start_frame")
            box.operator("audio_light_sync.analyze", icon="MODIFIER", text="ANALYZE BPM + FREQUENCY")
            if props.audio_loaded:
                box.label(text="BPM: {:.2f}".format(props.detected_bpm), icon="TIME")
                box.label(text="Diagnostic: {:.0f}%".format(props.bpm_confidence * 100.0), icon="INFO")
                box.label(text="Dominant: {:.1f} Hz".format(props.dominant_frequency), icon="GRAPH")
                box.label(text="Duration: {:.2f}s".format(props.duration))
                box.label(text="Frames analyzed: {}".format(props.analyzed_frames))
            else:
                box.label(text=props.status, icon="INFO")

        # WAVEFORM
        box = layout.box()
        row = box.row(align=True)
        row.prop(props, "show_waveform", text="", icon="TRIA_DOWN" if props.show_waveform else "TRIA_RIGHT", emboss=False)
        row.label(text="2. WAVEFORM", icon="SEQ_SEQUENCER")
        if props.show_waveform:
            row = box.row(align=True)
            row.prop(props, "waveform_channel")
            row.prop(props, "waveform_volume")
            box.operator("audio_light_sync.add_waveform", icon="SOUND", text="ADD / REFRESH WAVEFORM")

        # SECTIONS
        box = layout.box()
        row = box.row(align=True)
        row.prop(props, "show_sections", text="", icon="TRIA_DOWN" if props.show_sections else "TRIA_RIGHT", emboss=False)
        row.label(text="3. AUDIO SECTIONS", icon="MARKER")

        if props.show_sections:
            row = box.row(align=True)
            row.operator("audio_light_sync.add_section", icon="ADD", text="Add Section")
            row.operator("audio_light_sync.import_markers", icon="FILE_REFRESH", text="Import Timeline Markers")

            if not props.sections:
                box.label(text="Create START/END sections, then assign a response and lights.", icon="INFO")
            else:
                for i, section in enumerate(props.sections):
                    sec = box.box()
                    header = sec.row(align=True)
                    header.prop(section, "expanded", text="", icon="TRIA_DOWN" if section.expanded else "TRIA_RIGHT", emboss=False)
                    header.label(text="{}  {}–{}".format(section.name, section.start_frame, section.end_frame))
                    op = header.operator("audio_light_sync.remove_section", text="", icon="X")
                    op.index = i

                    if section.expanded:
                        sec.prop(section, "name")
                        row = sec.row(align=True)
                        row.prop(section, "start_frame")
                        row.prop(section, "end_frame")
                        op = sec.operator("audio_light_sync.sync_section", icon="FILE_REFRESH", text="Update Timeline Markers")
                        op.index = i

                        sec.prop(section, "response")
                        sec.prop(section, "reaction_style")
                        if section.reaction_style == "PUNCH":
                            sec.prop(section, "punch_decay")
                        sec.prop(section, "target_mode")
                        if section.target_mode == "INDIVIDUAL":
                            sec.prop(section, "light", text="Light")
                        else:
                            sec.prop(section, "collection", text="Collection")

                        row = sec.row(align=True)
                        row.prop(section, "gain")
                        row.prop(section, "threshold")
                        row = sec.row(align=True)
                        row.prop(section, "base_energy")
                        row.prop(section, "max_energy")
                        row = sec.row(align=True)
                        row.prop(section, "smoothing")
                        row.prop(section, "keyframe_step")

                        op = sec.operator("audio_light_sync.generate_section", icon="KEY_HLT", text="GENERATE THIS SECTION")
                        op.index = i

            box.separator()
            box.operator("audio_light_sync.generate_playhead", icon="PLAY", text="GENERATE SECTION AT PLAYHEAD")
            box.prop(props, "clear_section_only")

        # SIMPLE HELP
        box = layout.box()
        row = box.row(align=True)
        row.prop(props, "show_workflow", text="", icon="TRIA_DOWN" if props.show_workflow else "TRIA_RIGHT", emboss=False)
        row.label(text="Workflow", icon="INFO")
        if props.show_workflow:
            box.label(text="1. Select WAV → Analyze")
            box.label(text="2. Add/import START + END markers")
            box.label(text="3. Choose Bass / Vocals & Melodies / Volume")
            box.label(text="4. Assign Individual Light or Collection")
            box.label(text="5. Choose Smooth / Fast / Strobe / Punch")
            box.label(text="6. Generate only that section")
            box.label(text="Note: frequency bands are mixed-audio analysis, not true stem separation.")

        box = layout.box()
        box.operator("audio_light_sync.clear", icon="TRASH", text="CLEAR ALL GENERATED LIGHTING")


# ============================================================
# REGISTRATION
# ============================================================

classes = (
    ALS_Section,
    ALS_Properties,
    ALS_OT_load_audio,
    ALS_OT_analyze,
    ALS_OT_add_section,
    ALS_OT_remove_section,
    ALS_OT_import_markers,
    ALS_OT_sync_section,
    ALS_OT_generate_section,
    ALS_OT_generate_playhead,
    ALS_OT_add_waveform,
    ALS_OT_clear,
    ALS_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.audio_light_sync = bpy.props.PointerProperty(type=ALS_Properties)


def unregister():
    if hasattr(bpy.types.Scene, "audio_light_sync"):
        del bpy.types.Scene.audio_light_sync
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
