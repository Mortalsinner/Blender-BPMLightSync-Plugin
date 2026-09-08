bl_info = {
    "name": "Audio BPM Light Sync",
    "author": "OpenAI",
    "version": (2, 0, 0),
    "blender": (3, 2, 0),
    "location": "View3D > Sidebar > Audio Light",
    "description": "Audio-reactive lighting and BPM animation for visualizers.",
    "category": "Animation",
}

import bpy
import math
import os
import wave
import struct
import audioop
from mathutils import Color


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def read_audio_samples(filepath, target_rate=16000):
    """Read WAV audio and return mono samples at target_rate.

    Blender 3.2's Python can use the standard library, so this
    version intentionally supports WAV directly without numpy.
    """
    if not filepath:
        raise ValueError("No audio file selected.")

    if not os.path.exists(filepath):
        raise ValueError("Audio file does not exist.")

    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".wav":
        raise ValueError(
            "Audio Analysis currently supports WAV files. "
            "Convert MP3/M4A/etc. to WAV first."
        )

    with wave.open(filepath, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()

        raw = wf.readframes(frame_count)

    if sample_width not in (1, 2, 4):
        raise ValueError("Unsupported WAV sample format.")

    # Convert to mono.
    if channels > 1:
        raw = audioop.tomono(
            raw,
            sample_width,
            0.5,
            0.5,
        )

    # Resample for faster analysis.
    if sample_rate != target_rate:
        raw, _ = audioop.ratecv(
            raw,
            sample_width,
            1,
            sample_rate,
            target_rate,
            None,
        )

    # Convert samples to normalized floats.
    if sample_width == 1:
        values = struct.unpack(
            "<{}B".format(len(raw)),
            raw,
        )
        samples = [(v - 128) / 128.0 for v in values]

    elif sample_width == 2:
        count = len(raw) // 2
        values = struct.unpack(
            "<{}h".format(count),
            raw,
        )
        samples = [v / 32768.0 for v in values]

    else:
        count = len(raw) // 4
        values = struct.unpack(
            "<{}i".format(count),
            raw,
        )
        samples = [v / 2147483648.0 for v in values]

    return samples, target_rate


def rms(samples):
    if not samples:
        return 0.0

    total = 0.0
    for x in samples:
        total += x * x

    return math.sqrt(total / len(samples))


def zero_crossing_rate(samples):
    if len(samples) < 2:
        return 0.0

    crossings = 0

    previous = samples[0]

    for value in samples[1:]:
        if (value >= 0) != (previous >= 0):
            crossings += 1
        previous = value

    return crossings / float(len(samples))


def normalize(values):
    if not values:
        return []

    low = min(values)
    high = max(values)

    if high - low < 0.000001:
        return [0.0 for _ in values]

    return [
        (value - low) / (high - low)
        for value in values
    ]


def moving_average(values, window=3):
    if not values:
        return []

    result = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        section = values[start:i + 1]
        result.append(sum(section) / len(section))

    return result


def goertzel_power(samples, sample_rate, frequency):
    """Estimate energy around one frequency without external dependencies."""
    if not samples:
        return 0.0

    # Goertzel is efficient for a small set of frequencies and works well
    # for the lightweight, dependency-free analyzer used by this add-on.
    k = int(0.5 + (len(samples) * frequency / float(sample_rate)))
    omega = (2.0 * math.pi * k) / len(samples)
    coeff = 2.0 * math.cos(omega)

    q1 = 0.0
    q2 = 0.0
    for sample in samples:
        q0 = coeff * q1 - q2 + sample
        q2 = q1
        q1 = q0

    power = q1 * q1 + q2 * q2 - coeff * q1 * q2
    return max(0.0, power) / max(1, len(samples) ** 2)


AUDIO_BANDS = {
    # Instrument response -> representative frequency points.
    # These are frequency-biased responses, not true source separation.
    "KICK":       (45, 70, 100, 120),
    "SNARE":      (180, 250, 500, 1000, 1800, 2500),
    "HIHAT":      (4500, 6000, 7500),
    "TOMS":       (90, 130, 180, 240, 320),
    "CYMBALS":    (3500, 5000, 7000),
    "GUITAR_RHYTHM": (120, 180, 250, 350, 500, 700),
    "GUITAR_MELODY": (500, 800, 1200, 1800, 2400, 3200),
    "BASS_GUITAR":  (45, 70, 90, 120, 160, 220, 300),
    "PIANO":       (100, 200, 400, 700, 1000, 1600, 2400, 3200),
    "SYNTH":       (200, 400, 800, 1200, 1800, 2600, 3400),
}


def analyze_frequency_responses(samples, sample_rate, samples_per_frame, frame_count):
    """Build lightweight frequency-response curves for instrument-like controls."""
    responses = {name: [] for name in AUDIO_BANDS}

    # Analyze a modest number of representative frequencies per frame.
    # This keeps Blender 3.2 responsive without numpy/scipy.
    for frame_index in range(frame_count):
        center = int(frame_index * samples_per_frame)
        window_size = max(256, samples_per_frame * 4)
        start = max(0, center - window_size // 2)
        end = min(len(samples), center + window_size // 2)
        chunk = samples[start:end]

        if len(chunk) < 32:
            for name in responses:
                responses[name].append(0.0)
            continue

        raw = {}
        for name, frequencies in AUDIO_BANDS.items():
            total = 0.0
            count = 0
            for frequency in frequencies:
                if frequency < sample_rate * 0.5:
                    total += goertzel_power(chunk, sample_rate, frequency)
                    count += 1
            raw[name] = total / float(max(1, count))

        for name, value in raw.items():
            responses[name].append(value)

    # Normalize each response independently so quiet instruments can still react.
    for name in responses:
        responses[name] = moving_average(normalize(responses[name]), 2)

    return responses

def analyze_audio(filepath, fps, start_frame, end_frame):
    """Analyze WAV volume and estimate beat events.

    This is intentionally lightweight and dependency-free.
    It uses short-time energy + adaptive peak detection.
    """
    samples, sample_rate = read_audio_samples(filepath)

    duration = len(samples) / float(sample_rate)

    frame_count = max(1, end_frame - start_frame + 1)

    # Analyze roughly one audio window per Blender frame.
    samples_per_frame = max(
        1,
        int(sample_rate / max(fps, 1.0)),
    )

    # Cap work for extremely long files.
    max_analysis_frames = 12000

    if frame_count > max_analysis_frames:
        frame_count = max_analysis_frames

    energy = []
    zcr = []

    for frame_index in range(frame_count):

        center = int(
            frame_index
            * samples_per_frame
        )

        window_size = samples_per_frame * 2

        start = max(
            0,
            center - window_size // 2,
        )

        end = min(
            len(samples),
            center + window_size // 2,
        )

        chunk = samples[start:end]

        if not chunk:
            energy.append(0.0)
            zcr.append(0.0)
        else:
            energy.append(rms(chunk))
            zcr.append(zero_crossing_rate(chunk))

    energy = normalize(energy)
    energy = moving_average(energy, 2)

    frequency_responses = analyze_frequency_responses(
        samples,
        sample_rate,
        samples_per_frame,
        frame_count,
    )

    # Adaptive beat detection.
    beats = []

    min_gap = max(
        2,
        int(fps * 60.0 / 200.0),
    )

    for i in range(2, len(energy) - 2):

        local_start = max(
            0,
            i - int(fps * 0.35),
        )

        local_end = min(
            len(energy),
            i + int(fps * 0.35),
        )

        neighborhood = energy[local_start:local_end]

        if not neighborhood:
            continue

        average = sum(neighborhood) / len(neighborhood)
        value = energy[i]

        is_peak = (
            value > energy[i - 1]
            and value >= energy[i + 1]
        )

        threshold = average + 0.12

        if is_peak and value > threshold:

            if not beats or i - beats[-1] >= min_gap:
                beats.append(i)

    return {
        "duration": duration,
        "energy": energy,
        "zcr": zcr,
        "frequency_responses": frequency_responses,
        "beats": beats,
        "sample_rate": sample_rate,
    }


# ============================================================
# ANIMATION HELPERS
# ============================================================

def clear_generated_animation(obj):
    """Remove only F-curves created by this add-on."""
    if not obj.animation_data or not obj.animation_data.action:
        return

    action = obj.animation_data.action
    target_group = "Audio Light Sync"

    for fc in list(action.fcurves):
        if fc.group and fc.group.name == target_group:
            action.fcurves.remove(fc)

    if not action.fcurves:
        obj.animation_data_clear()


def keyframe_energy(light, frame, value):
    light.energy = value
    light.keyframe_insert(
        data_path="energy",
        frame=frame,
        group="Audio Light Sync",
    )

    if light.animation_data and light.animation_data.action:
        for fc in light.animation_data.action.fcurves:
            if fc.data_path == "energy":
                if fc.group:
                    fc.group.name = "Audio Light Sync"
                for key in fc.keyframe_points:
                    key.interpolation = "BEZIER"


def keyframe_color(light, frame, color):
    light.color = color

    light.keyframe_insert(
        data_path="color",
        frame=frame,
        group="Audio Light Sync",
    )

    if light.animation_data and light.animation_data.action:
        for fc in light.animation_data.action.fcurves:
            if fc.data_path == "color":
                if fc.group:
                    fc.group.name = "Audio Light Sync"
                for key in fc.keyframe_points:
                    key.interpolation = "LINEAR"


def hsv_color(hue, saturation=1.0, value=1.0):
    color = Color()
    color.hsv = (
        hue % 1.0,
        saturation,
        value,
    )
    return color


# ============================================================
# PROPERTIES
# ============================================================

class ALS_Properties(bpy.types.PropertyGroup):

    audio_file: bpy.props.StringProperty(
        name="Audio File",
        subtype="FILE_PATH",
    )

    audio_loaded: bpy.props.BoolProperty(
        name="Audio Loaded",
        default=False,
    )

    analysis_status: bpy.props.StringProperty(
        name="Status",
        default="No audio analyzed",
    )

    bpm: bpy.props.FloatProperty(
        name="BPM",
        default=128.0,
        min=20.0,
        max=400.0,
    )

    beat_division: bpy.props.EnumProperty(
        name="Beat Division",
        items=[
            ("1", "1/1", "One event per beat"),
            ("2", "1/2", "Two events per beat"),
            ("4", "1/4", "Four events per beat"),
            ("8", "1/8", "Eight events per beat"),
            ("16", "1/16", "Sixteen events per beat"),
        ],
        default="4",
    )

    start_frame: bpy.props.IntProperty(
        name="Start Frame",
        default=1,
        min=0,
    )

    end_frame: bpy.props.IntProperty(
        name="End Frame",
        default=250,
        min=1,
    )

    base_energy: bpy.props.FloatProperty(
        name="Base Energy",
        default=500.0,
        min=0.0,
    )

    flash_energy: bpy.props.FloatProperty(
        name="Flash Energy",
        default=2500.0,
        min=0.0,
    )

    reaction_strength: bpy.props.FloatProperty(
        name="Reaction Strength",
        default=1.0,
        min=0.0,
        max=2.0,
    )

    beat_threshold: bpy.props.FloatProperty(
        name="Beat Threshold",
        description="Sensitivity of automatic beat detection",
        default=0.12,
        min=0.01,
        max=1.0,
    )

    audio_smoothing: bpy.props.IntProperty(
        name="Smoothing",
        description="Smooth audio-driven intensity",
        default=2,
        min=0,
        max=20,
    )

    audio_gain: bpy.props.FloatProperty(
        name="Audio Gain",
        description="Multiply audio reaction",
        default=1.0,
        min=0.0,
        max=5.0,
    )

    audio_response: bpy.props.EnumProperty(
        name="Response",
        items=[
            ("VOLUME", "Volume", "React to overall audio energy"),
            ("BEAT", "Beat", "React primarily to detected beats"),
            ("HYBRID", "Hybrid", "Combine volume and beat response"),

            # Drums
            ("KICK", "Drums • Kick / Bass Drum", "React to low-frequency kick energy"),
            ("SNARE", "Drums • Snare", "React to snare-like mid/high frequency energy"),
            ("HIHAT", "Drums • Hi-Hat", "React to hi-hat-like high-frequency energy"),
            ("TOMS", "Drums • Toms", "React to tom-like low/mid frequency energy"),
            ("CYMBALS", "Drums • Cymbals", "React to cymbal-like high-frequency energy"),

            # Guitar / bass
            ("GUITAR_RHYTHM", "Guitar • Rhythm", "React to lower/mid guitar frequency energy"),
            ("GUITAR_MELODY", "Guitar • Melody", "React to higher guitar/melodic frequency energy"),
            ("BASS_GUITAR", "Bass Guitar", "React to bass-guitar frequency energy"),

            # Keys / electronic
            ("PIANO", "Keyboard / Piano", "React to piano/keyboard frequency energy"),
            ("SYNTH", "Synth (Optional)", "React to broad synth-like frequency energy"),
        ],
        default="HYBRID",
    )

    color_shift: bpy.props.BoolProperty(
        name="Color Shift",
        default=False,
    )

    color_hue_step: bpy.props.FloatProperty(
        name="Hue Step",
        default=0.08,
        min=0.0,
        max=1.0,
    )

    randomize_lights: bpy.props.BoolProperty(
        name="Stagger Lights",
        default=False,
    )

    stagger_frames: bpy.props.IntProperty(
        name="Stagger Frames",
        default=1,
        min=0,
        max=30,
    )

    clear_existing: bpy.props.BoolProperty(
        name="Clear Existing",
        default=True,
    )

    selected_only: bpy.props.BoolProperty(
        name="Selected Lights Only",
        default=True,
    )

    analyzed_beats: bpy.props.IntProperty(
        name="Detected Beats",
        default=0,
    )

    estimated_bpm: bpy.props.FloatProperty(
        name="Estimated BPM",
        default=0.0,
    )


# ============================================================
# OPERATORS
# ============================================================

class ALS_OT_load_audio(bpy.types.Operator):
    bl_idname = "audio_light_sync.load_audio"
    bl_label = "Select Audio"
    bl_description = "Select a WAV file for analysis"

    filepath: bpy.props.StringProperty(
        subtype="FILE_PATH"
    )

    filter_glob: bpy.props.StringProperty(
        default="*.wav",
        options={"HIDDEN"},
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.audio_light_sync

        props.audio_file = self.filepath
        props.audio_loaded = False
        props.analysis_status = (
            "Audio selected. Click Analyze Audio."
        )

        return {"FINISHED"}


class ALS_OT_analyze(bpy.types.Operator):
    bl_idname = "audio_light_sync.analyze"
    bl_label = "Analyze Audio"
    bl_description = "Analyze audio volume and detect beats"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.audio_light_sync

        if not props.audio_file:
            self.report(
                {"WARNING"},
                "Select a WAV audio file first."
            )
            return {"CANCELLED"}

        try:
            fps = scene.render.fps / scene.render.fps_base

            result = analyze_audio(
                props.audio_file,
                fps,
                props.start_frame,
                props.end_frame,
            )

        except Exception as exc:
            self.report(
                {"ERROR"},
                str(exc),
            )
            props.analysis_status = "Analysis failed"
            return {"CANCELLED"}

        props.audio_loaded = True
        props.analyzed_beats = len(result["beats"])

        # Estimate BPM from detected beat intervals.
        if len(result["beats"]) >= 2:

            intervals = []

            for a, b in zip(
                result["beats"][:-1],
                result["beats"][1:],
            ):
                intervals.append(b - a)

            average_interval = (
                sum(intervals) / len(intervals)
            )

            if average_interval > 0:
                estimated = (
                    60.0
                    * fps
                    / average_interval
                )

                props.estimated_bpm = estimated

        props.analysis_status = (
            "Analyzed: {} beats".format(
                props.analyzed_beats
            )
        )

        self.report(
            {"INFO"},
            props.analysis_status,
        )

        return {"FINISHED"}


class ALS_OT_generate_audio(bpy.types.Operator):
    bl_idname = "audio_light_sync.generate_audio"
    bl_label = "Generate From Audio"
    bl_description = "Generate light animation from analyzed audio"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.audio_light_sync

        if not props.audio_loaded:
            self.report(
                {"WARNING"},
                "Analyze audio first."
            )
            return {"CANCELLED"}

        lights = (
            [
                obj for obj in context.selected_objects
                if obj.type == "LIGHT"
            ]
            if props.selected_only
            else [
                obj for obj in scene.objects
                if obj.type == "LIGHT"
            ]
        )

        if not lights:
            self.report(
                {"WARNING"},
                "Select at least one light."
            )
            return {"CANCELLED"}

        try:
            fps = scene.render.fps / scene.render.fps_base

            result = analyze_audio(
                props.audio_file,
                fps,
                props.start_frame,
                props.end_frame,
            )

        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        energy = result["energy"]
        beats = result["beats"]
        frequency_responses = result.get("frequency_responses", {})

        if props.audio_smoothing > 0:
            energy = moving_average(
                energy,
                props.audio_smoothing,
            )

        beat_set = set(beats)

        scene.frame_start = props.start_frame
        scene.frame_end = min(
            props.end_frame,
            props.start_frame + len(energy) - 1,
        )

        for obj_index, obj in enumerate(lights):

            light = obj.data

            if props.clear_existing:
                clear_generated_animation(obj)

            original_color = tuple(light.color)

            for i, audio_value in enumerate(energy):

                frame = props.start_frame + i

                if frame > props.end_frame:
                    break

                value = max(
                    0.0,
                    min(
                        1.0,
                        audio_value * props.audio_gain,
                    ),
                )

                is_beat = i in beat_set

                beat_value = 1.0 if is_beat else 0.0

                if props.audio_response == "VOLUME":
                    response = value

                elif props.audio_response == "BEAT":
                    response = beat_value

                elif props.audio_response == "HYBRID":
                    response = max(
                        value,
                        beat_value * 0.95,
                    )

                else:
                    # Instrument modes use frequency-biased analysis. This is
                    # intentionally lightweight and dependency-free; it does
                    # not attempt machine-learning source separation.
                    instrument_curve = frequency_responses.get(
                        props.audio_response,
                        [],
                    )

                    if i < len(instrument_curve):
                        response = instrument_curve[i] * props.audio_gain
                    else:
                        response = 0.0

                    # Keep beat transients useful for percussive modes.
                    if props.audio_response in {
                        "KICK", "SNARE", "HIHAT", "TOMS", "CYMBALS"
                    }:
                        response = max(
                            response,
                            beat_value * 0.35,
                        )

                # Staggering makes groups of lights feel less robotic.
                # For instrument modes, stagger the selected frequency response
                # rather than falling back to full-spectrum energy.
                if props.randomize_lights:
                    stagger = (
                        obj_index
                        * props.stagger_frames
                    )
                    source_index = max(
                        0,
                        i - stagger,
                    )

                    if props.audio_response in {"VOLUME", "BEAT", "HYBRID"}:
                        if source_index < len(energy):
                            value2 = energy[source_index]
                        else:
                            value2 = value

                        response = max(
                            response,
                            value2 * 0.75,
                        )
                    else:
                        curve = frequency_responses.get(
                            props.audio_response,
                            [],
                        )
                        if source_index < len(curve):
                            response = max(
                                response,
                                curve[source_index] * props.audio_gain * 0.75,
                            )

                response = max(
                    0.0,
                    min(
                        1.0,
                        response
                        * props.reaction_strength,
                    ),
                )

                light_energy = (
                    props.base_energy
                    + (
                        props.flash_energy
                        - props.base_energy
                    )
                    * response
                )

                keyframe_energy(
                    light,
                    frame,
                    light_energy,
                )

                if props.color_shift and is_beat:

                    hue = (
                        i
                        * props.color_hue_step
                        + obj_index * 0.17
                    )

                    color = hsv_color(hue)

                    keyframe_color(
                        light,
                        frame,
                        color,
                    )

            light.energy = props.base_energy
            light.color = original_color

        self.report(
            {"INFO"},
            "Generated audio-reactive animation for {} light(s).".format(
                len(lights)
            ),
        )

        return {"FINISHED"}


class ALS_OT_generate_bpm(bpy.types.Operator):
    bl_idname = "audio_light_sync.generate_bpm"
    bl_label = "Generate BPM"
    bl_description = "Generate deterministic BPM light animation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.audio_light_sync

        lights = (
            [
                obj for obj in context.selected_objects
                if obj.type == "LIGHT"
            ]
            if props.selected_only
            else [
                obj for obj in scene.objects
                if obj.type == "LIGHT"
            ]
        )

        if not lights:
            self.report(
                {"WARNING"},
                "Select at least one light."
            )
            return {"CANCELLED"}

        fps = scene.render.fps / scene.render.fps_base
        frames_per_beat = fps * 60.0 / props.bpm
        subdivision = int(props.beat_division)
        spacing = frames_per_beat / subdivision

        total = int(
            math.floor(
                (
                    props.end_frame
                    - props.start_frame
                )
                / spacing
            )
        )

        for obj_index, obj in enumerate(lights):

            light = obj.data

            if props.clear_existing:
                clear_generated_animation(obj)

            original_color = tuple(light.color)

            keyframe_energy(
                light,
                props.start_frame,
                props.base_energy,
            )

            for event_index in range(total + 1):

                frame = (
                    props.start_frame
                    + event_index * spacing
                )

                if props.randomize_lights:
                    frame += (
                        obj_index
                        * props.stagger_frames
                    )

                if frame > props.end_frame:
                    break

                keyframe_energy(
                    light,
                    frame,
                    props.flash_energy,
                )

                decay = frame + (
                    spacing * props.flash_decay
                )

                if decay <= props.end_frame:
                    keyframe_energy(
                        light,
                        decay,
                        props.base_energy,
                    )

                if props.color_shift:

                    hue = (
                        event_index
                        * props.color_hue_step
                        + obj_index * 0.17
                    )

                    keyframe_color(
                        light,
                        frame,
                        hsv_color(hue),
                    )

                    if decay <= props.end_frame:
                        keyframe_color(
                            light,
                            decay,
                            original_color,
                        )

            light.energy = props.base_energy
            light.color = original_color

        scene.frame_start = props.start_frame
        scene.frame_end = props.end_frame

        self.report(
            {"INFO"},
            "Generated BPM animation."
        )

        return {"FINISHED"}


class ALS_OT_clear(bpy.types.Operator):
    bl_idname = "audio_light_sync.clear"
    bl_label = "Clear Generated Animation"
    bl_description = "Remove animation generated by this add-on"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        lights = [
            obj for obj in context.selected_objects
            if obj.type == "LIGHT"
        ]

        if not lights:
            self.report(
                {"WARNING"},
                "Select lights first."
            )
            return {"CANCELLED"}

        count = 0

        for obj in lights:

            if not obj.animation_data:
                continue

            action = obj.animation_data.action

            if not action:
                continue

            for fc in list(action.fcurves):

                if fc.group and fc.group.name == "Audio Light Sync":
                    action.fcurves.remove(fc)
                    count += 1

            if not action.fcurves:
                obj.animation_data_clear()

        self.report(
            {"INFO"},
            "Removed {} generated curve(s).".format(count)
        )

        return {"FINISHED"}


class ALS_OT_create_rig(bpy.types.Operator):
    bl_idname = "audio_light_sync.create_rig"
    bl_label = "Create Light Rig"
    bl_description = "Create a simple three-light visualizer rig"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        collection_name = "Audio Light Sync"

        collection = bpy.data.collections.get(
            collection_name
        )

        if collection is None:
            collection = bpy.data.collections.new(
                collection_name
            )
            context.scene.collection.children.link(
                collection
            )

        rig = [
            (
                "ALS_Key",
                (0, -3, 3),
                (
                    math.radians(35),
                    0,
                    0,
                ),
                1000,
            ),
            (
                "ALS_Fill",
                (3, 0, 2),
                (
                    math.radians(65),
                    0,
                    math.radians(90),
                ),
                700,
            ),
            (
                "ALS_Back",
                (-3, 1, 3),
                (
                    math.radians(55),
                    0,
                    math.radians(-60),
                ),
                1200,
            ),
        ]

        created = []

        for name, location, rotation, energy in rig:

            data = bpy.data.lights.new(
                name=name,
                type="AREA",
            )

            data.energy = energy
            data.shape = "DISK"
            data.size = 4.0

            obj = bpy.data.objects.new(
                name,
                data,
            )

            collection.objects.link(obj)

            obj.location = location
            obj.rotation_euler = rotation

            created.append(obj)

        bpy.ops.object.select_all(
            action="DESELECT"
        )

        for obj in created:
            obj.select_set(True)

        context.view_layer.objects.active = created[0]

        self.report(
            {"INFO"},
            "Created visualizer light rig."
        )

        return {"FINISHED"}


# ============================================================
# UI
# ============================================================

class ALS_PT_panel(bpy.types.Panel):
    bl_label = "Audio Light Sync"
    bl_idname = "ALS_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Audio Light"

    def draw(self, context):

        layout = self.layout
        props = context.scene.audio_light_sync

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        box = layout.box()
        box.label(
            text="Audio Input",
            icon="SOUND",
        )

        row = box.row()
        row.operator(
            "audio_light_sync.load_audio",
            icon="FILE_FOLDER",
            text="Select WAV",
        )

        if props.audio_file:
            filename = os.path.basename(
                props.audio_file
            )

            box.label(
                text=filename,
                icon="FILE_SOUND",
            )

        row = box.row()
        row.operator(
            "audio_light_sync.analyze",
            icon="MODIFIER",
            text="Analyze Audio",
        )

        if props.audio_loaded:

            box.label(
                text=props.analysis_status,
                icon="CHECKMARK",
            )

            box.label(
                text="Detected Beats: {}".format(
                    props.analyzed_beats
                ),
            )

            if props.estimated_bpm > 0:
                box.label(
                    text="Estimated BPM: {:.1f}".format(
                        props.estimated_bpm
                    ),
                )

        else:
            box.label(
                text=props.analysis_status,
                icon="INFO",
            )

        # ----------------------------------------------------
        # AUDIO RESPONSE
        # ----------------------------------------------------

        box = layout.box()
        box.label(
            text="Audio Response",
            icon="GRAPH",
        )

        box.prop(
            props,
            "audio_response",
        )

        box.prop(
            props,
            "audio_gain",
        )

        box.prop(
            props,
            "audio_smoothing",
        )

        box.prop(
            props,
            "beat_threshold",
        )

        box.label(
            text="Instrument modes are frequency-based approximations.",
            icon="INFO",
        )

        # ----------------------------------------------------
        # LIGHTING
        # ----------------------------------------------------

        box = layout.box()
        box.label(
            text="Light Reaction",
            icon="LIGHT",
        )

        box.prop(
            props,
            "base_energy",
        )

        box.prop(
            props,
            "flash_energy",
        )

        box.prop(
            props,
            "reaction_strength",
        )

        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        box = layout.box()
        box.label(
            text="Color",
            icon="COLOR",
        )

        box.prop(
            props,
            "color_shift",
        )

        if props.color_shift:
            box.prop(
                props,
                "color_hue_step",
            )

        # ----------------------------------------------------
        # TIMELINE
        # ----------------------------------------------------

        box = layout.box()
        box.label(
            text="Timeline",
            icon="PREVIEW_RANGE",
        )

        row = box.row(align=True)

        row.prop(
            props,
            "start_frame",
        )

        row.prop(
            props,
            "end_frame",
        )

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        box = layout.box()
        box.label(
            text="Target",
            icon="OUTLINER_OB_LIGHT",
        )

        box.prop(
            props,
            "selected_only",
        )

        box.prop(
            props,
            "clear_existing",
        )

        box.prop(
            props,
            "randomize_lights",
        )

        if props.randomize_lights:
            box.prop(
                props,
                "stagger_frames",
            )

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        layout.separator()

        row = layout.row()
        row.scale_y = 1.5

        row.operator(
            "audio_light_sync.generate_audio",
            icon="KEY_HLT",
            text="GENERATE FROM AUDIO",
        )

        row = layout.row()

        row.operator(
            "audio_light_sync.generate_bpm",
            icon="TIME",
            text="Generate BPM",
        )

        row = layout.row()

        row.operator(
            "audio_light_sync.clear",
            icon="TRASH",
            text="Clear Generated Animation",
        )

        # ----------------------------------------------------
        # QUICK SETUP
        # ----------------------------------------------------

        box = layout.box()
        box.label(
            text="Quick Setup",
            icon="TOOL_SETTINGS",
        )

        box.operator(
            "audio_light_sync.create_rig",
            icon="LIGHT_AREA",
        )

        # ----------------------------------------------------
        # WORKFLOW
        # ----------------------------------------------------

        box = layout.box()
        box.label(
            text="Workflow",
            icon="INFO",
        )

        box.label(text="1. Select your lights")
        box.label(text="2. Select a WAV file")
        box.label(text="3. Analyze Audio")
        box.label(text="4. Choose response mode")
        box.label(text="5. Generate From Audio")


# ============================================================
# REGISTRATION
# ============================================================

classes = (
    ALS_Properties,
    ALS_OT_load_audio,
    ALS_OT_analyze,
    ALS_OT_generate_audio,
    ALS_OT_generate_bpm,
    ALS_OT_clear,
    ALS_OT_create_rig,
    ALS_PT_panel,
)


def register():

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.audio_light_sync = (
        bpy.props.PointerProperty(
            type=ALS_Properties
        )
    )


def unregister():

    del bpy.types.Scene.audio_light_sync

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
