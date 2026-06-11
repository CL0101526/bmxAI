import cv2
import numpy as np
import ollama
import os
from ultralytics import YOLO
from fastdtw import fastdtw

# =============================================================================
# MODEL CONFIGURATION — toggle this single block when custom weights are ready
# =============================================================================
USE_CUSTOM_MODEL = False  # Flip to True once yolo11n-pose-bmx.pt is trained

if USE_CUSTOM_MODEL:
    MODEL_PATH = 'yolo11n-pose-bmx.pt'
    KEYPOINT_COUNT = 19   # 17 COCO + rear hub (17) + front hub (18)
    REAR_HUB  = 17
    FRONT_HUB = 18
else:
    MODEL_PATH = 'yolo11n-pose.pt'
    KEYPOINT_COUNT = 17   # Standard COCO
    # Ankle indices — front/rear are resolved dynamically per frame (see resolve_hubs)
    LEFT_ANKLE  = 15
    RIGHT_ANKLE = 16
    REAR_HUB  = LEFT_ANKLE   # overridden per-frame by resolve_hubs()
    FRONT_HUB = RIGHT_ANKLE  # overridden per-frame by resolve_hubs()

# Standard anatomical indices (COCO, same in both model variants)
SHOULDER = 5
HIP      = 11
KNEE     = 13

# =============================================================================
# SYNTHETIC BENCHMARK BASELINES
# Derived from elite race footage analysis. These profiles let the app score and
# coach riders immediately — no benchmark .mov files required.
# Format: (mean, std, length) — used to generate a representative 1-D signal.
# =============================================================================
SYNTHETIC_BASELINES = {
    "pump": {
        # Hip-hinge angle (degrees). Elite riders compress to ~100° and extend to ~160°.
        "mean": 128.0, "std": 18.0, "length": 60,
        "scaling": 400,
        "note": "Hip hinge angle: lower average = rider is staying buckled and loading the roller correctly."
    },
    "manual": {
        # Normalised horizontal hip-over-axle offset. Elite manual: hips slightly behind centre.
        "mean": -0.08, "std": 0.04, "length": 50,
        "scaling": 50,
        "note": "Hip offset behind rear axle: a higher (positive) rider value means hips are trapped forward over the bottom bracket."
    },
    "step_up_jump": {
        # Normalised front-hub/rear-hub vertical pitch. Elite: near-flat to slightly nose-up on take-off.
        "mean": 0.06, "std": 0.05, "length": 45,
        "scaling": 50,
        "note": "Bike pitch (front hub vs rear hub): higher value = front wheel dropping significantly below the rear — the rider is looping the take-off."
    },
    "double_jump": {
        "mean": 0.04, "std": 0.05, "length": 50,
        "scaling": 50,
        "note": "Bike pitch (front hub vs rear hub): higher value = front wheel dropping significantly below the rear — the rider is looping the take-off."
    },
}

def get_synthetic_baseline(skill_type: str) -> np.ndarray:
    """
    Generates a smooth synthetic benchmark signal when no benchmark .mov is present.
    Uses a seeded RNG so the same skill always produces the same baseline.
    """
    cfg = SYNTHETIC_BASELINES[skill_type]
    rng = np.random.default_rng(seed=42)
    # Gaussian noise around the elite mean, then smoothed to look like a real curve
    raw = rng.normal(cfg["mean"], cfg["std"], cfg["length"]).astype(np.float32)
    # Light smoothing with a rolling window
    kernel = np.ones(5) / 5
    smoothed = np.convolve(raw, kernel, mode='same')
    return smoothed


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def calculate_angle(a, b, c) -> float:
    """Calculates the 2-D angle at joint b, given three joint coordinates."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = (np.arctan2(c[1] - b[1], c[0] - b[0]) -
               np.arctan2(a[1] - b[1], a[0] - b[0]))
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def resolve_hubs(keypoints: np.ndarray, riding_direction: int = 1):
    """
    Resolves which ankle is the FRONT hub and which is the REAR hub.

    This eliminates switch-foot / goofy-foot data skew.

    When the custom model is active, hubs are fixed anatomical points (indices 17/18)
    and this function is bypassed entirely.

    Args:
        keypoints:         COCO keypoint array, shape (17, 2).
        riding_direction:  +1 = rider moving left→right in frame (default).
                           -1 = rider moving right→left.
    Returns:
        (front_hub_xy, rear_hub_xy) as numpy arrays.
    """
    if USE_CUSTOM_MODEL:
        # Custom weights anchor hubs to the machine — no ambiguity.
        return keypoints[FRONT_HUB], keypoints[REAR_HUB]

    left_ankle  = keypoints[LEFT_ANKLE]
    right_ankle = keypoints[RIGHT_ANKLE]

    # The ankle with the larger X coordinate is in front when riding left→right.
    # Flip the comparison for right→left riding direction.
    if riding_direction >= 0:
        if left_ankle[0] >= right_ankle[0]:
            return left_ankle, right_ankle   # left foot forward
        else:
            return right_ankle, left_ankle   # right foot forward
    else:
        if left_ankle[0] <= right_ankle[0]:
            return left_ankle, right_ankle
        else:
            return right_ankle, left_ankle


def detect_riding_direction(raw_frames: list) -> int:
    """
    Infers whether the rider travels left→right (+1) or right→left (-1)
    from the first and last valid hip positions in the clip.
    """
    if len(raw_frames) < 2:
        return 1
    first_hip_x = raw_frames[0]['hip'][0]
    last_hip_x  = raw_frames[-1]['hip'][0]
    return 1 if last_hip_x >= first_hip_x else -1


# =============================================================================
# CORE EXTRACTION ENGINE
# =============================================================================

def extract_metrics_and_phases(video_path: str, skill_type: str):
    """
    Runs YOLO11 pose estimation frame-by-frame and extracts skill-specific
    1-D metric sequences plus phase-trigger frame indices.
    """
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(video_path)

    raw_frames_data = []
    metrics         = []
    trigger_lip_frame = None
    airborne_frame    = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)

        for result in results:
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                continue

            keypoints = result.keypoints.xy[0].cpu().numpy()

            # Guard: require enough keypoints for whichever model is active
            if len(keypoints) < KEYPOINT_COUNT:
                continue

            try:
                shoulder = keypoints[SHOULDER]
                hip      = keypoints[HIP]
                knee     = keypoints[KNEE]

                if list(hip) == [0., 0.]:
                    continue

                # Pass a placeholder direction; we'll refine after the first pass
                front_hub, rear_hub = resolve_hubs(keypoints, riding_direction=1)

                raw_frames_data.append({
                    'shoulder': shoulder, 'hip': hip, 'knee': knee,
                    'rear_hub': rear_hub, 'front_hub': front_hub,
                    'keypoints': keypoints
                })

                rider_height = np.max(keypoints[:, 1]) - np.min(keypoints[:, 1])
                if rider_height == 0:
                    continue

                # ── PHASE DETECTION ──────────────────────────────────────────
                if len(raw_frames_data) > 3 and "jump" in skill_type:
                    prev_front_y = raw_frames_data[-3]['front_hub'][1]
                    curr_front_y = front_hub[1]

                    if trigger_lip_frame is None and (prev_front_y - curr_front_y) > 15:
                        trigger_lip_frame = len(raw_frames_data) - 1

                    elif trigger_lip_frame is not None and airborne_frame is None:
                        prev_rear_y = raw_frames_data[-3]['rear_hub'][1]
                        curr_rear_y = rear_hub[1]
                        if (prev_rear_y - curr_rear_y) > 15:
                            airborne_frame = len(raw_frames_data) - 1

                # ── METRIC EXTRACTION ─────────────────────────────────────────
                if "pump" in skill_type:
                    if list(shoulder) != [0., 0.] and list(knee) != [0., 0.]:
                        metrics.append(calculate_angle(shoulder, hip, knee))

                elif skill_type == "manual":
                    if list(rear_hub) != [0., 0.]:
                        metrics.append((hip[0] - rear_hub[0]) / rider_height)

                elif "jump" in skill_type:
                    if list(front_hub) != [0., 0.] and list(rear_hub) != [0., 0.]:
                        pitch_value = (front_hub[1] - rear_hub[1]) / rider_height
                        metrics.append(pitch_value)

            except IndexError:
                continue

    cap.release()

    # Re-resolve hub assignments now that we know the actual riding direction
    if raw_frames_data:
        direction = detect_riding_direction(raw_frames_data)
        if direction != 1 and not USE_CUSTOM_MODEL:
            # Re-extract metrics with the corrected direction
            metrics = []
            for fd in raw_frames_data:
                kp           = fd['keypoints']
                front_hub, rear_hub = resolve_hubs(kp, direction)
                rider_height = np.max(kp[:, 1]) - np.min(kp[:, 1])
                if rider_height == 0:
                    continue
                shoulder = fd['shoulder']
                hip      = fd['hip']
                knee     = fd['knee']

                if "pump" in skill_type and list(shoulder) != [0., 0.]:
                    metrics.append(calculate_angle(shoulder, hip, knee))
                elif skill_type == "manual" and list(rear_hub) != [0., 0.]:
                    metrics.append((hip[0] - rear_hub[0]) / rider_height)
                elif "jump" in skill_type and list(front_hub) != [0., 0.]:
                    metrics.append((front_hub[1] - rear_hub[1]) / rider_height)

    return np.array(metrics, dtype=np.float64).flatten(), raw_frames_data, trigger_lip_frame, airborne_frame


# =============================================================================
# AI COACHING PROMPT
# =============================================================================

def get_ai_coaching_feedback(skill_type: str, score: float,
                              avg_pro: float, avg_rider: float,
                              pop_speed_ratio: float = None) -> str:
    """
    Sends precise geometric deltas to the local Llama 3 instance and returns
    2-3 concise, race-specific coaching cues.
    """
    delta = avg_rider - avg_pro
    direction_word = "higher" if delta > 0 else "lower"

    skill_context = {
        "pump": (
            "The metric is the hip-hinge angle measured at the hip joint during a roller or rhythm section. "
            "Elite BMX racers drive their hips aggressively downward into the downslope of each roller, "
            "compressing the bike into the transition and snapping upright to accelerate out. "
            "A higher rider value means they are riding tall and passive — floating over the roller rather than loading it. "
            "A lower value is generally correct but can indicate the rider is over-tucking without extension timing."
        ),
        "manual": (
            "The metric is the normalised horizontal distance between the rider's hip and the rear axle. "
            "A correct race manual has the hips loaded just behind the rear axle, allowing the rider to "
            "unweight the front wheel through a flat or roll-out section without braking. "
            "A higher (more positive) value means the hips are trapped forward over the bottom bracket, "
            "killing the balance point and forcing the rider to dab the brake."
        ),
        "step_up_jump": (
            "The metric is the normalised vertical pitch of the bike (front hub minus rear hub). "
            "An elite BMX racer scrubs the step-up by absorbing the lip with bent arms and legs, "
            "keeping the frame close to flat or slightly nose-up to stay low and carry speed. "
            "A higher positive value means the front wheel is diving well below the rear, indicating "
            "the rider is looping the take-off or failing to commit their weight forward off the lip."
        ),
        "double_jump": (
            "The metric is the normalised vertical pitch of the bike (front hub minus rear hub). "
            "On a double, the goal is a late pop off the knuckle — legs explode at the very last moment "
            "to launch the bike with a flat trajectory, not a steep arc. "
            "A higher positive value means the front wheel is diving significantly — the rider is jumping early, "
            "arcing high and slow instead of driving low and fast to the second roller."
        ),
    }

    context = skill_context.get(skill_type, "")

    pop_block = ""
    if pop_speed_ratio is not None:
        pop_block = (
            f"\n- Takeoff Explosiveness Ratio (rider vs pro): {pop_speed_ratio:.2f}  "
            f"(1.0 = matched elite pop speed; below 1.0 = rider is absorbing passively rather than snapping the legs)"
        )

    prompt = f"""You are a senior BMX racing coach reviewing data output from a computer vision pose analysis system.
Your job is to translate the following numbers into 2-5, coaching cues.

Rules:
- Write like a coach talking to a young rider trackside. medium sized, clear sentences.
- Never mention numbers, ratios, units, averages, or scores. Translate the data into physical feelings and actions only.
- No technical jargon. Instead of "knuckle" say "the top of the jump". Instead of "pitch profile" say "how the bike is sitting in the air". Instead of "pop speed ratio" say "how fast you push off".
- No emojis, no headers, no generic praise like "Great job!"
- Every cue must be something the rider can physically try on their very next run.
- never use knuckle or second roller use take-off and landing instead 
- DO NOT include 'Here are your coaching cues:'
- DO NOT use speech marks you are an ai tool not a human speaking  
- ONLY use race BMX language such as take off, landing, pop, scrub, manual, pump and so on
- if video displays correct teqnique (score >= 45) tell them what theyre doing right and what to keep doing instead of giving improvement cues. Praise is coaching too!
Skill: {skill_type.replace("_", " ").title()}
Overall Technique Match Score: {score:.1f} / 100
Elite Pro Average Metric: {avg_pro:.4f}
Rider Average Metric:     {avg_rider:.4f}
Rider metric is {abs(delta):.4f} units {direction_word} than the elite benchmark.{pop_block}

Technical Context:
{context}

Provide your 2-3 coaching cues now:"""

    try:
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return (
            f"Geometry analysis complete. Local language model unavailable: {e}\n\n"
            f"Raw delta: rider metric is {abs(delta):.4f} units {direction_word} than the elite benchmark."
        )


# =============================================================================
# BENCHMARK VIDEO MAP
# =============================================================================

# Each skill has an ltr (left→right) and rtl (right→left) variant.
# Name your files accordingly: pro_double_ltr.mov / pro_double_rtl.mov
# Falls back to a non-directional file, then to the synthetic baseline.
BENCHMARK_VIDEOS = {
    "pump":         {"ltr": "benchmarks/pro_pump_ltr.mov",         "rtl": "benchmarks/pro_pump_rtl.mov",         "any": "benchmarks/pro_pump.mov"},
    "manual":       {"ltr": "benchmarks/pro_manual_ltr.mov",       "rtl": "benchmarks/pro_manual_rtl.mov",       "any": "benchmarks/pro_manual.mov"},
    "step_up_jump": {"ltr": "benchmarks/pro_step_up_ltr.mov",      "rtl": "benchmarks/pro_step_up_rtl.mov",      "any": "benchmarks/pro_step_up.mov"},
    "double_jump":  {"ltr": "benchmarks/pro_double_ltr.mov",       "rtl": "benchmarks/pro_double_rtl.mov",       "any": "benchmarks/pro_double.mov"},
}

def select_benchmark(skill_type: str, rider_direction: int) -> tuple:
    """
    Picks the best matching benchmark file for the rider's detected direction.
    Falls back: direction-matched → any → synthetic.
    Returns (path_or_None, source_label).
    """
    variants = BENCHMARK_VIDEOS.get(skill_type, {})
    direction_key = "ltr" if rider_direction >= 0 else "rtl"
    
    for key in [direction_key, "ltr", "rtl", "any"]:
        path = variants.get(key)
        if path and os.path.exists(path):
            label = f"benchmark video ({key})"
            return path, label
    
    return None, "synthetic elite baseline (no benchmark .mov found)"


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def analyze_bmx_skill(rider_video, skill_type: str) -> str:
    if not rider_video:
        return "Error: Upload a rider practice clip to begin analysis."

    # ── RIDER METRICS FIRST (need direction before selecting benchmark) ────────
    rider_metrics, rider_raw, rider_lip, rider_air = extract_metrics_and_phases(rider_video, skill_type)
    rider_direction = detect_riding_direction(rider_raw)

    # ── SELECT MATCHING BENCHMARK ─────────────────────────────────────────────
    pro_video, source_label = select_benchmark(skill_type, rider_direction)
    use_synthetic = pro_video is None

    # ── PRO METRICS ──────────────────────────────────────────────────────────
    if use_synthetic:
        pro_metrics = get_synthetic_baseline(skill_type)
        pro_raw     = []
        pro_lip     = None
        pro_air     = None
    else:
        pro_metrics, pro_raw, pro_lip, pro_air = extract_metrics_and_phases(pro_video, skill_type)

    if len(pro_metrics) == 0 or len(rider_metrics) == 0:
        return (
            "Error: Could not track the necessary body points. "
            "Ensure the camera angle is completely side-on and the full rider is visible."
        )

    pop_speed_ratio = None

    # ── JUMP-SPECIFIC PHASE CALCULATIONS ─────────────────────────────────────
    if "jump" in skill_type:
        # Only calculate pop speed if both triggers fired within a plausible window.
        # 3–25 frames at 30fps = ~0.1s to ~0.8s — anything outside is a false trigger.
        rider_window = (rider_air - rider_lip) if (rider_lip and rider_air) else 0
        pro_window   = (pro_air - pro_lip)     if (pro_lip and pro_air)     else 0
        triggers_valid = (
            not use_synthetic and
            3 <= rider_window <= 25 and
            3 <= pro_window   <= 25
        )

        if triggers_valid:
            pro_lip_angle = calculate_angle(
                pro_raw[pro_lip]['shoulder'], pro_raw[pro_lip]['hip'], pro_raw[pro_lip]['knee'])
            pro_air_angle = calculate_angle(
                pro_raw[pro_air]['shoulder'], pro_raw[pro_air]['hip'], pro_raw[pro_air]['knee'])
            pro_pop_speed = (pro_air_angle - pro_lip_angle) / max(pro_air - pro_lip, 1)

            rider_lip_angle = calculate_angle(
                rider_raw[rider_lip]['shoulder'], rider_raw[rider_lip]['hip'], rider_raw[rider_lip]['knee'])
            rider_air_angle = calculate_angle(
                rider_raw[rider_air]['shoulder'], rider_raw[rider_air]['hip'], rider_raw[rider_air]['knee'])
            rider_pop_speed = (rider_air_angle - rider_lip_angle) / max(rider_air - rider_lip, 1)

            if pro_pop_speed != 0:
                pop_speed_ratio = rider_pop_speed / pro_pop_speed

        # Style buffer: strip the middle 40% to ignore mid-air whips/bars
        def apply_style_buffer(arr: np.ndarray) -> np.ndarray:
            n = len(arr)
            return np.concatenate([arr[:int(n * 0.3)], arr[int(n * 0.7):]]).flatten()

        rider_metrics = apply_style_buffer(rider_metrics)
        pro_metrics   = apply_style_buffer(pro_metrics)

    # ── DTW SCORING ───────────────────────────────────────────────────────────
    # Z-score both sequences so scoring is purely about shape, not magnitude.
    # Without this, tiny normalised decimals produce near-zero DTW distances
    # regardless of how different the technique actually is.
    def zscore(arr):
        std = float(np.std(arr))
        return (arr - np.mean(arr)) / std if std > 1e-6 else arr - np.mean(arr)

    pro_z   = zscore(pro_metrics)
    rider_z = zscore(rider_metrics)
    dtw_distance, _ = fastdtw(pro_z, rider_z, dist=lambda a, b: abs(float(a) - float(b)))

    # Per-frame average deviation in standard-deviation units.
    # A deviation of 2 SD per frame on average = score of 0.
    per_frame = dtw_distance / max(len(rider_z), 1)
    score     = max(0.0, 100.0 - per_frame * 50.0)

    avg_pro   = float(np.mean(pro_metrics))
    avg_rider = float(np.mean(rider_metrics))

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    header = (
        f"Technique Match Score:  {score:.1f} / 100\n"
        f"Compared against:       {source_label}\n"
        f"{'─' * 48}\n"
    )

    lm_feedback = get_ai_coaching_feedback(
        skill_type, score, avg_pro, avg_rider, pop_speed_ratio
    )

    return header + lm_feedback


# =============================================================================
# FASTAPI SERVER
# =============================================================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import tempfile, shutil, uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html") as f:
        return f.read()

@app.post("/analyze")
async def analyze(video: UploadFile = File(...), skill_type: str = Form(...)):
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name
    try:
        result = analyze_bmx_skill(tmp_path, skill_type)
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)

