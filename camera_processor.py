"""
camera_processor.py — Background camera capture, ML inference, and MJPEG streaming.

Threading model (producer / consumer):
    - One daemon thread per camera runs _process_camera().  It captures frames, runs MediaPipe inference, draws overlays, and encodes the result to JPEG, storing it in _jpeg_cache keyed by cam_key.
    - generate_frames() is a generator called by Flask's streaming response. It reads from _jpeg_cache without touching the camera or the ML models, so it never stalls waiting for a slow frame.
    - _cache_lock protects _jpeg_cache; state_lock protects all per-camera state dicts (timers, recording flags, alert paths).

State machine (per camera):
    "IDLE"               — face present, no suspicious activity.
    "No Face Detected"   — no face for ≥ 3 s; alert fired and evidence recorded.
    "Hand Raised"        — hand visible while face is present for ≥ 3 s; alert fired.

Call init(flask_app, db, Alert) once at startup to inject Flask dependencies before any camera thread is started.
"""

# Import necessary libraries
import cv2
import datetime
import numpy as np
import mediapipe as mp
import os
import landmarker
import time
import threading
from collections import deque
from threading import Lock


# Configurable thresholds and settings — override via .env
NO_FACE_ALERT_SECONDS = int(os.environ.get("NO_FACE_ALERT_SECONDS", 3))
HAND_RAISED_ALERT_SECONDS = int(os.environ.get("HAND_RAISED_ALERT_SECONDS", 3))
FACE_DETECTION_CONFIDENCE = float(os.environ.get("FACE_DETECTION_CONFIDENCE", 0.5))
HAND_DETECTION_CONFIDENCE = float(os.environ.get("HAND_DETECTION_CONFIDENCE", 0.5))
EVIDENCE_VIDEO_FPS = float(os.environ.get("EVIDENCE_VIDEO_FPS", 15.0))
EVIDENCE_VIDEO_CODEC = os.environ.get("EVIDENCE_VIDEO_CODEC", "mp4v")

# Videowriter setup
forucc = cv2.VideoWriter_fourcc(*EVIDENCE_VIDEO_CODEC)

# Create output directory for evidence videos if it doesn't exist.
output_dir = os.path.join(landmarker.BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)

# Per-camera state dicts — all keyed by cam_key (int index or URL string).
# t1_by_cam:          datetime when the "no face" window started for each camera.
# t1_hand_by_cam:     datetime when the "hand raised" window started for each camera.
# state_by_cam:       current state-machine label ("IDLE", "No Face Detected", "Hand Raised").
# recording_by_cam:   True while pre-alert frames are being buffered for evidence.
# evidence_queue_by_cam: rolling deque of raw frames awaiting MP4 export.
# alert_evidence_paths: basenames of saved evidence MP4s (shown on the alerts page).
t1_by_cam = {}
t1_hand_by_cam = {}
state_by_cam = {}
recording_by_cam = {}
evidence_queue_by_cam = {}
alert_evidence_paths = []
# Single lock protects all of the dicts above to avoid race conditions between
# the capture thread and Flask request handlers reading state.
state_lock = Lock()

# MJPEG streaming cache — _process_camera() writes the latest JPEG bytes here;
# generate_frames() reads from it.  A separate lock keeps writes and reads atomic.
_jpeg_cache: dict = {}
_cache_lock = Lock()
# Registry of running processor threads so _ensure_processor() can check liveness
# without starting duplicates.
_processor_threads: dict = {}
_threads_lock = Lock()

# Run ML inference only once every N frames to improve throughput.
# Frames in between reuse the previous inference result — good enough at 30 fps
# because subjects don't move faster than the detector's effective range.
INFERENCE_EVERY_N = int(os.environ.get("INFERENCE_EVERY_N", 2))

# Maximum frames per second delivered to the browser.
STREAM_TARGET_FPS = int(os.environ.get("STREAM_TARGET_FPS", 30))

# JPEG encode parameters (quality 70 balances size vs. fidelity).
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, int(os.environ.get("JPEG_QUALITY", 70))]

# Injected Flask references — set by calling init().
# Stored at module level so the background threads (which have no request context)
# can open an app context and write to the database.
_app = None
_db = None
_Alert = None


def init(flask_app, db, Alert) -> None:
    """
    Injects the Flask app, SQLAlchemy db, and Alert model so that
    _process_camera can persist alerts without importing from app.py.

    :param flask_app: The Flask application instance.
    :param db: The SQLAlchemy database instance.
    :param Alert: The Alert database model class.
    :return: None
    """

    global _app, _db, _Alert
    _app = flask_app
    _db = db
    _Alert = Alert


def _process_camera(cam_key) -> None:
    """
    Background thread: captures frames, runs ML inference every
    INFERENCE_EVERY_N frames, draws annotations, and stores the latest
    JPEG bytes in _jpeg_cache for generate_frames() to stream independently.

    :param cam_key: Integer device index or URL/path string for the camera.
    :return: None
    """

    global t1_by_cam, t1_hand_by_cam, state_by_cam, recording_by_cam, evidence_queue_by_cam, forucc, alert_evidence_paths

    # CAP_DSHOW (DirectShow) gives lower latency and more reliable enumeration on
    # Windows for USB/built-in cameras.  For IP streams or file paths we fall back
    # to CAP_ANY so OpenCV picks the best available backend automatically.
    backend = cv2.CAP_DSHOW if isinstance(cam_key, int) else cv2.CAP_ANY
    cam = cv2.VideoCapture(cam_key, backend)
    attempts = 0
    # Retry a few times in case the device is briefly busy (e.g. another app just
    # released it) before giving up.
    while not cam.isOpened():
        attempts += 1
        if attempts > 5:
            print(f"Camera {cam_key}: failed to open after 5 attempts.")
            return
        cam = cv2.VideoCapture(cam_key, backend)
        time.sleep(1)
    print(f"Camera {cam_key}: ready")

    with state_lock:
        state_by_cam[cam_key] = "IDLE"
        recording_by_cam[cam_key] = False
        evidence_queue_by_cam[cam_key] = deque()

    def alert(alert_type: str, frame: np.ndarray) -> None:
        print(f"ALERT: {alert_type} at {datetime.datetime.now()}")
        _, buffer = cv2.imencode(".png", frame)
        new_alert = _Alert(
            alert_type=alert_type,
            alert_image=buffer.tobytes(),
            cam_no=str(cam_key),
            timestamp=datetime.datetime.now(),
        )
        with _app.app_context():
            _db.session.add(new_alert)
            _db.session.commit()

    def _save_evidence(queued_frames: list) -> None:
        """Writes evidence frames to MP4; runs in its own daemon thread so it
        never blocks the capture loop."""
        release_path = os.path.join(
            output_dir,
            f"evidence_cam{cam_key}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
        )
        h, w = queued_frames[0].shape[:2]
        out = cv2.VideoWriter(release_path, forucc, EVIDENCE_VIDEO_FPS, (w, h))
        if not out.isOpened():
            print(f"Failed to open VideoWriter for {release_path}")
            return
        for f in queued_frames:
            out.write(f)
        out.release()
        if os.path.exists(release_path) and os.path.getsize(release_path) > 0:
            with state_lock:
                base = os.path.basename(release_path)
                if base not in alert_evidence_paths:
                    alert_evidence_paths.append(base)
        else:
            print(f"Failed to save evidence video for camera {cam_key}")

    def stop_recording() -> None:
        """Grabs queued frames under the lock, then offloads MP4 writing to a
        daemon thread so the capture loop is never blocked by disk I/O."""
        with state_lock:
            queued_frames = list(evidence_queue_by_cam.get(cam_key, deque()))
            if not queued_frames:
                return
            recording_by_cam[cam_key] = False
            evidence_queue_by_cam[cam_key] = deque()
        threading.Thread(
            target=_save_evidence, args=(queued_frames,), daemon=True
        ).start()

    frame_counter = 0
    last_face_detected = False
    last_landmark_result = None
    last_hand_result = None
    fps_history: deque = deque(maxlen=15)  # rolling window for smoothed FPS
    t_prev = time.monotonic()

    try:
        with (
            landmarker.FaceDetector.create_from_options(
                landmarker.face_detector_options
            ) as face_detector,
            landmarker.FaceLandmarker.create_from_options(
                landmarker.face_landmark_options
            ) as face_landmarker,
            landmarker.HandLandmarker.create_from_options(
                landmarker.hand_landmark_options
            ) as hand_landmarker,
        ):
            while True:
                ret, frame = cam.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                # Record the pre-flip raw frame for evidence
                with state_lock:
                    if recording_by_cam.get(cam_key):
                        evidence_queue_by_cam[cam_key].append(frame)

                frame = cv2.flip(frame, 1)
                frame_counter += 1
                run_inference = frame_counter % INFERENCE_EVERY_N == 0

                # cv2.cvtColor is surprisingly expensive at full resolution, so we
                # only convert to RGB when we actually need it:
                #   • on inference frames (MediaPipe requires RGB input), or
                #   • when there are active detections whose landmarks need to be
                #     redrawn on the current (non-inference) frame.
                # If none of these conditions hold we skip the conversion entirely
                # and pass None — callers check for None before using rgb_frame.
                needs_rgb = (
                    run_inference
                    or (last_face_detected and last_landmark_result is not None)
                    or (
                        last_hand_result is not None and last_hand_result.hand_landmarks
                    )
                )
                rgb_frame = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if needs_rgb else None
                )

                if run_inference and rgb_frame is not None:
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb_frame
                    )

                    fdr = face_detector.detect(mp_image)
                    if not fdr.detections:
                        face_detected = False
                    elif (
                        fdr.detections[0].categories[0].score
                        > FACE_DETECTION_CONFIDENCE
                    ):
                        face_detected = True
                    else:
                        face_detected = False

                    last_landmark_result = (
                        face_landmarker.detect(mp_image) if face_detected else None
                    )
                    last_hand_result = hand_landmarker.detect(mp_image)
                    last_face_detected = face_detected
                else:
                    face_detected = last_face_detected

                drew_anything = False

                # Face landmarks + state machine
                if face_detected and last_landmark_result is not None:
                    landmarker.draw_face_landmarks_on_image(
                        rgb_frame, last_landmark_result
                    )
                    drew_anything = True
                    with state_lock:
                        t1_by_cam.pop(cam_key, None)
                        state_by_cam[cam_key] = "IDLE"
                    stop_recording()
                elif not face_detected:
                    with state_lock:
                        if cam_key not in t1_by_cam:
                            t1_by_cam[cam_key] = datetime.datetime.now()
                        no_face_start = t1_by_cam[cam_key]
                        recording_by_cam[cam_key] = True
                        current_state = state_by_cam.get(cam_key)
                    t2 = datetime.datetime.now()
                    if (
                        t2 - no_face_start
                        >= datetime.timedelta(seconds=NO_FACE_ALERT_SECONDS)
                        and current_state != "No Face Detected"
                    ):
                        alert("No Face Detected", frame)
                        with state_lock:
                            state_by_cam[cam_key] = "No Face Detected"
                            t1_by_cam[cam_key] = t2

                # Hand landmarks + state machine
                if (
                    last_hand_result is not None
                    and last_hand_result.hand_landmarks
                    and last_hand_result.handedness[0][0].score
                    > HAND_DETECTION_CONFIDENCE
                ):
                    if rgb_frame is None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    landmarker.draw_hand_landmarks_on_image(rgb_frame, last_hand_result)
                    drew_anything = True
                    with state_lock:
                        if cam_key not in t1_hand_by_cam:
                            t1_hand_by_cam[cam_key] = datetime.datetime.now()
                        hand_start = t1_hand_by_cam[cam_key]
                        current_state = state_by_cam.get(cam_key, "")
                    t2 = datetime.datetime.now()
                    if (
                        t2 - hand_start
                        >= datetime.timedelta(seconds=HAND_RAISED_ALERT_SECONDS)
                        and current_state == "IDLE"
                        and face_detected
                    ):
                        alert("Hand Raised", frame)
                        with state_lock:
                            state_by_cam[cam_key] = "Hand Raised"
                            t1_hand_by_cam[cam_key] = t2
                else:
                    with state_lock:
                        t1_hand_by_cam.pop(cam_key, None)

                # All landmark drawing functions modify rgb_frame in-place.
                # We defer the single BGR conversion to here so we only pay the
                # cost once per frame regardless of how many overlays were drawn.
                if drew_anything:
                    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # Smoothed FPS using a rolling window (avoids single-frame spikes)
                t_now = time.monotonic()
                fps_history.append(1.0 / max(t_now - t_prev, 1e-6))
                t_prev = t_now
                fps = sum(fps_history) / len(fps_history)

                with state_lock:
                    state_label = state_by_cam.get(cam_key, "IDLE")

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"State: {state_label}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                _, buf = cv2.imencode(".jpg", frame, JPEG_PARAMS)
                with _cache_lock:
                    _jpeg_cache[cam_key] = buf.tobytes()
    finally:
        cam.release()
        with _cache_lock:
            _jpeg_cache.pop(cam_key, None)


def _ensure_processor(cam_key) -> None:
    """
    Starts the background processing thread for cam_key if not already running.

    :param cam_key: Integer device index or URL/path string for the camera.
    :return: None
    """

    with _threads_lock:
        t = _processor_threads.get(cam_key)
        if t is None or not t.is_alive():
            t = threading.Thread(
                target=_process_camera,
                args=(cam_key,),
                daemon=True,
                name=f"cam-processor-{cam_key}",
            )
            _processor_threads[cam_key] = t
            t.start()


def generate_frames(cam_key):
    """
    Streams the latest processed JPEG for cam_key at up to STREAM_TARGET_FPS.
    All capture and ML inference is handled by a background thread so this
    generator never blocks on camera I/O or model inference — giving smooth,
    stable output regardless of how long a single inference takes.

    :param cam_key: Integer device index or URL/path string for the camera.
    :return: A generator yielding the video frames.
    """

    _ensure_processor(cam_key)
    interval = 1.0 / STREAM_TARGET_FPS

    while True:
        t0 = time.monotonic()
        with _cache_lock:
            jpeg = _jpeg_cache.get(cam_key)
        if jpeg is not None:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        elapsed = time.monotonic() - t0
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
