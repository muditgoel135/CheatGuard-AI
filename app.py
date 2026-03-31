# Import necessary libraries
import base64
import json
from flask import (
    Flask,
    Response,
    render_template,
    send_file,
    send_from_directory,
    redirect,
    request,
)
from flask_sqlalchemy import SQLAlchemy
import cv2
import datetime
import numpy as np
import mediapipe as mp
import os
import landmarker
from collections import deque
from io import BytesIO
import zipfile
import time
import threading
from threading import Lock


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
db = SQLAlchemy(app)


# Database model for alerts
class Alert(db.Model):
    """
    Database model for storing alerts generated.
    """

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    cam_no = db.Column(db.String(100), nullable=False, default=0)
    alert_type = db.Column(db.String(100), nullable=False)
    alert_image = db.Column(db.LargeBinary, nullable=False)

    def __repr__(self) -> str:
        """
        String representation of the Alert object for debugging and display purposes.

        :return: A string representation of the Alert object, including camera number, timestamp, and alert type.
        :rtype: str
        """

        return f"Alert('{self.cam_no}, {self.timestamp}', '{self.alert_type}')"


# Create the database tables
with app.app_context():
    db.create_all()


def scan_local_cameras(max_index: int = 10) -> list:
    """
    Scans for locally connected cameras across a range of indices.
    Unlike a sequential scan, this does not stop at the first gap, so
    non-sequential USB devices on Windows are detected correctly.

    :param max_index: Highest device index to probe (exclusive).
    :type max_index: int
    :return: List of working device indices.
    :rtype: list[int]
    """

    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                found.append(i)
        else:
            cap.release()
    return found


# Path to persistent camera registry
CAMERAS_JSON = os.path.join(landmarker.BASE_DIR, "cameras.json")


def load_cameras() -> list:
    """
    Loads the camera registry from cameras.json.
    On first run (file absent), auto-detects local cameras and writes the file.

    :return: List of camera dicts with keys: id, source, name.
    :rtype: list[dict]
    """

    if os.path.exists(CAMERAS_JSON):
        with open(CAMERAS_JSON, "r") as f:
            return json.load(f)

    # First run: auto-detect and persist
    cams = []
    for idx in scan_local_cameras():
        cams.append({"id": idx, "source": idx, "name": f"Camera {idx + 1}"})
    save_cameras(cams)
    return cams


def save_cameras(cams: list) -> None:
    """
    Persists the camera registry to cameras.json.

    :param cams: List of camera dicts to save.
    :type cams: list[dict]
    """

    with open(CAMERAS_JSON, "w") as f:
        json.dump(cams, f, indent=2)


def _next_cam_id(cams: list) -> int:
    """
    Returns the next available camera ID.

    :param cams: Current camera list.
    :type cams: list[dict]
    :return: Next available integer ID.
    :rtype: int
    """

    return max((c["id"] for c in cams), default=-1) + 1


# Videowriter setup
forucc = cv2.VideoWriter_fourcc(*"mp4v")


# Create output directory for evidence videos if it doesn't exist.
output_dir = os.path.join(landmarker.BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)


# Track no-face timers per camera source (int index or URL/path).
t1_by_cam = {}
t1_hand_by_cam = {}
state_by_cam = {}
recording_by_cam = {}
evidence_queue_by_cam = {}  # Optional: To store recent frames for evidence if needed.
alert_evidence_paths = []
state_lock = Lock()

# Per-camera streaming cache: background threads write, generators read.
_jpeg_cache: dict = {}
_cache_lock = Lock()
_processor_threads: dict = {}
_threads_lock = Lock()

# Run ML inference only once every N frames to improve throughput.
INFERENCE_EVERY_N = 2

# Maximum frames per second delivered to the browser.
STREAM_TARGET_FPS = 30

# JPEG encode parameters (quality 70 balances size vs. fidelity).
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 70]


def _process_camera(cam_key) -> None:
    """
    Background thread: captures frames, runs ML inference every
    INFERENCE_EVERY_N frames, draws annotations, and stores the latest
    JPEG bytes in _jpeg_cache for generate_frames() to stream independently.

    :param cam_key: Integer device index or URL/path string for the camera.
    """
    global t1_by_cam, t1_hand_by_cam, state_by_cam, recording_by_cam, evidence_queue_by_cam, forucc, alert_evidence_paths

    # Use CAP_DSHOW on Windows for integer indices (lower latency); FFMPEG for URLs.
    backend = cv2.CAP_DSHOW if isinstance(cam_key, int) else cv2.CAP_ANY
    cam = cv2.VideoCapture(cam_key, backend)
    attempts = 0
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
        new_alert = Alert(
            alert_type=alert_type,
            alert_image=buffer.tobytes(),
            cam_no=str(cam_key),
            timestamp=datetime.datetime.now(),
        )
        with app.app_context():
            db.session.add(new_alert)
            db.session.commit()

    def _save_evidence(queued_frames: list) -> None:
        """Writes evidence frames to MP4; runs in its own daemon thread so it
        never blocks the capture loop."""
        release_path = os.path.join(
            output_dir,
            f"evidence_cam{cam_key}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
        )
        h, w = queued_frames[0].shape[:2]
        out = cv2.VideoWriter(release_path, forucc, 15.0, (w, h))
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

                # Convert to RGB only when inference or active detections require it;
                # skips the conversion on quiet frames with no detections.
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
                    elif fdr.detections[0].categories[0].score > 0.5:
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
                        t2 - no_face_start >= datetime.timedelta(seconds=3)
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
                    and last_hand_result.handedness[0][0].score > 0.5
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
                        t2 - hand_start >= datetime.timedelta(seconds=3)
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
                        if state_by_cam.get(cam_key) == "Hand Raised" and face_detected:
                            state_by_cam[cam_key] = "IDLE"

                # Single BGR conversion after all in-place RGB drawing
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
    """Starts the background processing thread for cam_key if not already running."""
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


# Flask routes
@app.route("/")
def index() -> str:
    """
    Renders the main page with video feeds and alert counts for each camera.

    :return: The rendered HTML for the main page.
    :rtype: str
    """

    cams = load_cameras()
    output = ""
    for cam in cams:
        cam_id = cam["id"]
        cam_name = cam["name"]
        cam_source = cam["source"]
        output += f"""
            <h2>{cam_name}</h2>
            <img src='/video_feed/{cam_id}' width='100%'>
            <p>{Alert.query.filter_by(cam_no=str(cam_source)).count()} alerts</p>
            <a href="/alerts/{cam_source}" class="btn btn-primary">View Alerts</a>&nbsp;
            <form method="post" action="/clear_alerts/{cam_source}" style="display:inline;">
                <button type="submit" class="btn btn-danger">Clear Alerts</button>
            </form>&nbsp;
            <form method="post" action="/remove_camera/{cam_id}" style="display:inline;">
                <button type="submit" class="btn btn-warning">Remove Camera</button>
            </form><hr>
        """

    return render_template("index.html", content=output, alerts=Alert.query.all())


@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id: int) -> Response:
    """
    Route to serve the video feed for a specific camera.

    :param cam_id: The registry ID of the camera to stream.
    :type cam_id: int

    :return: A streaming response containing the video feed.
    :rtype: Response
    """

    cams = load_cameras()
    entry = next((c for c in cams if c["id"] == cam_id), None)
    if entry is None:
        return "Camera not found", 404

    # Resolve source: stored integers stay int, digit-strings are coerced, URLs stay str.
    source = entry["source"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Return the video feed as a multipart response.
    return app.response_class(
        generate_frames(source), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/clear_alerts", methods=["POST"])
def clear_alerts() -> str:
    """
    Clears all alerts from the database.

    :return: A message confirming that all alerts have been cleared, with a link to go back to the main page.
    :rtype: str
    """

    # Use app context to ensure the database session is available when deleting alerts.
    with app.app_context():
        num_rows_deleted = db.session.query(Alert).delete()
        db.session.commit()
    return f"Cleared {num_rows_deleted} alerts! <a href='/'>Go Back</a>"


@app.route("/clear_alerts/<cam_no>", methods=["POST"])
def clear_alerts_by_cam(cam_no) -> str:
    """
    Clears all alerts for a specific camera from the database.

    :param cam_no: The camera number for which to clear alerts.

    :return: A message confirming that alerts for the specified camera have been cleared, with a link to go back to the main page.
    If the camera number is invalid or an error occurs during deletion, an error message is returned instead of crashing the application.
    :rtype: str
    """

    try:
        # Delete alerts for the specified camera
        with app.app_context():
            num_rows_deleted = Alert.query.filter_by(cam_no=str(cam_no)).delete()
            db.session.commit()

    # Handle any exceptions that may occur during the deletion process and return an error message instead of crashing the application.
    except Exception as e:
        return f"Error clearing alerts for camera {cam_no}: {e}"

    # Send the confirmation message to the user
    return f"Cleared {num_rows_deleted} alerts for camera {cam_no}! <a href='/'>Go Back</a>"


@app.route("/alerts")
def alerts() -> str:
    """
    Renders the alerts page showing all alerts in descending order of timestamp.

    :return: The rendered HTML for the alerts page, containing all alerts and available evidence paths for download.
    :rtype: str
    """

    # Query all alerts from the database in descending order of timestamp and prepare them for display on the alerts page.
    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    view_alerts = []

    # Loop through the alerts to prepare them for display, encoding the alert images to base64 for rendering in HTML.
    for alert in alerts:
        view_alerts.append(
            {
                "id": alert.id,
                "timestamp": alert.timestamp,
                "cam_no": alert.cam_no,
                "alert_type": alert.alert_type,
                "alert_image": base64.b64encode(alert.alert_image).decode("utf-8"),
            }
        )

    # Update the evidence paths
    with state_lock:
        evidence_paths = list(alert_evidence_paths)

    # Send the page to the user with all alerts and the available evidence paths for download.
    return render_template(
        "alerts.html", alerts=view_alerts, alert_evidence_paths=evidence_paths
    )


@app.route("/alerts/<cam_no>")
def alerts_by_cam(cam_no: str) -> str:
    """
    Renders the alerts page showing alerts for a specific camera in descending order of timestamp.

    :param cam_no: The camera number for which to display alerts.

    :return: The rendered HTML for the alerts page, containing alerts for the specified camera and available evidence paths for download.
    :rtype: str
    """

    # Query alerts for the specified camera from the database in descending order of timestamp and prepare them for display on the alerts page.
    alerts = (
        Alert.query.filter_by(cam_no=str(cam_no)).order_by(Alert.timestamp.desc()).all()
    )

    view_alerts = []

    # Loop through the alerts to prepare them for display, encoding the alert images to base64 for rendering in HTML.
    for alert in alerts:
        view_alerts.append(
            {
                "id": alert.id,
                "timestamp": alert.timestamp,
                "cam_no": alert.cam_no,
                "alert_type": alert.alert_type,
                "alert_image": base64.b64encode(alert.alert_image).decode("utf-8"),
            }
        )

    # Update the evidence paths
    with state_lock:
        evidence_paths = list(alert_evidence_paths)

    # Send the page to the user with the alerts for the specified camera and the available evidence paths for download.
    return render_template(
        "alerts.html",
        alerts=view_alerts,
        cam_no=cam_no,
        alert_evidence_paths=evidence_paths,
    )


@app.route("/delete_alert/<int:alert_id>", methods=["POST"])
def delete_alert(alert_id: int) -> str:
    """
    Deletes a specific alert from the database.

    :param alert_id: The ID of the alert to be deleted.

    :return: A message confirming that the specified alert has been deleted, with a link to view the remaining alerts.
    If the alert ID is not found or an error occurs during deletion, an error message is returned instead of crashing the application.
    :rtype: str
    """

    with app.app_context():
        # Query the alert by ID
        alert = db.session.get(Alert, alert_id)

        # Ensure the alert exists before attempting to delete it to avoid errors.
        if alert:
            db.session.delete(alert)
            db.session.commit()

        # If the alert was not found, return a message indicating so, instead of attempting to delete and causing an error.
        else:
            return (
                f"Alert with id {alert_id} not found! <a href='/alerts'>View Alerts</a>"
            )

    # Send the user back to the alerts page with a message confirming deletion of the alert.
    return f"Deleted alert with id {alert_id}! <a href='/alerts'>View Alerts</a>"


@app.route("/download/<path:filepath>")
def download_file(filepath: str) -> Response:
    """
    Route to download an alert evidence file.

    :param filepath: The path to the file to be downloaded.

    :return: A response that sends the specified file for download.
    :rtype: Response
    """

    return send_from_directory(output_dir, filepath, as_attachment=True)


@app.route("/download_all_alerts")
def download_all_alerts() -> Response:
    """
    Route to download all alert evidence files as a zip archive.

    :return: A response that sends an in-memory zip file containing all alert evidence files for download.
    :rtype: Response
    """

    # Create an in-memory zip file containing all evidence files in the output directory.
    memory_file = BytesIO()

    # Walk through the output directory and add all files to the zip archive, keeping the folder structure intact.
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)

                # keeps folder structure inside zip
                arcname = os.path.relpath(file_path, output_dir)
                zf.write(file_path, arcname=arcname)

    # Reset the pointer of the in-memory file to the beginning before sending it for download.
    memory_file.seek(0)

    # Send the in-memory zip file for download with an appropriate filename.
    return send_file(
        memory_file,
        as_attachment=True,
        download_name="all_alert_evidence.zip",
    )


@app.route("/add_camera", methods=["POST"])
def add_camera() -> Response:
    """
    Adds a new camera to the registry.
    Accepts a ``source`` (integer index or URL string) and an optional ``name``.

    :return: Redirect to the main page.
    :rtype: Response
    """

    source = request.form.get("source", "").strip()
    name = request.form.get("name", "").strip() or f"Camera {source}"
    if not source:
        return "No source provided", 400

    # Coerce digit strings to int so OpenCV receives the right type.
    parsed = int(source) if source.isdigit() else source

    cams = load_cameras()
    cams.append({"id": _next_cam_id(cams), "source": parsed, "name": name})
    save_cameras(cams)
    return redirect("/")


@app.route("/remove_camera/<int:cam_id>", methods=["POST"])
def remove_camera(cam_id: int) -> Response:
    """
    Removes a camera from the registry by its ID.

    :param cam_id: The registry ID of the camera to remove.
    :type cam_id: int

    :return: Redirect to the main page.
    :rtype: Response
    """

    cams = [c for c in load_cameras() if c["id"] != cam_id]
    save_cameras(cams)
    return redirect("/")


@app.route("/refresh_cameras", methods=["POST"])
def refresh_cameras() -> Response:
    """
    Re-scans local (wired/USB) cameras and adds any newly discovered ones to the registry.
    Cameras already in the registry are left unchanged.

    :return: Redirect to the main page.
    :rtype: Response
    """

    cams = load_cameras()
    existing_sources = {c["source"] for c in cams}
    for idx in scan_local_cameras():
        if idx not in existing_sources:
            cams.append(
                {"id": _next_cam_id(cams), "source": idx, "name": f"Camera {idx + 1}"}
            )
    save_cameras(cams)
    return redirect("/")


# Run the Flask app
if __name__ == "__main__":
    app.run(threaded=True)
