"""
app.py — Flask web application for CheatGuard AI.

Responsibilities:
    - Defines the Alert database model and initialises the SQLite database.
    - Manages the camera registry (cameras.json): add, remove, refresh cameras.
    - Serves live MJPEG video feeds by delegating to camera_processor.
    - Provides routes for viewing, clearing, and downloading alert evidence.

Separation of concerns:
    camera_processor.py handles all OpenCV capture, MediaPipe inference, and
    JPEG encoding in background threads.  This file only deals with HTTP routing
    and database access.
"""

# Import necessary libraries
import base64
import json
from dotenv import load_dotenv
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
from sqlalchemy import func
import cv2
import os
import landmarker
from io import BytesIO
import zipfile
import camera_processor
import queue as queue_module
from threading import Lock as _Lock


# Load environment variables
load_dotenv()


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

if not app.config.get("SECRET_KEY"):
    raise RuntimeError(
        "SECRET_KEY is not set. Create a .env file with SECRET_KEY=<long-random-string>."
    )


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

# camera_processor runs background threads that write alerts to the database.
# We pass the app, db, and Alert references after db.create_all() so the tables
# exist before any camera thread could fire an alert.
camera_processor.init(app, db, Alert)

# SSE state — protected by _sse_lock
_sse_clients: list = []
_sse_lock = _Lock()


def _push_sse_event(data: str) -> None:
    """
    Push an event string to all connected SSE clients.

    :param data: The event data string to send to clients.
    :type data: str
    :return: None
    """
    with _sse_lock:
        for q in list(_sse_clients):
            try:
                q.put_nowait(data)
            except Exception:
                pass


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


# The camera registry is stored as a JSON file alongside the application rather
# than in the database so it persists across db.drop_all() calls and can be
# edited by hand without a migration.
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
            cams = json.load(f)
        for cam in cams:
            cam.setdefault("enabled", True)
        return cams

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
    :return: None
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


# Flask routes
@app.route("/")
def index() -> str:
    """
    Renders the main page with video feeds and alert counts for each camera.

    :return: The rendered HTML for the main page.
    :rtype: str
    """

    cams = load_cameras()
    alert_counts_raw = (
        db.session.query(Alert.cam_no, func.count(Alert.id))
        .group_by(Alert.cam_no)
        .all()
    )
    alert_counts = {cam_no: count for cam_no, count in alert_counts_raw}
    cam_data = [
        {
            "id": c["id"],
            "name": c["name"],
            "source": c["source"],
            "alert_count": alert_counts.get(str(c["source"]), 0),
            "enabled": c.get("enabled", True),
        }
        for c in cams
    ]
    return render_template("index.html", cams=cam_data, alerts=Alert.query.all())


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

    # JSON serialisation always produces strings for numeric values stored as ints
    # before a save/load cycle.  Coerce digit-only strings back to int so OpenCV
    # receives the correct type for local device indices; real URLs stay as str.
    source = entry["source"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Return the video feed as a multipart response.
    return app.response_class(
        camera_processor.generate_frames(source),
        mimetype="multipart/x-mixed-replace; boundary=frame",
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
            num_rows_deleted = (
                db.session.query(Alert).filter_by(cam_no=str(cam_no)).delete()
            )
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
    with camera_processor.state_lock:
        evidence_paths = list(camera_processor.alert_evidence_paths.keys())

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
    with camera_processor.state_lock:
        evidence_paths = list(camera_processor.alert_evidence_paths.keys())

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

    safe_name = os.path.basename(filepath)
    return send_from_directory(
        camera_processor.output_dir, safe_name, as_attachment=True
    )


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
        for root, dirs, files in os.walk(camera_processor.output_dir):
            for file in files:
                file_path = os.path.join(root, file)

                # keeps folder structure inside zip
                arcname = os.path.relpath(file_path, camera_processor.output_dir)
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


@app.route("/alert_stream")
def alert_stream() -> Response:
    """
    SSE endpoint — each connected dashboard tab gets live alert events.

    :return: A streaming response that provides server-sent events for live alert updates.
    :rtype: Response
    """

    def event_generator():
        q = queue_module.Queue(maxsize=50)
        with _sse_lock:
            _sse_clients.append(q)
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue_module.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _sse_lock:
                _sse_clients.remove(q)

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/toggle_camera/<int:cam_id>", methods=["POST"])
def toggle_camera(cam_id: int) -> Response:
    """
    Toggles the enabled/disabled state of a camera in the registry.

    :param cam_id: The registry ID of the camera to toggle.
    :type cam_id: int
    :return: Redirect to the main page.
    :rtype: Response
    """

    cams = load_cameras()
    for cam in cams:
        if cam["id"] == cam_id:
            cam["enabled"] = not cam.get("enabled", True)
            break
    save_cameras(cams)
    return redirect("/")


# Run the Flask app
if __name__ == "__main__":
    app.run(threaded=True)
