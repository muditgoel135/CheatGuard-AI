# CheatGuard AI

A Flask-based exam proctoring application that monitors live camera feeds for suspicious activity using MediaPipe computer vision models. When a face disappears or a hand is raised for a sustained period, an alert is automatically generated, a screenshot is saved to the database, and a video evidence clip is recorded to disk.

---

## Features

- Live MJPEG video stream from multiple cameras simultaneously
- Real-time face and hand landmark overlays drawn on-stream
- Two alert types: **No Face Detected** and **Hand Raised**
- Per-camera state machine with configurable thresholds (default: 3 seconds)
- Alert screenshots stored as PNG blobs in SQLite
- MP4 evidence videos saved automatically when suspicious activity ends
- Web UI to view, filter, delete, and download alerts and evidence
- Camera registry supporting local USB cameras and IP/RTSP streams
- Auto-detection of newly plugged USB cameras

---

## How It Works

### Threading Model

CheatGuard AI uses a producer/consumer threading architecture to keep the web server and ML inference fully independent:

```text
_process_camera() thread (one per camera)
  ├── Captures frames from OpenCV VideoCapture
  ├── Runs MediaPipe inference every 2nd frame
  ├── Draws face/hand landmark overlays
  ├── Encodes frame to JPEG → _jpeg_cache[cam_key]
  └── Fires alerts and buffers evidence frames

generate_frames() generator (one per HTTP streaming response)
  └── Reads from _jpeg_cache at up to 30 FPS → browser
```

The generators that serve MJPEG to the browser never touch the camera or the ML models — they only read the latest cached JPEG. This means a slow inference frame never causes the stream to stutter.

### State Machine (per camera)

Each camera independently tracks one of three states:

| State | Meaning |
| --- | --- |
| `IDLE` | Face is present, no suspicious activity |
| `No Face Detected` | No face visible for ≥ 3 seconds — alert fired |
| `Hand Raised` | Hand visible while face is present for ≥ 3 seconds — alert fired |

Once an alert fires, the state remains set until conditions return to normal, preventing duplicate alerts during a single sustained incident.

### ML Pipeline

Three MediaPipe models run per inference frame:

| Model | File | Size | Purpose |
| --- | --- | --- | --- |
| Face Detector | `face_detection_short_range.tflite` | 225 KB | Checks whether a face is present (confidence > 0.5) |
| Face Landmarker | `face_landmarker.task` | 3.6 MB | Extracts a 478-point face mesh for visual annotation |
| Hand Landmarker | `hand_landmarker.task` | 7.5 MB | Detects up to 2 hands with a 21-point skeleton each |

To reduce CPU load, inference runs only on every 2nd frame. Non-inference frames reuse the most recent result. RGB colour conversion (which is expensive at full resolution) is also skipped on frames where no drawing or inference is needed.

### Alert & Evidence Flow

```text
Suspicious condition met (≥ 3 s threshold)
  │
  ├── PNG screenshot of current frame → Alert row in SQLite
  │     fields: timestamp, cam_no, alert_type, alert_image (PNG binary)
  │
  └── Raw frame buffering enabled → evidence_queue[cam_key]

Condition clears (face returns / hand lowers)
  └── Daemon thread writes buffered frames → output/evidence_cam{n}_{ts}.mp4
        (15 FPS, MP4v codec, original resolution)
```

Evidence MP4 writing is offloaded to a separate daemon thread so the capture loop is never blocked by disk I/O.

---

## Project Structure

```text
CheatGuard-AI/
├── app.py                  # Flask app, routes, database model, camera registry
├── camera_processor.py     # Background capture, ML inference, MJPEG streaming
├── landmarker.py           # MediaPipe model config and landmark drawing helpers
├── cameras.json            # Persistent camera registry (auto-created on first run)
├── requirements.txt        # Python dependencies
├── .env                    # SECRET_KEY (create this — see Setup)
├── detection_models/
│   ├── face_detection_short_range.tflite
│   ├── face_landmarker.task
│   └── hand_landmarker.task
├── instance/
│   └── site.db             # SQLite alert database (auto-created)
├── output/                 # Evidence MP4 files (auto-created)
├── static/
│   └── styles.css
└── templates/
    ├── index.html          # Dashboard with live feeds and camera controls
    └── alerts.html         # Alert viewer with screenshots and evidence downloads
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- Windows recommended (camera scanning uses DirectShow); Linux/macOS work for IP streams

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create the `.env` file

```text
SECRET_KEY=your-secret-key-here
```

Replace the value with a long random string. This is used by Flask for session security.

### Run the app

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

On first launch, CheatGuard AI will automatically scan device indices 0–9 for connected cameras and write the results to `cameras.json`. The SQLite database and `output/` directory are also created automatically.

---

## Camera Management

### Local USB / Integrated Cameras

Cameras are identified by their device index (0, 1, 2, …). The **Refresh Local Cameras** button on the dashboard re-scans the system and adds any newly detected cameras to the registry without removing existing ones.

### IP / RTSP Streams

Use **Add Camera** on the dashboard and enter the stream URL as the source:

```text
rtsp://192.168.1.100:554/stream
http://192.168.1.100:8080/video
```

### cameras.json

The registry is stored as plain JSON alongside the application rather than in SQLite, so it survives a database reset and can be edited by hand. Each entry has three fields:

```json
[
  { "id": 0, "source": 0,    "name": "Desk Camera" },
  { "id": 1, "source": "rtsp://10.0.0.5:554/live", "name": "Exam Hall" }
]
```

---

## Web Interface

### Dashboard (`/`)

- Live MJPEG feed for every registered camera, with FPS and current state overlaid
- Alert count per camera with links to the camera's alert log
- Add camera form (accepts device index or stream URL)
- Remove camera and Refresh Local Cameras buttons

### Alerts (`/alerts`, `/alerts/<cam_no>`)

- Filterable by camera
- Each alert shows: timestamp, camera, alert type, and the PNG screenshot captured at the moment of the alert
- Per-alert delete button
- List of downloadable MP4 evidence files
- Download all evidence as a single ZIP archive

---

## Routes Reference

| Method | Route | Description |
| --- | --- | --- |
| GET | `/` | Dashboard |
| GET | `/video_feed/<cam_id>` | MJPEG stream for a camera |
| POST | `/add_camera` | Add a new camera (`source`, optional `name`) |
| POST | `/remove_camera/<cam_id>` | Remove a camera by registry ID |
| POST | `/refresh_cameras` | Scan for new local cameras |
| GET | `/alerts` | View all alerts |
| GET | `/alerts/<cam_no>` | View alerts for one camera |
| POST | `/clear_alerts` | Delete all alerts |
| POST | `/clear_alerts/<cam_no>` | Delete alerts for one camera |
| POST | `/delete_alert/<alert_id>` | Delete a single alert |
| GET | `/download/<filepath>` | Download an evidence MP4 |
| GET | `/download_all_alerts` | Download all evidence as ZIP |

---

## Configuration

The following constants in `camera_processor.py` control detection behaviour:

| Constant | Default | Description |
| --- | --- | --- |
| `INFERENCE_EVERY_N` | `2` | Run ML inference on every Nth frame |
| `STREAM_TARGET_FPS` | `30` | Max frames per second sent to the browser |
| `JPEG_PARAMS` | quality 70 | JPEG encoding quality for the stream |

The alert thresholds (3 seconds for both No Face and Hand Raised) are hardcoded in the `_process_camera()` function as `datetime.timedelta(seconds=3)`.

---

## Dependencies

| Package | Purpose |
| --- | --- |
| Flask | Web framework and HTTP routing |
| Flask-SQLAlchemy | ORM for the SQLite alert database |
| opencv-python | Camera capture, frame processing, JPEG/MP4 encoding |
| mediapipe | Face detection, face landmarking, hand landmarking |
| numpy | Array operations on frames |
| python-dotenv | Loading `SECRET_KEY` from `.env` |

See `requirements.txt` for pinned versions of all transitive dependencies.

---

## License

See [LICENSE](LICENSE).
