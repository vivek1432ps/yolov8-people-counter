# YOLOv8 People-Counting System

A Python, OpenCV, and Ultralytics YOLO demo that detects people in a video or webcam feed and displays the number of detections in each frame. Includes both a graphical interface and a command-line mode.

## Features
- **GUI:** choose a video file or webcam, adjust the confidence threshold, preview detections, start/stop processing
- Person detection with bounding boxes using Ultralytics YOLO
- Per-frame people count overlay
- Annotated video output saved to `outputs/people_counted.mp4` for video-file input
- Optional command-line mode via `app.py`

## Requirements
- Windows, macOS, or Linux
- Python 3.10 or 3.11 recommended
- Webcam optional; a local video works too
- CPU supported; a compatible NVIDIA GPU may accelerate inference

## Setup on Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current terminal, then activate again.

## Launch the graphical interface
```powershell
python gui.py
```
1. Click **Choose video…** to select an MP4, AVI, MOV, or MKV file, or click **Use webcam**.
2. Adjust the confidence threshold if needed.
3. Click **Start detection**. The first run downloads the pretrained `yolov8n.pt` model.
4. Click **Stop** to end processing.
5. For video files, find the annotated result at `outputs/people_counted.mp4`.

## Command-line alternative
```powershell
python app.py --source media/sample.mp4
python app.py --source 0
``
Press `q` in the OpenCV window to stop.

## Important limitation
This starter counts detections in each frame. It does **not** count unique people across a whole video. A person can be counted in many frames. Unique entry/exit counting requires object tracking (such as ByteTrack or BoT-SORT) and a defined counting line.

## Screenshots for your portfolio
Save real screenshots in a `screenshots/` directory:
1. `01_gui.png` — GUI with a video selected.
2. `02_detection.png` — GUI showing bounding boxes and the current frame count.
3. `03_output.png` — a frame from the saved annotated video.

Use `Win + Shift + S` on Windows. Only add screenshots from a run you have actually completed.

## Resume description
**YOLOv8 People-Counting System | Python, Tkinter, OpenCV, Ultralytics YOLO**
- Built a GUI-based computer-vision demo to detect people in video and webcam feeds.
- Added input selection, configurable confidence threshold, live preview, per-frame count, and annotated-video export.

## License
Review Ultralytics licensing terms before commercial deployment: https://www.ultralytics.com/license
