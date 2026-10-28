# YOLOv8 People-Counting System

A Python and OpenCV demo that detects people in a video or webcam feed and displays the number of people detected in each frame.

## Features
- Video-file and webcam input
- Person detection with bounding boxes using Ultralytics YOLO
- Per-frame people count overlay
- Annotated video output saved to `outputs/people_counted.mp4`

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

## Run on a video
Create a `media` folder and put your own video at `media/sample.mp4`. Then run:
```powershell
python app.py --source media/sample.mp4
```

## Run on your webcam
```powershell
python app.py --source 0
```

Press `q` in the video window to stop. On the first run, Ultralytics downloads the pretrained `yolov8n.pt` weights.

## Output
Video results are written to `outputs/people_counted.mp4`. Do not commit large video files or model weights to GitHub unless you intentionally want to host them.

## Important limitation
This starter counts detections in each frame. It does **not** count unique people across a whole video. A person can be counted in many frames. Unique entry/exit counting requires object tracking (such as ByteTrack or BoT-SORT) and a defined counting line.

## Screenshots for your portfolio
Save real screenshots in a `screenshots/` directory:
1. `01_detection.png` — running video with bounding boxes and people count.
2. `02_terminal.png` — terminal command and successful execution.
3. `03_output.png` — a frame from the saved annotated video.

Use `Win + Shift + S` on Windows to capture screenshots. Only add screenshots from a run you have actually completed.

## Resume description
**YOLOv8 People-Counting System | Python, OpenCV, Ultralytics YOLO**
- Built a computer-vision demo for detecting people in video and webcam feeds using a pretrained YOLO model.
- Added bounding-box visualization, per-frame detection counts, command-line input selection, and annotated-video export.

## License
Review Ultralytics licensing terms before commercial deployment: https://www.ultralytics.com/license
