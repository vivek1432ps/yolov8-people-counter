import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


class PeopleCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 People Counter")
        self.root.geometry("1050x760")
        self.root.minsize(800, 600)

        self.source = None
        self.stop_event = threading.Event()
        self.worker = None
        self.photo = None
        self.running = False

        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="YOLOv8 People Counter", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="Detect people in a video file or webcam feed. Counts are per frame, not unique visitors.",
            wraplength=900,
        ).pack(anchor="w", pady=(4, 12))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 10))
        self.file_label = ttk.Label(controls, text="No video selected", width=42)
        self.file_label.grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.choose_button = ttk.Button(controls, text="Choose video…", command=self.choose_video)
        self.choose_button.grid(row=0, column=1, padx=4)
        self.webcam_button = ttk.Button(controls, text="Use webcam", command=self.choose_webcam)
        self.webcam_button.grid(row=0, column=2, padx=4)

        ttk.Label(controls, text="Confidence threshold:").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.confidence = tk.DoubleVar(value=0.35)
        self.confidence_scale = ttk.Scale(
            controls, from_=0.1, to=0.9, variable=self.confidence, orient="horizontal", length=260
        )
        self.confidence_scale.grid(row=1, column=1, columnspan=2, sticky="w", pady=(12, 0))
        self.confidence_value = ttk.Label(controls, text="0.35")
        self.confidence_value.grid(row=1, column=3, padx=8, pady=(12, 0))
        self.confidence.trace_add("write", self.update_confidence_label)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(0, 10))
        self.start_button = ttk.Button(actions, text="Start detection", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.status = tk.StringVar(value="Choose a video or select the webcam.")
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=12)

        self.preview = ttk.Label(outer, text="Preview will appear here", anchor="center")
        self.preview.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="For video files, annotated output is saved to outputs/people_counted.mp4.",
            foreground="#555",
        ).pack(anchor="w", pady=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.refresh_preview)

    def update_confidence_label(self, *_):
        self.confidence_value.configure(text=f"{self.confidence.get():.2f}")

    def choose_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.source = path
            self.file_label.configure(text=Path(path).name)
            self.status.set("Video selected. Press Start detection.")

    def choose_webcam(self):
        self.source = 0
        self.file_label.configure(text="Default webcam (camera 0)")
        self.status.set("Webcam selected. Press Start detection.")

    def start(self):
        if self.running:
            return
        if self.source is None:
            messagebox.showinfo("Select input", "Choose a video file or select the webcam first.")
            return
        self.stop_event.clear()
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.choose_button.configure(state="disabled")
        self.webcam_button.configure(state="disabled")
        self.status.set("Loading YOLO model…")
        self.worker = threading.Thread(target=self.process_video, daemon=True)
        self.worker.start()

    def process_video(self):
        capture = None
        writer = None
        try:
            model = YOLO("yolov8n.pt")
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                raise RuntimeError("Could not open the selected input. Check the file or camera permissions.")

            is_webcam = isinstance(self.source, int)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not fps or fps < 1 or fps > 120:
                fps = 25.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

            if not is_webcam:
                output = Path("outputs") / "people_counted.mp4"
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {output}")

            self.root.after(0, lambda: self.status.set("Detection running…"))
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(
                    source=frame, classes=[0], conf=float(self.confidence.get()), verbose=False
                )[0]
                annotated = result.plot()
                count = 0 if result.boxes is None else len(result.boxes)
                cv2.putText(
                    annotated, f"People detected: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
                )
                if writer is not None:
                    if (annotated.shape[1], annotated.shape[0]) != (width, height):
                        annotated = cv2.resize(annotated, (width, height))
                    writer.write(annotated)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((960, 540))
                self.root.after(0, self.show_frame, image, count)

            if writer is not None:
                self.root.after(0, lambda: self.status.set("Finished. Output: outputs/people_counted.mp4"))
            else:
                self.root.after(0, lambda: self.status.set("Detection stopped."))
        except Exception as exc:
            self.root.after(0, lambda err=str(exc): messagebox.showerror("Detection error", err))
            self.root.after(0, lambda: self.status.set("Could not complete detection."))
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            self.root.after(0, self.finish)

    def show_frame(self, image, count):
        if not self.root.winfo_exists():
            return
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text="")
        if self.running:
            self.status.set(f"Detection running — {count} people detected in current frame.")

    def refresh_preview(self):
        if self.root.winfo_exists():
            self.root.after(150, self.refresh_preview)

    def finish(self):
        self.running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.choose_button.configure(state="normal")
        self.webcam_button.configure(state="normal")

    def stop(self):
        self.stop_event.set()
        self.status.set("Stopping after the current frame…")

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PeopleCounterGUI(root)
    root.mainloop()
