import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(value: str):
    """Convert numeric source strings such as '0' into webcam indexes."""
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(
        description="Detect people and display per-frame counts with YOLO."
    )
    parser.add_argument(
        "--source",
        default="media/sample.mp4",
        help="Video/image path or webcam index (use 0 for the default webcam).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model weights (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detection confidence threshold between 0 and 1.",
    )
    args = parser.parse_args()

    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")

    source = parse_source(args.source)
    model = YOLO(args.model)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise SystemExit(
            f"Could not open source: {args.source}. Check the file path or webcam permissions."
        )

    is_webcam = isinstance(source, int)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = None
    output_path = Path("outputs") / "people_counted.mp4"

    if not is_webcam:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise SystemExit(f"Could not create output video: {output_path}")

        print(f"Annotated video will be saved to: {output_path.resolve()}")

    print("Press q in the video window to stop.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                source=frame,
                classes=[0],  # COCO class 0 is person
                conf=args.confidence,
                verbose=False,
            )[0]

            annotated = result.plot()
            count = 0 if result.boxes is None else len(result.boxes)
            cv2.putText(
                annotated,
                f"People detected: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLOv8 People Counter (press q to quit)", annotated)

            if writer is not None:
                if (annotated.shape[1], annotated.shape[0]) != (width, height):
                    annotated = cv2.resize(annotated, (width, height))
                writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if writer is not None:
        print(f"Finished. Check: {output_path.resolve()}")


if __name__ == "__main__":
    main()
