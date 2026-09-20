"""Try the blink detector on your own webcam (or a video file) and see the numbers live.

    python -m videokyc.tools.blink_demo                 # default camera
    python -m videokyc.tools.blink_demo --video clip.mp4
    python -m videokyc.tools.blink_demo --dump ear.csv  # also log time,EAR for tuning thresholds

Shows the eye aspect ratio (EAR), the adaptive open-eye baseline and the blink counter. Press q to quit.
Use it to check the defaults in ``BlinkConfig`` suit your camera before relying on them.
"""

from __future__ import annotations

import argparse
import csv
import time

import cv2

from ..face.config import FaceConfig
from ..liveness.blink import BlinkConfig, BlinkDetector
from ..liveness.landmarks import MediaPipeEyeTracker


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video file instead of the camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--blinks", type=int, default=2, help="blinks required to pass")
    ap.add_argument("--dump", help="write time,ear rows to this CSV")
    args = ap.parse_args(argv)

    eyes = MediaPipeEyeTracker(FaceConfig().model_dir)
    det = BlinkDetector(BlinkConfig(required_blinks=args.blinks))
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the video source.")

    writer = csv.writer(open(args.dump, "w", newline="")) if args.dump else None  # noqa: SIM115
    start = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic() - start
        ear = eyes.ear(frame)
        det.update(ear, t)
        if writer:
            writer.writerow([f"{t:.3f}", "" if ear is None else f"{ear:.4f}"])

        base = det.baseline
        lines = [
            f"EAR {ear:.3f}" if ear is not None else "no face",
            f"baseline {base:.3f}  close<{base * det.config.close_ratio:.3f}" if base else "calibrating...",
            f"blinks {det.blinks}/{args.blinks}" + ("  PASSED" if det.passed else ""),
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("blink demo (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
