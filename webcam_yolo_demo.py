"""Real-time laptop webcam test using YOLO-Pose for archery biomechanics & phase detection."""

from __future__ import annotations

import argparse
from collections import Counter
import os
import time

import cv2
from ultralytics import YOLO

from src.vision_pose_pipeline import ArcheryPoseMetrics, ArcheryVisionAnalyzer


PHASE_COLORS = {
    "Idle": (200, 200, 200),
    "Draw": (0, 215, 255),
    "Aim (Hold)": (0, 255, 0),
    "Release": (0, 80, 255),
}


def draw_hud(frame, metrics: ArcheryPoseMetrics | None, fps: float) -> None:
    """Render biomechanical overlay box on top-left of frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (360, 185), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"Archery YOLO-Pose | FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if metrics is None:
        cv2.putText(frame, "Status: No full upper-body pose", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 1)
        cv2.putText(frame, "Tip: Step back so shoulders & wrists", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "     are clearly visible.", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Press 'q' or ESC to quit", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return

    color = PHASE_COLORS.get(metrics.phase, (255, 255, 255))
    cv2.putText(frame, f"Phase: {metrics.phase}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Bow Elbow:     {metrics.bow_elbow_angle:5.1f} deg", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Draw Elbow:    {metrics.draw_elbow_angle:5.1f} deg", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Shoulder Tilt: {metrics.shoulder_tilt_deg:5.1f} deg", (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(
        frame,
        f"Tremor (std):  {metrics.bow_hand_steadiness:.4f} | q=quit",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Archery AI Webcam Test with YOLO-Pose")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="YOLO pose model weights")
    parser.add_argument("--bow-side", type=str, default="left", choices=["left", "right"], help="Hand holding the bow")
    parser.add_argument("--conf", type=float, default=0.5, help="Person detection confidence")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    args = parser.parse_args()

    print(f"Loading YOLO-Pose model: {args.model} ...")
    model = YOLO(args.model)
    analyzer = ArcheryVisionAnalyzer(bow_side=args.bow_side)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    if not cap.isOpened() and os.name == "nt":
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}. Check camera permissions or index.")

    print("Webcam opened! Press 'q' or ESC in the video window to stop.")
    phase_counts: Counter[str] = Counter()
    aim_bow_angles: list[float] = []
    prev_time = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break
            frame_idx += 1

            frame = cv2.flip(frame, 1)  # Mirror view for natural laptop testing
            results = model.predict(frame, conf=args.conf, verbose=False)
            result = results[0]
            annotated = result.plot()

            metrics = None
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                # Take the first (most prominent) detected person
                kps_xy = result.keypoints.xy[0].cpu().numpy()
                kps_conf = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else None
                metrics = analyzer.analyze_frame(kps_xy, kps_conf)
                if metrics is not None:
                    phase_counts[metrics.phase] += 1
                    if metrics.phase == "Aim (Hold)":
                        aim_bow_angles.append(metrics.bow_elbow_angle)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(annotated, metrics, fps)
            cv2.imshow("Archery AI - YOLO Pose Webcam", annotated)

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n=== Session Summary ===")
    print("Phase frame counts:", dict(phase_counts))
    if aim_bow_angles:
        print(f"Mean Bow Elbow Angle during Aim: {sum(aim_bow_angles)/len(aim_bow_angles):.1f} deg")


if __name__ == "__main__":
    main()
