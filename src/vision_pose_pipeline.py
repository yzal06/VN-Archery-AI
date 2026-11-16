"""Vision-based archery biomechanics & phase recognition using YOLO Pose.

Replaces wrist-worn accelerometer with 2D skeletal keypoints from webcam/video,
while adapting the paper's SmoothDiff concept to normalized joint kinematics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np


# COCO 17-keypoint indices used by Ultralytics YOLO-Pose
KP_NOSE = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12


@dataclass
class ArcheryPoseMetrics:
    bow_elbow_angle: float
    draw_elbow_angle: float
    shoulder_tilt_deg: float
    bow_hand_steadiness: float
    draw_smooth_speed: float
    phase: str


def compute_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle ABC in degrees (at joint b)."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


class ArcheryVisionAnalyzer:
    """Tracks keypoint kinematics over a sliding window to estimate archery form & phase."""

    def __init__(self, bow_side: str = "left", smooth_window: int = 8, min_conf: float = 0.35):
        """
        Args:
            bow_side: 'left' if holding bow with left hand (right-handed archer), else 'right'.
            smooth_window: Rolling window size (frames) for SmoothDiff-style motion smoothing.
            min_conf: Minimum keypoint confidence threshold.
        """
        self.bow_side = bow_side.lower()
        self.smooth_window = smooth_window
        self.min_conf = min_conf

        self.bow_wrist_history: deque[np.ndarray] = deque(maxlen=smooth_window * 2)
        self.draw_wrist_history: deque[np.ndarray] = deque(maxlen=smooth_window * 2)
        self.draw_speed_history: deque[float] = deque(maxlen=smooth_window)
        self._last_phase = "Idle"

    def _side_indices(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if self.bow_side == "left":
            bow = (KP_L_SHOULDER, KP_L_ELBOW, KP_L_WRIST)
            draw = (KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST)
        else:
            bow = (KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST)
            draw = (KP_L_SHOULDER, KP_L_ELBOW, KP_L_WRIST)
        return bow, draw

    def analyze_frame(self, keypoints_xy: np.ndarray, keypoints_conf: np.ndarray | None = None) -> ArcheryPoseMetrics | None:
        """Compute archery biomechanical metrics for one person's (17, 2) keypoints."""
        if keypoints_xy.shape[0] < 13:
            return None

        bow_idx, draw_idx = self._side_indices()
        required = [KP_NOSE, *bow_idx, *draw_idx]
        if keypoints_conf is not None:
            if any(keypoints_conf[idx] < self.min_conf for idx in required):
                return None

        l_sh, r_sh = keypoints_xy[KP_L_SHOULDER], keypoints_xy[KP_R_SHOULDER]
        shoulder_width = float(np.linalg.norm(r_sh - l_sh))
        if shoulder_width < 10.0:
            return None

        # 1. Joint angles
        bow_angle = compute_angle_deg(keypoints_xy[bow_idx[0]], keypoints_xy[bow_idx[1]], keypoints_xy[bow_idx[2]])
        draw_angle = compute_angle_deg(keypoints_xy[draw_idx[0]], keypoints_xy[draw_idx[1]], keypoints_xy[draw_idx[2]])

        # 2. Shoulder tilt relative to horizontal axis
        dy = r_sh[1] - l_sh[1]
        dx = r_sh[0] - l_sh[0]
        shoulder_tilt = float(abs(math.degrees(math.atan2(dy, dx))))
        if shoulder_tilt > 90.0:
            shoulder_tilt = 180.0 - shoulder_tilt

        # 3. Normalized wrist positions (scale-invariant by shoulder width)
        bow_wrist_norm = keypoints_xy[bow_idx[2]] / shoulder_width
        draw_wrist_norm = keypoints_xy[draw_idx[2]] / shoulder_width
        nose_norm = keypoints_xy[KP_NOSE] / shoulder_width

        self.bow_wrist_history.append(bow_wrist_norm)
        self.draw_wrist_history.append(draw_wrist_norm)

        # Bow hand steadiness (lower std = steadier aim)
        if len(self.bow_wrist_history) >= 3:
            arr = np.array(self.bow_wrist_history)
            bow_steadiness = float(np.mean(np.std(arr, axis=0)))
        else:
            bow_steadiness = 0.0

        # SmoothDiff-inspired draw wrist speed
        if len(self.draw_wrist_history) >= 2:
            speed = float(np.linalg.norm(self.draw_wrist_history[-1] - self.draw_wrist_history[-2]))
        else:
            speed = 0.0
        self.draw_speed_history.append(speed)
        draw_smooth_speed = float(np.mean(self.draw_speed_history))

        # 4. Rule-based baseline phase estimation
        wrist_to_face = float(np.linalg.norm(draw_wrist_norm - nose_norm))
        phase = self._classify_phase(bow_angle, draw_angle, wrist_to_face, draw_smooth_speed)

        return ArcheryPoseMetrics(
            bow_elbow_angle=bow_angle,
            draw_elbow_angle=draw_angle,
            shoulder_tilt_deg=shoulder_tilt,
            bow_hand_steadiness=bow_steadiness,
            draw_smooth_speed=draw_smooth_speed,
            phase=phase,
        )

    def _classify_phase(
        self, bow_angle: float, draw_angle: float, wrist_to_face: float, smooth_speed: float
    ) -> str:
        # Bow arm extended (>145 deg), draw arm flexed (<105 deg) & hand near anchor point
        is_bow_extended = bow_angle > 140.0
        is_draw_anchored = draw_angle < 110.0 and wrist_to_face < 1.35

        if is_bow_extended and is_draw_anchored:
            phase = "Aim (Hold)" if smooth_speed < 0.03 else "Draw"
        elif self._last_phase in ("Aim (Hold)", "Draw") and smooth_speed >= 0.08:
            phase = "Release"
        else:
            phase = "Idle"

        self._last_phase = phase
        return phase
