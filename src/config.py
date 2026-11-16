"""Configuration constants matching the reference paper (arXiv:2511.14057v1)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArcheryConfig:
    # Sensor sampling rates (Hz) - Section 2.1
    acc_fs: int = 20
    ppg_fs: int = 20

    # Accelerometer preprocessing & SmoothDiff - Section 2.3.1 & 2.4.1.1
    smooth_window_size: int = 20

    # Motion recognition sliding window - Section 3.1.1
    motion_window_size: int = 80   # 4 seconds at 20 Hz
    motion_step_size: int = 20     # 1 second stride
    motion_pos_ratio: float = 0.5  # >50% positive points -> label 1

    # Motion prediction post-processing - Section 2.4.1.3
    motion_prob_threshold: float = 0.9
    min_phase_windows: int = 1
    max_phase_windows: int = 10

    # PPG preprocessing & Stress detection - Section 2.3.2 & 3.2.1
    ppg_lowcut_hz: float = 0.6
    ppg_highcut_hz: float = 10.0
    ppg_filter_order: int = 3
    stress_window_sec: int = 30
    stress_high_threshold: int = 4  # Likert 1-3: Low (0), 4-5: High (1)

    # Train/Test split - Section 3.1.2 & 3.2.2
    train_ratio: float = 0.7
    random_seed: int = 42
