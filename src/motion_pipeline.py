"""Motion phase recognition pipeline (Accelerometer + SmoothDiff + LSTM).

Implements Sections 2.3.1, 2.4.1, and 3.1 of arXiv:2511.14057v1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_total_acc(acc_xyz: np.ndarray) -> np.ndarray:
    """Compute Euclidean norm of 3-axis acceleration: sqrt(X^2 + Y^2 + Z^2)."""
    if acc_xyz.ndim != 2 or acc_xyz.shape[1] != 3:
        raise ValueError(f"Expected shape (T, 3), got {acc_xyz.shape}")
    return np.linalg.norm(acc_xyz, axis=1)


def compute_smooth_diff(total_acc: np.ndarray, window_size: int = 20) -> np.ndarray:
    """Compute Smoothed Differential Acceleration (SmoothDiff).

    1. First-order temporal difference: Diff[t] = TotalAcc[t] - TotalAcc[t-1], Diff[0] = 0.
    2. Causal rolling mean over `window_size` samples with zero-padded boundary.
    """
    diff = np.zeros_like(total_acc, dtype=np.float64)
    if len(total_acc) > 1:
        diff[1:] = total_acc[1:] - total_acc[:-1]

    smooth_diff = np.zeros_like(diff, dtype=np.float64)
    cumsum = np.cumsum(np.insert(diff, 0, 0.0))
    for t in range(len(diff)):
        start = max(0, t - window_size + 1)
        # Paper Eq. (4) divides by N (fixed window_size) with missing boundary values = 0
        smooth_diff[t] = (cumsum[t + 1] - cumsum[start]) / window_size
    return smooth_diff


def build_motion_features(acc_xyz: np.ndarray, smooth_window: int = 20) -> np.ndarray:
    """Build 5-channel motion feature matrix: [AccX, AccY, AccZ, TotalAcc, SmoothDiff]."""
    total_acc = compute_total_acc(acc_xyz)
    smooth_diff = compute_smooth_diff(total_acc, window_size=smooth_window)
    return np.column_stack([acc_xyz, total_acc, smooth_diff]).astype(np.float32)


def create_sliding_windows(
    features: np.ndarray,
    point_labels: np.ndarray,
    window_size: int = 80,
    step_size: int = 20,
    pos_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment time-series into sliding windows and assign binary labels.

    A window is labeled 1 if strictly > `pos_ratio` of its points are positive.
    """
    windows, labels = [], []
    num_points = len(features)
    for start in range(0, num_points - window_size + 1, step_size):
        end = start + window_size
        windows.append(features[start:end])
        ratio = np.mean(point_labels[start:end] > 0)
        labels.append(1 if ratio > pos_ratio else 0)
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.float32)


class MotionLSTM(nn.Module):
    """Unidirectional LSTM for binary archery motion phase detection."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = torch.tanh(out[:, -1, :])
        return torch.sigmoid(self.fc(last_hidden)).squeeze(-1)


def extract_predicted_phases(
    probs: np.ndarray,
    threshold: float = 0.9,
    min_len: int = 1,
    max_len: int = 10,
) -> list[tuple[int, int]]:
    """Post-process window probabilities into continuous (start_idx, end_idx) segments."""
    binary = (probs > threshold).astype(int)
    phases: list[tuple[int, int]] = []
    in_phase = False
    start = 0

    for i, val in enumerate(binary):
        if val == 1 and not in_phase:
            in_phase = True
            start = i
        elif val == 0 and in_phase:
            in_phase = False
            length = i - start
            if min_len <= length <= max_len:
                phases.append((start, i - 1))

    if in_phase:
        length = len(binary) - start
        if min_len <= length <= max_len:
            phases.append((start, len(binary) - 1))

    return phases


def compute_pqd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Prediction Quantity Deviation (PQD) defined in Eq. (11)."""
    p_true = float(np.sum(y_true))
    p_pred = float(np.sum(y_pred))
    if p_true == 0:
        return 1.0 if p_pred == 0 else 0.0
    return 1.0 - abs(p_pred - p_true) / p_true
