"""End-to-end baseline demo on synthetic 20 Hz Accelerometer & PPG data."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import ArcheryConfig
from src.motion_pipeline import (
    MotionLSTM,
    build_motion_features,
    compute_pqd,
    create_sliding_windows,
    extract_predicted_phases,
)
from src.stress_pipeline import (
    bandpass_filter_ppg,
    binarize_stress_labels,
    build_stress_mlp,
    extract_hrv_features,
    extract_rr_intervals,
)


def generate_synthetic_motion_data(cfg: ArcheryConfig, duration_sec: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Simulate 3-axis 20 Hz accelerometer stream with synthetic draw-to-release bursts."""
    rng = np.random.default_rng(cfg.random_seed)
    n_samples = duration_sec * cfg.acc_fs
    acc_xyz = rng.normal(0.0, 0.05, size=(n_samples, 3))
    acc_xyz[:, 2] += 9.81  # Gravity baseline
    labels = np.zeros(n_samples, dtype=int)

    # Inject 6 shooting bursts (~4s each)
    for start_sec in range(10, duration_sec - 10, 18):
        s = start_sec * cfg.acc_fs
        e = s + 4 * cfg.acc_fs
        t = np.linspace(0, np.pi, e - s)
        acc_xyz[s:e, 0] += 1.5 * np.sin(t)
        acc_xyz[s:e, 1] += 1.2 * np.sin(2 * t)
        acc_xyz[s:e, 2] += 0.8 * np.cos(t)
        labels[s:e] = 1

    return acc_xyz, labels


def generate_synthetic_stress_data(cfg: ArcheryConfig, n_windows: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Simulate 30s 20 Hz PPG windows and 1-5 Likert stress scores."""
    rng = np.random.default_rng(cfg.random_seed)
    win_len = cfg.stress_window_sec * cfg.ppg_fs
    t = np.arange(win_len) / cfg.ppg_fs

    X_feats, likert = [], []
    for i in range(n_windows):
        is_high = i >= n_windows // 2
        hr_hz = rng.uniform(1.5, 1.9) if is_high else rng.uniform(1.0, 1.3)
        raw_ppg = np.sin(2 * np.pi * hr_hz * t) + 0.15 * rng.normal(size=win_len)
        filtered = bandpass_filter_ppg(
            raw_ppg, fs=cfg.ppg_fs, lowcut=cfg.ppg_lowcut_hz, highcut=cfg.ppg_highcut_hz, order=cfg.ppg_filter_order
        )
        rr_sec = extract_rr_intervals(filtered, fs=cfg.ppg_fs)
        feats = extract_hrv_features(rr_sec, filtered, fs=cfg.ppg_fs)
        X_feats.append(feats)
        likert.append(rng.integers(4, 6) if is_high else rng.integers(1, 4))

    return np.array(X_feats, dtype=np.float32), binarize_stress_labels(np.array(likert), cfg.stress_high_threshold)


def run_motion_demo(cfg: ArcheryConfig) -> None:
    print("=== 1. Motion Recognition Pipeline (SmoothDiff + LSTM) ===")
    acc_xyz, point_labels = generate_synthetic_motion_data(cfg)
    features = build_motion_features(acc_xyz, smooth_window=cfg.smooth_window_size)
    X, y = create_sliding_windows(
        features, point_labels, window_size=cfg.motion_window_size, step_size=cfg.motion_step_size, pos_ratio=cfg.motion_pos_ratio
    )

    split = int(len(X) * cfg.train_ratio)
    X_train, X_test = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train, y_test = torch.tensor(y[:split]), y[split:]

    model = MotionLSTM(input_dim=5, hidden_dim=32, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    model.train()
    for _ in range(15):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = model(X_test).numpy()
    preds = (probs > cfg.motion_prob_threshold).astype(int)
    phases = extract_predicted_phases(
        probs, threshold=cfg.motion_prob_threshold, min_len=cfg.min_phase_windows, max_len=cfg.max_phase_windows
    )

    print(f"Windows: {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Accuracy (SLA): {accuracy_score(y_test, preds):.4f}")
    print(f"F1-score:       {f1_score(y_test, preds, zero_division=0):.4f}")
    print(f"PQD:            {compute_pqd(y_test, preds):.4f}")
    print(f"Detected phase window intervals: {phases}\n")


def run_stress_demo(cfg: ArcheryConfig) -> None:
    print("=== 2. Stress Detection Pipeline (PPG + HRV + MLP) ===")
    X, y = generate_synthetic_stress_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=cfg.train_ratio, random_state=cfg.random_seed, stratify=y
    )

    clf = build_stress_mlp(random_state=cfg.random_seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print(f"Samples: {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1-score: {f1_score(y_test, preds, zero_division=0):.4f}\n")


if __name__ == "__main__":
    config = ArcheryConfig()
    torch.manual_seed(config.random_seed)
    run_motion_demo(config)
    run_stress_demo(config)
