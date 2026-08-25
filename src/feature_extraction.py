"""
feature_extraction.py

Feature: mu (8-12Hz) and beta (13-30Hz) band power per channel, computed
via Welch's method. [ASSUMPTION — the README does not specify any feature
extraction method for this dataset; this is a standard, widely-used MI
feature set from BCI literature, not a dataset-native specification.]

Output feature vector length = N_CHANNELS * 2 (mu power + beta power per channel).
"""

import numpy as np
from scipy.signal import welch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def band_power(epoch: np.ndarray, fs: float, band: tuple) -> np.ndarray:
    """
    Compute average power in `band` for each channel of `epoch` [channels, samples].
    Returns array of shape [channels].
    """
    freqs, psd = welch(epoch, fs=fs, nperseg=min(256, epoch.shape[-1]), axis=-1)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return psd[:, band_mask].mean(axis=-1)


def extract_features(epoch: np.ndarray, fs: float) -> np.ndarray:
    """
    Extract mu + beta band power per channel for a single window.
    epoch: np.ndarray [N_CHANNELS, window_samples]
    Returns: np.ndarray of shape [N_CHANNELS * 2]
    """
    mu = band_power(epoch, fs, (8.0, 12.0))     # [ASSUMPTION] mu sub-band
    beta = band_power(epoch, fs, (13.0, 30.0))  # [ASSUMPTION] beta sub-band
    return np.concatenate([mu, beta])


def extract_features_batch(windows: list, fs: float) -> tuple:
    """
    Extract features for a list of labeled window dicts (from labeling.label_all_windows).

    Returns
    -------
    X: np.ndarray [n_windows, n_features]
    y: list[str] of length n_windows (labels)
    meta: list of dicts (session/run/decoder per window, for traceability / splitting)
    """
    X, y, meta = [], [], []
    for w in windows:
        feats = extract_features(w["eeg"], fs)
        X.append(feats)
        y.append(w["label"])
        meta.append({
            "session": w["session"],
            "run": w["run"],
            "decoder": w["decoder"],
            "trial_id": w["trial_id"],
            "t_start": w["t_start"],
            "t_end": w["t_end"],
        })
    return np.array(X), y, meta
