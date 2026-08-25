"""
preprocessing.py

Steps implemented (each tagged with its source):

1. Trial extraction  [README]
   README: "These events can be used to cut the run into individual trials...
   by taking only the data from each TrialStart to the next TrialEnd."

2. Band-pass filtering to mu/beta (8-30Hz)  [ASSUMPTION]
   The README states the RAW data was already filtered 0.1-200Hz with a 60Hz
   notch by the original authors. Narrowing further to the mu/beta band is a
   standard motor-imagery feature-extraction step from BCI literature, NOT
   something the README specifies for this dataset. Documented as our own
   methodology decision.

3. Sliding-window epoching  [ASSUMPTION]
   Window length / overlap are not specified anywhere in the README.
   Values are set in config.py and must be justified/reported by you.
"""

import numpy as np
from scipy.signal import butter, filtfilt

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def extract_trials(run: dict) -> list:
    """
    Cut the continuous EEG in a run dict into individual trials using
    TrialStart/TrialEnd event markers. [README-documented method]

    Returns
    -------
    list of dicts: [{ "eeg": np.ndarray [62, T_trial], "times": np.ndarray [T_trial] }, ...]
    """
    trials = []
    starts = [lat for lat, typ in zip(run["event_latency"], run["event_type"]) if typ == "TrialStart"]
    ends = [lat for lat, typ in zip(run["event_latency"], run["event_type"]) if typ == "TrialEnd"]

    if len(starts) != len(ends):
        print(f"[WARN] Mismatched TrialStart/TrialEnd counts in {run['source_file']} "
              f"({len(starts)} starts, {len(ends)} ends). Trimming to shorter length.")
        n = min(len(starts), len(ends))
        starts, ends = starts[:n], ends[:n]

    times = run["times"]
    data = run["data"]

    for t_start, t_end in zip(starts, ends):
        mask = (times >= t_start) & (times <= t_end)
        if mask.sum() == 0:
            continue
        trials.append({
            "eeg": data[:, mask],
            "times": times[mask],
        })
    return trials


def bandpass_filter(eeg: np.ndarray, fs: float,
                     low: float = config.MI_BAND_LOW,
                     high: float = config.MI_BAND_HIGH,
                     order: int = config.FILTER_ORDER) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter to each channel. [ASSUMPTION: mu/beta band]"""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, eeg, axis=-1)


def sliding_windows(trial_eeg: np.ndarray, trial_times: np.ndarray, fs: float,
                     window_sec: float = config.WINDOW_SEC,
                     overlap: float = config.WINDOW_OVERLAP) -> list:
    """
    Slice a single trial's EEG into overlapping fixed-length windows. [ASSUMPTION]

    Returns
    -------
    list of dicts: [{ "eeg": np.ndarray [62, win_samples], "t_start": float, "t_end": float }, ...]
    """
    win_samples = int(window_sec * fs)
    step_samples = int(win_samples * (1 - overlap))
    step_samples = max(step_samples, 1)

    n_samples = trial_eeg.shape[1]
    windows = []
    for start_idx in range(0, n_samples - win_samples + 1, step_samples):
        end_idx = start_idx + win_samples
        windows.append({
            "eeg": trial_eeg[:, start_idx:end_idx],
            "t_start": trial_times[start_idx],
            "t_end": trial_times[end_idx - 1],
        })
    return windows


def preprocess_run(run: dict) -> list:
    """
    Full preprocessing for one run: extract trials -> band-pass filter -> sliding windows.

    Returns
    -------
    list of window dicts (see sliding_windows), each also tagged with the
    run's decoder/session/subject metadata for traceability, and with a
    unique `trial_id` identifying which individual TrialStart/TrialEnd
    segment the window came from.

    [CODE ADDITION] `trial_id` did not previously exist on window dicts.
    It is added here so that train/val/test splitting can be done at the
    trial level (see src/classifier.py: trial_val_test_split), keeping all
    overlapping windows of one trial on the same side of a split boundary.
    This does not change any preprocessing math — it only adds bookkeeping.
    """
    trials = extract_trials(run)
    all_windows = []
    for trial_idx, trial in enumerate(trials):
        filtered = bandpass_filter(trial["eeg"], run["fs"])
        windows = sliding_windows(filtered, trial["times"], run["fs"])
        trial_id = f"{run['subject']}_{run['session']}_{run['decoder']}_{run['run']}_T{trial_idx:02d}"
        for w in windows:
            w["subject"] = run["subject"]
            w["session"] = run["session"]
            w["run"] = run["run"]
            w["decoder"] = run["decoder"]
            w["trial_id"] = trial_id
        all_windows.extend(windows)
    return all_windows
