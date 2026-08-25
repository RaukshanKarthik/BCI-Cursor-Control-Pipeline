"""
labeling.py

Label source: direction from CURSOR position to TARGET position,
averaged across the window's time span. [Confirmed design decision — NOT
specified in the README, which provides no discrete class labels at all.]

Rationale (discussed and confirmed with user): cursorvel is the OLD
decoder's output (README: "scaled outputs from the online DL decoders"),
so it is not a neutral ground truth of subject intent. targetpos alone
reflects only the environment's drift, not the subject. The
cursor->target vector is the closest available proxy for "which
direction the subject needs to imagine moving right now" in this
continuous pursuit paradigm.

Binning rule: 4 classes (Right/Up/Left/Down), 90-degree sectors
centered on each axis, using atan2. [ASSUMPTION - bin width choice]
"""

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _angle_to_class(dx: float, dy: float) -> str:
    """Map a 2D direction vector to one of 4 classes using 90-degree sectors."""
    if dx == 0 and dy == 0:
        # No net direction — assign to nearest previous class is not possible here,
        # so default is documented explicitly as an edge case.
        return "Idle_Zero"  # handled/filtered by caller; see note in label_window()
    angle = np.degrees(np.arctan2(dy, dx))  # -180..180
    angle = angle % 360
    if 45 <= angle < 135:
        return "Up"
    elif 135 <= angle < 225:
        return "Left"
    elif 225 <= angle < 315:
        return "Down"
    else:
        return "Right"


def label_window(window: dict, run: dict) -> str:
    """
    Compute the direction label for a single EEG window by averaging the
    cursor->target vector over the window's time span, using the run's
    cursorpos/targetpos/postimes (25Hz) timeseries.

    Returns one of config.CLASSES, or None if the window falls outside
    the position-timestamp range (edge case, should be rare).
    """
    postimes = run["postimes"]
    mask = (postimes >= window["t_start"]) & (postimes <= window["t_end"])
    if mask.sum() == 0:
        return None

    cx = run["cursorpos_x"][mask]
    cy = run["cursorpos_y"][mask]
    tx = run["targetpos_x"][mask]
    ty = run["targetpos_y"][mask]

    dx = np.mean(tx - cx)
    dy = np.mean(ty - cy)

    label = _angle_to_class(dx, dy)
    if label == "Idle_Zero":
        # Extremely rare edge case: cursor exactly on target for the whole window.
        # Default to the class nearest to zero movement is ambiguous by definition;
        # we drop such windows from training rather than guess. [ASSUMPTION]
        return None
    return label


def label_all_windows(windows: list, run: dict) -> list:
    """Attach a 'label' key to each window dict in place. Drops unlabeled windows."""
    labeled = []
    for w in windows:
        lbl = label_window(w, run)
        if lbl is not None:
            w["label"] = lbl
            labeled.append(w)
    return labeled
