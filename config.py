"""
config.py
Central configuration for the S18 BCI pipeline.

Every value below is tagged as one of:
  [README]     - directly stated in the dataset README
  [ASSUMPTION] - a design choice made by us because the README does not specify it.
                 These MUST be reported as methodology decisions in your project report,
                 not presented as dataset-native facts.
"""

# ----------------------------------------------------------------------
# Dataset structure (from README)
# ----------------------------------------------------------------------
SUBJECT_ID = "S18"                     # [our choice] prototype subject
N_CHANNELS = 62                        # [README] 64-electrode cap minus M1/M2 mastoids
EEG_FS = 1000                          # [README] EEG sampling rate, Hz
POS_FS = 25                            # [README] cursorpos/targetpos/cursorvel sampling rate, Hz (40ms)
TRIAL_DURATION_SEC = 60                # [README] each CP trial lasts 60 seconds

# Runs present for S18 (from your file listing) — sessions x decoders x runs
SESSIONS = ["Se01", "Se02", "Se03", "Se04"]     # [derived from filenames]
DECODERS_SE01 = ["AR", "Chance", "CL", "TL"]     # [derived from filenames]
DECODERS_SE02_04 = ["Chance", "CL", "DL", "TL"]  # [derived from filenames]
N_RUNS_PER_DECODER = 4                            # [derived from filenames] (Chance has only R01)

# ----------------------------------------------------------------------
# Preprocessing [ASSUMPTION — not specified in README beyond the
# original 0.1-200Hz bandpass + 60Hz notch already applied by the authors]
# ----------------------------------------------------------------------
MI_BAND_LOW = 8.0          # [ASSUMPTION] mu band lower edge, Hz — standard MI literature choice
MI_BAND_HIGH = 30.0        # [ASSUMPTION] beta band upper edge, Hz — standard MI literature choice
FILTER_ORDER = 4           # [ASSUMPTION] Butterworth filter order

# ----------------------------------------------------------------------
# Windowing / epoching [ASSUMPTION]
# ----------------------------------------------------------------------
WINDOW_SEC = 1.0            # [ASSUMPTION] sliding window length for classification, seconds
WINDOW_OVERLAP = 0.5        # [ASSUMPTION] fraction of overlap between consecutive windows

# ----------------------------------------------------------------------
# Labeling [our decision, confirmed with you]
# ----------------------------------------------------------------------
# Label = direction from cursor position to target position, averaged over the window.
# 4 classes: Right, Up, Left, Down (90-degree sectors centered on the axes).
CLASSES = ["Right", "Up", "Left", "Down"]

# ----------------------------------------------------------------------
# Cursor control mapping [ASSUMPTION]
# ----------------------------------------------------------------------
CLASS_TO_VECTOR = {
    "Right": (1.0, 0.0),
    "Up":    (0.0, 1.0),
    "Left":  (-1.0, 0.0),
    "Down":  (0.0, -1.0),
}
CURSOR_STEP_SPEED = 0.05    # [ASSUMPTION] fraction of screen width moved per classification step
WORKSPACE_MIN = -0.5        # [README] normalized workspace bounds
WORKSPACE_MAX = 0.5         # [README]

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
