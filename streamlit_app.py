"""
streamlit_app.py

Standalone Streamlit application that VISUALIZES ONLY — it does not run
any part of the pipeline itself. Run `main.py` first to generate real
outputs from your manually-supplied S18 file.

It reads three separate files produced by main.py and shows them in
separate sections/tabs so train / validation / test are never mixed:
  - outputs/S18_metrics_summary.json   (window counts, val/test accuracy)
  - outputs/S18_val_predictions.csv    (validation-set predictions)
  - outputs/S18_test_predictions.csv   (final test-set predictions)
and animates the classifier-driven cursor (via CursorController) as a
replay simulation over the test set — this is offline replay, not a
live EEG stream.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.cursor_control import CursorController

st.set_page_config(page_title="S18 BCI Cursor Control Demo", layout="centered")
st.title("BCI Cursor Control — Pipeline Results")
st.caption(
    "Visualizes saved outputs from main.py only. This app does not train or "
    "re-evaluate anything — it just displays what main.py already produced."
)

summary_path = os.path.join(config.OUTPUT_DIR, f"{config.SUBJECT_ID}_metrics_summary.json")
val_path = os.path.join(config.OUTPUT_DIR, f"{config.SUBJECT_ID}_val_predictions.csv")
test_path = os.path.join(config.OUTPUT_DIR, f"{config.SUBJECT_ID}_test_predictions.csv")

missing = [p for p in [summary_path, val_path, test_path] if not os.path.exists(p)]
if missing:
    st.error(
        "Missing output file(s):\n" + "\n".join(f"- {p}" for p in missing) +
        "\n\nRun `python main.py --zip <path_to_S18.zip>` (or `--dir`) first "
        "to generate them from your manually-supplied file."
    )
    st.stop()

with open(summary_path) as f:
    summary = json.load(f)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

tab_train, tab_val, tab_test = st.tabs(["Train (summary)", "Validation results", "Test results (final)"])

# ------------------------------------------------------------------
# TRAIN — window/trial counts only; no accuracy claim on training
# data is shown, since fit-on-train accuracy is not a generalization
# metric and reporting it here risks being mistaken for one.
# ------------------------------------------------------------------
with tab_train:
    st.subheader("Training split")
    st.metric("Training windows", summary["train_windows"])
    st.metric("Training trials", summary["train_trials"])
    st.write(
        f"{summary['train_windows']} / {summary['n_windows_total']} windows "
        f"({summary['train_windows']/summary['n_windows_total']:.1%}) used for training."
    )
    st.caption("Split is 60/20/20 by trial — see README_PIPELINE.md for why.")

# ------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------
with tab_val:
    st.subheader("Validation split")
    st.metric("Validation windows", summary["val_windows"])
    st.metric("Validation accuracy", f"{summary['val_accuracy']:.4f}")
    st.text(classification_report(val_df["true_label"], val_df["predicted_label"],
                                   labels=config.CLASSES, zero_division=0))
    st.write("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(val_df["true_label"], val_df["predicted_label"], labels=config.CLASSES)
    st.dataframe(pd.DataFrame(cm, index=config.CLASSES, columns=config.CLASSES))

# ------------------------------------------------------------------
# TEST — final, held-out result
# ------------------------------------------------------------------
with tab_test:
    st.subheader("Test split (final, held-out)")
    st.metric("Test windows", summary["test_windows"])
    st.metric("Test accuracy", f"{summary['test_accuracy']:.4f}")
    st.text(classification_report(test_df["true_label"], test_df["predicted_label"],
                                   labels=config.CLASSES, zero_division=0))
    st.write("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(test_df["true_label"], test_df["predicted_label"], labels=config.CLASSES)
    st.dataframe(pd.DataFrame(cm, index=config.CLASSES, columns=config.CLASSES))

    st.markdown("---")
    st.subheader("Cursor replay (test set)")

    st.sidebar.header("Playback settings")
    speed = st.sidebar.slider("Steps per second", 1, 20, 5)
    smoothing = st.sidebar.slider("Movement smoothing", 0.0, 0.9, 0.0, step=0.1,
                                   help="0 = pure discrete steps, higher = smoother blended movement")
    use_predicted = st.sidebar.radio("Drive cursor using:", ["Predicted class", "True class (ground truth)"])
    label_col = "predicted_label" if use_predicted == "Predicted class" else "true_label"

    if st.button("▶ Run replay"):
        controller = CursorController(start_pos=(0.0, 0.0))
        placeholder = st.empty()
        chart_placeholder = st.empty()

        xs, ys = [], []
        for i, row in test_df.iterrows():
            x, y = controller.step(row[label_col], smoothing=smoothing)
            xs.append(x)
            ys.append(y)

            with placeholder.container():
                st.write(f"Step {i+1}/{len(test_df)} — class: **{row[label_col]}** — cursor: ({x:.3f}, {y:.3f})")

            chart_df = pd.DataFrame({"x": xs, "y": ys})
            with chart_placeholder.container():
                st.scatter_chart(chart_df, x="x", y="y", height=400)

            time.sleep(1.0 / speed)

        st.success("Replay complete.")
    else:
        st.info("Click 'Run replay' to start the cursor animation.")
