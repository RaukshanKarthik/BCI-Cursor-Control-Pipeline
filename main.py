"""
main.py

Orchestrates the full offline pipeline:

  zip file -> unzip -> load runs -> preprocess (trials, filter, windows)
  -> label (cursor->target direction) -> extract features
  -> 60/20/20 train/val/test split (by trial)
  -> train on TRAIN -> validate on VAL -> [only then] evaluate on TEST
  -> save model + separate prediction logs for Streamlit

This pipeline only runs against a real S18 file that you provide via
--zip or --dir. There is no synthetic/dummy data path in this script.

Usage:
    python main.py --zip data/S18.zip
    python main.py --dir data/S18_extracted     (if already unzipped)
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import unzip_dataset, load_all_runs
from src.preprocessing import preprocess_run
from src.labeling import label_all_windows
from src.feature_extraction import extract_features_batch
from src.classifier import trial_val_test_split, train_classifier, evaluate_classifier, save_model


def run_pipeline(data_dir: str, subject_id: str = config.SUBJECT_ID):
    print(f"[1/7] Loading all runs for {subject_id} from {data_dir} ...")
    runs = load_all_runs(data_dir, subject_id)
    print(f"      Loaded {len(runs)} runs.")

    all_windows = []
    print("[2/7] Preprocessing (trial extraction + band-pass filter + sliding windows) ...")
    for run in runs:
        windows = preprocess_run(run)
        labeled = label_all_windows(windows, run)
        all_windows.extend(labeled)
    print(f"      Produced {len(all_windows)} labeled windows total.")

    if len(all_windows) == 0:
        raise RuntimeError("No labeled windows were produced. Check preprocessing/labeling logic against real data.")

    print("[3/7] Extracting features (mu/beta band power per channel) ...")
    X, y, meta = extract_features_batch(all_windows, fs=config.EEG_FS)
    print(f"      Feature matrix shape: {X.shape}")

    print("[4/7] Splitting 60/20/20 by trial (train/val/test) ...")
    split = trial_val_test_split(X, y, meta, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)
    n_total = split["n_windows_total"]
    print(f"      Train: {len(split['y_train'])} windows ({len(split['y_train'])/n_total:.1%}) "
          f"across {len(split['train_trials'])} trials")
    print(f"      Val:   {len(split['y_val'])} windows ({len(split['y_val'])/n_total:.1%}) "
          f"across {len(split['val_trials'])} trials")
    print(f"      Test:  {len(split['y_test'])} windows ({len(split['y_test'])/n_total:.1%}) "
          f"across {len(split['test_trials'])} trials  [HELD OUT — not touched until step 7/7]")

    print("[5/7] Training classifier on TRAIN split only ...")
    clf, scaler = train_classifier(split["X_train"], split["y_train"])

    print("[6/7] Validating on VAL split (test set still untouched) ...")
    val_results = evaluate_classifier(clf, scaler, split["X_val"], split["y_val"])
    print(f"      Validation accuracy: {val_results['accuracy']:.4f}")
    print(val_results["report"])

    print("=" * 70)
    print("[7/7] FINAL TEST EVALUATION — test split used for the first time now.")
    print("=" * 70)
    test_results = evaluate_classifier(clf, scaler, split["X_test"], split["y_test"])
    print(f"      Test accuracy: {test_results['accuracy']:.4f}")
    print(test_results["report"])

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    save_model(clf, scaler, os.path.join(config.MODEL_DIR, subject_id))

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Reconstruct meta rows for each split by re-deriving indices (order-preserving
    # with trial_val_test_split, since it slices X/y in the same index order it built).
    trial_to_indices = {}
    for i, m in enumerate(meta):
        trial_to_indices.setdefault(m["trial_id"], []).append(i)

    def meta_rows_for(trial_list):
        idxs = [i for tid in trial_list for i in trial_to_indices[tid]]
        return [meta[i] for i in idxs]

    val_log = pd.DataFrame(meta_rows_for(split["val_trials"]))
    val_log["true_label"] = val_results["y_test"]
    val_log["predicted_label"] = val_results["y_pred"]
    val_log_path = os.path.join(config.OUTPUT_DIR, f"{subject_id}_val_predictions.csv")
    val_log.to_csv(val_log_path, index=False)

    test_log = pd.DataFrame(meta_rows_for(split["test_trials"]))
    test_log["true_label"] = test_results["y_test"]
    test_log["predicted_label"] = test_results["y_pred"]
    test_log_path = os.path.join(config.OUTPUT_DIR, f"{subject_id}_test_predictions.csv")
    test_log.to_csv(test_log_path, index=False)

    summary = {
        "subject_id": subject_id,
        "n_windows_total": n_total,
        "train_windows": len(split["y_train"]),
        "val_windows": len(split["y_val"]),
        "test_windows": len(split["y_test"]),
        "train_trials": len(split["train_trials"]),
        "val_trials": len(split["val_trials"]),
        "test_trials": len(split["test_trials"]),
        "val_accuracy": val_results["accuracy"],
        "test_accuracy": test_results["accuracy"],
    }
    summary_path = os.path.join(config.OUTPUT_DIR, f"{subject_id}_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nValidation prediction log saved to {val_log_path}")
    print(f"Test prediction log saved to {test_log_path}")
    print(f"Metrics summary saved to {summary_path}")
    print("(all three are read separately by the Streamlit app)")

    return clf, scaler, val_results, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, default=None, help="Path to subject's zip file")
    parser.add_argument("--dir", type=str, default=None, help="Path to already-extracted .mat directory")
    parser.add_argument("--subject", type=str, default=config.SUBJECT_ID)
    args = parser.parse_args()

    if args.zip:
        extract_dir = os.path.join(config.DATA_DIR, f"{args.subject}_extracted")
        print(f"Unzipping {args.zip} -> {extract_dir}")
        unzip_dataset(args.zip, extract_dir)
        data_dir = extract_dir
    elif args.dir:
        data_dir = args.dir
    else:
        parser.error("Provide either --zip or --dir")

    run_pipeline(data_dir, args.subject)
