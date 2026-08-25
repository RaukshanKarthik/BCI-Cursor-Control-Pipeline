# S18 BCI Cursor Control Pipeline — Documentation

## What this is

A working code scaffold for:
`Raw EEG (62ch, 1kHz) -> Preprocessing -> Feature Extraction -> Classifier -> Class -> Cursor movement command -> Streamlit visualization`

This pipeline is intended to run only against a real, manually-supplied S18
`.mat` file — there is no synthetic/dummy data path in this codebase.
Field access in `src/data_loader.py` follows the README's documented "eeg"
struct exactly, but has NOT been verified against a real file — this must
be your first step.

## How to actually use this with real data

1. Place your S18 zip (or extracted `.mat` files) in the `data/` folder.
2. Sanity-check field parsing on ONE real file first:
   ```
   python -m src.data_loader data/S18_extracted/S18_Se01_AR_R01.mat
   ```
   Check the printed shapes/fields match what you expect (62 channels, etc.).
   If MATLAB struct nesting differs from what `loadmat` produces, adjust the
   `_unwrap`/field access lines in `load_run()` accordingly.
3. Run the full pipeline (trains, validates, then runs the final held-out
   test evaluation, in that order — the test split is never touched until
   validation is complete):
   ```
   python main.py --zip data/S18.zip
   ```
   or, if already extracted:
   ```
   python main.py --dir data/S18_extracted
   ```
   This produces, in `outputs/`:
   - `S18_val_predictions.csv` — validation-set predictions
   - `S18_test_predictions.csv` — final test-set predictions
   - `S18_metrics_summary.json` — window counts + val/test accuracy
4. Launch the Streamlit demo (reads the three files above separately):
   ```
   streamlit run streamlit_app.py
   ```

## Assumptions made (must be reported in your methodology, not presented as dataset facts)

| Decision | Value used | Why |
|---|---|---|
| Feature extraction | mu (8-12Hz) + beta (13-30Hz) band power per channel | Standard MI-BCI literature choice; not specified in README |
| Window length / overlap | 1.0s / 50% | Not specified in README; tunable in `config.py` |
| Classifier | RandomForest (200 trees) | Baseline; swappable for CSP+LDA, EEGNet, etc. |
| Label source | Cursor->Target direction vector, averaged per window | Confirmed design choice (session discussion) — cursorvel rejected as label source since README states it is "the scaled outputs from the online DL decoders" (old decoder output, not neutral ground truth) |
| Class binning | 4 classes, 90-degree sectors (Right/Up/Left/Down) | User-confirmed |
| Train/val/test split | 60/20/20 by trial (whole `TrialStart`-`TrialEnd` segments assigned to one split each, shuffled with fixed seed) | Avoids window-overlap leakage (windows overlap 50% within a trial) while allowing an exact 60/20/20 ratio; by-session was rejected because only 4 sessions exist and Se01 uses a different decoder set than Se02-04, confounding split with decoder identity |
| Cursor step mapping | Fixed-magnitude step per predicted class | Produces discrete/stepped movement — a known limitation of classification-driven control (discussed earlier); smoothing parameter available in `CursorController.step()` as a partial mitigation |

## Known limitations

- Untested against real `.mat` structure — MATLAB struct nesting via `scipy.io.loadmat`
  can behave differently across MATLAB versions; verify field access on a real file first.
- Cursor->target label at window edges: if cursor and target coincide exactly
  (net direction vector = zero), that window is **dropped** rather than
  arbitrarily assigned a class (see `labeling.py`).
- No artifact rejection (eye blink/EMG) is implemented — the README does not
  document any such step being pre-applied, and none is added here. This is
  a reasonable next addition if classification accuracy on real data is poor.
- Discrete classification will produce stepped, not smooth, cursor movement
  by design — see the smoothing parameter in `CursorController` and the
  hybrid classifier+regressor extension discussed earlier as ways to address this.
