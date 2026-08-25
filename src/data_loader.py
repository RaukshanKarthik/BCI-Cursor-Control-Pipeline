"""
data_loader.py

Loads S18 MATLAB v7.3/HDF5 EEG run files using h5py.

The real S18 files contain:

    eeg/data          : (samples, channels) = (T, 62)
    eeg/times         : sample indices
    eeg/fs            : 1000 Hz
    eeg/event/*       : MATLAB object references
    eeg/cursorpos/*
    eeg/targetpos/*
    eeg/postimes      : sample indices at 25 Hz
    eeg/channellabels : MATLAB object references

The rest of the pipeline expects EEG data as (channels, samples).
"""

import os
import zipfile
import glob

import h5py
import numpy as np


def unzip_dataset(zip_path: str, extract_to: str) -> str:
    """Unzip a subject's dataset archive into extract_to."""
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    return extract_to


def list_run_files(directory: str, subject_id: str) -> list:
    """Find all .mat run files for a given subject."""
    pattern = os.path.join(directory, f"{subject_id}_*.mat")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No .mat files found for subject '{subject_id}' "
            f"in '{directory}'."
        )

    return files


def _decode_char_field(dataset) -> str:
    """
    Decode a MATLAB char-array field stored DIRECTLY as uint16 character
    codes (not as an HDF5 reference requiring lookup).

    [VERIFIED against a real S18 file — 2026-08-25] Inspecting the real
    file's structure (data/S18_extracted/S18/S18_Se02_CL_R03.mat) showed
    that eeg/subject, eeg/session, eeg/run, eeg/decoder, eeg/study are
    each a plain (N, 1) uint16 dataset containing the character codes of
    the string directly (e.g. eeg/subject = [83, 49, 56] = "S18").
    This REPLACES the original assumption that these fields held object
    references needing a second h5file[ref] lookup — that assumption was
    wrong and caused a TypeError on real data.
    """
    arr = np.asarray(dataset[()]).flatten()
    return "".join(chr(int(x)) for x in arr)


def _decode_matlab_ref(h5file, ref):
    """
    Decode a MATLAB v7.3 object reference (used for fields that DO
    require a lookup, e.g. per-element entries inside event/* and
    channellabels — each element there is a separate HDF5 reference,
    unlike subject/session/run/decoder/study which store their value
    directly; see _decode_char_field above).

    MATLAB strings are stored as uint16 character codes.
    Numeric values are returned as numpy arrays.
    """
    obj = h5file[ref]
    arr = np.asarray(obj[()])

    # MATLAB character/string data.
    if np.issubdtype(arr.dtype, np.integer):
        values = arr.flatten()

        # Convert MATLAB UTF-16/ASCII character codes to Python string.
        try:
            return "".join(chr(int(x)) for x in values)
        except Exception:
            return values

    return arr


def _decode_scalar_ref(h5file, ref):
    """Decode a reference containing a scalar numeric value."""
    arr = np.asarray(h5file[ref][()])
    return float(arr.reshape(-1)[0])


def _decode_event_refs(h5file, refs, decode_as_string=False):
    """Decode a column of MATLAB object references."""
    result = []

    for ref in refs:
        if decode_as_string:
            result.append(_decode_matlab_ref(h5file, ref))
        else:
            result.append(_decode_scalar_ref(h5file, ref))

    return result


def load_run(mat_path: str) -> dict:
    """
    Load one real S18 MATLAB v7.3 run.

    Returns
    -------
    dict
        data: (62, T)
        times: (T,)
        event_latency: sample indices
        event_duration: sample counts
        event_type: list[str]
        fs: sampling frequency
        cursor/target position arrays at 25 Hz
        channel labels
    """

    with h5py.File(mat_path, "r") as f:
        eeg = f["eeg"]

        # --------------------------------------------------------------
        # EEG
        # --------------------------------------------------------------
        #
        # HDF5 file stores:
        #     (samples, channels)
        #
        # Pipeline expects:
        #     (channels, samples)
        #
        data = np.asarray(eeg["data"][()], dtype=float).T

        times = np.asarray(eeg["times"][()], dtype=float).reshape(-1)

        fs = float(np.asarray(eeg["fs"][()]).reshape(-1)[0])

        # --------------------------------------------------------------
        # Events
        # --------------------------------------------------------------
        latency_refs = eeg["event"]["latency"][:, 0]
        duration_refs = eeg["event"]["duration"][:, 0]
        type_refs = eeg["event"]["type"][:, 0]

        event_latency = np.array(
            _decode_event_refs(
                f,
                latency_refs,
                decode_as_string=False,
            ),
            dtype=float,
        )

        event_duration = np.array(
            _decode_event_refs(
                f,
                duration_refs,
                decode_as_string=False,
            ),
            dtype=float,
        )

        event_type = _decode_event_refs(
            f,
            type_refs,
            decode_as_string=True,
        )

        # --------------------------------------------------------------
        # Channel labels
        # --------------------------------------------------------------
        channel_refs = eeg["channellabels"][0]

        channellabels = [
            _decode_matlab_ref(f, ref)
            for ref in channel_refs
        ]

        # --------------------------------------------------------------
        # Position / cursor data
        # --------------------------------------------------------------
        cursorpos_x = np.asarray(
            eeg["cursorpos"]["x"][()], dtype=float
        ).reshape(-1)

        cursorpos_y = np.asarray(
            eeg["cursorpos"]["y"][()], dtype=float
        ).reshape(-1)

        targetpos_x = np.asarray(
            eeg["targetpos"]["x"][()], dtype=float
        ).reshape(-1)

        targetpos_y = np.asarray(
            eeg["targetpos"]["y"][()], dtype=float
        ).reshape(-1)

        cursorvel_x = np.asarray(
            eeg["cursorvel"]["x"][()], dtype=float
        ).reshape(-1)

        cursorvel_y = np.asarray(
            eeg["cursorvel"]["y"][()], dtype=float
        ).reshape(-1)

        postimes = np.asarray(
            eeg["postimes"][()], dtype=float
        ).reshape(-1)

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------
        subject = _decode_char_field(eeg["subject"])
        session = _decode_char_field(eeg["session"])
        run = _decode_char_field(eeg["run"])
        decoder = _decode_char_field(eeg["decoder"])
        study = _decode_char_field(eeg["study"])

        return {
            "data": data,
            "times": times,
            "event_latency": event_latency,
            "event_duration": event_duration,
            "event_type": event_type,
            "subject": subject,
            "session": session,
            "run": run,
            "decoder": decoder,
            "fs": fs,
            "study": study,
            "cursorpos_x": cursorpos_x,
            "cursorpos_y": cursorpos_y,
            "targetpos_x": targetpos_x,
            "targetpos_y": targetpos_y,
            "postimes": postimes,
            "channellabels": channellabels,
            "cursorvel_x": cursorvel_x,
            "cursorvel_y": cursorvel_y,
            "source_file": os.path.basename(mat_path),
        }


def load_all_runs(directory: str, subject_id: str) -> list:
    """Load every run file for a subject."""
    files = list_run_files(directory, subject_id)

    runs = []

    for f in files:
        try:
            runs.append(load_run(f))
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")

    return runs


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m src.data_loader "
            "<path_to_single_mat_file>"
        )
        sys.exit(1)

    r = load_run(sys.argv[1])

    print("Loaded run:", r["source_file"])

    for k, v in r.items():
        if isinstance(v, np.ndarray):
            print(
                f"  {k}: shape={v.shape}, "
                f"dtype={v.dtype}"
            )
        else:
            print(f"  {k}: {v}")