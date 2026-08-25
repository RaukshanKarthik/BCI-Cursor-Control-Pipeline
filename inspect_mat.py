"""
inspect_mat.py

Read-only inspection tool. Does NOT modify anything. Prints the real
structure (keys, dtypes, shapes) of the "eeg" group in a real S18 .mat
file, so the pipeline's data_loader.py can be fixed against ACTUAL
structure instead of assumptions.

Usage:
    python inspect_mat.py data/S18_extracted/S18/S18_Se02_CL_R03.mat
"""
import sys
import h5py
import numpy as np


def describe(name, obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, h5py.Dataset):
        print(f"{pad}[Dataset] {name}  shape={obj.shape}  dtype={obj.dtype}")
        # If it looks small, show a tiny preview
        try:
            if obj.size > 0 and obj.size <= 5:
                print(f"{pad}    value preview: {obj[()]}")
        except Exception as e:
            print(f"{pad}    (couldn't preview: {e})")
    elif isinstance(obj, h5py.Group):
        print(f"{pad}[Group]   {name}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_mat.py <path_to_mat_file>")
        sys.exit(1)

    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        print("=" * 70)
        print(f"TOP-LEVEL KEYS in {path}:")
        print(list(f.keys()))
        print("=" * 70)

        if "eeg" not in f:
            print("No top-level 'eeg' group found! Top-level keys were printed above.")
            return

        eeg = f["eeg"]
        print("\nKEYS INSIDE 'eeg':")
        print(list(eeg.keys()))
        print("=" * 70)

        print("\nFULL STRUCTURE of 'eeg' (recursive):")
        eeg.visititems(lambda name, obj: describe(name, obj, indent=1))

        print("\n" + "=" * 70)
        print("DETAILED LOOK at specific fields the pipeline reads:")
        for field in ["subject", "session", "run", "decoder", "study", "fs", "data", "times"]:
            if field in eeg:
                item = eeg[field]
                print(f"\n--- eeg/{field} ---")
                if isinstance(item, h5py.Dataset):
                    print(f"  type: Dataset, shape={item.shape}, dtype={item.dtype}")
                    try:
                        val = item[()]
                        print(f"  raw value (first bit): {np.asarray(val).flatten()[:10]}")
                    except Exception as e:
                        print(f"  (couldn't read value: {e})")
                elif isinstance(item, h5py.Group):
                    print(f"  type: Group, sub-keys: {list(item.keys())}")
            else:
                print(f"\n--- eeg/{field} --- NOT FOUND")

        print("\n" + "=" * 70)
        print("event / cursorpos / targetpos / channellabels structure:")
        for field in ["event", "cursorpos", "targetpos", "cursorvel", "channellabels", "postimes"]:
            if field in eeg:
                item = eeg[field]
                print(f"\n--- eeg/{field} ---")
                if isinstance(item, h5py.Dataset):
                    print(f"  type: Dataset, shape={item.shape}, dtype={item.dtype}")
                elif isinstance(item, h5py.Group):
                    print(f"  type: Group, sub-keys: {list(item.keys())}")
                    for sk in item.keys():
                        sub = item[sk]
                        if isinstance(sub, h5py.Dataset):
                            print(f"    {sk}: Dataset shape={sub.shape} dtype={sub.dtype}")
            else:
                print(f"\n--- eeg/{field} --- NOT FOUND")


if __name__ == "__main__":
    main()
