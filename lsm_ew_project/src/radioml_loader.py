"""
src/radioml_loader.py
=======================
Handles loading and preprocessing of the RadioML 2018.01A dataset --
the public benchmark dataset used for the MODULATION CLASSIFICATION
task (Head 1 of the multi-task readout).

ABOUT THE DATASET:
RadioML 2018.01A contains 2,555,904 IQ signal examples spanning 24
modulation types and 26 SNR levels (-20 dB to +30 dB in 2 dB steps). It
is distributed as a single HDF5 file (a binary format for storing large
numerical arrays efficiently, commonly used in scientific computing).

WHERE TO GET THE FILE:
    1. Go to https://www.deepsig.ai/datasets (search "RadioML 2018.01A"
       if the link moves -- DeepSig periodically reorganises their site).
    2. Download the file "2018.01A.tar.bz2" (approximately 6 GB).
    3. Extract it. Inside, you will find a file named exactly:
           GOLD_XYZ_OSC.0001_1024.hdf5
    4. Place that .hdf5 file at:
           <project_root>/data/radioml/GOLD_XYZ_OSC.0001_1024.hdf5
This exact path is defined in config.py as PATHS["radioml_file"].

DATASET INTERNAL STRUCTURE (inside the HDF5 file):
    'X' : shape (2555904, 1024, 2)  -- the IQ signals themselves
          (1024 time samples, 2 channels: [I, Q])
    'Y' : shape (2555904, 24)        -- ONE-HOT encoded modulation labels
          (e.g. [0,0,1,0,...,0] means "this example is class index 2")
    'Z' : shape (2555904, 1)         -- SNR value (dB) for each example

WHAT THIS FILE DOES:
Since the full dataset (2.5 million examples) is large (~18 GB once
loaded into memory as float arrays) and slow to process through the
LSM reservoir, this module provides functions to load a MANAGEABLE
SUBSET -- e.g. 50,000 examples, filtered to SNR >= a chosen threshold
-- which is sufficient for a BTech research project while keeping
experiment turnaround time reasonable. The subset size and SNR filter
are configurable in config.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import CONFIG


# ==============================================================================
# Dataset Availability Check
# ==============================================================================

def check_radioml_available(filepath: str = None) -> bool:
    """Returns True if the RadioML HDF5 file exists at the expected path."""
    if filepath is None:
        filepath = CONFIG["paths"]["radioml_file"]
    return os.path.isfile(filepath)


def print_radioml_missing_help(filepath: str = None):
    """
    Prints a clear, actionable error message explaining exactly how to
    obtain and place the RadioML dataset, if it is missing. This
    satisfies the project requirement: "If the dataset is missing,
    display a helpful error message."
    """
    if filepath is None:
        filepath = CONFIG["paths"]["radioml_file"]

    message = f"""
{'=' * 78}
RADIOML 2018.01A DATASET NOT FOUND
{'=' * 78}

Expected file location:
    {filepath}

This file was NOT found. To fix this:

  STEP 1: Download the dataset
      Visit: https://www.deepsig.ai/datasets
      Search for "RadioML 2018.01A" (sometimes listed as "2018.01A").
      Download the archive file (approximately 6 GB), typically named
      something like "2018.01A.tar.bz2".

  STEP 2: Extract the archive
      Use any archive tool (7-Zip on Windows, Archive Utility on Mac,
      or `tar -xjf 2018.01A.tar.bz2` on Linux/Mac terminal).
      Inside, you should find a file named exactly:
          GOLD_XYZ_OSC.0001_1024.hdf5

  STEP 3: Place the file in this project
      Move/copy that .hdf5 file to:
          {os.path.dirname(filepath)}
      so that the final path is exactly:
          {filepath}

  STEP 4: Re-run your script
      Once the file is in place, re-run whatever script you were
      running -- it will detect the dataset automatically.

NOTE: The synthetic EW jamming and emitter datasets (used for the
jamming-classification and emitter-ID tasks) do NOT require any
external download -- they are generated automatically by
src/synthetic_ew_dataset.py the first time you run any experiment
script. Only the MODULATION CLASSIFICATION task (Head 1) requires
this external RadioML file.
{'=' * 78}
"""
    print(message)


# ==============================================================================
# Dataset Loading
# ==============================================================================

def load_radioml_subset(
    filepath: str = None,
    n_samples: int = None,
    min_snr: float = None,
    seed: int = None,
    verbose: bool = True,
):
    """
    Loads a random, class-balanced-as-possible SUBSET of the RadioML
    2018.01A dataset, filtered to a minimum SNR.

    WHY FILTER BY MINIMUM SNR FOR THE MAIN EXPERIMENTS?
    Very low SNR examples (e.g. -20 dB, where noise power is 100x the
    signal power) are extremely difficult for ANY classifier -- spiking
    or otherwise -- to get right, and including too many of them in a
    SMALL subset can make the subset unrepresentative of "typical"
    operating conditions. We default to using only SNR >= 0 dB examples
    for the main accuracy benchmarks, but separately evaluate the FULL
    SNR range when producing the "accuracy vs SNR" plot (see
    experiments/evaluate_models.py), which is the scientifically
    important curve for understanding model robustness to noise.

    Parameters
    ----------
    filepath : str, optional
        Path to the HDF5 file. Defaults to config path.
    n_samples : int, optional
        How many examples to load. Defaults to config value.
    min_snr : float, optional
        Minimum SNR (dB) to include. Defaults to config value. Pass
        None explicitly (i.e. min_snr=-100) if you want the FULL SNR
        range (e.g. for the accuracy-vs-SNR sweep).
    seed : int, optional
        Random seed for subset sampling.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 1024, 2)
    Y : np.ndarray, shape (n_samples,)       -- integer class labels (0-23)
    Z : np.ndarray, shape (n_samples,)       -- SNR values in dB
    class_names : list of str                -- the 24 modulation names
    """
    import h5py

    if filepath is None:
        filepath = CONFIG["paths"]["radioml_file"]
    if n_samples is None:
        n_samples = CONFIG["radioml"]["n_samples_subset"]
    if min_snr is None:
        min_snr = CONFIG["radioml"]["min_snr_for_subset"]
    if seed is None:
        seed = CONFIG["seed"]

    if not check_radioml_available(filepath):
        print_radioml_missing_help(filepath)
        raise FileNotFoundError(
            f"RadioML dataset file not found at: {filepath}\n"
            "See the message above for instructions on how to obtain it."
        )

    class_names = CONFIG["radioml"]["modulation_names"]
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Opening RadioML HDF5 file: {filepath}")

    with h5py.File(filepath, "r") as f:
        total_examples = f["X"].shape[0]

        if verbose:
            print(f"Full dataset contains {total_examples:,} examples.")
            print(f"Filtering to SNR >= {min_snr} dB, then sampling {n_samples:,} examples...")

        # Loading the FULL "Z" (SNR) array is cheap -- it's only one
        # float per example, ~2.5M floats = ~20 MB, easily fits in memory.
        # We use this to find which example INDICES satisfy our SNR
        # filter, WITHOUT loading the (much larger) X array yet.
        Z_full = f["Z"][:].flatten()
        valid_indices = np.where(Z_full >= min_snr)[0]

        if verbose:
            print(f"{len(valid_indices):,} examples satisfy the SNR filter.")

        if len(valid_indices) < n_samples:
            print(
                f"WARNING: only {len(valid_indices)} examples satisfy the SNR "
                f"filter, fewer than the requested {n_samples}. Using all "
                "available examples that pass the filter."
            )
            n_samples = len(valid_indices)

        # Randomly choose n_samples indices from the valid pool, WITHOUT
        # replacement (so no example is duplicated).
        chosen_indices = rng.choice(valid_indices, size=n_samples, replace=False)
        # Sort indices because HDF5 fancy-indexing is much faster when
        # indices are in increasing order (avoids excessive random disk
        # seeks).
        chosen_indices = np.sort(chosen_indices)

        if verbose:
            print("Loading selected examples from disk (this may take a moment)...")

        X = f["X"][chosen_indices].astype(np.float32)
        Y_onehot = f["Y"][chosen_indices]
        Z = f["Z"][chosen_indices].flatten().astype(np.float32)

    # Convert one-hot labels (e.g. [0,0,1,0,...]) to a single integer
    # class index (e.g. 2) using argmax, which finds the position of the
    # maximum value (the "1") along each row.
    Y = np.argmax(Y_onehot, axis=1).astype(np.int64)

    if verbose:
        print(f"Loaded subset: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
        unique_classes, counts = np.unique(Y, return_counts=True)
        print(f"Number of distinct classes present in subset: {len(unique_classes)}/24")

    return X, Y, Z, class_names


def save_radioml_subset(X, Y, Z, output_dir: str = None):
    """
    Saves a loaded RadioML subset to .npy files, so subsequent runs can
    load the SUBSET quickly without re-reading from the (large) original
    HDF5 file every time.
    """
    if output_dir is None:
        output_dir = CONFIG["paths"]["radioml_dir"]
    os.makedirs(output_dir, exist_ok=True)

    tag = f"subset_{CONFIG['radioml']['n_samples_subset']}_{CONFIG['radioml']['min_snr_for_subset']}"

    np.save(os.path.join(output_dir, f"{tag}_X.npy"), X)
    np.save(os.path.join(output_dir, f"{tag}_Y.npy"), Y)
    np.save(os.path.join(output_dir, f"{tag}_Z.npy"), Z)
    print(f"Saved RadioML subset (cached) to: {output_dir}")


def load_cached_radioml_subset(data_dir: str = None):
    """
    Loads a previously cached subset (saved via save_radioml_subset), if
    one exists. Returns None if no cache is found, signalling the
    calling code that it should call load_radioml_subset() instead
    (which requires the full HDF5 file).
    """
    if data_dir is None:
        data_dir = CONFIG["paths"]["radioml_dir"]

    tag = f"subset_{CONFIG['radioml']['n_samples_subset']}_{CONFIG['radioml']['min_snr_for_subset']}"

    x_path = os.path.join(data_dir, f"{tag}_X.npy")
    y_path = os.path.join(data_dir, f"{tag}_Y.npy")
    z_path = os.path.join(data_dir, f"{tag}_Z.npy")

    if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(z_path)):
        return None

    X = np.load(x_path)
    Y = np.load(y_path)
    Z = np.load(z_path)
    return X, Y, Z


def get_radioml_data(force_reload: bool = False, verbose: bool = True):
    """
    The MAIN entry point other scripts should use to get RadioML data.
    Tries the cache first (fast); falls back to loading from the
    original HDF5 file (slower, but only happens once) if no cache
    exists or force_reload=True.
    """
    if not force_reload:
        cached = load_cached_radioml_subset()
        if cached is not None:
            if verbose:
                print("Loaded RadioML subset from cache (fast path).")
            X, Y, Z = cached
            class_names = CONFIG["radioml"]["modulation_names"]
            return X, Y, Z, class_names

    X, Y, Z, class_names = load_radioml_subset(verbose=verbose)
    save_radioml_subset(X, Y, Z)
    return X, Y, Z, class_names


# ==============================================================================
# Script Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("Checking for RadioML 2018.01A dataset...\n")

    filepath = CONFIG["paths"]["radioml_file"]

    if not check_radioml_available(filepath):
        print_radioml_missing_help(filepath)
    else:
        print(f"Found RadioML dataset at: {filepath}")
        print("Loading a subset to verify everything works...\n")
        X, Y, Z, class_names = get_radioml_data(verbose=True)
        print("\nDataset loaded successfully.")
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}, unique classes: {np.unique(Y)}")
        print(f"Z (SNR) range: {Z.min()} to {Z.max()} dB")
