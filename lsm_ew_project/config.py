"""
config.py
==========
Central configuration file for the entire LSM-EW project.

WHY THIS FILE EXISTS:
Instead of hard-coding numbers (like "1000 neurons" or "spectral radius 0.9")
inside every script, we define them ONCE here. Every other file imports
from this file. This means:
  1. You change a setting in exactly one place.
  2. Every experiment uses consistent settings unless you deliberately
     override them (e.g. during the hyperparameter sweep).
  3. Anyone reading this file can understand the entire experimental
     setup of the project in under five minutes.

HOW TO USE:
    from config import CONFIG
    n_neurons = CONFIG["reservoir"]["N"]
"""

import os
import random
import numpy as np

# ----------------------------------------------------------------------------
# Reproducibility: Random Seed
# ----------------------------------------------------------------------------
# A "random seed" is a starting number for a pseudo-random number generator.
# If we always start from the same seed, every "random" choice the computer
# makes (shuffling data, initialising weights, etc.) becomes EXACTLY repeatable.
# This is essential for research: someone else (or you, six months later)
# should be able to re-run your code and get the same numbers.
GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """
    Set the random seed for every library that generates randomness in
    this project (Python's built-in random, NumPy, and PyTorch if available).

    Call this function at the very start of every script, before generating
    any data or building any model.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # no-op safely if no GPU present
    except ImportError:
        # torch might not be installed yet when this is first imported;
        # that's fine, we just skip GPU seeding in that case.
        pass


# ----------------------------------------------------------------------------
# Project Directory Paths
# ----------------------------------------------------------------------------
# os.path.dirname(__file__) gives the folder this config.py file lives in.
# We build every other path RELATIVE to that, so the project works no matter
# where you copy the folder on your computer.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "project_root": PROJECT_ROOT,
    "data_root": os.path.join(PROJECT_ROOT, "data"),
    "radioml_dir": os.path.join(PROJECT_ROOT, "data", "radioml"),
    "radioml_file": os.path.join(
        PROJECT_ROOT, "data", "radioml", "GOLD_XYZ_OSC.0001_1024.hdf5"
    ),
    "synthetic_dir": os.path.join(PROJECT_ROOT, "data", "synthetic_ew"),
    "results_dir": os.path.join(PROJECT_ROOT, "results"),
    "figures_dir": os.path.join(PROJECT_ROOT, "results", "figures"),
    "reports_dir": os.path.join(PROJECT_ROOT, "results", "reports"),
    "logs_dir": os.path.join(PROJECT_ROOT, "results", "logs"),
    "models_dir": os.path.join(PROJECT_ROOT, "models"),
    "configs_dir": os.path.join(PROJECT_ROOT, "configs"),
}


def ensure_directories_exist() -> None:
    """
    Create every directory listed in PATHS if it does not already exist.
    This means you never have to manually create folders — every script
    that needs to write a file calls this first.
    """
    for key, path in PATHS.items():
        if key.endswith("_dir") or key.endswith("_root"):
            os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------------
# RadioML 2018.01A Dataset Configuration
# ----------------------------------------------------------------------------
# RadioML 2018.01A is a public benchmark dataset of radio modulation signals.
# Each example is a complex-valued time series of 1024 samples (I and Q
# channels), labelled with one of 24 modulation types, recorded at SNR
# (Signal-to-Noise Ratio) values from -20 dB to +30 dB.
RADIOML_CONFIG = {
    "n_samples_per_signal": 1024,   # number of IQ samples in each signal
    "n_channels": 2,                # I (in-phase) and Q (quadrature)
    "n_classes": 24,                # 24 modulation types
    "modulation_names": [
        "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
        "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
        "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
        "AM-DSB-SC", "FM", "GMSK", "OQPSK",
    ],
    "snr_range": list(range(-20, 32, 2)),   # -20 dB to +30 dB in 2 dB steps
    # For initial experiments we use a SUBSET of the full 2.5 million
    # examples (full dataset is ~18 GB in memory) to keep things tractable
    # on a normal laptop. Increase this once your pipeline is validated.
    "n_samples_subset": 50_000,
    "min_snr_for_subset": 0,   # only use SNR >= 0 dB for the main experiments
}

# ----------------------------------------------------------------------------
# Synthetic Electronic Warfare (EW) Dataset Configuration
# ----------------------------------------------------------------------------
SYNTHETIC_EW_CONFIG = {
    "n_samples_per_signal": 1024,
    "n_channels": 2,
    "jamming_types": [
        "tone", "sweep", "noise", "repeat", "pulse", "barrage",
    ],
    "n_classes": 6,
    "snr_levels_db": list(range(-10, 25, 5)),   # -10, -5, 0, 5, 10, 15, 20
    "n_examples_per_class_per_snr": 800,
    # 6 classes x 7 snr levels x 800 = 33,600 total examples
    "n_emitter_profiles": 10,   # for the emitter-identification task
}

# ----------------------------------------------------------------------------
# Spike Encoding Configuration
# ----------------------------------------------------------------------------
ENCODING_CONFIG = {
    "methods": ["rate", "ttfs", "hybrid"],
    "rate_lambda_max": 0.9,     # maximum firing probability per timestep
    "ttfs_block_size": 8,       # samples grouped per TTFS "neuron"
    "hybrid_theta": 0.5,        # amplitude threshold separating TTFS/rate regimes
    "n_timesteps": 128,         # number of simulation timesteps used (T)
}

# ----------------------------------------------------------------------------
# Liquid State Machine (LSM) Reservoir Configuration
# ----------------------------------------------------------------------------
# These are the NOMINAL (default) reservoir parameters. The hyperparameter
# sweep (see src/hyperparameter_search.py) will vary these one at a time
# to study sensitivity, but every other experiment uses these defaults.
RESERVOIR_CONFIG = {
    "N": 500,                  # total number of LIF neurons in the reservoir
    "ei_ratio": 0.8,             # 80% excitatory, 20% inhibitory (Dale's principle)
    "p_conn": 0.10,               # connection probability (sparsity) between neurons
    "spectral_radius": 0.9,      # scales recurrent weights -> controls reservoir dynamics
    "tau_m_exc_ms": 20.0,         # membrane time constant for excitatory neurons (ms)
    "tau_m_inh_ms": 10.0,         # membrane time constant for inhibitory neurons (ms)
    "v_th": 1.0,                  # firing threshold (normalised units)
    "v_reset": 0.0,               # reset potential after a spike
    "tau_ref_ms": 2.0,             # refractory period (ms) -- neuron cannot fire again
    "dt_ms": 1.0,                  # simulation timestep size (ms)
    # input_scale controls how strongly a single input spike drives a
    # reservoir neuron's membrane potential. NOTE: this must be tuned
    # relative to v_th (the firing threshold) and the leak factor alpha.
    # Because we use sparse binary spike inputs (mostly zeros) combined
    # with the exponential-Euler update V = alpha*V + (1-alpha)*I, a
    # SINGLE input spike's instantaneous contribution to V is only
    # (1-alpha) * input_scale -- which is a SMALL fraction unless
    # input_scale is large enough to compensate. A value of 5.0-8.0
    # (rather than the more "intuitive"-looking small value 0.1) is
    # what is actually needed here to bring neurons close to threshold
    # over realistic spike rates. This was confirmed empirically: with
    # input_scale=0.1 the reservoir NEVER fires (dead network); with
    # input_scale=6.0 it produces a healthy 5-20% active fraction.
    "input_scale": 6.0,
    "n_timesteps": 128,             # T -- how many timesteps to simulate per input signal
}

# ----------------------------------------------------------------------------
# Hyperparameter Sweep Configuration
# ----------------------------------------------------------------------------
HYPERPARAM_SWEEP_CONFIG = {
    "N_values": [100, 200, 500, 1000, 2000],
    "spectral_radius_values": [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5],
    "p_conn_values": [0.02, 0.05, 0.10, 0.20, 0.40],
    "input_scale_values": [1.0, 3.0, 6.0, 10.0, 15.0],
    "tau_m_values": [5.0, 10.0, 20.0, 40.0, 80.0],
    # For speed, the sweep uses a SMALL subset of data (this is standard
    # practice -- sensitivity studies don't need the full dataset, they
    # need many configurations tested quickly).
    "n_samples_for_sweep": 2000,
}

# ----------------------------------------------------------------------------
# Baseline Model Configuration
# ----------------------------------------------------------------------------
BASELINE_CONFIG = {
    "cnn": {
        "n_epochs": 15,
        "batch_size": 256,
        "learning_rate": 1e-3,
    },
    "lstm": {
        "n_epochs": 5,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "hidden_size": 64,
        "num_layers": 1,
    },
    "esn": {
        # Echo State Network uses the SAME reservoir code as the LSM but
        # with continuous (non-spiking) tanh neurons instead of LIF spiking
        # neurons -- this is the classic "rate-based" reservoir computing
        # model that LSMs were inspired by.
        "N": 1000,
        "spectral_radius": 0.9,
        "p_conn": 0.10,
    },
    "svm_expert_features": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
    },
}

# ----------------------------------------------------------------------------
# Energy Estimation Configuration
# ----------------------------------------------------------------------------
# These constants come from published hardware specifications and are used
# to ESTIMATE (not measure on real hardware) the energy a neuromorphic chip
# would consume running this workload.
ENERGY_CONFIG = {
    # Energy per synaptic operation on Intel Loihi 2, from Intel's published
    # characterisation data. 1 pJ (picojoule) = 1e-12 Joules.
    "loihi2_energy_per_synop_joules": 23e-12,
    # Typical GPU thermal design power, used to estimate baseline DL energy.
    "gpu_tdp_watts": 250.0,
    # Typical embedded CPU power draw, used for SVM / expert-feature baseline.
    "cpu_tdp_watts": 15.0,
}

# ----------------------------------------------------------------------------
# Adversarial Robustness Configuration
# ----------------------------------------------------------------------------
ADVERSARIAL_CONFIG = {
    "awgn_attack_snr_db": [10, 5, 0, -5, -10],
    "freq_perturbation_max_offset": 0.02,   # max fractional frequency shift
    "distortion_levels": [0.0, 0.1, 0.2, 0.3, 0.5],  # nonlinear distortion strength
}

# ----------------------------------------------------------------------------
# Bundle everything into a single CONFIG dictionary for convenient importing
# ----------------------------------------------------------------------------
CONFIG = {
    "seed": GLOBAL_SEED,
    "paths": PATHS,
    "radioml": RADIOML_CONFIG,
    "synthetic_ew": SYNTHETIC_EW_CONFIG,
    "encoding": ENCODING_CONFIG,
    "reservoir": RESERVOIR_CONFIG,
    "hyperparam_sweep": HYPERPARAM_SWEEP_CONFIG,
    "baseline": BASELINE_CONFIG,
    "energy": ENERGY_CONFIG,
    "adversarial": ADVERSARIAL_CONFIG,
}


if __name__ == "__main__":
    # Running "python config.py" directly will just print the configuration
    # and create all directories -- a quick way to sanity-check the setup.
    set_global_seed()
    ensure_directories_exist()
    print("Configuration loaded successfully.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Global seed: {GLOBAL_SEED}")
    print("\nDirectory structure created:")
    for key, path in PATHS.items():
        if key.endswith("_dir") or key.endswith("_root"):
            exists = "OK" if os.path.isdir(path) else "MISSING"
            print(f"  [{exists}] {path}")
