"""
src/hyperparameter_search.py
==============================
Performs SENSITIVITY ANALYSIS on the LSM reservoir's key hyperparameters,
one at a time. This is a standard research methodology called a
"one-at-a-time (OAT) sweep": vary ONE parameter across a range of values
while holding all OTHERS fixed at their nominal (default) value, and
observe the effect on classification accuracy.

WHY THIS MATTERS FOR THE PAPER:
Simply reporting "our LSM achieves X% accuracy" tells a reader very
little about WHY it works or how ROBUST that result is to the specific
choice of hyperparameters. A sensitivity study answers questions like:
    - Does accuracy improve smoothly with more neurons, or does it
      plateau (telling us how big a reservoir is actually "worth it")?
    - Is there really a performance peak near spectral radius ~0.9 (the
      "edge of chaos" hypothesis), or is the relationship flatter than
      theory predicts?
    - How sensitive is the result to connectivity sparsity -- do we need
      a finely-tuned p_conn, or is a wide range "good enough"?

THE FIVE PARAMETERS SWEPT (matching the project's required scope):
    1. N               -- reservoir size (number of neurons)
    2. spectral_radius (rho) -- recurrent weight scaling
    3. p_conn           -- connection probability (sparsity)
    4. input_scale (alpha)   -- input drive strength
    5. tau_m             -- membrane time constant

FOR SPEED, this sweep uses a SMALL subset of data (configurable in
config.py's HYPERPARAM_SWEEP_CONFIG) -- sensitivity studies are about
relative trends across many configurations, not about squeezing out the
absolute best possible accuracy, so a smaller, faster-to-process dataset
is standard and appropriate practice here.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import CONFIG, set_global_seed, ensure_directories_exist
from src.reservoir import LSMReservoir
from src.encoding import hybrid_encode
from src.readouts import build_jamming_readout
from src.synthetic_ew_dataset import load_synthetic_dataset


# ==============================================================================
# Core Evaluation Function for One Hyperparameter Configuration
# ==============================================================================

def evaluate_one_configuration(
    X_signals: np.ndarray,
    y_labels: np.ndarray,
    reservoir_kwargs: dict,
    n_timesteps: int,
    test_size: float = 0.25,
    seed: int = None,
) -> float:
    """
    Builds a reservoir with the given hyperparameters, encodes + processes
    every signal through it, trains a readout classifier, and returns the
    resulting TEST accuracy. This single function is called repeatedly,
    once per hyperparameter VALUE being tested, with all other parameters
    held at their nominal values (passed in via reservoir_kwargs).
    """
    if seed is None:
        seed = CONFIG["seed"]

    rng = np.random.default_rng(seed)

    reservoir = LSMReservoir(seed=seed, **reservoir_kwargs)

    states = np.zeros((len(X_signals), reservoir.N * 4))
    for i, signal in enumerate(X_signals):
        spikes = hybrid_encode(signal, rng=rng)
        states[i] = reservoir.simulate(spikes, n_timesteps=n_timesteps)

    X_train, X_test, y_train, y_test = train_test_split(
        states, y_labels, test_size=test_size, random_state=seed, stratify=y_labels
    )

    readout = build_jamming_readout()
    readout.fit(X_train, y_train)
    y_pred = readout.predict(X_test)

    return accuracy_score(y_test, y_pred)


# ==============================================================================
# Sweep Functions -- One Per Hyperparameter
# ==============================================================================

def _get_sweep_subset(n_samples: int, seed: int = None):
    """
    Loads (or generates) the synthetic jamming dataset and returns a
    random, class-balanced-as-possible subset of size n_samples, used
    consistently across every sweep so results are comparable.
    """
    if seed is None:
        seed = CONFIG["seed"]

    X, Y, Z, jam_names = load_synthetic_dataset()

    rng = np.random.default_rng(seed)
    n_available = len(X)
    n_samples = min(n_samples, n_available)
    indices = rng.choice(n_available, size=n_samples, replace=False)

    return X[indices], Y[indices]


def sweep_reservoir_size(X_subset, y_subset, nominal_kwargs: dict, n_timesteps: int) -> list:
    """Sweeps N (reservoir size) while holding all other parameters fixed."""
    results = []
    for N in CONFIG["hyperparam_sweep"]["N_values"]:
        kwargs = {**nominal_kwargs, "N": N}
        start = time.time()
        acc = evaluate_one_configuration(X_subset, y_subset, kwargs, n_timesteps)
        elapsed = time.time() - start
        print(f"  N={N:>5d} -> accuracy={acc:.4f}  ({elapsed:.1f}s)")
        results.append({"parameter": "N", "value": N, "accuracy": acc, "elapsed_seconds": elapsed})
    return results


def sweep_spectral_radius(X_subset, y_subset, nominal_kwargs: dict, n_timesteps: int) -> list:
    """Sweeps the spectral radius (rho) while holding all other parameters fixed."""
    results = []
    for rho in CONFIG["hyperparam_sweep"]["spectral_radius_values"]:
        kwargs = {**nominal_kwargs, "spectral_radius": rho}
        start = time.time()
        acc = evaluate_one_configuration(X_subset, y_subset, kwargs, n_timesteps)
        elapsed = time.time() - start
        print(f"  rho={rho:.2f} -> accuracy={acc:.4f}  ({elapsed:.1f}s)")
        results.append({"parameter": "spectral_radius", "value": rho, "accuracy": acc, "elapsed_seconds": elapsed})
    return results


def sweep_connection_sparsity(X_subset, y_subset, nominal_kwargs: dict, n_timesteps: int) -> list:
    """Sweeps p_conn (connection probability) while holding all other parameters fixed."""
    results = []
    for p_conn in CONFIG["hyperparam_sweep"]["p_conn_values"]:
        kwargs = {**nominal_kwargs, "p_conn": p_conn}
        start = time.time()
        acc = evaluate_one_configuration(X_subset, y_subset, kwargs, n_timesteps)
        elapsed = time.time() - start
        print(f"  p_conn={p_conn:.2f} -> accuracy={acc:.4f}  ({elapsed:.1f}s)")
        results.append({"parameter": "p_conn", "value": p_conn, "accuracy": acc, "elapsed_seconds": elapsed})
    return results


def sweep_input_scale(X_subset, y_subset, nominal_kwargs: dict, n_timesteps: int) -> list:
    """Sweeps input_scale (alpha) while holding all other parameters fixed."""
    results = []
    for alpha in CONFIG["hyperparam_sweep"]["input_scale_values"]:
        kwargs = {**nominal_kwargs, "input_scale": alpha}
        start = time.time()
        acc = evaluate_one_configuration(X_subset, y_subset, kwargs, n_timesteps)
        elapsed = time.time() - start
        print(f"  alpha={alpha:.2f} -> accuracy={acc:.4f}  ({elapsed:.1f}s)")
        results.append({"parameter": "input_scale", "value": alpha, "accuracy": acc, "elapsed_seconds": elapsed})
    return results


def sweep_membrane_time_constant(X_subset, y_subset, nominal_kwargs: dict, n_timesteps: int) -> list:
    """
    Sweeps tau_m (membrane time constant). NOTE: since the reservoir uses
    DIFFERENT tau_m for excitatory vs inhibitory neurons by default
    (tau_m_exc_ms and tau_m_inh_ms), for this sweep we set BOTH to the
    same swept value, so we are testing the effect of "overall" membrane
    time constant magnitude rather than the excitatory/inhibitory split.
    """
    results = []
    for tau_m in CONFIG["hyperparam_sweep"]["tau_m_values"]:
        kwargs = {**nominal_kwargs, "tau_m_exc_ms": tau_m, "tau_m_inh_ms": tau_m}
        start = time.time()
        acc = evaluate_one_configuration(X_subset, y_subset, kwargs, n_timesteps)
        elapsed = time.time() - start
        print(f"  tau_m={tau_m:.1f}ms -> accuracy={acc:.4f}  ({elapsed:.1f}s)")
        results.append({"parameter": "tau_m", "value": tau_m, "accuracy": acc, "elapsed_seconds": elapsed})
    return results


# ==============================================================================
# Main Sweep Orchestration
# ==============================================================================

def run_full_hyperparameter_sweep(output_csv: str = None) -> pd.DataFrame:
    """
    Runs ALL FIVE hyperparameter sweeps in sequence and saves the combined
    results to a single CSV file.
    """
    set_global_seed()
    ensure_directories_exist()

    sweep_cfg = CONFIG["hyperparam_sweep"]
    reservoir_cfg = CONFIG["reservoir"]

    print("Loading data subset for hyperparameter sweep...")
    X_subset, y_subset = _get_sweep_subset(sweep_cfg["n_samples_for_sweep"])
    print(f"Using {len(X_subset)} examples for the sweep.\n")

    # Nominal (default) reservoir settings -- held fixed except for
    # whichever single parameter is currently being swept.
    nominal_kwargs = {
        "ei_ratio": reservoir_cfg["ei_ratio"],
        "p_conn": reservoir_cfg["p_conn"],
        "spectral_radius": reservoir_cfg["spectral_radius"],
        "tau_m_exc_ms": reservoir_cfg["tau_m_exc_ms"],
        "tau_m_inh_ms": reservoir_cfg["tau_m_inh_ms"],
        "v_th": reservoir_cfg["v_th"],
        "v_reset": reservoir_cfg["v_reset"],
        "tau_ref_ms": reservoir_cfg["tau_ref_ms"],
        "dt_ms": reservoir_cfg["dt_ms"],
        "input_scale": reservoir_cfg["input_scale"],
        "N": reservoir_cfg["N"],
    }
    n_timesteps = reservoir_cfg["n_timesteps"]

    all_results = []

    print("=" * 70)
    print("SWEEP 1/5: Reservoir size (N)")
    print("=" * 70)
    kwargs_no_N = {k: v for k, v in nominal_kwargs.items() if k != "N"}
    all_results += sweep_reservoir_size(X_subset, y_subset, kwargs_no_N, n_timesteps)

    print("\n" + "=" * 70)
    print("SWEEP 2/5: Spectral radius (rho)")
    print("=" * 70)
    kwargs_no_rho = {k: v for k, v in nominal_kwargs.items() if k != "spectral_radius"}
    all_results += sweep_spectral_radius(X_subset, y_subset, kwargs_no_rho, n_timesteps)

    print("\n" + "=" * 70)
    print("SWEEP 3/5: Connection sparsity (p_conn)")
    print("=" * 70)
    kwargs_no_p = {k: v for k, v in nominal_kwargs.items() if k != "p_conn"}
    all_results += sweep_connection_sparsity(X_subset, y_subset, kwargs_no_p, n_timesteps)

    print("\n" + "=" * 70)
    print("SWEEP 4/5: Input scale (alpha)")
    print("=" * 70)
    kwargs_no_alpha = {k: v for k, v in nominal_kwargs.items() if k != "input_scale"}
    all_results += sweep_input_scale(X_subset, y_subset, kwargs_no_alpha, n_timesteps)

    print("\n" + "=" * 70)
    print("SWEEP 5/5: Membrane time constant (tau_m)")
    print("=" * 70)
    kwargs_no_tau = {
        k: v for k, v in nominal_kwargs.items()
        if k not in ("tau_m_exc_ms", "tau_m_inh_ms")
    }
    all_results += sweep_membrane_time_constant(X_subset, y_subset, kwargs_no_tau, n_timesteps)

    df = pd.DataFrame(all_results)

    if output_csv is None:
        output_csv = os.path.join(CONFIG["paths"]["results_dir"], "hyperparameter_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nFull sweep results saved to: {output_csv}")

    return df


# ==============================================================================
# Script Entry Point
# ==============================================================================

if __name__ == "__main__":
    df = run_full_hyperparameter_sweep()
    print("\nSummary of best value found per parameter:")
    for param in df["parameter"].unique():
        sub = df[df["parameter"] == param]
        best_row = sub.loc[sub["accuracy"].idxmax()]
        print(f"  {param}: best value = {best_row['value']}, accuracy = {best_row['accuracy']:.4f}")
