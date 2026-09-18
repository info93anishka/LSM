"""
src/energy_analysis.py
========================
Implements ENERGY and LATENCY ANALYSIS comparing the LSM against all
four baseline models (CNN, LSTM, ESN, Expert+SVM).

WHY THIS IS THE MOST IMPORTANT RESULT IN THE WHOLE PROJECT:
Modern deep learning models (CNN, LSTM) often achieve HIGHER accuracy
than spiking/reservoir approaches on many benchmarks. So why would
anyone use an LSM instead? The answer is almost always ENERGY
EFFICIENCY, not accuracy. Neuromorphic hardware (like Intel's Loihi 2
chip) is event-driven: it only consumes energy when a spike actually
occurs, and reservoir-computing approaches like the LSM produce very
SPARSE spike trains (see src/encoding.py's sparsity metrics). This
means that even if the LSM is slightly less accurate than a CNN, it can
be ORDERS OF MAGNITUDE more energy-efficient -- which matters enormously
for battery-powered EW (Electronic Warfare) receivers, drones, or other
power-constrained platforms where a GPU is simply not an option.

WHAT THIS MODULE MEASURES FOR EACH MODEL:
    1. Training time      -- how long training took (wall-clock seconds)
    2. Inference latency   -- time to classify ONE signal (milliseconds)
    3. Spike sparsity       -- (LSM only) fraction of positions with a spike
    4. Estimated energy per inference -- using published hardware
       constants (see config.py's ENERGY_CONFIG)

ENERGY ESTIMATION METHODOLOGY (IMPORTANT CAVEAT):
We do NOT have access to physical Loihi 2 hardware, GPUs with power
meters, etc. for this BTech project. Instead, we use a standard
ESTIMATION methodology found throughout the neuromorphic computing
literature:
    - For the LSM: count total SYNAPTIC OPERATIONS (synops) -- every
      time a spike travels across a synapse -- and multiply by Loihi 2's
      published energy-per-synop figure (23 picojoules, from Intel's
      characterisation data).
    - For GPU-based models (CNN, LSTM): multiply measured wall-clock
      inference TIME by a typical GPU's Thermal Design Power (TDP),
      giving Energy = Power x Time. This is a coarse but standard
      proxy used when direct power measurement hardware isn't available.
    - For the CPU-based SVM baseline: same Energy = Power x Time
      approach, using a typical embedded CPU's TDP.
These are ESTIMATES, not direct hardware measurements -- this
distinction should always be stated clearly when reporting these
numbers in the final paper.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import CONFIG


# ==============================================================================
# LSM Energy Estimation
# ==============================================================================

def estimate_lsm_energy(
    reservoir,
    mean_spike_sparsity: float,
    n_timesteps: int,
) -> dict:
    """
    Estimates the energy consumed by ONE LSM inference (one signal being
    classified) if it were run on Intel Loihi 2 neuromorphic hardware.

    THE MATH (Synaptic Operation Counting):
    A "synaptic operation" (synop) occurs every time a spike from one
    neuron travels across one active synapse to reach another neuron.
    The TOTAL number of synaptic operations during one inference is
    estimated as:

        total_spikes = mean_spike_sparsity * N * n_timesteps
        synops_per_spike = N * p_conn      (average number of OUTGOING
                                             connections per neuron)
        total_synops = total_spikes * synops_per_spike

    Then:
        energy_joules = total_synops * energy_per_synop_joules

    where energy_per_synop_joules is Loihi 2's published per-operation
    energy cost (~23 picojoules), taken from config.py's ENERGY_CONFIG.

    Parameters
    ----------
    reservoir : LSMReservoir
        Used to read N and p_conn.
    mean_spike_sparsity : float
        The AVERAGE fraction of (neuron, timestep) positions that
        contained a spike, measured empirically by running the
        reservoir over real test signals (NOT a theoretical assumption
        -- see measure_lsm_spike_sparsity() below).
    n_timesteps : int
        Number of simulation timesteps per inference.

    Returns
    -------
    dict with keys: total_synops, energy_joules, energy_microjoules
    """
    energy_per_synop = CONFIG["energy"]["loihi2_energy_per_synop_joules"]

    total_spikes = mean_spike_sparsity * reservoir.N * n_timesteps
    synops_per_spike = reservoir.N * reservoir.p_conn
    total_synops = total_spikes * synops_per_spike

    energy_joules = total_synops * energy_per_synop

    return {
        "total_synops": total_synops,
        "energy_joules": energy_joules,
        "energy_microjoules": energy_joules * 1e6,
    }


def measure_lsm_spike_sparsity(spike_trains: list) -> float:
    """
    Measures the ACTUAL mean spike sparsity across a list of encoded
    spike trains (the fraction of all positions, across all signals,
    that contain a spike). This empirical measurement feeds into
    estimate_lsm_energy() above, rather than assuming a sparsity value.
    """
    total_positions = 0
    total_spikes = 0
    for spikes in spike_trains:
        total_positions += spikes.size
        total_spikes += spikes.sum()
    return float(total_spikes) / float(total_positions)


def measure_lsm_inference_latency(reservoir, encode_fn, sample_signal: np.ndarray, n_repeats: int = 20) -> float:
    """
    Measures the WALL-CLOCK time (in seconds) to run ONE signal through
    the full LSM pipeline (encoding + reservoir simulation), averaged
    over several repeats for a stable estimate. NOTE: this measures
    latency on a CONVENTIONAL CPU running our Python simulation -- it
    does NOT represent the latency a real neuromorphic chip would
    achieve (which would typically be much faster due to dedicated
    parallel hardware). This CPU-simulation latency number is included
    for completeness / transparency, but the energy estimate above
    (based on hardware specs, not simulation speed) is the
    scientifically meaningful comparison point.
    """
    rng = np.random.default_rng(CONFIG["seed"])

    start = time.time()
    for _ in range(n_repeats):
        spikes = encode_fn(sample_signal, rng=rng)
        _ = reservoir.simulate(spikes)
    elapsed = (time.time() - start) / n_repeats
    return elapsed


# ==============================================================================
# Baseline Energy Estimation (GPU / CPU power x time)
# ==============================================================================

def estimate_baseline_energy(inference_latency_seconds: float, device: str) -> dict:
    """
    Estimates energy for a baseline model using the simple formula:
        Energy (Joules) = Power (Watts) x Time (seconds)

    device should be one of "cuda" (GPU) or "cpu". We look up the
    appropriate Thermal Design Power (TDP) constant from config.py.
    """
    if "cuda" in device.lower():
        power_watts = CONFIG["energy"]["gpu_tdp_watts"]
    else:
        power_watts = CONFIG["energy"]["cpu_tdp_watts"]

    energy_joules = power_watts * inference_latency_seconds

    return {
        "power_watts": power_watts,
        "energy_joules": energy_joules,
        "energy_microjoules": energy_joules * 1e6,
    }


def measure_svm_inference_latency(model, sample_input: np.ndarray, n_repeats: int = 50) -> float:
    """Measures inference latency for the Expert-Features+SVM baseline (single sample)."""
    # sample_input should be shape (1, T, 2) -- one signal
    if sample_input.ndim == 2:
        sample_input = sample_input[np.newaxis, ...]
    start = time.time()
    for _ in range(n_repeats):
        _ = model.predict(sample_input, verbose=False)
    elapsed = (time.time() - start) / n_repeats
    return elapsed


# ==============================================================================
# Full Comparison Table Builder
# ==============================================================================

def build_energy_latency_comparison(
    lsm_info: dict,
    cnn_timing: dict,
    lstm_timing: dict,
    svm_latency_seconds: float,
    esn_latency_seconds: float = None,
) -> pd.DataFrame:
    """
    Assembles the final ENERGY AND LATENCY COMPARISON TABLE across all
    five approaches (LSM + 4 baselines), ready to save as
    results/energy_results.csv and results/latency_results.csv.

    Parameters
    ----------
    lsm_info : dict
        Output of combining estimate_lsm_energy() and
        measure_lsm_inference_latency() for the LSM.
    cnn_timing, lstm_timing : dict
        The timing_info dictionaries returned by
        src/baselines.py's train_cnn_classifier / train_lstm_classifier.
    svm_latency_seconds : float
        Single-sample inference latency for the Expert+SVM baseline.
    esn_latency_seconds : float, optional
        Single-sample inference latency for the ESN baseline.
    """
    rows = []

    # --- LSM row ---
    rows.append({
        "model": "LSM (Liquid State Machine)",
        "hardware_assumed": "Intel Loihi 2 (neuromorphic, energy estimate) / CPU (latency measured)",
        "inference_latency_ms": lsm_info["inference_latency_seconds"] * 1000,
        "energy_per_inference_microjoules": lsm_info["energy_microjoules"],
        "training_time_seconds": "N/A (only readout is trained; see readout training time)",
    })

    # --- CNN row ---
    cnn_energy = estimate_baseline_energy(cnn_timing["inference_latency_seconds"], cnn_timing["device"])
    rows.append({
        "model": "1D CNN",
        "hardware_assumed": f"{cnn_timing['device'].upper()} (power x time estimate)",
        "inference_latency_ms": cnn_timing["inference_latency_seconds"] * 1000,
        "energy_per_inference_microjoules": cnn_energy["energy_microjoules"],
        "training_time_seconds": cnn_timing["training_time_seconds"],
    })

    # --- LSTM row ---
    lstm_energy = estimate_baseline_energy(lstm_timing["inference_latency_seconds"], lstm_timing["device"])
    rows.append({
        "model": "Bidirectional LSTM",
        "hardware_assumed": f"{lstm_timing['device'].upper()} (power x time estimate)",
        "inference_latency_ms": lstm_timing["inference_latency_seconds"] * 1000,
        "energy_per_inference_microjoules": lstm_energy["energy_microjoules"],
        "training_time_seconds": lstm_timing["training_time_seconds"],
    })

    # --- SVM (Expert Features) row ---
    svm_energy = estimate_baseline_energy(svm_latency_seconds, "cpu")
    rows.append({
        "model": "Expert Features + RBF-SVM",
        "hardware_assumed": "CPU (power x time estimate)",
        "inference_latency_ms": svm_latency_seconds * 1000,
        "energy_per_inference_microjoules": svm_energy["energy_microjoules"],
        "training_time_seconds": "N/A (not separately timed)",
    })

    # --- ESN row (optional) ---
    if esn_latency_seconds is not None:
        esn_energy = estimate_baseline_energy(esn_latency_seconds, "cpu")
        rows.append({
            "model": "Echo State Network (ESN)",
            "hardware_assumed": "CPU (power x time estimate)",
            "inference_latency_ms": esn_latency_seconds * 1000,
            "energy_per_inference_microjoules": esn_energy["energy_microjoules"],
            "training_time_seconds": "N/A (only readout is trained)",
        })

    df = pd.DataFrame(rows)

    # Compute the energy ratio relative to the LSM, for the headline
    # "our approach is N times more efficient" result.
    lsm_energy_value = df.loc[df["model"].str.contains("LSM"), "energy_per_inference_microjoules"].values[0]
    df["energy_ratio_vs_lsm"] = df["energy_per_inference_microjoules"] / lsm_energy_value

    return df


# ==============================================================================
# Script Entry Point (standalone demonstration with dummy numbers)
# ==============================================================================

if __name__ == "__main__":
    print("Demonstrating energy/latency estimation with representative dummy timing values...\n")

    from src.reservoir import LSMReservoir

    reservoir = LSMReservoir(N=200, seed=CONFIG["seed"])

    # Use a plausible measured sparsity value (in the real pipeline this
    # comes from measure_lsm_spike_sparsity() on actual encoded data).
    dummy_sparsity = 0.02
    lsm_energy = estimate_lsm_energy(reservoir, dummy_sparsity, n_timesteps=128)
    print("LSM energy estimate (N=200, T=128, measured sparsity=2%):")
    for k, v in lsm_energy.items():
        print(f"  {k}: {v:.6g}")

    dummy_lsm_latency = 0.005   # 5 ms, representative of CPU simulation
    lsm_info = {**lsm_energy, "inference_latency_seconds": dummy_lsm_latency}

    dummy_cnn_timing = {"training_time_seconds": 45.2, "inference_latency_seconds": 0.0021, "device": "cpu"}
    dummy_lstm_timing = {"training_time_seconds": 78.5, "inference_latency_seconds": 0.0065, "device": "cpu"}
    dummy_svm_latency = 0.0008
    dummy_esn_latency = 0.003

    comparison_df = build_energy_latency_comparison(
        lsm_info, dummy_cnn_timing, dummy_lstm_timing, dummy_svm_latency, dummy_esn_latency
    )

    print("\nFull comparison table (dummy values for demonstration):")
    print(comparison_df.to_string(index=False))

    print("\nEnergy analysis module executed successfully.")
