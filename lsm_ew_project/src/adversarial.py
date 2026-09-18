"""
src/adversarial.py
====================
Implements ADVERSARIAL ROBUSTNESS TESTING for the trained LSM pipeline.

WHY ROBUSTNESS TESTING MATTERS FOR AN EW (ELECTRONIC WARFARE) PROJECT:
In a real electronic warfare scenario, an adversary actively wants to
DEFEAT your classifier -- by adding noise, shifting frequencies, or
distorting their jamming signal to look like something else (or to look
like nothing at all). A classifier that only works on "clean" test data
is not useful for this domain. This module systematically degrades test
signals in three realistic ways and measures how much accuracy drops --
this characterises how BRITTLE or ROBUST the trained models are.

THE THREE ATTACKS IMPLEMENTED:
    1. AWGN attack            -- inject additional Gaussian noise
                                  (simulates a noisier real-world channel,
                                  or a deliberate noise-jamming counter-
                                  measure against our own classifier)
    2. Frequency perturbation -- apply a small random frequency shift
                                  (simulates Doppler shift, oscillator
                                  drift, or deliberate frequency evasion)
    3. Signal distortion      -- apply nonlinear amplitude distortion
                                  (simulates amplifier saturation/clipping,
                                  multipath fading artifacts, or
                                  deliberate signal shaping to evade
                                  detection)

For each attack, we sweep across several SEVERITY levels and measure
classification accuracy at each level, producing a "degradation curve"
that is one of the most important results in the final report.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import accuracy_score

from config import CONFIG


# ==============================================================================
# Attack 1: Additive White Gaussian Noise (AWGN)
# ==============================================================================

def awgn_attack(iq_signal: np.ndarray, target_snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Adds EXTRA Gaussian noise to an already-clean (or already-noisy)
    signal so that the additional noise alone corresponds to the given
    target_snr_db, relative to the ORIGINAL signal's power.

    This reuses the exact same mathematics as
    src/synthetic_ew_dataset.py's add_awgn() function -- see that
    file's docstring for the full derivation. We duplicate a focused
    version here (rather than importing it) to keep this module fully
    self-contained for adversarial-specific use, and because here we
    explicitly think of it as an ATTACK being applied to an existing
    signal, not as part of original dataset generation.
    """
    complex_signal = iq_signal[:, 0] + 1j * iq_signal[:, 1]
    signal_power = np.mean(np.abs(complex_signal) ** 2)
    if signal_power < 1e-12:
        signal_power = 1e-12

    noise_power = signal_power / (10 ** (target_snr_db / 10.0))
    noise_std = np.sqrt(noise_power / 2.0)

    noise = noise_std * (
        rng.standard_normal(len(complex_signal)) + 1j * rng.standard_normal(len(complex_signal))
    )
    attacked = complex_signal + noise

    result = np.zeros_like(iq_signal)
    result[:, 0] = attacked.real
    result[:, 1] = attacked.imag
    return result


# ==============================================================================
# Attack 2: Frequency Perturbation
# ==============================================================================

def frequency_perturbation_attack(
    iq_signal: np.ndarray, max_offset: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Applies a small random FREQUENCY SHIFT to the signal, simulating
    Doppler shift, oscillator drift, or deliberate frequency evasion.

    THE MATH:
    A frequency shift in the time domain is achieved by multiplying the
    complex signal by a complex exponential at the shift frequency:
        x_shifted(t) = x(t) * exp(j * 2*pi * f_shift * t)
    This works because multiplying by exp(j*2*pi*f_shift*t) in the time
    domain corresponds to SHIFTING the signal's entire spectrum by
    f_shift in the frequency domain (a standard property of the Fourier
    transform known as the "frequency shifting theorem"). A classifier
    relying heavily on absolute frequency position (rather than relative/
    shape-based features) will be more vulnerable to this attack.

    f_shift is drawn randomly from [-max_offset, +max_offset] (in
    normalised frequency units, same convention as the rest of this
    project).
    """
    complex_signal = iq_signal[:, 0] + 1j * iq_signal[:, 1]
    n_samples = len(complex_signal)
    t = np.arange(n_samples)

    f_shift = rng.uniform(-max_offset, max_offset)
    shift_factor = np.exp(1j * 2 * np.pi * f_shift * t)

    attacked = complex_signal * shift_factor

    result = np.zeros_like(iq_signal)
    result[:, 0] = attacked.real
    result[:, 1] = attacked.imag
    return result


# ==============================================================================
# Attack 3: Signal Distortion (Nonlinear Amplitude Distortion)
# ==============================================================================

def signal_distortion_attack(iq_signal: np.ndarray, distortion_level: float) -> np.ndarray:
    """
    Applies NONLINEAR AMPLITUDE DISTORTION to the signal, simulating
    amplifier saturation/compression or deliberate signal shaping.

    THE MATH:
    We use a standard soft-clipping nonlinearity based on the hyperbolic
    tangent function, applied to the signal's AMPLITUDE while preserving
    its PHASE:
        x(t) = A(t) * exp(j*phi(t))           [polar decomposition]
        A_distorted(t) = A(t) - distortion_level * tanh(A(t))
        x_distorted(t) = A_distorted(t) * exp(j*phi(t))

    WHY THIS FORM?
    tanh(A) grows almost linearly for small A but saturates (flattens
    out) for large A. Subtracting a scaled tanh(A) term therefore leaves
    SMALL amplitudes nearly unchanged but COMPRESSES large amplitudes --
    exactly the behaviour of a real amplifier as it approaches its power
    limit and "clips" the peaks of a signal. distortion_level controls
    how strong this compression effect is (0 = no distortion at all).
    """
    complex_signal = iq_signal[:, 0] + 1j * iq_signal[:, 1]
    amplitude = np.abs(complex_signal)
    phase = np.angle(complex_signal)

    distorted_amplitude = amplitude - distortion_level * np.tanh(amplitude)
    distorted_amplitude = np.maximum(distorted_amplitude, 0)   # amplitude can't go negative

    distorted_signal = distorted_amplitude * np.exp(1j * phase)

    result = np.zeros_like(iq_signal)
    result[:, 0] = distorted_signal.real
    result[:, 1] = distorted_signal.imag
    return result


# ==============================================================================
# Full Robustness Evaluation Pipeline
# ==============================================================================

def evaluate_robustness(
    X_test_signals: np.ndarray,
    y_test_labels: np.ndarray,
    encode_fn,
    reservoir,
    readout_head,
    n_timesteps: int = None,
    seed: int = None,
) -> dict:
    """
    Runs all three adversarial attacks (at all configured severity
    levels) against a trained LSM pipeline (encoder + reservoir +
    readout) and records the resulting accuracy at each severity level.

    This function is intentionally GENERIC over the encoding function,
    reservoir, and readout head, so it can be reused to test robustness
    of ANY trained LSM configuration (e.g. the jamming-classification
    head, or in principle the modulation head too).

    Parameters
    ----------
    X_test_signals : np.ndarray, shape (n_test, T, 2)
        Clean test signals (before any attack is applied).
    y_test_labels : np.ndarray, shape (n_test,)
    encode_fn : callable
        One of the encoding functions from src/encoding.py (e.g. hybrid_encode).
    reservoir : LSMReservoir
        An already-built, trained-pipeline-matching reservoir.
    readout_head : ReadoutHead
        An already-fitted readout classifier matching this reservoir's
        output dimensionality.
    n_timesteps : int, optional
        Passed through to reservoir.simulate().
    seed : int, optional

    Returns
    -------
    dict with keys "clean_accuracy", "awgn", "frequency", "distortion" --
    each attack key maps to a list of {severity, accuracy} dictionaries.
    """
    if seed is None:
        seed = CONFIG["seed"]
    rng = np.random.default_rng(seed)

    adv_cfg = CONFIG["adversarial"]

    def _run_pipeline(signals):
        states = np.zeros((len(signals), reservoir.N))
        for i, sig in enumerate(signals):
            spikes = encode_fn(sig, rng=rng) if encode_fn is not None else sig
            states[i] = reservoir.simulate(spikes, n_timesteps=n_timesteps)
        preds = readout_head.predict(states)
        return accuracy_score(y_test_labels, preds)

    results = {}

    print("Evaluating CLEAN (unattacked) accuracy...")
    results["clean_accuracy"] = _run_pipeline(X_test_signals)
    print(f"  Clean accuracy: {results['clean_accuracy']:.4f}")

    print("\nEvaluating AWGN attack robustness...")
    awgn_results = []
    for snr in adv_cfg["awgn_attack_snr_db"]:
        attacked_signals = [awgn_attack(sig, snr, rng) for sig in X_test_signals]
        acc = _run_pipeline(attacked_signals)
        print(f"  Attack SNR={snr:>4d} dB -> accuracy={acc:.4f}")
        awgn_results.append({"severity": snr, "accuracy": acc})
    results["awgn"] = awgn_results

    print("\nEvaluating frequency perturbation attack robustness...")
    freq_results = []
    max_offset = adv_cfg["freq_perturbation_max_offset"]
    offset_levels = np.linspace(0, max_offset, 5)
    for offset in offset_levels:
        attacked_signals = [frequency_perturbation_attack(sig, offset, rng) for sig in X_test_signals]
        acc = _run_pipeline(attacked_signals)
        print(f"  Max freq offset={offset:.4f} -> accuracy={acc:.4f}")
        freq_results.append({"severity": float(offset), "accuracy": acc})
    results["frequency"] = freq_results

    print("\nEvaluating signal distortion attack robustness...")
    dist_results = []
    for level in adv_cfg["distortion_levels"]:
        attacked_signals = [signal_distortion_attack(sig, level) for sig in X_test_signals]
        acc = _run_pipeline(attacked_signals)
        print(f"  Distortion level={level:.2f} -> accuracy={acc:.4f}")
        dist_results.append({"severity": level, "accuracy": acc})
    results["distortion"] = dist_results

    return results


def robustness_results_to_dataframe(results: dict):
    """Converts the nested robustness results dictionary into a flat
    pandas DataFrame, suitable for saving to CSV and plotting."""
    import pandas as pd

    rows = []
    rows.append({"attack": "clean", "severity": 0.0, "accuracy": results["clean_accuracy"]})
    for attack_name in ("awgn", "frequency", "distortion"):
        for entry in results[attack_name]:
            rows.append({"attack": attack_name, "severity": entry["severity"], "accuracy": entry["accuracy"]})
    return pd.DataFrame(rows)


# ==============================================================================
# Script Entry Point (sanity check using a small reservoir + dummy data)
# ==============================================================================

if __name__ == "__main__":
    print("Testing adversarial attack functions on a synthetic test signal...\n")

    rng = np.random.default_rng(CONFIG["seed"])
    t = np.linspace(0, 1, 256)
    test_signal = np.zeros((256, 2))
    test_signal[:, 0] = np.cos(2 * np.pi * 5 * t)
    test_signal[:, 1] = np.sin(2 * np.pi * 5 * t)

    print("Original signal power:", np.mean(test_signal[:, 0]**2 + test_signal[:, 1]**2))

    awgn_result = awgn_attack(test_signal, target_snr_db=5.0, rng=rng)
    print("After AWGN attack (SNR=5dB), power:", np.mean(awgn_result[:, 0]**2 + awgn_result[:, 1]**2))

    freq_result = frequency_perturbation_attack(test_signal, max_offset=0.02, rng=rng)
    print("After frequency perturbation, signal shape unchanged:", freq_result.shape)

    dist_result = signal_distortion_attack(test_signal, distortion_level=0.3)
    original_amplitude = np.sqrt(test_signal[:, 0]**2 + test_signal[:, 1]**2).mean()
    distorted_amplitude = np.sqrt(dist_result[:, 0]**2 + dist_result[:, 1]**2).mean()
    print(f"Original mean amplitude: {original_amplitude:.4f}, "
          f"after distortion: {distorted_amplitude:.4f} (should be lower)")

    print("\nAll three attack functions executed successfully.")
