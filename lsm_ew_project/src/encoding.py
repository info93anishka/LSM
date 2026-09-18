"""
src/encoding.py
=================
Implements the three spike encoding schemes used to convert raw IQ
(In-phase/Quadrature) radio signals into SPIKE TRAINS that the Liquid
State Machine reservoir can process.

WHY DO WE NEED THIS AT ALL?
A Liquid State Machine is built from spiking neurons. Spiking neurons only
understand ONE kind of input: a sequence of 0s and 1s over time (a spike
train), where 1 means "an event happened at this exact timestep" and 0
means "nothing happened." Our raw data, however, is a sequence of
floating-point numbers (the IQ samples). The job of an ENCODER is to
convert one into the other -- floating point values into spike trains --
while preserving as much useful information as possible.

This file implements three different encoding philosophies (see the
detailed docstring inside each function for the mathematics):
    1. RATE CODING       -- amplitude controls how OFTEN a neuron spikes
    2. TTFS (Time-To-First-Spike) -- amplitude controls WHEN a neuron spikes
    3. HYBRID             -- combines both, switching based on amplitude

Each function takes a (T, 2) array (T timesteps, 2 channels: I and Q)
of normalised real-valued samples, and returns a (T, 2) array of binary
spikes (0 or 1 at every position).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import CONFIG


# ==============================================================================
# Helper: Normalisation
# ==============================================================================

def normalise_to_unit_range(x: np.ndarray) -> np.ndarray:
    """
    Rescales an array so all its values fall between 0 and 1.

    Equation:
        x_norm = (x - min(x)) / (max(x) - min(x))

    This is called MIN-MAX NORMALISATION. We need this because both our
    encoding schemes interpret values as either "firing probabilities"
    (which must be between 0 and 1) or "fractions of a time window"
    (also between 0 and 1) -- so the raw signal amplitude, which could be
    any real number, must first be squashed into this range.

    If the array is constant (max == min), we return an array of zeros
    to avoid a division-by-zero error.
    """
    x_min = x.min()
    x_max = x.max()
    spread = x_max - x_min
    if spread < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / spread


# ==============================================================================
# Encoding Method 1: Rate Coding
# ==============================================================================

def rate_encode(
    iq_signal: np.ndarray,
    lambda_max: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    RATE CODING
    ============
    Converts a signal's AMPLITUDE into a neuron's FIRING RATE (how often
    it spikes per unit time). This is the most intuitive encoding and the
    closest analogue to how a standard artificial neuron's activation
    relates to a spiking neuron's average output.

    THE MATH:
    For each timestep t, we treat the normalised amplitude x_norm(t) as
    a firing PROBABILITY (scaled by a maximum rate lambda_max):

        firing_probability(t) = x_norm(t) * lambda_max

    Then at every timestep, we flip a biased coin: a spike occurs if a
    random number drawn uniformly from [0, 1] is LESS than that
    probability. This procedure -- generating spikes at each timestep
    independently with a probability based on a target rate -- defines
    what is called a Poisson process (an idealised model of "memoryless"
    random events), which is also how many real biological neurons'
    spike trains are statistically described.

        spike(t) = 1   if  U(0,1) < firing_probability(t)
                 = 0   otherwise

    INTERPRETATION:
    A high-amplitude sample produces MANY spikes (looks like rapid
    firing). A low-amplitude sample produces FEW or no spikes. Averaged
    over a long enough time window, the FRACTION of timesteps that have
    a spike approximately reconstructs the original normalised amplitude
    -- this is why it's called "rate" coding: information is recovered
    from the spiking RATE, not from individual spike timings.

    TRADE-OFF:
    Accurate reconstruction of amplitude requires averaging over MANY
    timesteps, which is informationally inefficient -- this is the
    central weakness of rate coding compared to TTFS.

    Parameters
    ----------
    iq_signal : np.ndarray, shape (T, 2)
        Raw (not yet normalised) IQ signal, columns are [I, Q].
    lambda_max : float
        Maximum firing probability per timestep (defaults to config value).
    rng : np.random.Generator
        Random number generator (pass one in for reproducibility).

    Returns
    -------
    np.ndarray, shape (T, 2), dtype int8
        Binary spike train (0 or 1 at every position).
    """
    if lambda_max is None:
        lambda_max = CONFIG["encoding"]["rate_lambda_max"]
    if rng is None:
        rng = np.random.default_rng()

    n_timesteps, n_channels = iq_signal.shape
    spikes = np.zeros((n_timesteps, n_channels), dtype=np.int8)

    for ch in range(n_channels):
        # We encode the MAGNITUDE of each sample (its absolute value),
        # since firing probability cannot be negative, but the sign
        # information is implicitly preserved because I and Q channels
        # are encoded separately and together define the complex value.
        amplitude = np.abs(iq_signal[:, ch])
        normalised_amplitude = normalise_to_unit_range(amplitude)
        firing_probability = normalised_amplitude * lambda_max

        random_draws = rng.random(n_timesteps)
        spikes[:, ch] = (random_draws < firing_probability).astype(np.int8)

    return spikes


# ==============================================================================
# Encoding Method 2: Time-To-First-Spike (TTFS)
# ==============================================================================

def ttfs_encode(
    iq_signal: np.ndarray,
    block_size: int = None,
) -> np.ndarray:
    """
    TIME-TO-FIRST-SPIKE (TTFS) ENCODING
    =====================================
    Instead of encoding amplitude as a RATE, TTFS encodes amplitude as a
    DELAY -- the time between the start of the observation window and the
    FIRST (and only) spike that neuron emits.

    THE MATH:
    We divide the T timesteps into "blocks" of size B (e.g. B=8). Each
    block is treated as the receptive window for one "TTFS neuron." Within
    that block, we compute the average normalised amplitude:

        amplitude_block = mean(x_norm[block_start : block_start + B])

    The neuron fires EXACTLY ONCE within its block, at a time offset that
    is INVERSELY proportional to the amplitude:

        t_fire = round( (1 - amplitude_block) * (B - 1) )

    INTERPRETATION OF THIS FORMULA:
        - If amplitude_block = 1 (maximum strength): t_fire = 0
          --> the neuron fires IMMEDIATELY at the very start of the block.
        - If amplitude_block = 0 (no signal): t_fire = B - 1
          --> the neuron fires at the very LAST possible moment (or you
              can treat near-zero amplitudes as "doesn't fire at all" --
              see the threshold note below).
    This is the defining principle of TTFS: "important / strong information
    arrives FAST, weak information arrives SLOW (or not at all)" -- directly
    analogous to how, in many biological sensory systems, more intense
    stimuli trigger faster neural responses.

    WHY THIS IS ENERGY-EFFICIENT:
    Each neuron emits AT MOST ONE spike for its entire block, versus rate
    coding's potentially many spikes per block. Fewer spikes = fewer
    synaptic operations downstream = lower energy on real neuromorphic
    hardware (since hardware energy scales with the NUMBER of spike
    events processed).

    Parameters
    ----------
    iq_signal : np.ndarray, shape (T, 2)
    block_size : int
        Number of timesteps grouped into one TTFS "neuron" (defaults to
        config value).

    Returns
    -------
    np.ndarray, shape (T, 2), dtype int8
    """
    if block_size is None:
        block_size = CONFIG["encoding"]["ttfs_block_size"]

    n_timesteps, n_channels = iq_signal.shape
    spikes = np.zeros((n_timesteps, n_channels), dtype=np.int8)

    n_blocks = n_timesteps // block_size

    for ch in range(n_channels):
        amplitude = np.abs(iq_signal[:, ch])
        normalised_amplitude = normalise_to_unit_range(amplitude)

        for b in range(n_blocks):
            block_start = b * block_size
            block_end = block_start + block_size
            amplitude_block = normalised_amplitude[block_start:block_end].mean()

            # t_fire computed as described in the docstring above.
            t_fire_offset = int(round((1.0 - amplitude_block) * (block_size - 1)))
            fire_index = block_start + t_fire_offset

            if fire_index < n_timesteps:
                spikes[fire_index, ch] = 1

    return spikes


# ==============================================================================
# Encoding Method 3: Hybrid Rate/TTFS Encoding
# ==============================================================================

def hybrid_encode(
    iq_signal: np.ndarray,
    theta: float = None,
    lambda_max: float = None,
    block_size: int = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    HYBRID RATE/TTFS ENCODING (the novel encoding proposed in this project)
    ==========================================================================
    Motivation: real RF signals are SPARSE in a specific sense -- they have
    short bursts of high amplitude (carrier pulses, signal peaks) separated
    by long stretches of low amplitude (noise floor, inter-pulse gaps).
    Using a single encoding strategy uniformly across the whole signal is
    wasteful: TTFS is excellent for the high-amplitude, information-rich
    parts (where you want speed and efficiency), but it is FRAGILE for
    low-amplitude, noisy parts (since a tiny noise fluctuation can shift
    when the "first spike" occurs, corrupting the encoded information).
    Rate coding is the opposite: noisy-but-robust due to averaging over
    many timesteps, which works well for the less critical low-amplitude
    regions.

    THE RULE:
    For each block of size B, compute the block's average normalised
    amplitude as in TTFS encoding. Then:

        if amplitude_block >= theta:
            use TTFS encoding for this block      (HIGH amplitude -> fast, sparse)
        else:
            use RATE encoding for this block       (LOW amplitude -> robust)

    theta (the threshold) is a tunable hyperparameter; we use 0.5 by
    default (the midpoint of the normalised amplitude range), meaning
    "above-average" amplitude blocks get the efficient TTFS treatment
    and "below-average" blocks get the robust rate-coding treatment.

    EXPECTED BENEFIT:
    Lower total spike count (energy) than pure rate coding, while being
    more ROBUST to noise than pure TTFS, because the noisiest parts of
    the signal (low amplitude) use the noise-tolerant rate-coding scheme.

    Parameters
    ----------
    iq_signal : np.ndarray, shape (T, 2)
    theta : float
        Amplitude threshold separating the TTFS and rate-coding regimes.
    lambda_max : float
        Maximum firing probability for the rate-coded regions.
    block_size : int
        Block size for both the threshold decision and TTFS sub-encoding.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (T, 2), dtype int8
    """
    cfg = CONFIG["encoding"]
    if theta is None:
        theta = cfg["hybrid_theta"]
    if lambda_max is None:
        lambda_max = cfg["rate_lambda_max"]
    if block_size is None:
        block_size = cfg["ttfs_block_size"]
    if rng is None:
        rng = np.random.default_rng()

    n_timesteps, n_channels = iq_signal.shape
    spikes = np.zeros((n_timesteps, n_channels), dtype=np.int8)
    n_blocks = n_timesteps // block_size

    for ch in range(n_channels):
        amplitude = np.abs(iq_signal[:, ch])
        normalised_amplitude = normalise_to_unit_range(amplitude)

        for b in range(n_blocks):
            block_start = b * block_size
            block_end = block_start + block_size
            block_slice = slice(block_start, block_end)
            amplitude_block = normalised_amplitude[block_slice].mean()

            if amplitude_block >= theta:
                # --- High amplitude: use TTFS for this block ---
                t_fire_offset = int(round((1.0 - amplitude_block) * (block_size - 1)))
                fire_index = block_start + t_fire_offset
                if fire_index < n_timesteps:
                    spikes[fire_index, ch] = 1
            else:
                # --- Low amplitude: use rate coding for this block ---
                local_amplitudes = normalised_amplitude[block_slice]
                firing_probabilities = local_amplitudes * lambda_max
                random_draws = rng.random(block_end - block_start)
                local_spikes = (random_draws < firing_probabilities).astype(np.int8)
                spikes[block_slice, ch] = local_spikes

    return spikes


# ==============================================================================
# Encoding Dispatcher (convenience function)
# ==============================================================================

# A lookup table mapping string names to their encoding functions, so other
# parts of the code can do: encode_fn = ENCODERS["hybrid"]; spikes = encode_fn(signal)
ENCODERS = {
    "rate": rate_encode,
    "ttfs": ttfs_encode,
    "hybrid": hybrid_encode,
}


def encode_signal(iq_signal: np.ndarray, method: str = "hybrid", **kwargs) -> np.ndarray:
    """
    Convenience dispatcher: encode_signal(signal, method="rate") instead of
    having to import and call rate_encode/ttfs_encode/hybrid_encode directly.
    """
    if method not in ENCODERS:
        raise ValueError(
            f"Unknown encoding method '{method}'. Choose from: {list(ENCODERS.keys())}"
        )
    # TTFS encoding is deterministic (no randomness) so it does not accept an
    # rng argument. Strip it from kwargs when calling ttfs_encode.
    if method == "ttfs":
        kwargs = {k: v for k, v in kwargs.items() if k != "rng"}
    return ENCODERS[method](iq_signal, **kwargs)


# ==============================================================================
# Spike Sparsity Metric
# ==============================================================================

def compute_spike_sparsity(spikes: np.ndarray) -> float:
    """
    Computes the SPARSITY of a spike train: the fraction of all
    (timestep, channel) positions that contain a spike.

    Equation:
        sparsity = (total number of spikes) / (total number of positions)

    Lower sparsity is generally better for energy efficiency on
    neuromorphic hardware, since hardware energy consumption scales with
    the number of spike events that must be processed and routed.
    """
    return float(np.mean(spikes))


def compare_encoding_sparsity(iq_signal: np.ndarray, rng: np.random.Generator = None):
    """
    Encodes the SAME signal using all three methods and reports the
    sparsity of each, for direct comparison. Used by the
    experiment script that generates the "encoding comparison" figure.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {}
    for method_name, encode_fn in ENCODERS.items():
        if method_name == "ttfs":
            spikes = encode_fn(iq_signal)
        else:
            spikes = encode_fn(iq_signal, rng=rng)
        results[method_name] = compute_spike_sparsity(spikes)

    return results


# ==============================================================================
# Script Entry Point (for quick manual testing)
# ==============================================================================

if __name__ == "__main__":
    print("Testing spike encoding methods on a synthetic test signal...\n")

    rng = np.random.default_rng(CONFIG["seed"])

    # Build a simple test signal: a sine wave with some added noise.
    t = np.linspace(0, 1, CONFIG["encoding"]["n_timesteps"])
    test_signal = np.zeros((len(t), 2))
    test_signal[:, 0] = np.sin(2 * np.pi * 5 * t) + 0.1 * rng.standard_normal(len(t))
    test_signal[:, 1] = np.cos(2 * np.pi * 5 * t) + 0.1 * rng.standard_normal(len(t))

    sparsity_results = compare_encoding_sparsity(test_signal, rng=rng)

    print("Spike sparsity comparison (fraction of positions with a spike):")
    for method, sparsity in sparsity_results.items():
        print(f"  {method:>8s}: {sparsity:.4f}  ({sparsity*100:.1f}% of positions spike)")

    print("\nLower sparsity generally means lower energy consumption on")
    print("neuromorphic hardware. TTFS should show the lowest sparsity.")
