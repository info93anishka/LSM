"""
src/synthetic_ew_dataset.py
=============================
Generates the SYNTHETIC ELECTRONIC WARFARE (EW) dataset used throughout
this project. Unlike RadioML (which we download), this dataset does not
exist anywhere -- we create it ourselves using signal-processing equations
that describe real jamming behaviour.

BACKGROUND -- WHAT IS A JAMMING SIGNAL?
A "jammer" is a transmitter whose entire purpose is to disrupt a victim
receiver, either by overpowering it with noise, mimicking its signal, or
otherwise corrupting what it receives. Different jamming TYPES have
characteristic mathematical structures, which is exactly what a classifier
needs to learn to tell them apart.

This file implements SIX jamming types:
    1. Tone jamming     -- one or more pure sine-wave tones
    2. Sweep jamming     -- a tone whose frequency changes over time (chirp)
    3. Noise jamming     -- band-limited random noise
    4. Repeat jamming    -- a captured signal replayed with a delay (DRFM)
    5. Pulse jamming     -- short high-power bursts repeated periodically
    6. Barrage jamming   -- wideband high-power noise across the whole band

EACH SIGNAL is a COMPLEX-VALUED time series:
    x(t) = I(t) + j * Q(t)
where I(t) is the "in-phase" component and Q(t) is the "quadrature"
component (90 degrees phase-shifted). This I/Q representation is the
universal format used in software-defined radio and matches the format
of the RadioML dataset, so both datasets are directly comparable.

HOW TO RUN THIS FILE DIRECTLY:
    python src/synthetic_ew_dataset.py
This will generate the dataset and save it to data/synthetic_ew/.
"""

import os
import sys

# Allow this file to be run directly (python src/synthetic_ew_dataset.py)
# by adding the project root to Python's import search path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from config import CONFIG, set_global_seed, ensure_directories_exist


# ==============================================================================
# SECTION 1: Individual Jamming Signal Generators
# ==============================================================================
#
# Each function below returns a single complex-valued NumPy array of length
# N (number of samples), BEFORE noise is added. We add noise separately in
# a shared helper function so every jamming type gets identical noise
# handling (this is important for a FAIR comparison across classes).

def _time_vector(n_samples: int) -> np.ndarray:
    """
    Returns a normalised time vector t = [0, 1, 2, ..., N-1] / fs.
    We use a normalised sample rate fs = 1.0 throughout this project --
    this is standard practice in baseband signal simulation, where we care
    about RELATIVE frequency content, not absolute Hz values. If you later
    want to relate this to a real radio's sample rate (e.g. 1 MHz), you
    would simply multiply by that value when interpreting results.
    """
    fs = 1.0
    return np.arange(n_samples) / fs


def tone_jamming(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    TONE JAMMING
    ------------
    The simplest jamming waveform: a single continuous-wave (CW) sinusoid
    at some frequency f_j, designed to saturate a specific narrow channel.

    Equation:
        x(t) = exp(j * 2*pi*f_j*t)

    This is a complex exponential -- the standard way to represent a single
    pure-frequency tone in I/Q form. Its real part is cos(2*pi*f_j*t) and
    its imaginary part is sin(2*pi*f_j*t).

    f_j is drawn randomly between 0.05 and 0.45 (normalised frequency,
    where 0.5 is the Nyquist limit -- the highest frequency representable
    at our sample rate without aliasing).
    """
    t = _time_vector(n_samples)
    f_j = rng.uniform(0.05, 0.45)
    phase = rng.uniform(0, 2 * np.pi)   # random starting phase
    signal = np.exp(1j * (2 * np.pi * f_j * t + phase))
    return signal


def sweep_jamming(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    SWEEP JAMMING (also called "chirp jamming")
    ---------------------------------------------
    The carrier frequency changes LINEARLY over time, sweeping across a
    range of frequencies. This is effective against frequency-hopping
    victim radios because it eventually covers every hopped frequency.

    Equation (linear chirp):
        instantaneous frequency:  f(t) = f0 + k*t
        phase (integral of frequency): phi(t) = 2*pi*(f0*t + 0.5*k*t^2)
        signal: x(t) = exp(j * phi(t))

    We integrate frequency to get phase because frequency is the RATE OF
    CHANGE of phase -- this is just calculus: phase = integral of frequency
    over time (scaled by 2*pi to convert cycles to radians).

    f0 is the starting frequency, k is the sweep rate (how fast frequency
    changes per sample).
    """
    t = _time_vector(n_samples)
    f0 = rng.uniform(0.02, 0.15)
    k = rng.uniform(0.0005, 0.003)   # sweep rate
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    signal = np.exp(1j * phase)
    return signal


def noise_jamming(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    NOISE JAMMING
    --------------
    Band-limited Gaussian noise designed to raise the noise floor at the
    victim receiver, degrading its effective signal-to-noise ratio.

    Equation:
        x(t) = n_I(t) + j * n_Q(t),    n_I, n_Q ~ Normal(0, 1)

    Then we band-limit it by filtering, since real jammers cannot occupy
    infinite bandwidth -- we apply a simple low-pass filter (moving average)
    to constrain the noise to a realistic bandwidth.
    """
    signal = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    # Apply a simple moving-average low-pass filter to band-limit the noise.
    # A moving average of width w acts as a crude low-pass filter because it
    # smooths out (averages away) fast (high-frequency) fluctuations while
    # preserving slow (low-frequency) trends.
    window = rng.integers(3, 9)
    kernel = np.ones(window) / window
    real_filtered = np.convolve(signal.real, kernel, mode="same")
    imag_filtered = np.convolve(signal.imag, kernel, mode="same")
    return real_filtered + 1j * imag_filtered


def repeat_jamming(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    REPEAT JAMMING (Digital Radio Frequency Memory / DRFM jamming)
    -----------------------------------------------------------------
    The jammer CAPTURES a victim's own signal, then re-transmits ("repeats")
    it after a short delay, often with amplification. This is used for
    deception jamming -- e.g. creating false radar targets at incorrect
    ranges (since radar range is calculated from the round-trip delay,
    artificially adding delay makes the target appear farther away).

    Equation:
        original captured signal: s(t) = exp(j*2*pi*f_c*t)
        repeated signal: x(t) = G * s(t - delta_t)
    where delta_t is the repeat delay and G is the replay gain (amplification).

    We implement the delay by literally shifting the array and zero-padding
    the start (since there is nothing to repeat before the original signal
    arrived).
    """
    t = _time_vector(n_samples)
    f_c = rng.uniform(0.05, 0.35)
    original = np.exp(1j * 2 * np.pi * f_c * t)
    max_delay = max(11, n_samples // 4)
    delay_samples = rng.integers(10, max_delay)
    gain = rng.uniform(0.8, 1.5)

    signal = np.zeros(n_samples, dtype=complex)
    signal[delay_samples:] = gain * original[: n_samples - delay_samples]
    return signal


def pulse_jamming(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    PULSE JAMMING
    --------------
    Short, high-power bursts of energy repeated at a fixed interval,
    specifically designed to interfere with PULSED radar receivers
    (which only listen for echoes during specific time windows).

    Equation (rectangular pulse train modulated onto a carrier):
        x(t) = exp(j*2*pi*f_c*t) * rect_train(t)
    where rect_train(t) is 1 during a pulse (of width tau_p) and 0
    otherwise, repeating every T_r seconds (the "pulse repetition interval").
    """
    t = _time_vector(n_samples)
    f_c = rng.uniform(0.05, 0.4)
    carrier = np.exp(1j * 2 * np.pi * f_c * t)

    pulse_width = rng.integers(15, 60)
    pulse_period = rng.integers(80, 200)

    rect_train = np.zeros(n_samples)
    start = 0
    while start < n_samples:
        end = min(start + pulse_width, n_samples)
        rect_train[start:end] = 1.0
        start += pulse_period

    return carrier * rect_train


def barrage_jamming(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    BARRAGE JAMMING
    -----------------
    The "brute force" jamming approach: high-power WIDEBAND noise that
    covers the ENTIRE operating band of the victim, rather than a narrow
    slice. It uses more power than tone or noise jamming but requires no
    knowledge of the victim's exact frequency.

    Equation:
        x(t) = A * (n_I(t) + j*n_Q(t)),   n_I, n_Q ~ Normal(0, 1)
    where A is a higher gain factor than ordinary noise jamming (to reflect
    the high transmit power characteristic of barrage jamming), and -- in
    contrast to noise_jamming() -- we do NOT band-limit it, since the whole
    point is wideband coverage.
    """
    amplitude = rng.uniform(1.5, 3.0)
    signal = amplitude * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    return signal


# Map jamming type names to their generator functions. Order here defines
# the integer class label each jamming type gets (0 = tone, 1 = sweep, etc).
JAMMING_GENERATORS = {
    "tone": tone_jamming,
    "sweep": sweep_jamming,
    "noise": noise_jamming,
    "repeat": repeat_jamming,
    "pulse": pulse_jamming,
    "barrage": barrage_jamming,
}


# ==============================================================================
# SECTION 2: Noise Addition (to control SNR)
# ==============================================================================

def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal so that the
    resulting Signal-to-Noise Ratio matches the requested snr_db value.

    WHAT IS SNR?
    Signal-to-Noise Ratio measures how much stronger the "wanted" signal
    is compared to background noise, expressed in decibels (dB):
        SNR_dB = 10 * log10(signal_power / noise_power)
    Higher SNR = cleaner signal (easier to classify).
    Lower SNR = noisier signal (harder to classify).
    SNR = 0 dB means signal power EQUALS noise power.

    HOW THIS FUNCTION WORKS:
    1. Compute the average POWER of the input signal:
           P_signal = mean(|x|^2)
       (power of a complex number x is |x|^2 = x * conjugate(x))
    2. Rearrange the SNR formula to solve for the required noise power:
           P_noise = P_signal / 10^(SNR_dB / 10)
    3. Generate complex Gaussian noise with that power and add it to the
       signal. Since complex noise has two independent components (real
       and imaginary), each component gets HALF the total noise power
       (so they combine to the correct total).
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    if signal_power == 0:
        signal_power = 1e-12   # avoid division by zero for an all-zero signal

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std_per_component = np.sqrt(noise_power / 2.0)

    noise = noise_std_per_component * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return signal + noise


# ==============================================================================
# SECTION 3: Dataset Builder
# ==============================================================================

def generate_synthetic_ew_dataset(
    n_examples_per_class_per_snr: int = None,
    snr_levels_db: list = None,
    n_samples_per_signal: int = None,
    seed: int = None,
    verbose: bool = True,
):
    """
    Builds the complete synthetic EW dataset by calling each jamming
    generator many times across multiple SNR levels.

    Returns:
        X : np.ndarray of shape (n_total, n_samples_per_signal, 2)
            The IQ signals, with channel 0 = real (I) part and
            channel 1 = imaginary (Q) part.
        Y : np.ndarray of shape (n_total,)
            Integer jamming-type labels (0-5), see JAMMING_GENERATORS keys.
        Z : np.ndarray of shape (n_total,)
            The SNR (in dB) at which each example was generated.
    """
    cfg = CONFIG["synthetic_ew"]
    if n_examples_per_class_per_snr is None:
        n_examples_per_class_per_snr = cfg["n_examples_per_class_per_snr"]
    if snr_levels_db is None:
        snr_levels_db = cfg["snr_levels_db"]
    if n_samples_per_signal is None:
        n_samples_per_signal = cfg["n_samples_per_signal"]
    if seed is None:
        seed = CONFIG["seed"]

    rng = np.random.default_rng(seed)

    jamming_names = list(JAMMING_GENERATORS.keys())
    n_classes = len(jamming_names)
    n_total = n_classes * len(snr_levels_db) * n_examples_per_class_per_snr

    X = np.zeros((n_total, n_samples_per_signal, 2), dtype=np.float32)
    Y = np.zeros(n_total, dtype=np.int64)
    Z = np.zeros(n_total, dtype=np.float32)

    idx = 0
    iterator = snr_levels_db
    if verbose:
        iterator = tqdm(snr_levels_db, desc="Generating synthetic EW dataset")

    for snr in iterator:
        for class_idx, jam_name in enumerate(jamming_names):
            gen_fn = JAMMING_GENERATORS[jam_name]
            for _ in range(n_examples_per_class_per_snr):
                clean_signal = gen_fn(n_samples_per_signal, rng)
                noisy_signal = add_awgn(clean_signal, snr, rng)

                X[idx, :, 0] = noisy_signal.real
                X[idx, :, 1] = noisy_signal.imag
                Y[idx] = class_idx
                Z[idx] = snr
                idx += 1

    return X, Y, Z, jamming_names


def generate_emitter_dataset(
    n_emitter_profiles: int = None,
    n_examples_per_emitter: int = 500,
    n_samples_per_signal: int = None,
    seed: int = None,
    verbose: bool = True,
):
    """
    Generates a SEPARATE dataset for the emitter-identification task (Head 3
    of the multi-task classifier).

    BACKGROUND -- WHAT IS "EMITTER IDENTIFICATION"?
    Every real radio transmitter has tiny, involuntary hardware imperfections:
    its oscillator drifts slightly from the exact intended frequency, its
    amplifier introduces a small amount of nonlinear distortion, and its
    phase noise has a characteristic profile. These imperfections form a kind
    of "fingerprint" that -- in principle -- lets us tell two transmitters
    apart even if they are sending an IDENTICAL waveform. This is called
    Radio Frequency (RF) Fingerprinting and is used for transmitter
    authentication and anti-spoofing.

    HOW WE SIMULATE THIS:
    We define N synthetic "emitter profiles", each with its own:
        - carrier frequency offset (df)       -- oscillator drift
        - phase noise standard deviation (sigma_phi)
        - third-order nonlinearity coefficient (a3) -- amplifier distortion
    Every example from "emitter k" is generated using THAT emitter's
    specific (df, sigma_phi, a3) values, with some additional random
    variation on top (since even the same physical device's transmissions
    vary slightly each time).
    """
    if n_emitter_profiles is None:
        n_emitter_profiles = CONFIG["synthetic_ew"]["n_emitter_profiles"]
    if n_samples_per_signal is None:
        n_samples_per_signal = CONFIG["synthetic_ew"]["n_samples_per_signal"]
    if seed is None:
        seed = CONFIG["seed"] + 1   # offset seed so this dataset differs from the jamming one

    rng = np.random.default_rng(seed)

    # Define a fixed, reproducible "fingerprint" for each emitter.
    emitter_profiles = []
    for k in range(n_emitter_profiles):
        profile = {
            "carrier_freq": rng.uniform(0.1, 0.4),          # base carrier frequency
            "freq_offset": rng.uniform(-0.002, 0.002),       # oscillator drift
            "phase_noise_std": rng.uniform(0.01, 0.08),      # phase noise strength
            "nonlinearity_a3": rng.uniform(-0.05, 0.05),     # 3rd-order distortion coeff
        }
        emitter_profiles.append(profile)

    n_total = n_emitter_profiles * n_examples_per_emitter
    X = np.zeros((n_total, n_samples_per_signal, 2), dtype=np.float32)
    Y = np.zeros(n_total, dtype=np.int64)

    t = _time_vector(n_samples_per_signal)
    idx = 0
    iterator = range(n_emitter_profiles)
    if verbose:
        iterator = tqdm(iterator, desc="Generating emitter dataset")

    for emitter_id in iterator:
        prof = emitter_profiles[emitter_id]
        for _ in range(n_examples_per_emitter):
            # Effective carrier frequency includes this emitter's drift,
            # PLUS a small amount of random jitter (since drift itself
            # wanders slightly transmission to transmission).
            f_eff = prof["carrier_freq"] + prof["freq_offset"] + rng.normal(0, 0.0005)

            # Phase noise: random-walk phase fluctuation, characteristic
            # of imperfect oscillators. We generate small random phase
            # increments and accumulate (cumulative sum) them over time --
            # this creates a "drifting" phase rather than instantaneous
            # independent noise, which is physically more realistic.
            phase_noise = np.cumsum(
                rng.normal(0, prof["phase_noise_std"], n_samples_per_signal)
            )

            clean_phase = 2 * np.pi * f_eff * t + phase_noise
            clean_signal = np.exp(1j * clean_phase)

            # Apply a simple third-order nonlinearity to the AMPLITUDE,
            # modelling amplifier compression/distortion:
            #   y = x + a3 * |x|^2 * x
            # This is a standard simplified model of amplifier nonlinearity
            # (the cubic term is the dominant nonlinear distortion term in
            # a power amplifier's Taylor series expansion).
            a3 = prof["nonlinearity_a3"]
            distorted_signal = clean_signal + a3 * (np.abs(clean_signal) ** 2) * clean_signal

            # Add a modest amount of channel noise on top (SNR ~15 dB,
            # representing a reasonably good but not perfect intercept).
            final_signal = add_awgn(distorted_signal, snr_db=15.0, rng=rng)

            X[idx, :, 0] = final_signal.real
            X[idx, :, 1] = final_signal.imag
            Y[idx] = emitter_id
            idx += 1

    return X, Y, emitter_profiles


# ==============================================================================
# SECTION 4: Save / Load Helpers
# ==============================================================================

def save_synthetic_dataset(X, Y, Z, jamming_names, output_dir: str = None):
    """Saves the jamming dataset to disk as .npy files (NumPy's native format)."""
    if output_dir is None:
        output_dir = CONFIG["paths"]["synthetic_dir"]
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "jamming_X.npy"), X)
    np.save(os.path.join(output_dir, "jamming_Y.npy"), Y)
    np.save(os.path.join(output_dir, "jamming_Z.npy"), Z)
    with open(os.path.join(output_dir, "jamming_class_names.txt"), "w") as f:
        f.write("\n".join(jamming_names))

    print(f"Saved jamming dataset to: {output_dir}")
    print(f"  X shape: {X.shape}  (n_examples, n_samples_per_signal, 2)")
    print(f"  Y shape: {Y.shape}  (integer class labels 0-{len(jamming_names)-1})")
    print(f"  Z shape: {Z.shape}  (SNR in dB)")


def save_emitter_dataset(X, Y, profiles, output_dir: str = None):
    """Saves the emitter-identification dataset to disk."""
    if output_dir is None:
        output_dir = CONFIG["paths"]["synthetic_dir"]
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "emitter_X.npy"), X)
    np.save(os.path.join(output_dir, "emitter_Y.npy"), Y)

    import json
    with open(os.path.join(output_dir, "emitter_profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"Saved emitter dataset to: {output_dir}")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")


def load_synthetic_dataset(data_dir: str = None):
    """
    Loads the synthetic jamming dataset from disk. If the dataset does not
    exist yet, this function GENERATES it automatically (this satisfies the
    project requirement: "Generate the synthetic dataset automatically if
    it does not exist.")
    """
    if data_dir is None:
        data_dir = CONFIG["paths"]["synthetic_dir"]

    x_path = os.path.join(data_dir, "jamming_X.npy")
    y_path = os.path.join(data_dir, "jamming_Y.npy")
    z_path = os.path.join(data_dir, "jamming_Z.npy")
    names_path = os.path.join(data_dir, "jamming_class_names.txt")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print("Synthetic EW dataset not found on disk. Generating it now...")
        print("(This happens once -- subsequent runs will load the saved file.)")
        set_global_seed()
        X, Y, Z, jamming_names = generate_synthetic_ew_dataset()
        save_synthetic_dataset(X, Y, Z, jamming_names, output_dir=data_dir)
        return X, Y, Z, jamming_names

    X = np.load(x_path)
    Y = np.load(y_path)
    Z = np.load(z_path)
    with open(names_path) as f:
        jamming_names = f.read().splitlines()

    return X, Y, Z, jamming_names


def load_emitter_dataset(data_dir: str = None):
    """Loads the emitter dataset, generating it first if it doesn't exist."""
    if data_dir is None:
        data_dir = CONFIG["paths"]["synthetic_dir"]

    x_path = os.path.join(data_dir, "emitter_X.npy")
    y_path = os.path.join(data_dir, "emitter_Y.npy")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print("Emitter dataset not found on disk. Generating it now...")
        set_global_seed(CONFIG["seed"] + 1)
        X, Y, profiles = generate_emitter_dataset()
        save_emitter_dataset(X, Y, profiles, output_dir=data_dir)
        return X, Y, profiles

    X = np.load(x_path)
    Y = np.load(y_path)
    import json
    with open(os.path.join(data_dir, "emitter_profiles.json")) as f:
        profiles = json.load(f)

    return X, Y, profiles


# ==============================================================================
# SECTION 5: Script Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Running this file directly builds BOTH synthetic datasets from scratch
    # and saves them to disk.
    set_global_seed()
    ensure_directories_exist()

    print("=" * 70)
    print("GENERATING SYNTHETIC ELECTRONIC WARFARE DATASETS")
    print("=" * 70)

    print("\n[1/2] Generating jamming dataset (6 classes)...")
    X_jam, Y_jam, Z_jam, jam_names = generate_synthetic_ew_dataset()
    save_synthetic_dataset(X_jam, Y_jam, Z_jam, jam_names)

    print("\n[2/2] Generating emitter identification dataset (10 classes)...")
    X_emit, Y_emit, profiles = generate_emitter_dataset()
    save_emitter_dataset(X_emit, Y_emit, profiles)

    print("\nDone. Both datasets are saved under data/synthetic_ew/")
