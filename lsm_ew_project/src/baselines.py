"""
src/baselines.py
==================
Implements the FOUR BASELINE MODELS that the LSM pipeline is compared
against. Having strong, fairly-implemented baselines is essential for a
credible research claim -- without them, "our LSM gets 88% accuracy" is
a meaningless number. We need to know how that compares to standard
approaches.

THE FOUR BASELINES:
    1. Expert Features + RBF-SVM  -- classical signal processing approach
    2. 1D CNN                      -- deep learning, convolutional
    3. Bidirectional LSTM          -- deep learning, recurrent
    4. Echo State Network (ESN)    -- the NON-SPIKING ancestor of the LSM

WHY INCLUDE THE ESN SPECIFICALLY?
The Echo State Network is architecturally almost IDENTICAL to our LSM
reservoir (same random sparse recurrent connectivity, same spectral
radius scaling idea) but uses CONTINUOUS-VALUED tanh neurons instead of
SPIKING LIF neurons. Comparing the LSM directly against an ESN isolates
the EFFECT OF SPIKING specifically -- since everything else about the
architecture is kept as similar as possible, any accuracy or efficiency
difference we observe can be attributed to the binary, event-driven
nature of spikes rather than to some other confound.
"""
import os
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import CONFIG


# ==============================================================================
# Baseline 1: Expert Features + RBF-SVM
# ==============================================================================

def extract_expert_features(iq_signal: np.ndarray) -> np.ndarray:
    """
    Computes a set of classical, HAND-DESIGNED signal-processing features
    from a raw IQ signal. This represents the "traditional" (pre-deep-
    learning) approach to RF signal classification, where domain experts
    choose specific statistics believed to be discriminative.

    FEATURES COMPUTED (with the intuition behind each):
      1. Mean amplitude       -- overall signal "loudness"
      2. Std of amplitude     -- how much the envelope fluctuates
      3. Mean instantaneous frequency  -- the average rate of phase change
      4. Std of instantaneous frequency -- frequency stability/modulation
      5. Kurtosis of amplitude  -- "peakiness" of the amplitude distribution
         (helps distinguish constant-envelope modulations like FM/PSK
         from variable-envelope ones like QAM/AM)
      6. Skewness of amplitude  -- asymmetry of the amplitude distribution
      7. Mean of |FFT|          -- average spectral magnitude
      8. Std of |FFT|           -- spectral spread (bandwidth proxy)
      9. Spectral peak-to-average ratio -- how "tonal" vs "noisy" the
         spectrum looks (helps separate jamming types e.g. pure tone
         jamming has a huge spectral peak; noise/barrage jamming does not)
      10. Zero-crossing rate of the real part -- related to dominant
          frequency content

    INSTANTANEOUS FREQUENCY -- THE MATH:
    For a complex signal x(t) = A(t) * exp(j*phi(t)), the instantaneous
    frequency is the rate of change of phase:
        f_inst(t) = (1 / 2*pi) * d(phi)/dt
    We approximate the derivative using a simple finite difference of
    the UNWRAPPED phase (np.unwrap handles the 2*pi phase discontinuities
    that occur when phase naturally wraps from +pi to -pi).
    """
    I = iq_signal[:, 0]
    Q = iq_signal[:, 1]
    complex_signal = I + 1j * Q

    amplitude = np.abs(complex_signal)
    phase = np.unwrap(np.angle(complex_signal))
    inst_freq = np.diff(phase) / (2 * np.pi)   # finite-difference derivative

    fft_magnitude = np.abs(np.fft.fft(complex_signal))

    from scipy.stats import kurtosis, skew

    features = np.array([
        amplitude.mean(),
        amplitude.std(),
        inst_freq.mean(),
        inst_freq.std(),
        kurtosis(amplitude, nan_policy="omit"),
        skew(amplitude, nan_policy="omit"),
        fft_magnitude.mean(),
        fft_magnitude.std(),
        fft_magnitude.max() / (fft_magnitude.mean() + 1e-12),   # peak-to-average ratio
        np.sum(np.diff(np.sign(I)) != 0) / len(I),               # zero-crossing rate
    ])

    # Replace any NaN/inf that might arise from degenerate signals
    # (e.g. all-zero input) with 0, so the classifier never crashes on
    # malformed feature vectors.
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_expert_features_batch(X: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Applies extract_expert_features() to every signal in a dataset."""
    from tqdm import tqdm

    n_signals = X.shape[0]
    n_features = 10
    features = np.zeros((n_signals, n_features))

    iterator = range(n_signals)
    if verbose:
        iterator = tqdm(iterator, desc="Extracting expert features")

    for i in iterator:
        features[i] = extract_expert_features(X[i])

    return features


class ExpertFeatureSVM:
    """
    Wraps expert feature extraction + an RBF-kernel SVM classifier.

    WHY RBF KERNEL?
    The Radial Basis Function (RBF) kernel allows the SVM to learn
    NONLINEAR decision boundaries in the original (low-dimensional,
    10-feature) space:
        K(x, x') = exp(-gamma * ||x - x'||^2)
    This is appropriate here because hand-crafted features often have
    complex, nonlinear relationships with the class label (unlike the
    high-dimensional reservoir states, where a LINEAR readout already
    works well because the reservoir itself has already performed a
    nonlinear transformation of the raw signal).
    """

    def __init__(self, C: float = 1.0, gamma="scale"):
        self.scaler = StandardScaler()
        self.classifier = SVC(C=C, gamma=gamma, kernel="rbf", random_state=CONFIG["seed"])
        self.is_fitted = False

    def fit(self, X_raw_signals: np.ndarray, y_labels: np.ndarray, verbose: bool = True):
        features = extract_expert_features_batch(X_raw_signals, verbose=verbose)
        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, y_labels)
        self.is_fitted = True
        return self

    def predict(self, X_raw_signals: np.ndarray, verbose: bool = False) -> np.ndarray:
        features = extract_expert_features_batch(X_raw_signals, verbose=verbose)
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)


# ==============================================================================
# Baseline 2: 1D Convolutional Neural Network (CNN)
# ==============================================================================

def build_cnn_model(n_classes: int, input_length: int = 1024):
    """
    Builds a 1D CNN for end-to-end modulation/jamming classification
    directly from raw IQ signals.

    ARCHITECTURE:
    The input is treated as a 2-channel (I, Q) 1D signal of length 1024.
    Five convolutional blocks progressively extract higher-level temporal
    features, each block halving the sequence length via max-pooling
    (so by the end, a very long raw signal has been compressed into a
    small number of highly informative feature maps). A final fully
    connected layer maps these features to class scores.

    WHY CONVOLUTIONS FOR RF SIGNALS?
    A 1D convolution slides a small learnable filter across the time
    axis, computing a weighted sum at every position -- this is
    mathematically a (discrete) CROSS-CORRELATION operation, which makes
    convolutional layers naturally suited to detecting LOCAL, repeating
    temporal patterns (like the oscillations of a carrier wave) regardless
    of WHERE in the signal they occur -- a property called translation
    invariance.
    """
    import torch
    import torch.nn as nn

    class CNN1D(nn.Module):
        def __init__(self, n_classes, input_length):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),   # global average pooling
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            # x arrives as (batch, length, channels) -- PyTorch's Conv1d
            # expects (batch, channels, length), so we transpose (swap)
            # the last two dimensions first.
            x = x.transpose(1, 2)
            x = self.conv_layers(x)
            return self.classifier(x)

    return CNN1D(n_classes, input_length)


def train_cnn_classifier(X_train, y_train, X_val, y_val, n_classes: int, verbose: bool = True):
    """
    Trains the 1D CNN baseline using standard PyTorch training loop
    practices: Adam optimiser, cross-entropy loss, mini-batches.

    Returns the trained model plus a dictionary of timing information
    (used later for the energy/latency comparison).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    cfg = CONFIG["baseline"]["cnn"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_cnn_model(n_classes, X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    train_start = time.time()
    for epoch in range(cfg["n_epochs"]):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t.to(device))
                val_preds = val_outputs.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)
            print(
                f"  CNN epoch {epoch+1}/{cfg['n_epochs']}: "
                f"loss={epoch_loss/len(train_dl):.4f}, val_acc={val_acc:.3f}"
            )

    training_time = time.time() - train_start

    # Measure single-sample inference latency, used later for the
    # energy/latency comparison table.
    model.eval()
    with torch.no_grad():
        single_input = X_val_t[:1].to(device)
        n_repeats = 100
        start = time.time()
        for _ in range(n_repeats):
            _ = model(single_input)
        inference_latency = (time.time() - start) / n_repeats

    timing_info = {
        "training_time_seconds": training_time,
        "inference_latency_seconds": inference_latency,
        "device": str(device),
    }

    return model, timing_info


def evaluate_cnn_classifier(model, X_test, y_test) -> dict:
    """Evaluates a trained CNN model on held-out test data."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_t)
        preds = outputs.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, preds)
    return {"accuracy": accuracy, "y_pred": preds}


# ==============================================================================
# Baseline 3: Bidirectional LSTM
# ==============================================================================

def build_lstm_model(n_classes: int, hidden_size: int = 128, num_layers: int = 2):
    """
    Builds a Bidirectional LSTM classifier operating directly on the raw
    IQ sequence.

    WHY BIDIRECTIONAL?
    A standard (unidirectional) LSTM only has access to PAST context when
    making a prediction at time t. A BIDIRECTIONAL LSTM runs two LSTMs --
    one reading the sequence forward, one reading it backward -- and
    concatenates their hidden states. This means the final representation
    incorporates information from the ENTIRE signal (both what came
    before AND after any given point), which is valid here because we
    classify the WHOLE signal only after it has been completely received
    (this is an offline/batch classification setting, not real-time
    streaming, so non-causal bidirectional processing is fair).
    """
    import torch
    import torch.nn as nn

    class LSTMClassifier(nn.Module):
        def __init__(self, n_classes, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=2,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),   # *2 because bidirectional
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            # h_n has shape (num_layers * 2, batch, hidden_size). We take
            # the LAST layer's forward and backward hidden states (indices
            # -2 and -1) and concatenate them.
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            return self.fc(final_hidden)

    return LSTMClassifier(n_classes, hidden_size, num_layers)


def train_lstm_classifier(X_train, y_train, X_val, y_val, n_classes: int, verbose: bool = True):
    """Trains the Bidirectional LSTM baseline. Same training pattern as the CNN."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    cfg = CONFIG["baseline"]["lstm"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_lstm_model(n_classes, cfg["hidden_size"], cfg["num_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    train_start = time.time()
    for epoch in range(cfg["n_epochs"]):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t.to(device))
                val_preds = val_outputs.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)
            print(
                f"  LSTM epoch {epoch+1}/{cfg['n_epochs']}: "
                f"loss={epoch_loss/len(train_dl):.4f}, val_acc={val_acc:.3f}"
            )

    training_time = time.time() - train_start

    model.eval()
    with torch.no_grad():
        single_input = X_val_t[:1].to(device)
        n_repeats = 100
        start = time.time()
        for _ in range(n_repeats):
            _ = model(single_input)
        inference_latency = (time.time() - start) / n_repeats

    timing_info = {
        "training_time_seconds": training_time,
        "inference_latency_seconds": inference_latency,
        "device": str(device),
    }

    return model, timing_info


def evaluate_lstm_classifier(model, X_test, y_test) -> dict:
    """Evaluates a trained LSTM model on held-out test data."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_t)
        preds = outputs.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, preds)
    return {"accuracy": accuracy, "y_pred": preds}


# ==============================================================================
# Baseline 4: Echo State Network (ESN)
# ==============================================================================

class EchoStateNetwork:
    """
    Implements a classic Echo State Network (ESN) -- the CONTINUOUS-VALUED
    (non-spiking) ancestor of the Liquid State Machine.

    KEY DIFFERENCE FROM THE LSM:
    Instead of binary spiking LIF neurons, ESN neurons use a continuous
    activation function (tanh) and update CONTINUOUSLY at every timestep
    (no thresholding, no discrete spike events, no refractory period).

    THE ESN UPDATE EQUATION:
        x[t] = (1 - leak) * x[t-1] + leak * tanh(W @ x[t-1] + W_in @ u[t])

    where x[t] is the reservoir's continuous state vector at time t,
    u[t] is the (continuous-valued, NOT spike-encoded) input at time t,
    W is the recurrent weight matrix (built with the same spectral-radius
    scaling logic as our LSM), W_in is the input weight matrix, and
    "leak" is a leak rate controlling how much of the new tanh-activated
    update replaces the old state at each step (leak=1 means no memory
    of the past state is retained; leak close to 0 means very slow,
    smooth dynamics).

    Just like the LSM, the ESN's recurrent weights are NEVER trained --
    only a linear readout on top of the collected states is trained.
    This makes the ESN the perfect "spiking vs non-spiking" comparison
    point: same reservoir computing PHILOSOPHY, different NEURON MODEL.
    """

    def __init__(self, N: int = None, spectral_radius: float = None, p_conn: float = None, leak: float = 0.3, seed: int = None):
        cfg = CONFIG["baseline"]["esn"]
        self.N = N if N is not None else cfg["N"]
        self.spectral_radius = spectral_radius if spectral_radius is not None else cfg["spectral_radius"]
        self.p_conn = p_conn if p_conn is not None else cfg["p_conn"]
        self.leak = leak
        self.seed = seed if seed is not None else CONFIG["seed"]

        self._rng = np.random.default_rng(self.seed)
        self.W = self._build_recurrent_weights()
        self.W_in = None
        self._n_input_channels_cached = None

    def _build_recurrent_weights(self) -> np.ndarray:
        """Same spectral-radius-scaling logic as the LSM reservoir, but
        WITHOUT the excitatory/inhibitory sign constraint (ESNs do not
        traditionally follow Dale's principle -- this is itself one of
        the structural differences worth noting between the two models)."""
        N = self.N
        rng = self._rng

        W = rng.standard_normal((N, N)) / np.sqrt(N)
        connection_mask = rng.random((N, N)) < self.p_conn
        np.fill_diagonal(connection_mask, False)
        W = W * connection_mask

        eigenvalues = np.linalg.eigvals(W)
        current_spectral_radius = np.max(np.abs(eigenvalues))
        if current_spectral_radius > 1e-10:
            W = W * (self.spectral_radius / current_spectral_radius)

        return W

    def simulate(self, continuous_input: np.ndarray) -> np.ndarray:
        """
        Runs the ESN over a CONTINUOUS-VALUED (not spike-encoded) input
        signal, e.g. the raw normalised IQ samples directly.

        Returns the MEAN reservoir state over time (analogous to the
        LSM's mean-firing-rate state vector, but here it's the mean of
        the continuous tanh activations).
        """
        T, n_input_channels = continuous_input.shape

        if self.W_in is None or self._n_input_channels_cached != n_input_channels:
            self.W_in = self._rng.standard_normal((self.N, n_input_channels)) * 0.5
            self._n_input_channels_cached = n_input_channels

        x = np.zeros(self.N)
        state_accumulator = np.zeros(self.N)

        for t in range(T):
            u_t = continuous_input[t]
            pre_activation = self.W @ x + self.W_in @ u_t
            x = (1 - self.leak) * x + self.leak * np.tanh(pre_activation)
            state_accumulator += x

        return state_accumulator / T


# ==============================================================================
# Script Entry Point (sanity checks for all four baselines)
# ==============================================================================

if __name__ == "__main__":
    print("Testing all baseline model implementations with dummy data...\n")

    rng = np.random.default_rng(CONFIG["seed"])
    n_samples = 60
    n_classes = 6
    signal_length = 256   # shorter for a quick smoke test

    X_dummy = rng.standard_normal((n_samples, signal_length, 2)).astype(np.float32)
    y_dummy = rng.integers(0, n_classes, n_samples)

    split = int(0.7 * n_samples)
    X_train, X_test = X_dummy[:split], X_dummy[split:]
    y_train, y_test = y_dummy[:split], y_dummy[split:]

    # --- Test 1: Expert features + SVM ---
    print("[1/4] Testing Expert Features + RBF-SVM...")
    expert_svm = ExpertFeatureSVM()
    expert_svm.fit(X_train, y_train, verbose=False)
    preds = expert_svm.predict(X_test, verbose=False)
    print(f"      Test accuracy (dummy data): {accuracy_score(y_test, preds):.3f}\n")

    # --- Test 2: CNN ---
    print("[2/4] Testing 1D CNN (1 quick epoch for smoke test)...")
    import copy
    test_cnn_cfg = copy.deepcopy(CONFIG["baseline"]["cnn"])
    CONFIG["baseline"]["cnn"]["n_epochs"] = 1   # speed up smoke test
    model, timing = train_cnn_classifier(X_train, y_train, X_test, y_test, n_classes, verbose=True)
    results = evaluate_cnn_classifier(model, X_test, y_test)
    print(f"      Test accuracy (dummy data): {results['accuracy']:.3f}")
    print(f"      Training time: {timing['training_time_seconds']:.2f}s, "
          f"inference latency: {timing['inference_latency_seconds']*1000:.2f}ms\n")
    CONFIG["baseline"]["cnn"] = test_cnn_cfg   # restore original config

    # --- Test 3: LSTM ---
    print("[3/4] Testing Bidirectional LSTM (1 quick epoch for smoke test)...")
    test_lstm_cfg = copy.deepcopy(CONFIG["baseline"]["lstm"])
    CONFIG["baseline"]["lstm"]["n_epochs"] = 1
    model, timing = train_lstm_classifier(X_train, y_train, X_test, y_test, n_classes, verbose=True)
    results = evaluate_lstm_classifier(model, X_test, y_test)
    print(f"      Test accuracy (dummy data): {results['accuracy']:.3f}")
    print(f"      Training time: {timing['training_time_seconds']:.2f}s, "
          f"inference latency: {timing['inference_latency_seconds']*1000:.2f}ms\n")
    CONFIG["baseline"]["lstm"] = test_lstm_cfg

    # --- Test 4: ESN ---
    print("[4/4] Testing Echo State Network...")
    esn = EchoStateNetwork(N=100)
    state = esn.simulate(X_dummy[0])
    print(f"      ESN state vector shape: {state.shape}, mean={state.mean():.4f}, std={state.std():.4f}")

    print("\nAll four baseline implementations executed successfully.")
