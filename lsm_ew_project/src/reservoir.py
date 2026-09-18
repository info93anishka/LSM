"""
src/reservoir.py
==================
Implements the LIQUID STATE MACHINE (LSM) RESERVOIR -- the core component
of this entire project.

WHAT IS A LIQUID STATE MACHINE (CONCEPTUAL OVERVIEW)?
A Liquid State Machine has two parts:
    1. THE RESERVOIR ("the liquid"): a large, FIXED, randomly-connected
       network of spiking neurons. We never train its weights. Its only
       job is to take an input spike train and transform it into a rich,
       high-dimensional pattern of internal activity.
    2. THE READOUT: a simple, TRAINABLE classifier (in this project:
       logistic regression / SVM / ridge classifier -- see src/readouts.py)
       that looks at a snapshot of the reservoir's activity and decides
       what class the input belongs to.

WHY DOES THIS WORK? (THE "LIQUID" ANALOGY)
Imagine dropping a stone into a pond. The ripples that spread outward
encode information about the stone (its size, where it landed, how hard
it hit) in a complex spatio-temporal pattern. You don't need to "train"
the water -- its natural physics already create a rich, informative
response. A reservoir of spiking neurons behaves similarly: random
recurrent connectivity creates complex temporal dynamics that mix and
combine the input in nonlinear ways, producing a high-dimensional
representation from which a SIMPLE linear classifier can often extract
the answer -- even though the reservoir itself was never optimised for
the task.

THE NEURON MODEL: Leaky Integrate-and-Fire (LIF)
Every neuron in our reservoir follows the LIF model. See the detailed
mathematics in the LSMReservoir class docstring below.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import CONFIG


# ==============================================================================
# Reservoir Construction
# ==============================================================================

class LSMReservoir:
    """
    A Liquid State Machine reservoir: a fixed, randomly-connected
    population of Leaky Integrate-and-Fire (LIF) spiking neurons.

    THE LIF NEURON MODEL (the mathematics):
    Every neuron has an internal "membrane potential" V(t), like a
    bucket collecting electrical charge. The equation governing how V
    changes over time is:

        tau_m * dV/dt = -(V - V_rest) + R * I(t)

    In words: the membrane potential naturally DECAYS back toward a
    resting value (the "-(V - V_rest)" term -- this is the "leak"), but
    is pushed upward by incoming current I(t) (from other neurons'
    spikes, or external input). tau_m controls HOW FAST the leak happens
    -- a large tau_m means slow decay (long memory), small tau_m means
    fast decay (short memory).

    When V(t) reaches a threshold V_th, the neuron "fires" a spike, and
    V is immediately reset to V_reset. After firing, the neuron enters a
    brief REFRACTORY PERIOD (tau_ref) during which it cannot fire again,
    no matter how much input it receives -- this models the real
    biophysical fact that a neuron's ion channels need time to "recover"
    after firing.

    DISCRETE-TIME SIMULATION:
    Computers cannot solve continuous differential equations exactly, so
    we DISCRETISE time into small steps of size dt (e.g. dt = 1 ms) and
    use the EXPONENTIAL EULER method, which is more numerically stable
    than the simpler "forward Euler" method for this type of decay
    equation. The update rule becomes:

        alpha = exp(-dt / tau_m)
        V[t+1] = alpha * V[t] + (1 - alpha) * R * I[t]

    This says: at each timestep, the OLD membrane potential is
    multiplied by a decay factor alpha (between 0 and 1), and a fraction
    (1-alpha) of the new input current is added in. As dt becomes small
    relative to tau_m, this exactly reproduces the continuous-time
    leaky-integration behaviour.

    NETWORK STRUCTURE (Dale's Principle):
    Following a key principle of real cortical circuits known as
    "Dale's principle," every neuron is either PURELY EXCITATORY
    (its outgoing connections always push other neurons UP toward
    firing) or PURELY INHIBITORY (its outgoing connections always
    push other neurons DOWN away from firing). We use an 80/20 split:
    80% excitatory, 20% inhibitory -- this ratio is observed
    empirically in mammalian cortex and is known to produce stable,
    rich dynamics in reservoir models.

    SPECTRAL RADIUS SCALING:
    The recurrent weight matrix W is randomly generated, then RESCALED
    so that its SPECTRAL RADIUS (the largest absolute eigenvalue of W)
    equals a target value rho (typically ~0.9). 

    WHY DOES THE SPECTRAL RADIUS MATTER?
    Think of the reservoir as a dynamical system: if you perturb it
    slightly (e.g. with one input spike) and then watch what happens
    with no further input, does the resulting activity:
        - die out quickly?        (spectral radius << 1: "over-damped")
        - persist and gradually decay in a rich, complex way?
          (spectral radius ~ 1: the "edge of chaos")
        - grow without bound / become chaotic and unpredictable?
          (spectral radius >> 1: "chaotic regime")
    The "edge of chaos" (rho just below 1) is widely considered the
    sweet spot for reservoir computing: dynamics are rich and
    information-preserving for long enough to be useful, but don't
    explode into chaotic noise that erases the input's signature.
    """

    def __init__(
        self,
        N: int = None,
        ei_ratio: float = None,
        p_conn: float = None,
        spectral_radius: float = None,
        tau_m_exc_ms: float = None,
        tau_m_inh_ms: float = None,
        v_th: float = None,
        v_reset: float = None,
        tau_ref_ms: float = None,
        dt_ms: float = None,
        input_scale: float = None,
        seed: int = None,
    ):
        """
        Builds the reservoir's fixed random recurrent connectivity.
        Every parameter defaults to the value in config.py's
        RESERVOIR_CONFIG if not explicitly overridden (this lets the
        hyperparameter sweep script vary individual parameters easily).
        """
        cfg = CONFIG["reservoir"]
        self.N = N if N is not None else cfg["N"]
        self.ei_ratio = ei_ratio if ei_ratio is not None else cfg["ei_ratio"]
        self.p_conn = p_conn if p_conn is not None else cfg["p_conn"]
        self.spectral_radius = (
            spectral_radius if spectral_radius is not None else cfg["spectral_radius"]
        )
        self.tau_m_exc_ms = tau_m_exc_ms if tau_m_exc_ms is not None else cfg["tau_m_exc_ms"]
        self.tau_m_inh_ms = tau_m_inh_ms if tau_m_inh_ms is not None else cfg["tau_m_inh_ms"]
        self.v_th = v_th if v_th is not None else cfg["v_th"]
        self.v_reset = v_reset if v_reset is not None else cfg["v_reset"]
        self.tau_ref_ms = tau_ref_ms if tau_ref_ms is not None else cfg["tau_ref_ms"]
        self.dt_ms = dt_ms if dt_ms is not None else cfg["dt_ms"]
        self.input_scale = input_scale if input_scale is not None else cfg["input_scale"]
        self.seed = seed if seed is not None else CONFIG["seed"]

        self._rng = np.random.default_rng(self.seed)

        self.n_excitatory = int(self.N * self.ei_ratio)
        self.n_inhibitory = self.N - self.n_excitatory

        # Per-neuron membrane time constant: excitatory neurons get the
        # slower (longer-memory) time constant, inhibitory neurons get
        # the faster one -- this matches biological observations that
        # inhibitory interneurons tend to respond and decay more quickly.
        self.tau_m = np.concatenate([
            np.full(self.n_excitatory, self.tau_m_exc_ms),
            np.full(self.n_inhibitory, self.tau_m_inh_ms),
        ])

        # Build the recurrent weight matrix W, shape (N, N).
        # W[i, j] = weight from neuron j onto neuron i (i.e. how strongly
        # a spike from neuron j affects neuron i's membrane potential).
        self.W = self._build_recurrent_weights()

        # The exponential decay factor for each neuron's membrane,
        # precomputed once here so the simulation loop doesn't have to
        # recompute exp() at every timestep (this is purely a speed
        # optimisation -- the math is identical either way).
        self.alpha = np.exp(-self.dt_ms / self.tau_m)

        # Input weight matrix is built lazily, the first time we know
        # how many input channels there are (e.g. 2 for I/Q data).
        self.W_in = None
        self._n_input_channels_cached = None

    def _build_recurrent_weights(self) -> np.ndarray:
        """
        Constructs the (N, N) recurrent weight matrix following these steps:
          1. Start with random Gaussian weights.
          2. Apply sparsity: zero out most connections (only p_conn
             fraction survive), and remove self-connections.
          3. Enforce Dale's principle: the LAST n_inhibitory columns
             (i.e. all outgoing connections FROM inhibitory neurons) are
             forced to be negative.
          4. Rescale the entire matrix so its spectral radius equals
             the target value.
        """
        N = self.N
        rng = self._rng

        # Step 1: random Gaussian weights, scaled down a bit so the raw
        # (pre-spectral-scaling) magnitudes aren't extreme.
        W = rng.standard_normal((N, N)) / np.sqrt(N)

        # Step 2: sparsity mask. Each entry survives with probability
        # p_conn; the rest are zeroed. No self-connections (diagonal = 0).
        connection_mask = rng.random((N, N)) < self.p_conn
        np.fill_diagonal(connection_mask, False)
        W = W * connection_mask

        # Step 3: Dale's principle. Columns n_excitatory: onward
        # correspond to inhibitory neurons' OUTGOING weights -- force
        # these to be negative (multiply by -1, then take absolute
        # value first to guarantee sign correctness regardless of the
        # original random draw).
        W[:, self.n_excitatory:] = -np.abs(W[:, self.n_excitatory:])
        # Excitatory neurons' outgoing weights (first n_excitatory
        # columns) are forced positive for the same reason.
        W[:, : self.n_excitatory] = np.abs(W[:, : self.n_excitatory])

        # Step 4: rescale so the spectral radius (largest |eigenvalue|)
        # equals self.spectral_radius. We compute eigenvalues of the
        # CURRENT matrix, find the largest absolute one, then scale
        # every entry by (target / current) so the new largest
        # eigenvalue magnitude becomes exactly the target.
        eigenvalues = np.linalg.eigvals(W)
        current_spectral_radius = np.max(np.abs(eigenvalues))

        if current_spectral_radius > 1e-10:
            W = W * (self.spectral_radius / current_spectral_radius)

        return W

    def _build_input_weights(self, n_input_channels: int) -> np.ndarray:
        """
        Builds the (N, n_input_channels) input weight matrix the first
        time it's needed. Each reservoir neuron receives a random
        (mostly weak) connection from each input channel.
        """
        W_in = self._rng.standard_normal((self.N, n_input_channels)) * self.input_scale
        return W_in

    def simulate(self, spike_input: np.ndarray, n_timesteps: int = None) -> np.ndarray:
        """
        Runs the reservoir for n_timesteps, driven by the given input
        spike train, and returns a "reservoir state vector" summarising
        the network's activity -- this is what gets passed to the
        downstream readout classifiers.

        SIMULATION LOOP (step by step):
        At every timestep t:
          1. Compute the EXTERNAL input current to every neuron:
                 I_input[t] = W_in @ spike_input[t]
             (matrix-vector product: each neuron's input current is a
             weighted sum of which input channels just spiked)
          2. Compute the RECURRENT input current from neurons that
             fired in the PREVIOUS timestep:
                 I_recurrent[t] = W @ fired[t-1]
          3. Update every neuron's membrane potential using the
             exponential-Euler leaky-integration rule:
                 V[t] = alpha * V[t-1] + (1 - alpha) * (I_input[t] + I_recurrent[t])
          4. Force neurons in their refractory period to stay at
             V_reset, regardless of input.
          5. Check which neurons crossed threshold (V >= V_th): these
             neurons FIRE. Record this. Reset their membrane potential
             to V_reset and put them in a refractory period.
          6. Accumulate a running spike count for each neuron, for the
             final state-vector computation.

        AFTER the simulation loop finishes, the RESERVOIR STATE VECTOR
        is computed as each neuron's MEAN FIRING RATE over the whole
        simulation window:

            state[i] = (1 / T) * sum_t spike[i, t]

        This single N-dimensional vector is what represents "what the
        reservoir thought about this input" -- and it's the only thing
        that gets passed on to the (trainable) readout classifiers.

        Parameters
        ----------
        spike_input : np.ndarray, shape (T, n_input_channels)
            A binary spike train (from one of the encoding functions in
            src/encoding.py).
        n_timesteps : int, optional
            Number of timesteps to simulate. Defaults to T (the length
            of spike_input) if not specified, or can be set shorter to
            only use the first part of a longer signal (useful for
            speed during hyperparameter sweeps).

        Returns
        -------
        np.ndarray, shape (N,)
            The mean firing rate of every neuron over the simulation --
            the reservoir state vector.
        """
        T_available, n_input_channels = spike_input.shape
        T = n_timesteps if n_timesteps is not None else T_available
        T = min(T, T_available)

        # Build (or reuse) the input weight matrix. We cache it so that
        # repeated calls to simulate() with the same number of input
        # channels don't keep regenerating random input weights (which
        # would make the reservoir's behaviour inconsistent across
        # calls -- the WHOLE POINT of a fixed reservoir is that its
        # weights, including input weights, stay constant).
        if self.W_in is None or self._n_input_channels_cached != n_input_channels:
            self.W_in = self._build_input_weights(n_input_channels)
            self._n_input_channels_cached = n_input_channels

        V = np.zeros(self.N)                  # membrane potentials, start at 0
        refractory_remaining = np.zeros(self.N)  # ms remaining in refractory period
        spike_count = np.zeros(self.N)         # running total of spikes per neuron
        spike_history = np.zeros((T, self.N), dtype=np.uint8)
        fired_prev = np.zeros(self.N)          # which neurons fired LAST timestep

        for t in range(T):
            # --- Step 1: external input current ---
            current_input_spikes = spike_input[t].astype(float)
            I_input = self.W_in @ current_input_spikes

            # --- Step 2: recurrent current from PREVIOUS timestep's spikes ---
            I_recurrent = self.W @ fired_prev

            # --- Step 3: update membrane potential (exponential Euler) ---
            V = self.alpha * V + (1 - self.alpha) * (I_input + I_recurrent)

            # --- Step 4: enforce refractory period ---
            in_refractory = refractory_remaining > 0
            V[in_refractory] = self.v_reset
            refractory_remaining = np.maximum(refractory_remaining - self.dt_ms, 0.0)

            # --- Step 5: check threshold crossing -> firing ---
            fired_now = V >= self.v_th
            spike_history[t] = fired_now
            spike_count += fired_now.astype(float)
            V[fired_now] = self.v_reset
            refractory_remaining[fired_now] = self.tau_ref_ms

            fired_prev = fired_now.astype(float)

        # ---------- Rich Temporal Reservoir Features ----------

        mean_rate = spike_history.mean(axis=0)
        max_rate = spike_history.max(axis=0)
        std_rate = spike_history.std(axis=0)
        spike_count_norm = spike_count / T
        
        state_vector = np.concatenate([
            mean_rate,
            max_rate,
            std_rate,
            spike_count_norm
        ])
        
        return state_vector

    def get_reservoir_info(self) -> dict:
        """Returns a dictionary summarising this reservoir's configuration
        -- useful for logging and for the final research report."""
        return {
            "N": self.N,
            "n_excitatory": self.n_excitatory,
            "n_inhibitory": self.n_inhibitory,
            "p_conn": self.p_conn,
            "spectral_radius_target": self.spectral_radius,
            "spectral_radius_actual": float(np.max(np.abs(np.linalg.eigvals(self.W)))),
            "tau_m_exc_ms": self.tau_m_exc_ms,
            "tau_m_inh_ms": self.tau_m_inh_ms,
            "v_th": self.v_th,
            "input_scale": self.input_scale,
        }


# ==============================================================================
# Batch Processing Helper
# ==============================================================================

def extract_states_for_dataset(
    spike_trains: list,
    reservoir: LSMReservoir,
    n_timesteps: int = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Runs MANY spike-encoded signals through the SAME reservoir, one at a
    time, and stacks the resulting state vectors into a single matrix.

    This is the function used by every training script to convert a
    dataset of spike trains into a dataset of reservoir states, ready for
    the readout classifiers.

    Parameters
    ----------
    spike_trains : list of np.ndarray, each shape (T, n_channels)
    reservoir : LSMReservoir
        An already-constructed reservoir (so every signal is processed
        by the EXACT SAME fixed random network).
    n_timesteps : int, optional
        Passed through to reservoir.simulate().
    verbose : bool
        Whether to print progress.

    Returns
    -------
    np.ndarray, shape (n_signals, N)
    """
    from tqdm import tqdm

    n_signals = len(spike_trains)
    states = np.zeros((n_signals, reservoir.N * 4))

    iterator = range(n_signals)
    if verbose:
        iterator = tqdm(iterator, desc="Extracting reservoir states")

    for i in iterator:
        states[i] = reservoir.simulate(spike_trains[i], n_timesteps=n_timesteps)

    return states


# ==============================================================================
# Script Entry Point (sanity check)
# ==============================================================================

if __name__ == "__main__":
    print("Testing LSMReservoir construction and simulation...\n")

    from src.encoding import rate_encode

    # Build a small reservoir for a quick test (N=200 instead of the full
    # 1000, just so this sanity check runs in a couple of seconds).
    reservoir = LSMReservoir(N=200, seed=CONFIG["seed"])
    info = reservoir.get_reservoir_info()

    print("Reservoir configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Generate 5 different random test signals and confirm the reservoir
    # produces DIFFERENT state vectors for different inputs (a basic
    # sanity check that the reservoir is actually doing something useful,
    # rather than collapsing every input to the same output).
    print("\nRunning 5 random test signals through the reservoir...")
    rng = np.random.default_rng(CONFIG["seed"])

    states = []
    for i in range(5):
        dummy_signal = rng.standard_normal((128, 2))
        spikes = rate_encode(dummy_signal, rng=rng)
        state = reservoir.simulate(spikes)
        states.append(state)
        print(
            f"  Signal {i}: mean firing rate = {state.mean():.4f}, "
            f"std = {state.std():.4f}, "
            f"active neurons = {(state > 0).sum()}/{reservoir.N}"
        )

    states = np.array(states)
    pairwise_distances = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            dist = np.linalg.norm(states[i] - states[j])
            pairwise_distances.append(dist)

    print(f"\nAverage pairwise distance between state vectors: {np.mean(pairwise_distances):.4f}")
    print("(A larger, non-zero number here confirms the reservoir produces")
    print(" genuinely different responses to different inputs, as expected.)")
