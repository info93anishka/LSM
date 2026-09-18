"""
tests/test_all.py
==================
Pytest unit tests for every module in the LSM-EW project.

HOW TO RUN:
    # From the project root:
    pytest tests/test_all.py -v

    # Run just one test class:
    pytest tests/test_all.py::TestEncoding -v

All tests use tiny data sizes and very small models so the full test
suite completes in under 60 seconds on any modern laptop.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG, set_global_seed, ensure_directories_exist


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fix_seed():
    set_global_seed()
    ensure_directories_exist()


@pytest.fixture
def small_iq_signal():
    rng = np.random.default_rng(42)
    return rng.standard_normal((128, 2)).astype(np.float32)


@pytest.fixture
def small_iq_batch():
    rng = np.random.default_rng(42)
    return rng.standard_normal((20, 128, 2)).astype(np.float32)


@pytest.fixture
def small_reservoir():
    from src.reservoir import LSMReservoir
    return LSMReservoir(N=50, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_seed_is_integer(self):
        assert isinstance(CONFIG["seed"], int)

    def test_required_keys_present(self):
        for key in ("reservoir", "encoding", "radioml", "synthetic_ew", "energy"):
            assert key in CONFIG, f"Missing CONFIG key: {key}"

    def test_paths_are_strings(self):
        for k, v in CONFIG["paths"].items():
            assert isinstance(v, str), f"Path '{k}' is not a string"

    def test_reservoir_n_positive(self):
        assert CONFIG["reservoir"]["N"] > 0

    def test_spectral_radius_valid(self):
        rho = CONFIG["reservoir"]["spectral_radius"]
        assert 0 < rho < 2, f"Spectral radius {rho} seems unreasonable"


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestEncoding:
    def test_rate_encode_shape(self, small_iq_signal):
        from src.encoding import rate_encode
        rng = np.random.default_rng(42)
        spikes = rate_encode(small_iq_signal, rng=rng)
        assert spikes.shape == small_iq_signal.shape

    def test_rate_encode_binary(self, small_iq_signal):
        from src.encoding import rate_encode
        rng = np.random.default_rng(42)
        spikes = rate_encode(small_iq_signal, rng=rng)
        assert set(np.unique(spikes)).issubset({0, 1})

    def test_ttfs_encode_shape(self, small_iq_signal):
        from src.encoding import ttfs_encode
        spikes = ttfs_encode(small_iq_signal)
        assert spikes.shape == small_iq_signal.shape

    def test_ttfs_encode_sparse(self, small_iq_signal):
        from src.encoding import ttfs_encode
        spikes = ttfs_encode(small_iq_signal)
        # TTFS should be sparser than 50%
        assert spikes.mean() < 0.5

    def test_hybrid_encode_shape(self, small_iq_signal):
        from src.encoding import hybrid_encode
        rng = np.random.default_rng(42)
        spikes = hybrid_encode(small_iq_signal, rng=rng)
        assert spikes.shape == small_iq_signal.shape

    def test_sparsity_ordering(self, small_iq_signal):
        """TTFS should be sparser than or equal to hybrid, which <= rate."""
        from src.encoding import rate_encode, ttfs_encode, hybrid_encode, compute_spike_sparsity
        rng = np.random.default_rng(42)
        sp_rate   = compute_spike_sparsity(rate_encode(small_iq_signal, rng=rng))
        sp_ttfs   = compute_spike_sparsity(ttfs_encode(small_iq_signal))
        sp_hybrid = compute_spike_sparsity(hybrid_encode(small_iq_signal, rng=rng))
        assert sp_ttfs <= sp_hybrid + 0.05, "TTFS should be sparser than hybrid"
        assert sp_hybrid <= sp_rate + 0.05, "Hybrid should be sparser than or close to rate"

    def test_encode_dispatcher(self, small_iq_signal):
        from src.encoding import encode_signal
        rng = np.random.default_rng(42)
        for method in ("rate", "ttfs", "hybrid"):
            spikes = encode_signal(small_iq_signal, method=method, rng=rng)
            assert spikes.shape == small_iq_signal.shape, f"Shape mismatch for method={method}"

    def test_invalid_method_raises(self, small_iq_signal):
        from src.encoding import encode_signal
        with pytest.raises(ValueError):
            encode_signal(small_iq_signal, method="nonexistent_method")


# ─────────────────────────────────────────────────────────────────────────────
# Reservoir
# ─────────────────────────────────────────────────────────────────────────────

class TestReservoir:
    def test_build_correct_shape(self, small_reservoir):
        N = small_reservoir.N
        assert small_reservoir.W.shape == (N, N)

    def test_spectral_radius_close_to_target(self, small_reservoir):
        actual_rho = np.max(np.abs(np.linalg.eigvals(small_reservoir.W)))
        target_rho = small_reservoir.spectral_radius
        assert abs(actual_rho - target_rho) < 0.01, \
            f"Spectral radius {actual_rho:.4f} far from target {target_rho}"

    def test_dale_excitatory_positive(self, small_reservoir):
        n_exc = small_reservoir.n_excitatory
        exc_weights = small_reservoir.W[:, :n_exc]
        # All non-zero excitatory weights should be non-negative
        nz = exc_weights[exc_weights != 0]
        assert np.all(nz >= 0), "Some excitatory weights are negative (Dale's principle violated)"

    def test_dale_inhibitory_nonpositive(self, small_reservoir):
        n_exc = small_reservoir.n_excitatory
        inh_weights = small_reservoir.W[:, n_exc:]
        nz = inh_weights[inh_weights != 0]
        assert np.all(nz <= 0), "Some inhibitory weights are positive (Dale's principle violated)"

    def test_no_self_connections(self, small_reservoir):
        assert np.all(np.diag(small_reservoir.W) == 0), "Reservoir has self-connections"

    def test_simulate_output_shape(self, small_reservoir, small_iq_signal):
        from src.encoding import hybrid_encode
        rng = np.random.default_rng(42)
        spikes = hybrid_encode(small_iq_signal, rng=rng)
        state = small_reservoir.simulate(spikes)
        assert state.shape == (small_reservoir.N,)

    def test_simulate_produces_firing(self, small_reservoir, small_iq_signal):
        from src.encoding import hybrid_encode
        rng = np.random.default_rng(42)
        spikes = hybrid_encode(small_iq_signal, rng=rng)
        state = small_reservoir.simulate(spikes)
        # At least some neurons should fire
        assert state.sum() > 0, "Reservoir produced zero firing (dead network)"

    def test_different_inputs_produce_different_states(self, small_reservoir):
        from src.encoding import rate_encode
        rng = np.random.default_rng(42)
        s1 = rng.standard_normal((128, 2))
        s2 = rng.standard_normal((128, 2)) * 3.0   # clearly different signal
        spk1 = rate_encode(s1, rng=rng)
        spk2 = rate_encode(s2, rng=rng)
        state1 = small_reservoir.simulate(spk1)
        state2 = small_reservoir.simulate(spk2)
        # States should not be identical
        assert not np.allclose(state1, state2, atol=1e-6), \
            "Reservoir produced identical states for different inputs"

    def test_get_reservoir_info_keys(self, small_reservoir):
        info = small_reservoir.get_reservoir_info()
        for key in ("N", "n_excitatory", "n_inhibitory", "p_conn",
                    "spectral_radius_target", "spectral_radius_actual"):
            assert key in info, f"Missing key in reservoir info: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic EW Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticEWDataset:
    def test_generate_small_jamming(self):
        from src.synthetic_ew_dataset import generate_synthetic_ew_dataset
        X, Y, Z, names = generate_synthetic_ew_dataset(
            n_examples_per_class_per_snr=3,
            snr_levels_db=[0, 10],
            n_samples_per_signal=64,
            verbose=False,
        )
        expected_n = 6 * 2 * 3
        assert X.shape == (expected_n, 64, 2)
        assert Y.shape == (expected_n,)
        assert Z.shape == (expected_n,)
        assert len(names) == 6

    def test_jamming_labels_in_range(self):
        from src.synthetic_ew_dataset import generate_synthetic_ew_dataset
        _, Y, _, names = generate_synthetic_ew_dataset(
            n_examples_per_class_per_snr=2, snr_levels_db=[0],
            n_samples_per_signal=32, verbose=False)
        assert Y.min() >= 0
        assert Y.max() < len(names)

    def test_all_6_classes_present(self):
        from src.synthetic_ew_dataset import generate_synthetic_ew_dataset
        _, Y, _, names = generate_synthetic_ew_dataset(
            n_examples_per_class_per_snr=2, snr_levels_db=[0, 5, 10],
            n_samples_per_signal=32, verbose=False)
        assert len(np.unique(Y)) == 6

    def test_generate_emitter_dataset(self):
        from src.synthetic_ew_dataset import generate_emitter_dataset
        X, Y, profiles = generate_emitter_dataset(
            n_emitter_profiles=3, n_examples_per_emitter=4,
            n_samples_per_signal=32, verbose=False)
        assert X.shape == (12, 32, 2)
        assert Y.shape == (12,)
        assert len(profiles) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Readouts
# ─────────────────────────────────────────────────────────────────────────────

class TestReadouts:
    def _make_dummy_states_labels(self, n=100, n_features=50, n_classes=6):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, n_features))
        y = rng.integers(0, n_classes, n)
        return X, y

    def test_modulation_readout_fits_and_predicts(self):
        from src.readouts import build_modulation_readout
        X, y = self._make_dummy_states_labels(n_classes=24)
        head = build_modulation_readout()
        head.fit(X, y)
        preds = head.predict(X)
        assert preds.shape == y.shape

    def test_jamming_readout_fits_and_predicts(self):
        from src.readouts import build_jamming_readout
        X, y = self._make_dummy_states_labels(n_classes=6)
        head = build_jamming_readout()
        head.fit(X, y)
        preds = head.predict(X)
        assert preds.shape == y.shape

    def test_emitter_readout_fits_and_predicts(self):
        from src.readouts import build_emitter_readout
        X, y = self._make_dummy_states_labels(n_classes=10)
        head = build_emitter_readout()
        head.fit(X, y)
        preds = head.predict(X)
        assert preds.shape == y.shape

    def test_predict_proba_shape(self):
        from src.readouts import build_modulation_readout
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 30))
        y = rng.integers(0, 5, 50)
        head = build_modulation_readout()
        head.fit(X, y)
        proba = head.predict_proba(X)
        assert proba.shape[0] == 50
        assert proba.shape[1] >= 2

    def test_evaluate_returns_accuracy(self):
        from src.readouts import build_jamming_readout
        from sklearn.metrics import accuracy_score
        X, y = self._make_dummy_states_labels(n_classes=6)
        head = build_jamming_readout()
        head.fit(X, y)
        results = head.evaluate(X, y)
        assert "accuracy" in results
        assert 0 <= results["accuracy"] <= 1

    def test_readout_not_fitted_raises(self):
        from src.readouts import build_jamming_readout
        head = build_jamming_readout()
        with pytest.raises(RuntimeError):
            head.predict(np.zeros((5, 50)))


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial attacks
# ─────────────────────────────────────────────────────────────────────────────

class TestAdversarial:
    def test_awgn_attack_shape(self, small_iq_signal):
        from src.adversarial import awgn_attack
        rng = np.random.default_rng(42)
        result = awgn_attack(small_iq_signal, target_snr_db=5.0, rng=rng)
        assert result.shape == small_iq_signal.shape

    def test_awgn_attack_changes_signal(self, small_iq_signal):
        from src.adversarial import awgn_attack
        rng = np.random.default_rng(42)
        result = awgn_attack(small_iq_signal, target_snr_db=5.0, rng=rng)
        assert not np.allclose(result, small_iq_signal)

    def test_freq_perturbation_shape(self, small_iq_signal):
        from src.adversarial import frequency_perturbation_attack
        rng = np.random.default_rng(42)
        result = frequency_perturbation_attack(small_iq_signal, 0.01, rng)
        assert result.shape == small_iq_signal.shape

    def test_distortion_attack_reduces_amplitude(self, small_iq_signal):
        from src.adversarial import signal_distortion_attack
        original_power = np.mean(small_iq_signal ** 2)
        result = signal_distortion_attack(small_iq_signal, distortion_level=0.5)
        distorted_power = np.mean(result ** 2)
        assert distorted_power <= original_power + 1e-6, \
            "Distortion attack should not increase signal power"

    def test_zero_distortion_leaves_signal_unchanged(self, small_iq_signal):
        from src.adversarial import signal_distortion_attack
        result = signal_distortion_attack(small_iq_signal, distortion_level=0.0)
        assert np.allclose(result, small_iq_signal, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Baselines (smoke tests — just wiring, not accuracy)
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselines:
    def test_expert_features_extraction(self):
        from src.baselines import extract_expert_features
        rng = np.random.default_rng(42)
        sig = rng.standard_normal((256, 2)).astype(np.float32)
        feats = extract_expert_features(sig)
        assert feats.shape == (10,)
        assert not np.any(np.isnan(feats))

    def test_esn_simulate_output(self):
        from src.baselines import EchoStateNetwork
        esn = EchoStateNetwork(N=30, seed=42)
        rng = np.random.default_rng(42)
        signal = rng.standard_normal((64, 2))
        state = esn.simulate(signal)
        assert state.shape == (30,)
        assert not np.all(state == 0)


# ─────────────────────────────────────────────────────────────────────────────
# Energy Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyAnalysis:
    def test_lsm_energy_estimate_positive(self):
        from src.energy_analysis import estimate_lsm_energy
        from src.reservoir import LSMReservoir
        res = LSMReservoir(N=50, seed=42)
        energy = estimate_lsm_energy(res, mean_spike_sparsity=0.05, n_timesteps=64)
        assert energy["energy_joules"] > 0
        assert energy["total_synops"] > 0

    def test_baseline_energy_estimate(self):
        from src.energy_analysis import estimate_baseline_energy
        result = estimate_baseline_energy(0.002, "cpu")
        assert result["energy_joules"] > 0

    def test_comparison_table_shape(self):
        from src.energy_analysis import build_energy_latency_comparison
        lsm_info = {"energy_joules": 1e-9, "energy_microjoules": 0.001,
                    "total_synops": 1000, "inference_latency_seconds": 0.003}
        cnn_timing  = {"training_time_seconds": 10, "inference_latency_seconds": 0.002, "device": "cpu"}
        lstm_timing = {"training_time_seconds": 20, "inference_latency_seconds": 0.005, "device": "cpu"}
        df = build_energy_latency_comparison(lsm_info, cnn_timing, lstm_timing, 0.001, 0.002)
        assert len(df) == 5
        assert "energy_ratio_vs_lsm" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
