"""
experiments/run_pipeline.py
==============================
THE MAIN SCRIPT — runs the complete research pipeline from start to finish.

HOW TO RUN:
  Full pipeline (takes 1-4 hours depending on hardware):
      python experiments/run_pipeline.py

  Quick demo (reduced sizes, runs in ~10 minutes):
      python experiments/run_pipeline.py --quick

  Skip expensive steps you've already run:
      python experiments/run_pipeline.py --skip-sweep --skip-adversarial

PIPELINE STAGES (in order):
  Stage 1  — Dataset preparation (generate EW, load RadioML)
  Stage 2  — Spike encoding comparison
  Stage 3  — Build LSM reservoir
  Stage 4  — Extract reservoir states for all three tasks
  Stage 5  — Train three multi-task readout heads
  Stage 6  — Train four baseline models
  Stage 7  — Evaluate LSM and baselines (accuracy, confusion matrices,
               ROC/PR curves, accuracy-vs-SNR)
  Stage 8  — Hyperparameter sensitivity sweep
  Stage 9  — Adversarial robustness testing
  Stage 10 — Energy and latency analysis
  Stage 11 — Generate all figures
  Stage 12 — Generate all CSV and text reports
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import CONFIG, set_global_seed, ensure_directories_exist
from src.synthetic_ew_dataset import load_synthetic_dataset, load_emitter_dataset
from src.radioml_loader import get_radioml_data, check_radioml_available
from src.encoding import hybrid_encode, rate_encode, ttfs_encode, compare_encoding_sparsity
from src.reservoir import LSMReservoir, extract_states_for_dataset
from src.readouts import MultiTaskReadout
from src.baselines import (
    ExpertFeatureSVM, train_cnn_classifier, evaluate_cnn_classifier,
    train_lstm_classifier, evaluate_lstm_classifier, EchoStateNetwork,
    extract_expert_features_batch,
)
from src.energy_analysis import (
    estimate_lsm_energy, measure_lsm_spike_sparsity,
    measure_lsm_inference_latency, build_energy_latency_comparison,
    measure_svm_inference_latency,
)
from src.adversarial import evaluate_robustness, robustness_results_to_dataframe
from src import visualization as viz
from src.reporting import (
    save_metrics_csv, save_model_comparison_csv,
    save_energy_and_latency_csvs, save_classification_report_txt,
    save_final_research_report,
)
from src.hyperparameter_search import run_full_hyperparameter_sweep
import joblib


# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LSM-EW full research pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with reduced dataset/reservoir size for a quick demo (~10 min)",
    )
    parser.add_argument("--skip-sweep",      action="store_true", help="Skip hyperparameter sweep")
    parser.add_argument("--skip-adversarial",action="store_true", help="Skip adversarial robustness")
    parser.add_argument("--skip-baselines",  action="store_true", help="Skip CNN/LSTM training")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner(stage: str):
    print(f"\n{'='*70}")
    print(f"  {stage}")
    print(f"{'='*70}")


def encode_dataset(X_signals, encoding_method: str, rng, n_timesteps: int):
    """Encodes a batch of IQ signals into spike trains."""
    from tqdm import tqdm
    spike_trains = []
    for sig in tqdm(X_signals, desc=f"Encoding ({encoding_method})"):
        if encoding_method == "rate":
            spk = rate_encode(sig, rng=rng)
        elif encoding_method == "ttfs":
            spk = ttfs_encode(sig)
        else:
            spk = hybrid_encode(sig, rng=rng)
        # Truncate to n_timesteps
        spike_trains.append(spk[:n_timesteps])
    return spike_trains


def build_snr_accuracy_curve(reservoir, readout_head, X_all, y_all, Z_all, encoding_method, rng, n_timesteps):
    """Computes accuracy at each unique SNR level for the accuracy-vs-SNR plot."""
    snr_levels = sorted(np.unique(Z_all))
    snr_acc = {}
    for snr in snr_levels:
        mask = Z_all == snr
        if mask.sum() < 20:
            continue
        X_snr = X_all[mask]
        y_snr = y_all[mask]
        states = np.zeros((len(X_snr), reservoir.N * 4))
        for i, sig in enumerate(X_snr):
            spk = hybrid_encode(sig, rng=rng)
            states[i] = reservoir.simulate(spk[:n_timesteps], n_timesteps=n_timesteps)
        preds = readout_head.predict(states)
        snr_acc[float(snr)] = accuracy_score(y_snr, preds)
    return snr_acc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── apply QUICK-mode overrides ────────────────────────────────────────────
    if args.quick:
        print("\n[QUICK MODE] Reducing sizes for a fast demonstration run.")
        CONFIG["radioml"]["n_samples_subset"]         = 2000
        CONFIG["radioml"]["min_snr_for_subset"]       = 0
        CONFIG["synthetic_ew"]["n_examples_per_class_per_snr"] = 100
        CONFIG["synthetic_ew"]["snr_levels_db"]       = [0, 10]
        CONFIG["reservoir"]["N"]                      = 1000
        CONFIG["reservoir"]["n_timesteps"]            = 128
        CONFIG["baseline"]["cnn"]["n_epochs"]         = 3
        CONFIG["baseline"]["lstm"]["n_epochs"]        = 3
        CONFIG["hyperparam_sweep"]["n_samples_for_sweep"] = 200
        CONFIG["hyperparam_sweep"]["N_values"]         = [100, 300]
        CONFIG["hyperparam_sweep"]["spectral_radius_values"] = [0.5, 0.9, 1.1]
        CONFIG["hyperparam_sweep"]["p_conn_values"]    = [0.05, 0.1, 0.2]
        CONFIG["hyperparam_sweep"]["input_scale_values"]=[3.0, 6.0, 10.0]
        CONFIG["hyperparam_sweep"]["tau_m_values"]     = [10.0, 20.0, 40.0]

    # ── setup ─────────────────────────────────────────────────────────────────
    set_global_seed()
    ensure_directories_exist()
    rng = np.random.default_rng(CONFIG["seed"])
    T   = CONFIG["reservoir"]["n_timesteps"]

    pipeline_start = time.time()
    all_metrics    = {}
    comparison_rows = []

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 1 — DATA PREPARATION
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 1 — Dataset Preparation")

    # ── Synthetic EW (jamming + emitter) ─────────────────────────────────────
    print("Loading synthetic jamming dataset...")
    X_jam, Y_jam, Z_jam, jam_names = load_synthetic_dataset()
    print(f"  Jamming dataset: {X_jam.shape}, classes: {jam_names}")

    print("Loading synthetic emitter dataset...")
    X_emit, Y_emit, _ = load_emitter_dataset()
    print(f"  Emitter dataset: {X_emit.shape}")

    # ── RadioML ───────────────────────────────────────────────────────────────
    radioml_available = check_radioml_available()
    if radioml_available:
        print("Loading RadioML 2018.01A subset...")
        X_rf, Y_rf, Z_rf, mod_names = get_radioml_data()
        print(f"  RadioML subset: {X_rf.shape}, classes: {len(mod_names)}")
    else:
        print("WARNING: RadioML not found — modulation task will be SKIPPED.")
        print("         Place GOLD_XYZ_OSC.0001_1024.hdf5 in data/radioml/ to enable.")
        X_rf = Y_rf = Z_rf = mod_names = None

    # ── Train/test splits ─────────────────────────────────────────────────────
    X_jam_tr, X_jam_te, Y_jam_tr, Y_jam_te, Z_jam_tr, Z_jam_te = train_test_split(
        X_jam, Y_jam, Z_jam, test_size=0.2, random_state=CONFIG["seed"], stratify=Y_jam
    )
    X_emit_tr, X_emit_te, Y_emit_tr, Y_emit_te = train_test_split(
        X_emit, Y_emit, test_size=0.2, random_state=CONFIG["seed"], stratify=Y_emit
    )
    if radioml_available:
        X_rf_tr, X_rf_te, Y_rf_tr, Y_rf_te, Z_rf_tr, Z_rf_te = train_test_split(
            X_rf, Y_rf, Z_rf, test_size=0.2, random_state=CONFIG["seed"], stratify=Y_rf
        )

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2 — SPIKE ENCODING COMPARISON
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 2 — Spike Encoding Comparison")

    sample_signal   = X_jam[0]
    sparsity_results = compare_encoding_sparsity(sample_signal[:T], rng)
    print("Spike sparsity results:")
    for method, sp in sparsity_results.items():
        print(f"  {method:>8s}: {sp:.4f} ({sp*100:.1f}%)")

    viz.plot_encoding_comparison(sample_signal[:T])
    viz.plot_spike_sparsity(sparsity_results)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 3 — BUILD RESERVOIR
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 3 — Build LSM Reservoir")

    reservoir_cfg = CONFIG["reservoir"]
    reservoir = LSMReservoir(
        N=reservoir_cfg["N"],
        ei_ratio=reservoir_cfg["ei_ratio"],
        p_conn=reservoir_cfg["p_conn"],
        spectral_radius=reservoir_cfg["spectral_radius"],
        tau_m_exc_ms=reservoir_cfg["tau_m_exc_ms"],
        tau_m_inh_ms=reservoir_cfg["tau_m_inh_ms"],
        v_th=reservoir_cfg["v_th"],
        v_reset=reservoir_cfg["v_reset"],
        tau_ref_ms=reservoir_cfg["tau_ref_ms"],
        dt_ms=reservoir_cfg["dt_ms"],
        input_scale=reservoir_cfg["input_scale"],
        seed=CONFIG["seed"],
    )
    reservoir_info = reservoir.get_reservoir_info()
    print("Reservoir built:")
    for k, v in reservoir_info.items():
        print(f"  {k}: {v}")

    # Save reservoir weights for reproducibility
    reservoir_w_path = os.path.join(CONFIG["paths"]["models_dir"], "reservoir_W.npy")
    np.save(reservoir_w_path, reservoir.W)
    print(f"  Reservoir weights saved to {reservoir_w_path}")

    # Reservoir activity visualisation
    viz.plot_reservoir_activity(reservoir, sample_signal[:T])

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 4 — EXTRACT RESERVOIR STATES
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 4 — Extract Reservoir States")

    def encode_and_extract(X_signals, label=""):
        """Encode every signal then run through reservoir, return states."""
        n = len(X_signals)
        states = np.zeros((n, reservoir.N * 4))
        spike_sparsities = []
        from tqdm import tqdm
        for i in tqdm(range(n), desc=f"Reservoir states ({label})"):
            spk = hybrid_encode(X_signals[i][:T], rng=rng)
            states[i] = reservoir.simulate(spk, n_timesteps=T)
            spike_sparsities.append(spk.mean())
        return states, spike_sparsities

    print("Extracting states — JAMMING train/test...")
    S_jam_tr, sp_list = encode_and_extract(X_jam_tr, "jamming-train")
    S_jam_te, _       = encode_and_extract(X_jam_te, "jamming-test")
    mean_sparsity = float(np.mean(sp_list))
    print(f"  Mean spike sparsity (hybrid encoding): {mean_sparsity:.4f}")

    print("Extracting states — EMITTER train/test...")
    S_emit_tr, _ = encode_and_extract(X_emit_tr, "emitter-train")
    S_emit_te, _ = encode_and_extract(X_emit_te, "emitter-test")

    if radioml_available:
        print("Extracting states — RADIOML train/test...")
        S_rf_tr, _ = encode_and_extract(X_rf_tr, "radioml-train")
        S_rf_te, _ = encode_and_extract(X_rf_te, "radioml-test")

    # Cache states to disk (so you can re-run downstream stages without
    # re-running the expensive reservoir simulation each time).
    states_dir = os.path.join(CONFIG["paths"]["models_dir"], "states")
    os.makedirs(states_dir, exist_ok=True)
    np.save(os.path.join(states_dir, "jam_train.npy"), S_jam_tr)
    np.save(os.path.join(states_dir, "jam_test.npy"),  S_jam_te)
    np.save(os.path.join(states_dir, "emit_train.npy"),S_emit_tr)
    np.save(os.path.join(states_dir, "emit_test.npy"), S_emit_te)
    if radioml_available:
        np.save(os.path.join(states_dir, "rf_train.npy"), S_rf_tr)
        np.save(os.path.join(states_dir, "rf_test.npy"),  S_rf_te)
    print(f"  States cached to {states_dir}")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 5 — TRAIN MULTI-TASK READOUT HEADS
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 5 — Train Multi-Task Readout Heads")

    readouts = MultiTaskReadout()
    # Quick-mode subset (temporary)
    S_jam_tr = S_jam_tr[:5000]
    Y_jam_tr = Y_jam_tr[:5000]
    S_jam_te = S_jam_te[:1000]
    Y_jam_te = Y_jam_te[:1000]
    

    print("S_jam_tr shape:", S_jam_tr.shape)
    print("S_jam_tr dtype:", S_jam_tr.dtype)
    print("NaN values:", np.isnan(S_jam_tr).sum())
    print("Inf values:", np.isinf(S_jam_tr).sum())
    

    S_jam_tr = S_jam_tr[:5000]
    Y_jam_tr = Y_jam_tr[:5000]
    readouts.fit_jamming(S_jam_tr, Y_jam_tr)
    jam_eval = readouts.jamming_head.evaluate(S_jam_te, Y_jam_te, jam_names)
    print(f"  Jamming accuracy (test): {jam_eval['accuracy']*100:.2f}%")

    readouts.fit_emitter(S_emit_tr, Y_emit_tr)
    emit_eval = readouts.emitter_head.evaluate(S_emit_te, Y_emit_te)
    print(f"  Emitter accuracy (test): {emit_eval['accuracy']*100:.2f}%")

    all_metrics["LSM (Jamming)"]  = {"accuracy": jam_eval["accuracy"]}
    all_metrics["LSM (Emitter)"]  = {"accuracy": emit_eval["accuracy"]}

    if radioml_available:
        readouts.fit_modulation(S_rf_tr, Y_rf_tr)
        mod_eval = readouts.modulation_head.evaluate(S_rf_te, Y_rf_te, mod_names)
        print(f"  Modulation accuracy (test): {mod_eval['accuracy']*100:.2f}%")
        all_metrics["LSM (Modulation)"] = {"accuracy": mod_eval["accuracy"]}

    readouts.save_all(CONFIG["paths"]["models_dir"])

    # Confusion matrices
    viz.plot_confusion_matrix(
        jam_eval["confusion_matrix"], jam_names,
        title="LSM Jamming Classification (test set)",
        filename="fig03a_confusion_matrix_jamming.png",
    )
    viz.plot_confusion_matrix(
        emit_eval["confusion_matrix"],
        [f"Emitter {i}" for i in range(CONFIG["synthetic_ew"]["n_emitter_profiles"])],
        title="LSM Emitter Identification (test set)",
        filename="fig03b_confusion_matrix_emitter.png",
    )

    # ROC and PR curves for jamming task
    y_score_jam = readouts.jamming_head.predict_proba(S_jam_te)
    viz.plot_roc_curves(Y_jam_te, y_score_jam, jam_names,
                        title="ROC Curves — LSM Jamming Classification",
                        filename="fig05_roc_curves_jamming.png")
    viz.plot_pr_curves(Y_jam_te, y_score_jam, jam_names,
                       title="Precision-Recall — LSM Jamming Classification",
                       filename="fig06_pr_curves_jamming.png")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 6 — TRAIN BASELINE MODELS
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 6 — Train Baseline Models (on Jamming task)")

    n_jam_classes = len(jam_names)
    cnn_timing = lstm_timing = None
    svm_latency = esn_latency = None

    if not args.skip_baselines:

        # Expert features + RBF-SVM
        print("\n[1/4] Expert Features + SVM...")
        svm_model = ExpertFeatureSVM()
        svm_model.fit(X_jam_tr, Y_jam_tr, verbose=True)
        svm_preds = svm_model.predict(X_jam_te, verbose=False)
        svm_acc   = accuracy_score(Y_jam_te, svm_preds)
        print(f"  SVM accuracy: {svm_acc*100:.2f}%")
        all_metrics["Expert + SVM (Jamming)"] = {"accuracy": svm_acc}
        svm_latency = measure_svm_inference_latency(svm_model, X_jam_te[:1])
        joblib.dump(svm_model, os.path.join(CONFIG["paths"]["models_dir"], "svm_baseline.joblib"))

        # 1D CNN
        print("\n[2/4] 1D CNN...")
        cnn_model, cnn_timing = train_cnn_classifier(
            X_jam_tr, Y_jam_tr, X_jam_te, Y_jam_te, n_jam_classes)
        cnn_res = evaluate_cnn_classifier(cnn_model, X_jam_te, Y_jam_te)
        print(f"  CNN accuracy: {cnn_res['accuracy']*100:.2f}%")
        all_metrics["1D CNN (Jamming)"] = {"accuracy": cnn_res["accuracy"]}
        import torch
        torch.save(cnn_model.state_dict(),
                   os.path.join(CONFIG["paths"]["models_dir"], "cnn_baseline.pt"))

        # BiLSTM
        print("\n[3/4] Bidirectional LSTM...")

        # Use a smaller dataset only in Quick Mode
        if args.quick:
            X_lstm_tr = X_jam_tr[:4000]
            Y_lstm_tr = Y_jam_tr[:4000]
            X_lstm_te = X_jam_te[:1000]
            Y_lstm_te = Y_jam_te[:1000]
        else:
            X_lstm_tr = X_jam_tr
            Y_lstm_tr = Y_jam_tr
            X_lstm_te = X_jam_te
            Y_lstm_te = Y_jam_te

        lstm_model, lstm_timing = train_lstm_classifier(
            X_lstm_tr,
            Y_lstm_tr,
            X_lstm_te,
            Y_lstm_te,
            n_jam_classes
        )

        lstm_res = evaluate_lstm_classifier(
            lstm_model,
            X_lstm_te,
            Y_lstm_te
        )

        print(f"  LSTM accuracy: {lstm_res['accuracy']*100:.2f}%")
        all_metrics["BiLSTM (Jamming)"] = {"accuracy": lstm_res["accuracy"]}

        torch.save(
            lstm_model.state_dict(),
            os.path.join(CONFIG["paths"]["models_dir"], "lstm_baseline.pt")
        )

        '''if not args.quick:
                    print("\n[3/4] Bidirectional LSTM...")
            lstm_model, lstm_timing = train_lstm_classifier(
                X_jam_tr, Y_jam_tr, X_jam_te, Y_jam_te, n_jam_classes
            )
            lstm_res = evaluate_lstm_classifier(lstm_model, X_jam_te, Y_jam_te)
            print(f"  LSTM accuracy: {lstm_res['accuracy']*100:.2f}%")
            all_metrics["BiLSTM (Jamming)"] = {"accuracy": lstm_res["accuracy"]}
        else:
            print("\n[3/4] Skipping LSTM in Quick Mode")

        print("\n[3/4] Bidirectional LSTM...")
        lstm_model, lstm_timing = train_lstm_classifier(
            X_jam_tr, Y_jam_tr, X_jam_te, Y_jam_te, n_jam_classes)
        lstm_res = evaluate_lstm_classifier(lstm_model, X_jam_te, Y_jam_te)
        print(f"  LSTM accuracy: {lstm_res['accuracy']*100:.2f}%")
        all_metrics["BiLSTM (Jamming)"] = {"accuracy": lstm_res["accuracy"]}
        torch.save(lstm_model.state_dict(),
                   os.path.join(CONFIG["paths"]["models_dir"], "lstm_baseline.pt"))'''

        # ESN
        print("\n[4/4] Echo State Network...")
        esn = EchoStateNetwork(N=CONFIG["baseline"]["esn"]["N"], seed=CONFIG["seed"])
        print("  Extracting ESN states — train...")
        from tqdm import tqdm
        esn_states_tr = np.array([esn.simulate(X_jam_tr[i][:T]) for i in tqdm(range(len(X_jam_tr)), desc="ESN train")])
        esn_states_te = np.array([esn.simulate(X_jam_te[i][:T]) for i in tqdm(range(len(X_jam_te)), desc="ESN test")])
        from src.readouts import build_jamming_readout
        esn_readout = build_jamming_readout()
        esn_readout.fit(esn_states_tr, Y_jam_tr)
        esn_preds = esn_readout.predict(esn_states_te)
        esn_acc   = accuracy_score(Y_jam_te, esn_preds)
        print(f"  ESN accuracy: {esn_acc*100:.2f}%")
        all_metrics["ESN (Jamming)"] = {"accuracy": esn_acc}
        esn_sample_start = time.time()
        for _ in range(20):
            _ = esn.simulate(X_jam_te[0][:T])
        esn_latency = (time.time() - esn_sample_start) / 20

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 7 — ACCURACY-vs-SNR CURVES
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 7 — Accuracy-vs-SNR Curves")

    print("Computing LSM accuracy at each SNR level...")
    lsm_snr_acc = build_snr_accuracy_curve(
        reservoir, readouts.jamming_head, X_jam, Y_jam, Z_jam,
        "hybrid", rng, T
    )

    snr_curves = {"LSM": lsm_snr_acc}

    if not args.skip_baselines and cnn_timing is not None:
        print("Computing CNN accuracy at each SNR level...")
        import torch
        cnn_model.eval()
        cnn_snr_acc = {}
        for snr in sorted(np.unique(Z_jam)):
            mask = Z_jam == snr
            if mask.sum() < 10:
                continue
            with torch.no_grad():
                X_snr_t = torch.FloatTensor(X_jam[mask])
                preds = cnn_model(X_snr_t).argmax(dim=1).numpy()
            cnn_snr_acc[float(snr)] = accuracy_score(Y_jam[mask], preds)
        snr_curves["CNN"] = cnn_snr_acc

    viz.plot_accuracy_vs_snr(
        snr_curves,
        title="Jamming Classification Accuracy vs. SNR",
        filename="fig04_accuracy_vs_snr_jamming.png",
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 8 — HYPERPARAMETER SWEEP
    # ═════════════════════════════════════════════════════════════════════════
    if not args.skip_sweep:
        banner("STAGE 8 — Hyperparameter Sensitivity Sweep")
        sweep_csv = os.path.join(CONFIG["paths"]["results_dir"], "hyperparameter_results.csv")
        df_sweep  = run_full_hyperparameter_sweep(output_csv=sweep_csv)
        viz.plot_hyperparameter_sensitivity(df_sweep)
    else:
        print("\nSTAGE 8 — Hyperparameter Sweep SKIPPED (--skip-sweep)")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 9 — ADVERSARIAL ROBUSTNESS
    # ═════════════════════════════════════════════════════════════════════════
    if not args.skip_adversarial:
        banner("STAGE 9 — Adversarial Robustness Testing")

        # Use at most 200 test examples to keep this stage manageable
        n_adv = min(200, len(X_jam_te))
        X_adv = X_jam_te[:n_adv]
        Y_adv = Y_jam_te[:n_adv]

        robustness_results = evaluate_robustness(
            X_test_signals=X_adv,
            y_test_labels=Y_adv,
            encode_fn=hybrid_encode,
            reservoir=reservoir,
            readout_head=readouts.jamming_head,
            n_timesteps=T,
        )
        rob_df = robustness_results_to_dataframe(robustness_results)
        rob_csv = os.path.join(CONFIG["paths"]["results_dir"], "adversarial_robustness.csv")
        rob_df.to_csv(rob_csv, index=False)
        print(f"Adversarial results saved to {rob_csv}")

        viz.plot_adversarial_robustness(
            robustness_results,
            clean_accuracy=robustness_results["clean_accuracy"],
        )
    else:
        robustness_results = {}
        print("\nSTAGE 9 — Adversarial Robustness SKIPPED (--skip-adversarial)")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 10 — ENERGY & LATENCY
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 10 — Energy & Latency Analysis")

    print("Measuring LSM inference latency...")
    lsm_inf_latency = measure_lsm_inference_latency(reservoir, hybrid_encode, sample_signal[:T])
    lsm_energy_est  = estimate_lsm_energy(reservoir, mean_sparsity, T)
    lsm_info = {**lsm_energy_est, "inference_latency_seconds": lsm_inf_latency}

    if cnn_timing is None:
        cnn_timing  = {"training_time_seconds": 0, "inference_latency_seconds": 0.002, "device": "cpu"}
    if lstm_timing is None:
        lstm_timing = {"training_time_seconds": 0, "inference_latency_seconds": 0.006, "device": "cpu"}
    if svm_latency is None:
        svm_latency = 0.001
    if esn_latency is None:
        esn_latency = lsm_inf_latency * 0.3   # ESN is faster (no sparsity overhead in Python)

    energy_df = build_energy_latency_comparison(
        lsm_info, cnn_timing, lstm_timing, svm_latency, esn_latency
    )
    save_energy_and_latency_csvs(energy_df)

    print("\nEnergy comparison table:")
    print(energy_df[["model", "energy_per_inference_microjoules", "energy_ratio_vs_lsm", "inference_latency_ms"]].to_string(index=False))

    viz.plot_energy_comparison(energy_df)
    viz.plot_latency_comparison(energy_df)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 11 — REMAINING FIGURES (already saved inline above;
    #              add any top-level summary figures here)
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 11 — Figures Complete")
    print(f"All figures saved to: {CONFIG['paths']['figures_dir']}")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 12 — SAVE ALL REPORTS AND CSVs
    # ═════════════════════════════════════════════════════════════════════════
    banner("STAGE 12 — Generate Reports")

    save_metrics_csv(all_metrics)

    comparison_rows = []
    for model_name, m in all_metrics.items():
        comparison_rows.append({"model": model_name, "accuracy": m["accuracy"]})
    save_model_comparison_csv(comparison_rows)

    y_true_dict    = {"Jamming": Y_jam_te, "Emitter": Y_emit_te}
    y_pred_dict    = {"Jamming": jam_eval["y_pred"], "Emitter": emit_eval["y_pred"]}
    class_names_dict = {"Jamming": jam_names,
                        "Emitter": [f"Emitter {i}" for i in range(CONFIG["synthetic_ew"]["n_emitter_profiles"])]}
    if radioml_available and readouts.modulation_head.is_fitted:
        y_true_dict["Modulation"]    = Y_rf_te
        y_pred_dict["Modulation"]    = mod_eval["y_pred"]
        class_names_dict["Modulation"] = mod_names

    save_classification_report_txt(y_true_dict, y_pred_dict, class_names_dict)

    save_final_research_report(
        reservoir_info=reservoir_info,
        metrics_dict=all_metrics,
        sparsity_dict=sparsity_results,
        energy_df=energy_df,
        robustness_summary=robustness_results,
    )

    # ─── Done ────────────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE  —  total time: {total_time/60:.1f} minutes")
    print(f"  Results: {CONFIG['paths']['results_dir']}")
    print(f"  Models:  {CONFIG['paths']['models_dir']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
