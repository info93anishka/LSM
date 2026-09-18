"""
src/visualization.py
======================
Generates ALL figures used in the research paper and final report.
Every function saves its output as a high-resolution PNG to results/figures/.

Figures produced:
  Fig 01 - Spike encoding comparison (3 methods side-by-side)
  Fig 02 - Reservoir activity heatmap (neuron firings over time)
  Fig 03 - Confusion matrix per trained head
  Fig 04 - Accuracy vs SNR curve (LSM vs baselines)
  Fig 05 - ROC curves (one-vs-rest per class)
  Fig 06 - Precision-Recall curves
  Fig 07 - Hyperparameter sensitivity (5 subplots)
  Fig 08 - Energy comparison bar chart
  Fig 09 - Latency comparison bar chart
  Fig 10 - Spike sparsity comparison (3 encoding methods)
  Fig 11 - Adversarial robustness degradation curves
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on servers with no display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from config import CONFIG, ensure_directories_exist

# ── consistent style for every figure ──────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "lsm":     "#1E4D8C",
    "cnn":     "#E05C2A",
    "lstm":    "#2EAA5E",
    "esn":     "#9B4DCA",
    "svm":     "#D4A017",
    "rate":    "#1E4D8C",
    "ttfs":    "#E05C2A",
    "hybrid":  "#2EAA5E",
}

FIG_DIR = CONFIG["paths"]["figures_dir"]


def _savefig(fig, filename: str):
    ensure_directories_exist()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 01 – Spike Encoding Comparison
# ──────────────────────────────────────────────────────────────────────────────
def plot_encoding_comparison(iq_signal: np.ndarray, seed: int = None):
    """
    Shows the raw IQ signal alongside the three spike encodings so the
    reader can see the difference visually.
    """
    from src.encoding import rate_encode, ttfs_encode, hybrid_encode

    if seed is None:
        seed = CONFIG["seed"]
    rng = np.random.default_rng(seed)

    T = min(128, iq_signal.shape[0])
    signal_slice = iq_signal[:T]

    spikes_rate   = rate_encode(signal_slice, rng=rng)
    spikes_ttfs   = ttfs_encode(signal_slice)
    spikes_hybrid = hybrid_encode(signal_slice, rng=rng)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    t = np.arange(T)

    # Raw I channel
    axes[0].plot(t, signal_slice[:, 0], color="#444", linewidth=1.2)
    axes[0].set_ylabel("Amplitude (I)")
    axes[0].set_title("Raw IQ signal (I channel)")

    # Spike rasters
    for ax, spk, method, color in zip(
        axes[1:],
        [spikes_rate, spikes_ttfs, spikes_hybrid],
        ["Rate coding", "TTFS coding", "Hybrid coding"],
        [COLORS["rate"], COLORS["ttfs"], COLORS["hybrid"]],
    ):
        spike_times = np.where(spk[:, 0] == 1)[0]
        ax.vlines(spike_times, 0, 1, color=color, linewidth=0.9, alpha=0.8)
        sparsity = spk.mean()
        ax.set_ylabel("Spike")
        ax.set_title(f"{method}  (sparsity = {sparsity:.3f})")
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 1])

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Spike encoding comparison (I channel, first 128 timesteps)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "fig01_spike_encoding_comparison.png")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 02 – Reservoir Activity Heatmap
# ──────────────────────────────────────────────────────────────────────────────
def plot_reservoir_activity(reservoir, iq_signal: np.ndarray, seed: int = None):
    """
    Runs one signal through the reservoir and visualises which neurons
    fired at which timesteps as a 2-D heatmap (neurons × timesteps).
    """
    from src.encoding import hybrid_encode

    if seed is None:
        seed = CONFIG["seed"]
    rng = np.random.default_rng(seed)

    T = min(128, iq_signal.shape[0])
    signal_slice = iq_signal[:T]
    spikes = hybrid_encode(signal_slice, rng=rng)

    # Collect per-timestep per-neuron firing during simulation
    N = reservoir.N
    activity = np.zeros((N, T))
    V = np.zeros(N)
    ref = np.zeros(N)
    fired_prev = np.zeros(N)
    alpha = reservoir.alpha

    if reservoir.W_in is None:
        reservoir.W_in = reservoir._build_input_weights(spikes.shape[1])
        reservoir._n_input_channels_cached = spikes.shape[1]

    for t in range(T):
        I_in  = reservoir.W_in @ spikes[t].astype(float)
        I_rec = reservoir.W @ fired_prev
        V = alpha * V + (1 - alpha) * (I_in + I_rec)
        V[ref > 0] = reservoir.v_reset
        ref = np.maximum(ref - reservoir.dt_ms, 0)
        fired = V >= reservoir.v_th
        activity[:, t] = fired.astype(float)
        V[fired] = reservoir.v_reset
        ref[fired] = reservoir.tau_ref_ms
        fired_prev = fired.astype(float)

    # Only show first 200 neurons for readability
    n_show = min(200, N)
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(activity[:n_show, :], aspect="auto", cmap="Blues",
                   interpolation="nearest", origin="upper")
    ax.set_xlabel("Timestep (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(f"Reservoir activity (first {n_show} of {N} neurons, one input signal)")
    fig.colorbar(im, ax=ax, label="Spike (0/1)")
    fig.tight_layout()
    return _savefig(fig, "fig02_reservoir_activity.png")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 03 – Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    confusion_matrix_array: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    filename: str = "fig03_confusion_matrix.png",
):
    n = len(class_names)
    fig_size = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Normalise rows to percentages so the color scale is meaningful even
    # when class sizes differ slightly.
    row_sums = confusion_matrix_array.sum(axis=1, keepdims=True)
    cm_norm = np.divide(confusion_matrix_array.astype(float), row_sums,
                        where=row_sums != 0) * 100

    sns.heatmap(
        cm_norm, annot=(n <= 12),   # skip per-cell numbers when too many classes
        fmt=".0f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.3, linecolor="#ddd",
        cbar_kws={"label": "% of true class"},
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return _savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 04 – Accuracy vs SNR
# ──────────────────────────────────────────────────────────────────────────────
def plot_accuracy_vs_snr(
    snr_accuracy_dict: dict,
    title: str = "Classification Accuracy vs. SNR",
    filename: str = "fig04_accuracy_vs_snr.png",
):
    """
    snr_accuracy_dict: {model_name: {snr_value: accuracy, ...}, ...}
    e.g. {"LSM": {-10: 0.25, 0: 0.72, 10: 0.91}, "CNN": {...}}
    """
    model_colors = {
        "LSM": COLORS["lsm"], "CNN": COLORS["cnn"],
        "LSTM": COLORS["lstm"], "ESN": COLORS["esn"], "SVM": COLORS["svm"],
    }
    model_markers = {"LSM": "o", "CNN": "s", "LSTM": "^", "ESN": "D", "SVM": "v"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, snr_acc in snr_accuracy_dict.items():
        snrs = sorted(snr_acc.keys())
        accs = [snr_acc[s] * 100 for s in snrs]
        color   = model_colors.get(model_name, "#888")
        marker  = model_markers.get(model_name, "o")
        ax.plot(snrs, accs, marker=marker, label=model_name,
                color=color, markersize=7, linewidth=2)

    ax.axvline(x=0, color="#999", linestyle="--", linewidth=1, label="SNR = 0 dB")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return _savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 05 – ROC Curves
# ──────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list,
    title: str = "ROC Curves (one-vs-rest)",
    filename: str = "fig05_roc_curves.png",
    max_classes_shown: int = 8,
):
    """
    Plots one ROC curve per class (one-vs-rest, OVR) using pre-computed
    probability/score arrays.
    y_score: shape (n_samples, n_classes) — probability or decision scores.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    classes = sorted(np.unique(y_true))
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.colormaps["tab10"]

    n_show = min(max_classes_shown, len(classes))
    for i, cls_idx in enumerate(classes[:n_show]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, cls_idx])
        roc_auc = auc(fpr, tpr)
        label = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        ax.plot(fpr, tpr, color=cmap(i / n_show),
                label=f"{label} (AUC={roc_auc:.2f})", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 06 – Precision-Recall Curves
# ──────────────────────────────────────────────────────────────────────────────
def plot_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list,
    title: str = "Precision-Recall Curves",
    filename: str = "fig06_pr_curves.png",
    max_classes_shown: int = 8,
):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    classes = sorted(np.unique(y_true))
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.colormaps["tab10"]

    n_show = min(max_classes_shown, len(classes))
    for i, cls_idx in enumerate(classes[:n_show]):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, cls_idx])
        ap = average_precision_score(y_bin[:, i], y_score[:, cls_idx])
        label = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        ax.plot(recall, precision, color=cmap(i / n_show),
                label=f"{label} (AP={ap:.2f})", linewidth=1.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 07 – Hyperparameter Sensitivity
# ──────────────────────────────────────────────────────────────────────────────
def plot_hyperparameter_sensitivity(
    df_results,
    filename: str = "fig07_hyperparameter_sensitivity.png",
):
    """
    df_results: pandas DataFrame with columns [parameter, value, accuracy].
    Produces a 1×5 panel, one subplot per swept parameter.
    """
    params = [
        ("N",               "Reservoir size $N$",          "N neurons"),
        ("spectral_radius",  "Spectral radius $\\rho$",     ""),
        ("p_conn",           "Connection prob. $p$",        ""),
        ("input_scale",      "Input scale $\\alpha$",       ""),
        ("tau_m",            "Membrane time const. $\\tau_m$ (ms)", ""),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for ax, (param_key, xlabel, _) in zip(axes, params):
        sub = df_results[df_results["parameter"] == param_key].sort_values("value")
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.plot(sub["value"], sub["accuracy"] * 100,
                "o-", color=COLORS["lsm"], markersize=7, linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.set_title(f"Sensitivity: {xlabel}")

    fig.suptitle("LSM Hyperparameter Sensitivity Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 08 – Energy Comparison
# ──────────────────────────────────────────────────────────────────────────────
def plot_energy_comparison(
    energy_df,
    filename: str = "fig08_energy_comparison.png",
):
    """energy_df: DataFrame with columns [model, energy_per_inference_microjoules]."""
    models = energy_df["model"].tolist()
    energies = energy_df["energy_per_inference_microjoules"].tolist()

    short_names = [m.split("(")[0].strip() for m in models]
    bar_colors = [COLORS["lsm"], COLORS["cnn"], COLORS["lstm"],
                  COLORS["svm"], COLORS["esn"]][:len(models)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(models)), energies, color=bar_colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_ylabel("Energy per inference (μJ)")
    ax.set_title("Estimated Energy per Inference Comparison", fontsize=13, fontweight="bold")
    ax.set_yscale("log")

    for bar, val in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{val:.2g} μJ", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return _savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 09 – Latency Comparison
# ──────────────────────────────────────────────────────────────────────────────
def plot_latency_comparison(
    energy_df,
    filename: str = "fig09_latency_comparison.png",
):
    """energy_df: same DataFrame used for energy; also has inference_latency_ms."""
    models    = energy_df["model"].tolist()
    latencies = energy_df["inference_latency_ms"].tolist()
    short_names = [m.split("(")[0].strip() for m in models]
    bar_colors  = [COLORS["lsm"], COLORS["cnn"], COLORS["lstm"],
                   COLORS["svm"], COLORS["esn"]][:len(models)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(models)), latencies, color=bar_colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_ylabel("Inference latency (ms)")
    ax.set_title("Inference Latency per Sample", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.2f} ms", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return _savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 10 – Spike Sparsity
# ──────────────────────────────────────────────────────────────────────────────
def plot_spike_sparsity(
    sparsity_dict: dict,
    filename: str = "fig10_spike_sparsity.png",
):
    """
    sparsity_dict: {method_name: sparsity_value}
    e.g. {"rate": 0.48, "ttfs": 0.12, "hybrid": 0.22}
    """
    methods    = list(sparsity_dict.keys())
    sparsities = [sparsity_dict[m] * 100 for m in methods]
    colors     = [COLORS.get(m, "#888") for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, sparsities, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, sparsities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Spike sparsity (% of positions firing)")
    ax.set_title("Spike Sparsity by Encoding Method\n(lower = more energy-efficient)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(sparsities) * 1.25)
    fig.tight_layout()
    return _savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 11 – Adversarial Robustness
# ──────────────────────────────────────────────────────────────────────────────
def plot_adversarial_robustness(
    robustness_results: dict,
    clean_accuracy: float,
    filename: str = "fig11_adversarial_robustness.png",
):
    """
    robustness_results: output dict from src/adversarial.evaluate_robustness()
    i.e. keys "awgn", "frequency", "distortion", each a list of
    {severity, accuracy} dicts.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    attack_configs = [
        ("awgn",       "AWGN Attack",              "Additional noise SNR (dB)"),
        ("frequency",  "Frequency Perturbation",    "Max freq offset (normalised)"),
        ("distortion", "Signal Distortion",         "Distortion level"),
    ]

    for ax, (attack_key, title, xlabel) in zip(axes, attack_configs):
        entries = robustness_results.get(attack_key, [])
        if not entries:
            ax.set_visible(False)
            continue
        severities = [e["severity"] for e in entries]
        accuracies  = [e["accuracy"] * 100 for e in entries]

        ax.axhline(clean_accuracy * 100, color="#888", linestyle="--",
                   linewidth=1.5, label=f"Clean ({clean_accuracy*100:.1f}%)")
        ax.plot(severities, accuracies, "o-", color=COLORS["lsm"],
                markersize=8, linewidth=2, label="Under attack")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)

    fig.suptitle("LSM Adversarial Robustness", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Script entry point — quick smoke test that generates stub figures
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    from src.reservoir import LSMReservoir
    from src.synthetic_ew_dataset import load_synthetic_dataset

    print("Generating sample visualizations with dummy data...\n")
    ensure_directories_exist()

    rng = np.random.default_rng(CONFIG["seed"])
    X, Y, Z, jam_names = load_synthetic_dataset()
    sample_signal = X[0]

    # Fig 01 – encoding comparison
    print("[1] Spike encoding comparison...")
    plot_encoding_comparison(sample_signal)

    # Fig 02 – reservoir activity
    print("[2] Reservoir activity heatmap...")
    res = LSMReservoir(N=100, seed=CONFIG["seed"])
    plot_reservoir_activity(res, sample_signal)

    # Fig 03 – confusion matrix (dummy 6×6)
    print("[3] Confusion matrix...")
    dummy_cm = rng.integers(0, 50, (6, 6))
    np.fill_diagonal(dummy_cm, rng.integers(80, 200, 6))
    plot_confusion_matrix(dummy_cm, jam_names, title="Jamming Classification (demo)")

    # Fig 04 – accuracy vs SNR (dummy data)
    print("[4] Accuracy vs SNR...")
    snr_acc = {}
    for model_name in ["LSM", "CNN", "LSTM", "ESN", "SVM"]:
        base = rng.uniform(0.5, 0.95)
        snr_acc[model_name] = {snr: max(0, min(1, base - 0.05 * abs(snr) / 10 + rng.normal(0, 0.02)))
                                for snr in range(-20, 32, 2)}
    plot_accuracy_vs_snr(snr_acc)

    # Fig 07 – hyperparameter sensitivity
    print("[7] Hyperparameter sensitivity...")
    rows = []
    for p, vals in [("N", [100,200,500,1000,2000]),
                    ("spectral_radius", [0.3,0.5,0.7,0.9,1.1,1.3]),
                    ("p_conn", [0.02,0.05,0.1,0.2,0.4]),
                    ("input_scale", [1,3,6,10,15]),
                    ("tau_m", [5,10,20,40,80])]:
        for v in vals:
            rows.append({"parameter": p, "value": v, "accuracy": rng.uniform(0.4, 0.9)})
    plot_hyperparameter_sensitivity(pd.DataFrame(rows))

    # Fig 10 – spike sparsity
    print("[10] Spike sparsity...")
    plot_spike_sparsity({"rate": 0.484, "ttfs": 0.125, "hybrid": 0.215})

    # Fig 11 – adversarial robustness (dummy)
    print("[11] Adversarial robustness...")
    dummy_robustness = {
        "awgn": [{"severity": s, "accuracy": max(0.1, 0.82 - 0.06 * abs(s) / 5)}
                  for s in [10, 5, 0, -5, -10]],
        "frequency": [{"severity": v, "accuracy": max(0.1, 0.82 - v * 8)}
                       for v in [0, 0.005, 0.01, 0.015, 0.02]],
        "distortion": [{"severity": v, "accuracy": max(0.1, 0.82 - v * 1.2)}
                        for v in [0.0, 0.1, 0.2, 0.3, 0.5]],
    }
    plot_adversarial_robustness(dummy_robustness, clean_accuracy=0.82)

    print(f"\nAll demo figures saved to: {FIG_DIR}")
