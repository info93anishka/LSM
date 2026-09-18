# Liquid State Machines for Real-Time Classification of Radar Emissions, Jamming Signals, and Threat Emitters

A complete, production-ready research pipeline implementing a **Liquid State Machine (LSM)** based neuromorphic classifier for Electronic Warfare (EW) signal analysis, built for a BTech research project.

This project implements:
- **Multi-task classification** from a single fixed spiking reservoir: modulation type (24 classes, RadioML 2018.01A), jamming type (6 classes, synthetic), and emitter identity (10 classes, synthetic RF fingerprinting)
- **Three spike encoding schemes**: Rate coding, Time-To-First-Spike (TTFS), and a novel Hybrid scheme
- **Four baseline models** for fair comparison: Expert Features + RBF-SVM, 1D CNN, Bidirectional LSTM, Echo State Network (ESN)
- **Hyperparameter sensitivity analysis** across 5 reservoir parameters
- **Adversarial robustness testing**: AWGN, frequency perturbation, and signal distortion attacks
- **Energy and latency analysis** estimating neuromorphic (Loihi 2) vs. GPU/CPU energy consumption
- **Full visualization suite**: confusion matrices, ROC/PR curves, accuracy-vs-SNR, sensitivity plots, energy comparisons
- **Automated reporting**: CSVs and a final research report, generated automatically

---

## 1. Folder Structure

```
lsm_ew_project/
│
├── config.py                      # Central configuration (all hyperparameters)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/
│   ├── radioml/                   # Place GOLD_XYZ_OSC.0001_1024.hdf5 here
│   └── synthetic_ew/               # Auto-generated jamming + emitter datasets
│
├── src/
│   ├── __init__.py
│   ├── synthetic_ew_dataset.py    # Generates 6 jamming types + 10 emitter profiles
│   ├── radioml_loader.py          # Loads/caches RadioML 2018.01A subsets
│   ├── encoding.py                # Rate / TTFS / Hybrid spike encoders
│   ├── reservoir.py               # LSM reservoir (LIF neurons, Dale's principle)
│   ├── readouts.py                # 3 readout heads (LogReg / LinearSVM / Ridge)
│   ├── baselines.py                # Expert+SVM, 1D CNN, BiLSTM, ESN
│   ├── hyperparameter_search.py   # 5-parameter sensitivity sweep
│   ├── adversarial.py              # AWGN / frequency / distortion attacks
│   ├── energy_analysis.py          # Energy & latency estimation
│   ├── visualization.py            # All figure-generation functions
│   └── reporting.py                # CSV and text report generation
│
├── experiments/
│   ├── __init__.py
│   └── run_pipeline.py            # MAIN SCRIPT — runs everything end-to-end
│
├── tests/
│   └── test_all.py                # 42 pytest unit tests covering every module
│
├── models/                         # Saved reservoir weights, readouts, baselines
├── configs/                         # (reserved for saved experiment configs)
└── results/
    ├── figures/                    # All generated PNG plots
    ├── reports/                    # classification_report.txt, final_research_report.txt
    ├── logs/
    ├── metrics.csv
    ├── model_comparison.csv
    ├── hyperparameter_results.csv
    ├── energy_results.csv
    └── latency_results.csv
```

---

## 2. Setup Instructions

### Step 1 — Install Python 3.12

Download from [python.org](https://www.python.org/downloads/). Confirm with:
```bash
python --version
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv lsm_env

# Windows:
lsm_env\Scripts\activate

# Mac/Linux:
source lsm_env/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

If you are on Linux and see "externally-managed-environment" errors:
```bash
pip install -r requirements.txt --break-system-packages
```

### Step 4 — Place the RadioML 2018.01A dataset (optional but recommended)

1. Visit **https://www.deepsig.ai/datasets** and download **RadioML 2018.01A** (~6 GB archive).
2. Extract it. You will find a file named exactly:
   ```
   GOLD_XYZ_OSC.0001_1024.hdf5
   ```
3. Place it at:
   ```
   lsm_ew_project/data/radioml/GOLD_XYZ_OSC.0001_1024.hdf5
   ```

> **If you skip this step**, the pipeline still runs completely — it will automatically skip the modulation-classification task (Head 1) and print a clear warning, while the jamming and emitter tasks (which use auto-generated synthetic data) run normally.

The synthetic EW jamming dataset and emitter dataset require **no download** — they are generated automatically the first time any script needs them.

---

## 3. Example Terminal Commands

### Run individual modules (each is self-testing)

```bash
python config.py                          # Verify configuration + create folders
python src/synthetic_ew_dataset.py         # Generate synthetic EW datasets
python src/encoding.py                     # Test spike encoding, print sparsity
python src/reservoir.py                    # Build reservoir, test firing behavior
python src/readouts.py                     # Test all 3 readout heads
python src/radioml_loader.py               # Check/load RadioML dataset
python src/baselines.py                    # Smoke-test all 4 baseline models
python src/hyperparameter_search.py        # Run full hyperparameter sweep
python src/adversarial.py                  # Test all 3 adversarial attacks
python src/energy_analysis.py              # Demo energy/latency estimation
python src/visualization.py                # Generate sample figures
python src/reporting.py                    # Generate sample CSV/text reports
```

### Run automated tests

```bash
pytest tests/test_all.py -v
```

### Run the FULL pipeline

```bash
# Quick demo (~10-15 minutes, reduced dataset/reservoir sizes):
python experiments/run_pipeline.py --quick

# Full research run (1-4 hours depending on hardware):
python experiments/run_pipeline.py

# Skip expensive stages if re-running:
python experiments/run_pipeline.py --skip-sweep --skip-adversarial --skip-baselines
```

---

## 4. Expected Outputs

After running `experiments/run_pipeline.py`, you should see:

**In `results/figures/`** (11+ PNG files):
- `fig01_spike_encoding_comparison.png`
- `fig02_reservoir_activity.png`
- `fig03a/b_confusion_matrix_*.png`
- `fig04_accuracy_vs_snr_jamming.png`
- `fig05/06_roc_pr_curves_jamming.png`
- `fig07_hyperparameter_sensitivity.png`
- `fig08_energy_comparison.png`
- `fig09_latency_comparison.png`
- `fig10_spike_sparsity.png`
- `fig11_adversarial_robustness.png`

**In `results/`** (CSV files):
- `metrics.csv` — per-model/task accuracy
- `model_comparison.csv` — side-by-side comparison
- `hyperparameter_results.csv` — full sweep results
- `energy_results.csv`, `latency_results.csv`
- `adversarial_robustness.csv`

**In `results/reports/`**:
- `classification_report.txt` — sklearn-style precision/recall/F1 per class
- `final_research_report.txt` — human-readable summary, ready for a supervisor

**In `models/`**:
- `reservoir_W.npy` — the fixed reservoir weight matrix (for reproducibility)
- `*_readout.joblib` — trained readout classifiers
- `*_baseline.pt` / `.joblib` — trained baseline models
- `states/` — cached reservoir state vectors (avoids re-running expensive simulation)

**Typical console output during a run:**
```
======================================================================
  STAGE 5 — Train Multi-Task Readout Heads
======================================================================
Training Head 2 (Jamming classification, Linear SVM)...
  Jamming accuracy (test): 91.06%
Training Head 3 (Emitter identification, Ridge Classifier)...
  Emitter accuracy (test): 84.50%
Training Head 1 (Modulation classification, Logistic Regression)...
  Modulation accuracy (test): 78.03%
```

---

## 5. Expected Figure Descriptions

| Figure | What it shows |
|---|---|
| `fig01` | Raw IQ signal alongside its Rate/TTFS/Hybrid spike-train encodings (vertical lines = spikes) |
| `fig02` | Heatmap of which reservoir neurons fired at which timestep for one input signal |
| `fig03a/b` | Confusion matrix heatmaps (blue intensity = % of true class predicted as each label) |
| `fig04` | Line plot: classification accuracy (%) vs. SNR (dB), one line per model |
| `fig05/06` | ROC and Precision-Recall curves, one line per class |
| `fig07` | Five side-by-side line plots showing accuracy sensitivity to N, ρ, p, α, τ_m |
| `fig08/09` | Bar charts comparing energy (μJ, log scale) and latency (ms) across all 5 models |
| `fig10` | Bar chart comparing spike sparsity % across Rate/TTFS/Hybrid encoding |
| `fig11` | Three line plots showing accuracy degradation under increasing attack severity |

---

## 6. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run scripts from the **project root**, not from inside `src/` |
| `pip install` fails with "externally-managed-environment" | Add `--break-system-packages` to the pip command |
| RadioML file not found | Check the exact path: `data/radioml/GOLD_XYZ_OSC.0001_1024.hdf5` (case-sensitive) |
| Reservoir produces zero firing ("dead" network) | Check `config.py`'s `input_scale` — it must be large enough (default 6.0) relative to `v_th` (default 1.0) given sparse binary spike inputs |
| Pipeline very slow on full dataset | Use `--quick` flag, or reduce `n_samples_subset` / `N` (reservoir size) in `config.py` |
| `CUDA out of memory` during CNN/LSTM training | Reduce `batch_size` in `config.py`'s `BASELINE_CONFIG`, or force CPU by setting `CUDA_VISIBLE_DEVICES=""` |
| Plots not displaying (only saved as files) | This is intentional — `visualization.py` uses the non-interactive `Agg` backend for server compatibility. Open the PNG files in `results/figures/` directly |
| `pytest` not found | `pip install pytest --break-system-packages` |
| Want to re-run only one stage | Use the cached files in `models/states/*.npy` and `data/synthetic_ew/*.npy` to skip regenerating data/states |

---

## 7. Reproducibility Notes

- **Global random seed**: set in `config.py` (`GLOBAL_SEED = 42`), applied via `set_global_seed()` to Python's `random`, NumPy, and PyTorch.
- **Reservoir weights**: saved to `models/reservoir_W.npy` after every full pipeline run.
- **All trained models**: saved to `models/` (readout heads as `.joblib`, CNN/LSTM as PyTorch `.pt` state dicts).
- **All configuration**: centralized in `config.py` — a single source of truth for every hyperparameter used anywhere in the project.

---

## 8. Citation / Acknowledgement Notes for Your Report

- RadioML 2018.01A: O'Shea, T. J., Roy, T., & Clancy, T. C. (2018). *Over-the-air deep learning based radio signal classification.* IEEE Journal of Selected Topics in Signal Processing.
- Loihi 2 energy figures: Intel Labs published characterization data (approximate, used here as an *estimate*, not a hardware measurement).
- Liquid State Machine concept: Maass, W., Natschläger, T., & Markram, H. (2002). *Real-time computing without stable states.* Neural Computation.
