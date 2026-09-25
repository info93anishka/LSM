# ⚡ Neuromorphic Electronic Warfare Signal Intelligence

## Liquid State Machines for Real-Time Classification of Radar Emissions, Jamming Signals, and Threat Emitters

<p align="center">
  <strong>A neuromorphic Electronic Warfare (EW) signal-intelligence pipeline built around Liquid State Machines (LSMs)</strong><br>
  <sub>Spike-based RF processing • Multi-task classification • Energy-aware inference • Adversarial evaluation</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Electronic%20Warfare-1f2937?style=for-the-badge" alt="Electronic Warfare">
  <img src="https://img.shields.io/badge/Architecture-Liquid%20State%20Machine-0f766e?style=for-the-badge" alt="Liquid State Machine">
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/Year-2026-7c3aed?style=for-the-badge" alt="2026">
</p>

---

## 01 — Project Overview

Electronic Warfare systems must identify and interpret radio-frequency activity under strict constraints on **latency, energy consumption, computational resources, and robustness**.

This project investigates whether a **Liquid State Machine (LSM)** can provide a practical neuromorphic alternative for RF signal intelligence. Instead of processing continuously valued signals through a conventional deep neural network, the pipeline converts RF/IQ signals into discrete spike trains, propagates those spikes through a fixed recurrent reservoir of spiking neurons, and performs classification using lightweight readout heads.

The architecture is designed around a central idea:

> **Compute rich temporal representations in a fixed spiking reservoir, then train only lightweight readout layers.**

The project evaluates this approach across modulation recognition, jamming detection, and RF emitter identification, while also examining energy, latency, hyperparameter sensitivity, spike sparsity, and adversarial robustness.

The underlying internship work was carried out in the context of **Neuromorphic Electronic Signal Intelligence** at the **Solid State Physics Laboratory (SSPL), Defence Research and Development Organisation (DRDO)**. The project report describes the eventual target of mapping the architecture to **Intel Loihi 2** neuromorphic hardware. Physical Loihi 2 execution was not performed in the present study; energy figures are proxy estimates based on published operation-level data. 

---

## 02 — What This Project Implements

| Component | Implementation |
|---|---|
| **Core architecture** | Liquid State Machine with spiking LIF neurons |
| **Reservoir** | 1,000-neuron core configuration |
| **Neuron model** | Leaky Integrate-and-Fire (LIF) |
| **Population structure** | 800 excitatory + 200 inhibitory neurons |
| **Connectivity** | Sparse recurrent connectivity |
| **Spectral radius study** | ρ = 0.5, 0.9, 1.1 |
| **Temporal processing** | Parallel fast/slow reservoir dynamics |
| **Encoding** | Rate, TTFS, Hybrid Rate/TTFS |
| **Modulation classes** | 24 RadioML classes |
| **Jamming classes** | 6 jamming types + non-jamming class |
| **Emitter classes** | 10 synthetic RF emitter identities |
| **Readout heads** | Multinomial Logistic Regression, Linear SVM, Ridge Classifier |
| **Baselines** | Expert Features + RBF-SVM, 1D CNN, Bidirectional LSTM |
| **Robustness tests** | AWGN, frequency offset, signal distortion |
| **Efficiency analysis** | Energy and CPU latency |
| **Analysis** | Hyperparameter sensitivity + spike sparsity |
| **Reporting** | CSV metrics, figures, classification reports |

The three classification tasks share the same reservoir representation, so the computationally expensive reservoir simulation is performed once per signal before the independent readout heads operate on the resulting feature vector.

---

## 03 — System Architecture

```text
                    ┌───────────────────────────────┐
                    │        RF / IQ Signal         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Spike Encoding          │
                    │                               │
                    │  • Rate Coding                │
                    │  • TTFS Coding                │
                    │  • Hybrid Rate / TTFS         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
              ┌────────────────────────────────────────────┐
              │            Liquid State Machine             │
              │                                             │
              │  ┌────────────────┐  ┌──────────────────┐  │
              │  │ Fast Reservoir │  │ Slow Reservoir   │  │
              │  │ Carrier detail │  │ Envelope detail  │  │
              │  └───────┬────────┘  └────────┬─────────┘  │
              │          └──────────┬─────────┘            │
              │                     ▼                      │
              │          Spiking Reservoir States          │
              └─────────────────────┬──────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Fixed-Length Feature Vector  │
                    │   Reservoir firing rates     │
                    └───────────────┬───────────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │ Head 1          │ │ Head 2          │ │ Head 3          │
        │ Modulation      │ │ Jamming         │ │ Emitter ID      │
        │ 24 classes      │ │ 7 classes       │ │ 10 classes      │
        │ Logistic Reg.   │ │ Linear SVM       │ │ Ridge           │
        └─────────────────┘ └─────────────────┘ └─────────────────┘
```

The reservoir operates on temporal spike dynamics rather than directly learning a conventional dense representation. A fixed simulation window of **128 ms** is used to calculate the average firing activity of each neuron, producing the feature vector passed to the readout stage.

---

## 04 — Why Liquid State Machines?

A conventional neural network jointly learns the feature representation and the final classifier. In contrast, an LSM separates these two functions.

The **reservoir** is a recurrent spiking network that dynamically processes incoming spikes. Its activity depends not only on the current input but also on previous inputs, allowing it to capture temporal dependencies and provide short-term memory.

The **readout layer** receives the reservoir state and performs the final classification. In this project, the reservoir itself does not require backpropagation-based training; only lightweight readout classifiers are trained.

This makes the architecture attractive for studying the combination of:

- temporal signal representation,
- sparse event-driven computation,
- low training cost,
- hardware-oriented neuromorphic processing,
- and energy-efficient inference.

---

## 05 — Research Motivation

The project addresses a specific EW deployment problem: signal intelligence systems may need to operate on platforms such as UAVs or remote sensors where **power and computational resources are constrained**.

Three conventional approaches were considered:

### Hand-Engineered Features + SVM

Uses amplitude, phase, cyclostationary and related signal features followed by a conventional classifier.

**Strength:** Fast, inexpensive, interpretable.  
**Limitation:** Requires manual feature engineering and can degrade under difficult signal conditions.

### Deep Learning

Includes CNNs and recurrent models such as LSTMs applied directly to RF/IQ signals or derived representations.

**Strength:** Strong representation-learning capability.  
**Limitation:** Higher computational and energy requirements.

### Echo State / Reservoir Computing

Uses a fixed recurrent reservoir but continuous-valued activations.

**Strength:** Low training cost.  
**Limitation:** Does not exploit discrete spike sparsity in the same way as a spiking reservoir.

The LSM approach investigated here combines the **fixed-reservoir principle of reservoir computing** with **spike-based neuromorphic computation**.

---

## 06 — Research Questions

The experimental pipeline was designed around several questions:

1. Can spike encoding preserve enough RF information for useful classification?
2. How do Rate, TTFS, and Hybrid encoding affect spike sparsity and classification?
3. Can a shared reservoir support multiple EW classification tasks?
4. How sensitive is classification to reservoir size, connectivity, spectral radius, input scaling, and membrane time constant?
5. Does spike-based processing show useful behaviour under signal perturbations?
6. How does estimated neuromorphic inference energy compare with conventional baselines?
7. Does software simulation provide a latency advantage, or is the expected advantage primarily hardware-dependent?

---

## 07 — Datasets

### RadioML 2018.01A

The project uses **RadioML 2018.01A** for modulation recognition.

- Approximately **2.5 million labelled examples**
- **24 modulation classes**
- SNR range approximately **−20 dB to +30 dB**
- IQ signal representation
- Used as the public benchmark for modulation classification

The required file is:

```text
GOLD_XYZ_OSC.0001_1024.hdf5
```

Expected location:

```text
data/radioml/GOLD_XYZ_OSC.0001_1024.hdf5
```

### Synthetic EW Dataset

Because a suitable public dataset covering the project's target jamming scenarios was not available, a synthetic EW dataset was generated.

The project report defines six jamming types:

```text
1. Tone jamming
2. Sweep jamming
3. Noise jamming
4. Repeat / deception jamming
5. Pulse jamming
6. Barrage jamming
```

The classification head additionally includes a **non-jamming class**, producing **7 jamming-detection classes**.

Synthetic signals incorporate channel effects such as:

- additive white Gaussian noise,
- fading,
- oscillator phase noise,
- variable signal-to-noise conditions.

The project also generates synthetic emitter profiles for RF fingerprinting and emitter identification.

---

## 08 — Spike Encoding

Three encoding strategies are implemented.

### Rate Coding

Signal amplitude controls firing probability or firing frequency.

**Advantages**
- Simple
- Relatively robust to noise
- Represents amplitude redundantly

**Trade-off**
- Requires more spikes and longer observation windows for fine-grained amplitude representation.

### Time-To-First-Spike (TTFS)

The timing of a neuron's first spike represents the input amplitude.

```text
Higher amplitude  → earlier spike
Lower amplitude   → later spike
```

**Advantages**
- Very low spike count
- Efficient temporal representation

**Trade-off**
- More sensitive to timing errors and noise.

### Hybrid Rate / TTFS

The project introduces a hybrid strategy motivated by the temporal sparsity of RF signals.

High-amplitude, easily identifiable signal regions use **TTFS**, while lower-amplitude or ambiguous regions use **Rate coding**.

This attempts to combine:

```text
TTFS → temporal efficiency
Rate → robustness / redundancy
```

Reported spike sparsity in the internship report:

| Encoding | Spike Sparsity |
|---|---:|
| Rate | 19% |
| TTFS | 9% |
| Hybrid | 17% |

---

## 09 — Liquid State Machine Architecture

### Reservoir Size

The core configuration uses:

```text
N = 1,000 neurons
```

The study examined reservoir sizes ranging from approximately **200 to 5,000 neurons**.

### Excitatory / Inhibitory Structure

The 1,000-neuron reservoir is divided as:

```text
800 Excitatory neurons
200 Inhibitory neurons
```

This gives an **80:20 excitatory/inhibitory balance**.

### Sparse Connectivity

The reservoir is not fully connected. Connectivity is intentionally sparse to reduce computational cost while retaining recurrent temporal dynamics.

### Spectral Radius

The study evaluates:

```text
ρ = 0.5
ρ = 0.9
ρ = 1.1
```

The spectral radius influences the persistence and stability of reservoir dynamics.

An important experimental observation was that classification accuracy changed only modestly across the tested spectral-radius values, suggesting that encoding and readout design may be more important accuracy bottlenecks for the tested configuration.

---

## 10 — Two-Timescale Reservoir Extension

RF signals contain information at multiple temporal scales.

```text
Fast dynamics
    ↓
Carrier-level information

Slow dynamics
    ↓
Signal-envelope information
```

To preserve both, the project uses a two-reservoir extension:

```text
                Input Spikes
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Fast Reservoir         Slow Reservoir
   short τm               long τm
          │                     │
          └──────────┬──────────┘
                     ▼
             Concatenated State
                     │
                     ▼
               Readout Heads
```

The fast reservoir is tuned toward carrier-level structure, while the slower reservoir captures envelope-level temporal structure.

---

## 11 — Multi-Task Readout System

A single reservoir representation feeds three independent classification heads.

| Head | Task | Classes | Classifier |
|---|---|---:|---|
| **Head 1** | Modulation recognition | 24 | Multinomial Logistic Regression |
| **Head 2** | Jamming detection | 7 | Linear SVM |
| **Head 3** | Emitter identification | 10 | Ridge Classifier |

The reservoir simulation is performed once per signal. The resulting feature vector is then reused by all three classifiers.

This design directly investigates whether **shared neuromorphic representations can support multiple related EW intelligence tasks without maintaining three separate reservoirs**.

---

## 12 — Baseline Models

To provide a meaningful comparison, the project evaluates the LSM against conventional approaches.

| Baseline | Description |
|---|---|
| **Expert Features + RBF-SVM** | Hand-engineered amplitude, phase and cyclostationary features |
| **1D CNN** | Five-layer convolutional network operating on raw IQ waveform |
| **Bidirectional LSTM** | Two-layer recurrent architecture for sequential RF data |
| **ESN** | Echo State Network / continuous reservoir baseline |

The baseline comparison is intended to separate the contribution of **spiking computation** from the broader concept of **reservoir computing**.

---

## 13 — Evaluation Framework

The evaluation covers six major dimensions.

### Classification

- Overall accuracy
- Per-class accuracy
- Accuracy versus SNR
- Confusion matrices
- ROC curves
- Precision-Recall curves

### Energy

Energy is estimated from synaptic operation counts using a published Loihi 2 operation-level energy figure.

> **Important:** The reported neuromorphic energy numbers are estimates/proxies, not measurements performed on physical Loihi 2 hardware.

### Latency

CPU wall-clock inference latency is measured for the software implementations.

### Spike Sparsity

Percentage of active neurons/spikes is measured to assess whether the implementation achieves the sparsity expected from event-driven computation.

### Hyperparameter Sensitivity

The study varies:

```text
N       → reservoir size
ρ       → spectral radius
Pconn   → connection sparsity
α       → input scaling
τm      → membrane time constant
```

### Adversarial Robustness

Three perturbation families are evaluated:

```text
AWGN
Frequency Offset
Signal Distortion
```

---

## 14 — Experimental Results

### Classification Accuracy

| Task | LSM | Expert + SVM | 1D CNN | LSTM |
|---|---:|---:|---:|---:|
| Modulation Recognition | **39.45%** | 51.93% | 62.85% | 49.07% |
| Jamming Detection | **41.06%** | 90.28% | 89.03% | 67.16% |
| Emitter Identification | **53.50%** | 67.14% | 92.54% | 61.00% |

The LSM did not match the classification accuracy of the conventional baselines in the evaluated configuration.

The strongest class-specific jamming result was **pulse jamming at 69%**, consistent with its pronounced temporal structure.

---

## 15 — Energy and Latency

| Metric | LSM | Expert + SVM | 1D CNN | LSTM |
|---|---:|---:|---:|---:|
| Estimated inference energy (µJ) | **3.30** | 30,881 | 36,458 | 233,850 |
| CPU latency (ms) | 17.28 | 2.06 | 2.43 | 15.59 |
| Spike sparsity | 24.9% | — | — | — |

The estimated LSM inference energy is approximately:

```text
9,400× lower than Expert + SVM
11,000× lower than 1D CNN
70,000× lower than LSTM
```

These energy ratios are based on the project's operation-count proxy and **should not be interpreted as physical Loihi 2 measurements**.

An important result is that the CPU implementation was not faster:

```text
LSM software latency = 17.28 ms
```

This is higher than the Expert+SVM and CNN implementations. The report attributes this to the overhead of explicitly simulating spiking dynamics on general-purpose CPU hardware. The expected hardware advantage therefore requires validation on dedicated neuromorphic hardware.

---

## 16 — Hyperparameter Sensitivity

The reported sensitivity experiments included:

| Parameter | Values Tested | Observed Accuracy |
|---|---|---|
| Reservoir size N | 700, 1000 | 0.36, 0.26 |
| Spectral radius ρ | 0.5, 0.9, 1.1 | 0.28, 0.26, 0.28 |
| Connectivity | 0.05, 0.10, 0.20 | 0.24, 0.26, 0.28 |
| Input scaling α | 3, 6, 10 | 0.22, 0.26, 0.22 |
| Membrane time τm | 10, 20, 40 ms | 0.28, 0.24, 0.24 |

Key observations:

- Spectral radius produced relatively small changes across the tested range.
- Increasing connectivity produced a modest improvement.
- Input scaling performed best around **α = 6** in the tested configuration.
- The **10 ms** membrane time constant slightly outperformed longer values.
- The limited number of tested configurations means these trends should not be treated as universal conclusions.

---

## 17 — Adversarial Robustness

Clean accuracy for the evaluated robustness configuration was:

```text
29.0%
```

### AWGN

| SNR / Attack Level | Accuracy |
|---:|---:|
| 10 dB | 23.5% |
| 5 dB | 19.0% |
| 0 dB | 11.5% |
| −5 dB | 18.5% |
| −10 dB | 17.5% |

### Frequency Offset

| Offset | Accuracy |
|---:|---:|
| 0 | 24.0% |
| 0.005 | 24.5% |
| 0.010 | 30.0% |
| 0.015 | 23.5% |
| 0.020 | 27.0% |

### Signal Distortion

| Distortion | Accuracy |
|---:|---:|
| 0 | 29.5% |
| 0.1 | 28.0% |
| 0.2 | 30.5% |
| 0.3 | 22.5% |
| 0.5 | 30.0% |

The robustness results are **mixed rather than conclusive**. Frequency-offset and distortion performance remained relatively close to the clean baseline, whereas AWGN caused substantial degradation. Given the modest clean accuracy and limited test configuration, further validation is required before claiming a structural robustness advantage.

---

## 18 — Repository Structure

```text
lsm_ew_project/
│
├── .gitignore
├── README.md
├── config.py
├── requirements.txt
│
├── configs/
│   └── .gitkeep
│
├── data/
│   ├── radioml/
│   │   └── GOLD_XYZ_OSC.0001_1024.hdf5   # not committed
│   └── synthetic_ew/                      # generated locally
│
├── src/
│   ├── __init__.py
│   ├── adversarial.py
│   ├── baselines.py
│   ├── encoding.py
│   ├── energy_analysis.py
│   ├── hyperparameter_search.py
│   ├── radioml_loader.py
│   ├── readouts.py
│   ├── reporting.py
│   ├── reservoir.py
│   ├── synthetic_ew_dataset.py
│   └── visualization.py
│
├── experiments/
│   ├── __init__.py
│   └── run_pipeline.py
│
├── models/                                 # generated locally
│   └── .gitkeep
│
├── results/                                # generated locally
│   ├── figures/
│   ├── reports/
│   ├── logs/
│   └── *.csv
│
└── tests/
    └── test_all.py
```

Large datasets, generated NumPy arrays, trained model files, cached states, and generated results are intentionally excluded through `.gitignore`.

---

## 19 — Installation

### Requirements

Recommended environment:

```text
Python 3.12
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a virtual environment:

```bash
python -m venv lsm_env
```

Windows:

```bash
lsm_env\Scripts\activate
```

Linux/macOS:

```bash
source lsm_env/bin/activate
```

---

## 20 — Dataset Setup

Download **RadioML 2018.01A** and place:

```text
GOLD_XYZ_OSC.0001_1024.hdf5
```

at:

```text
data/radioml/GOLD_XYZ_OSC.0001_1024.hdf5
```

The synthetic EW datasets are generated by the project and do not require an external download.

If RadioML is unavailable, the pipeline can still execute the synthetic jamming and emitter tasks while skipping the modulation-classification task.

---

## 21 — Running the Project

Run the configuration check:

```bash
python config.py
```

Generate synthetic EW data:

```bash
python src/synthetic_ew_dataset.py
```

Test the spike encoders:

```bash
python src/encoding.py
```

Test the reservoir:

```bash
python src/reservoir.py
```

Run the RadioML loader:

```bash
python src/radioml_loader.py
```

Run the baseline models:

```bash
python src/baselines.py
```

Run hyperparameter experiments:

```bash
python src/hyperparameter_search.py
```

Run adversarial evaluation:

```bash
python src/adversarial.py
```

Run energy analysis:

```bash
python src/energy_analysis.py
```

Generate visualisations:

```bash
python src/visualization.py
```

Generate reports:

```bash
python src/reporting.py
```

---

## 22 — Automated Pipeline

### Quick Run

For a reduced demonstration:

```bash
python experiments/run_pipeline.py --quick
```

### Full Run

For the complete experiment:

```bash
python experiments/run_pipeline.py
```

### Skip Expensive Stages

```bash
python experiments/run_pipeline.py \
    --skip-sweep \
    --skip-adversarial \
    --skip-baselines
```

---

## 23 — Testing

Run the complete test suite:

```bash
pytest tests/test_all.py -v
```

The project includes unit tests covering the major data, encoding, reservoir, classification, analysis, and reporting components.

---

## 24 — Output Artefacts

After a complete run, generated outputs include:

### Figures

```text
fig01_spike_encoding_comparison.png
fig02_reservoir_activity.png
fig03a_confusion_matrix_jamming.png
fig03b_confusion_matrix_emitter.png
fig04_accuracy_vs_snr_jamming.png
fig05_roc_curves_jamming.png
fig06_pr_curves_jamming.png
fig07_hyperparameter_sensitivity.png
fig08_energy_comparison.png
fig09_latency_comparison.png
fig10_spike_sparsity.png
fig11_adversarial_robustness.png
```

### Metrics

```text
metrics.csv
model_comparison.csv
hyperparameter_results.csv
energy_results.csv
latency_results.csv
adversarial_robustness.csv
```

### Reports

```text
classification_report.txt
final_research_report.txt
```

### Model Artefacts

Generated locally:

```text
reservoir_W.npy
*_readout.joblib
*_baseline.pt
*_baseline.joblib
states/*.npy
```

---

## 25 — Reproducibility

The experimental pipeline uses a global random seed:

```python
GLOBAL_SEED = 42
```

The configuration is centralized in:

```text
config.py
```

This provides a single location for the major experimental hyperparameters.

Reservoir weights and trained models can be saved locally to reproduce subsequent readout and evaluation stages without regenerating the entire pipeline.

---

## 26 — Current Limitations

This repository should be treated as a **research proof-of-concept**, not as an operational EW deployment system.

The current study has several important limitations:

- Classification accuracy remains below the evaluated conventional baselines.
- CPU simulation of the LSM does not provide a latency advantage.
- The reported Loihi 2 energy values are estimates rather than physical measurements.
- Measured spike sparsity of **24.9%** exceeded the intended target of below **5%**.
- Synthetic jamming data cannot fully substitute for real-world EW recordings.
- The single-task versus multi-task comparison requires further completion.
- The Echo State Network baseline requires completion for a cleaner reservoir-computing comparison.
- Physical deployment on Loihi 2 has not yet been experimentally validated.

These limitations are important because the central result of the project is not that the current LSM already outperforms conventional classifiers. Rather, the work demonstrates a substantial **estimated energy-efficiency opportunity** while identifying the architectural changes required to improve accuracy and validate the hardware advantage.

---

## 27 — Future Work

The project roadmap includes:

### Phase 2 — Neuromorphic Hardware Deployment

Map the validated architecture to **Intel Loihi 2** using the Lava software framework and replace proxy energy estimates with direct hardware measurements.

### Phase 3 — Adaptive Reservoir

Investigate online reservoir adaptation using mechanisms such as **Spike-Timing-Dependent Plasticity (STDP)**.

### Phase 4 — Real-Time SDR Integration

Connect the classifier to a **Software Defined Radio (SDR)** platform for laboratory over-the-air RF classification.

### Phase 5 — Multi-Antenna Extension

Extend the architecture to MIMO receiver arrays and incorporate spatial information such as direction of arrival.

### Phase 6 — On-Device Learning

Enable the deployed system to learn new emitter signatures through operator-confirmed encounters.

### Phase 7 — Memristive Hardware

Investigate fixed randomly wired reservoirs implemented using memristive crossbar architectures for ultra-low-power analogue neuromorphic processing.

---

## 28 — Research Significance

The project's results reveal a useful distinction between **energy efficiency, computational speed, and classification accuracy**.

The LSM achieved an estimated inference energy of:

```text
3.30 µJ / inference
```

while requiring:

```text
17.28 ms / inference
```

on the evaluated CPU implementation.

This demonstrates why neuromorphic computing should not be evaluated solely through conventional software benchmarks. The intended advantage arises from **event-driven sparse computation on specialised hardware**, rather than from simply simulating spiking neurons faster on a general-purpose processor.

The current implementation therefore provides a foundation for subsequent work focused on:

```text
Better encoding
      ↓
Higher spike sparsity
      ↓
Improved representation
      ↓
Higher classification accuracy
      ↓
Loihi 2 hardware validation
      ↓
Real-time RF / SDR demonstration
```

---

## 29 — Citation & Acknowledgements

### RadioML 2018.01A

O'Shea, T. J., Roy, T., & Clancy, T. C. (2018). *Over-the-air deep learning based radio signal classification.* IEEE Journal of Selected Topics in Signal Processing.

### Liquid State Machines

Maass, W., Natschläger, T., & Markram, H. (2002). *Real-time computing without stable states.* Neural Computation.

### Neuromorphic Energy

Loihi 2 energy figures used in this project are based on published Intel Labs characterization data and are treated as **estimates rather than physical measurements**.

---

## 30 — Project Context

**Project:** Neuromorphic Electronic Signal Intelligence  
**Core Method:** Liquid State Machines  
**Domain:** Electronic Warfare / RF Signal Intelligence  
**Institutional Context:** Solid State Physics Laboratory (SSPL), Defence Research and Development Organisation (DRDO)  
**Academic Context:** Electronics & Communication Engineering  
**Year:** 2026

---

<p align="center">
  <strong>Liquid State Machines × Spiking Neural Networks × RF Signal Intelligence</strong><br>
  <sub>Research prototype for low-power, real-time electronic warfare signal classification</sub>
</p>
