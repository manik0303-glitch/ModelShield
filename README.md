# ModelShield 🛡️

An adversarial robustness evaluation platform for CIFAR-10 image classification models, featuring:
- FGSM and PGD adversarial attack generation
- Dataset-level robustness evaluation
- Gradient-masking diagnostic suite (new in this branch)
- Controlled Gradient-Masking Simulation Lab (Validation Mode with BPDA)
- Defense Verification Lab (input quantization)
- Trust check pipeline with PASS/WARNING/FAIL flags

---

## 🆕 Gradient-Masking Diagnostic Suite

This branch adds a full gradient-masking detection feature to ModelShield, based on Athalye et al., *Obfuscated Gradients Give a False Sense of Security* (ICML 2018).

### What is Gradient Masking?
Gradient masking is a failure mode where gradient-based attacks appear ineffective because the model's input gradients are vanishing, shattered, or misleading — **not because the model is genuinely robust**.

### 6 Diagnostic Signals
| Signal | Method | Cost |
|---|---|---|
| **Gradient Health** | Per-sample L2 norm, near-zero fraction, NaN/Inf detection | 1 backward pass |
| **1D Loss Landscape Ray** | Evaluates loss monotonicity along `x + α·sign(∇L)` | 0 backward passes |
| **PGD Restart Sensitivity** | ASR spread across independent random starts | R-1 extra backward runs |
| **Attack Strength Consistency** | Step trajectory snapshotting + epsilon scaling | 1 long PGD + 2 sweeps |
| **FGSM vs PGD Discrepancy** | One-step vs. iterative attack comparison | Reuses base results |
| **Zero-Order Random Baseline** | Uniform noise vs. gradient-guided success | 0 backward passes |

---

## Project Structure

```text
ModelShield/
├── app.py                             # Streamlit dashboard (entry point)
├── attacks/
│   ├── bpda_pgd.py                    # 🆕 BPDA-assisted PGD (straight-through approximation)
│   ├── fgsm.py                        # FGSM adversarial attack
│   ├── pgd.py                         # PGD adversarial attack
│   └── ensemble.py                    # Attack ensemble & baseline metrics
├── defenses/
│   └── defense_methods.py             # Identity and quantization defenses
├── evaluation/
│   ├── gradient_masking.py            # Gradient-masking diagnostic engine (6 signals)
│   ├── masking_simulation.py          # 🆕 Controlled gradient-masking simulation runner
│   ├── test_gradient_masking.py       # Diagnostic engine unit tests
│   ├── trust_checks.py                # Perturbation/pixel/metric sanity checks
│   ├── trust_flag.py                  # Aggregates checks into PASS/WARNING/FAIL
│   ├── defense_evaluation.py          # Defense comparison evaluation
│   └── metrics.py                     # Clean accuracy utilities
├── tests/
│   └── test_masking_simulation.py     # 🆕 Simulation suite unit tests (25 tests)
├── model/
│   ├── load_model.py                  # Loads Carmon2019Unlabeled via RobustBench
│   └── dataset.py                     # CIFAR-10 data loading utilities
├── requirements.txt
├── ModelShield_Gradient_Masking_Implementation_Plan.md
└── ModelShield_Simulated_Gradient_Masking_Implementation_Plan.md
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** On first run, RobustBench will automatically download the pretrained model weights (~153 MB) and the CIFAR-10 dataset (~170 MB). These are cached locally and only downloaded once.

### 2. Run the Dashboard
```bash
streamlit run app.py
```

Access it at `http://localhost:8501` in your browser.

---

## Running the Tests

All 35 unit tests run **offline without any network access or model downloads**:
```bash
python -m pytest tests/ evaluation/ -v
```

Expected output:
```text
35 passed in ~6s
```
python -m pytest evaluation/test_gradient_masking.py -v
```

Expected output:
```text
evaluation/test_gradient_masking.py .......... [100%]
10 passed in 1.24s
```

---

## How to Use the Gradient-Masking Detector

1. Open the dashboard and navigate to **3. Test CIFAR-10 Dataset**.
2. Click **Run CIFAR-10 Evaluation** to load the baseline attack results.
3. Scroll to **🔬 Gradient-Masking Diagnostic Suite**.
4. (Optional) Configure the diagnostic sample cap and random search setting.
5. Click **Run Gradient-Masking Diagnostics**.

### Output Statuses
| Status | Meaning |
|---|---|
| ✅ `NO_STRONG_SIGNAL` | No suspicious gradient patterns detected in this run |
| ⚠️ `WARNING` | One or more suspicious signals observed — evaluate with adaptive attacks |
| ℹ️ `INCONCLUSIVE` | Insufficient clean-correct samples to assess gradients |

> **Important:** A `WARNING` does *not* prove masking is present. A `NO_STRONG_SIGNAL` does *not* prove the model is robust. These are heuristic diagnostics.

---

## Expected Results on `Carmon2019Unlabeled` (Genuine AT Model)

| Metric | Expected Value |
|---|---|
| Clean Accuracy | ~85–90% |
| FGSM Accuracy | ~60–70% |
| PGD Accuracy | ~55–65% |
| Overall Masking Status | `NO_STRONG_SIGNAL` |

---

## References

- Athalye, A., Carlini, N., & Wagner, D. (2018). [Obfuscated Gradients Give a False Sense of Security](https://arxiv.org/abs/1802.00420). *ICML 2018*.
- Croce, F., & Hein, M. (2020). [RobustBench](https://robustbench.github.io/).
- Carmon, Y. et al. (2019). Unlabeled Data Improves Adversarial Robustness.

