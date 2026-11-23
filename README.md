# ModelShield 🛡️

An adversarial robustness evaluation platform for CIFAR-10 image classification models, featuring:
- FGSM and PGD adversarial attack generation
- Dataset-level robustness evaluation
- Gradient-masking diagnostic suite
- ⚖️ Model Security Regression Gate (new in this branch)
- Defense Verification Lab (input quantization)
- Trust check pipeline with PASS/WARNING/FAIL flags

---

## ⚖️ Model Security Regression Gate

Feature 15 adds an automated, policy-driven regression gate that compares baseline and candidate evaluation results under equivalent conditions.

### Key Capabilities:
- **Condition Compatibility Verification:** Rejects evaluations with mismatched datasets, sample counts, $\epsilon$ perturbation budgets, or PGD iterations by default.
- **Directional Regression Detection:** Detects clean accuracy drops, PGD robust accuracy drops, and Attack Success Rate (ASR) surges.
- **Configurable Tolerances:** Two-tiered threshold configuration supporting early warning (`WARN`) and hard deployment block (`FAIL`).
- **Integrity & Diagnostics Enforcement:** Propagates ModelShield trust flag results and rejects candidates with suspicious gradient-masking signatures.
- **Authoritative Explanations & Reports:** Generates structured JSON reports for CI/CD runners and formatted Markdown summaries for pull request comments.

---

## 🔬 Gradient-Masking Diagnostic Suite

ModelShield includes a full gradient-masking detection feature based on Athalye et al., *Obfuscated Gradients Give a False Sense of Security* (ICML 2018).

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
│   ├── fgsm.py                        # FGSM adversarial attack
│   ├── pgd.py                         # PGD adversarial attack
│   └── ensemble.py                    # Attack ensemble & baseline metrics
├── defenses/
│   └── defense_methods.py             # Identity and quantization defenses
├── evaluation/
│   ├── regression_gate.py             # 🆕 Model security regression gate engine
│   ├── test_regression_gate.py        # 🆕 Unit tests (26 tests, offline)
│   ├── gradient_masking.py            # Gradient-masking diagnostic engine
│   ├── test_gradient_masking.py       # Diagnostic unit tests (10 tests, offline)
│   ├── trust_checks.py                # Perturbation/pixel/metric sanity checks
│   ├── trust_flag.py                  # Aggregates checks into PASS/WARNING/FAIL
│   ├── defense_evaluation.py          # Defense comparison evaluation
│   └── metrics.py                     # Clean accuracy utilities
├── model/
│   ├── load_model.py                  # Loads Carmon2019Unlabeled via RobustBench
│   └── dataset.py                     # CIFAR-10 data loading utilities
├── requirements.txt
└── ModelShield_Gradient_Masking_Implementation_Plan.md
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

All unit tests run **offline without any network access or model downloads**:

### Regression Gate & Live Evaluation Test Suites
```bash
python -m unittest evaluation.test_regression_gate -v
python -m unittest evaluation.test_live_eval -v
```

### Trust Checks & Trust Flag Tests
```bash
python evaluation/test_trust_checks.py
python evaluation/test_trust_flag.py
```

---

## How to Use the Model Security Regression Gate

### Option A: Live Model Checkpoint Upload & Automatic Evaluation (Recommended)
1. Open the dashboard and navigate to **⚖️ Model Security Regression Gate**.
2. In the **🚀 Live Model Evaluation** tab:
   - **Stage 1 (Select Baseline)**: Select the default `Carmon2019Unlabeled` baseline checkpoint (`models/cifar10/Linf/Carmon2019Unlabeled.pt`), reuse an active session snapshot, or upload a custom baseline checkpoint.
   - **Stage 2 (Upload Candidate)**: Upload your candidate PyTorch checkpoint (`.pt` or `.pth`). ModelShield automatically infers or lets you select the model architecture (`WideResNet-28-10`, `WideResNet-34-10`, `ResNet-18`, `ResNet-50`, or custom).
   - **Stage 3 (Configure Evaluation)**: Select CIFAR-10 test sample count, perturbation budget $\epsilon$, and PGD attack iterations.
   - **Stage 4 (Run Evaluation)**: Click **🚀 Evaluate Models & Run Security Gate**.
3. **Stage 5 (Compare Results)**: Review the PASS/WARN/FAIL verdict, comparative metric deltas, and condition compatibility audit.
4. **Stage 6 (Export Artifacts)**: Download generated Baseline & Candidate JSON snapshots, full JSON gate report, and Markdown summary.

### Option B: Snapshot Comparison & Manual JSON Upload
1. Switch to the **📁 Snapshot Comparison & JSON Upload** tab.
2. Load snapshots captured from the **Test CIFAR-10 Dataset** workspace or upload previously saved JSON evaluation snapshots.
3. Configure regression thresholds and click **Evaluate Gate on Loaded Snapshots**.

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
