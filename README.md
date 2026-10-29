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

### Regression Gate Test Suite (26 tests)
```bash
python -m unittest evaluation.test_regression_gate -v
```

### Trust Checks & Trust Flag Tests
```bash
python evaluation/test_trust_checks.py
python evaluation/test_trust_flag.py
```

---

## How to Use the Model Security Regression Gate

1. Open the dashboard and navigate to **3. Test CIFAR-10 Dataset**.
2. Run an evaluation on your baseline model, scroll to **⚖️ Model Security Regression Gate Snapshots**, and click **📌 Set as Regression Baseline**.
3. Re-run an evaluation on your candidate model or defense configuration, and click **🎯 Set as Regression Candidate**.
4. Switch to **⚖️ Model Security Regression Gate** from the sidebar navigation.
5. Review the loaded snapshots, adjust regression tolerances in **⚙️ Regression Gate Thresholds & Policies** if desired, and click **Run Security Regression Gate**.
6. Review the PASS/WARN/FAIL verdict, metric deltas table, and condition compatibility audit.
7. Download the **JSON Report** or **Markdown Summary** using the export buttons.

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
