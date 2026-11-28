# 🛡️ ModelShield: Adaptive Red-Team Mission Lab

ModelShield is an AI security testing and adversarial robustness evaluation platform. It automates adversarial red-teaming for machine learning vision models, moving beyond fixed attack scripts by introducing an **Adaptive Decision Engine**, a **Preflight Integrity Validator**, an interactive TryHackMe-inspired **Adaptive Red-Team Mission Lab**, and an **Evidence-Grounded AI Copilot**.

---

## 🏛️ System Architecture

```text
                               +-------------------------------------+
                               |   ModelShield Streamlit Dashboard   |
                               |  - Test Your Own Image              |
                               |  - Create Adversarial Image         |
                               |  - Test CIFAR-10 Dataset            |
                               |  - 🛡️ Defense Verification Lab     |
                               |  - ⚔️ Adaptive Red-Team Mission Lab |
                               +------------------+------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |      Red-Team Orchestrator          |
                               |        (orchestrator/engine.py)     |
                               +------------------+------------------+
                                                  |
           +-----------------------+--------------+-----------------------+
           |                       |                                      |
           v                       v                                      v
+---------------------+ +----------------------+        +----------------------+
| Preflight Validator | |   Adaptive Decision  |        | Investigation Copilot|
| - Dimensions & dims | |         Policy       |        | - Mission briefing   |
| - Pixel ranges [0,1]| | - Escalation rules   |        | - Decision explainer |
| - Finite check      | | - Budget enforcement |        | - Grounded Q&A       |
+---------------------+ +----------------------+        +----------------------+
           |                       |                                      |
           +-----------------------+--------------------------------------+
                                   |
                                   v
+----------------------------------------------------------------------------------+
|                              Core Evaluation Engine                              |
|  - Fast Gradient Sign Method (FGSM, attacks/fgsm.py)                             |
|  - Projected Gradient Descent (PGD, attacks/pgd.py)                              |
|  - Defense Evaluation (Input Quantization, defenses/defense_methods.py)          |
|  - Trust & Integrity Checks (evaluation/trust_checks.py, trust_flag.py)          |
+----------------------------------------------------------------------------------+
```

---

## 🚀 Key Features

### 1. Independent Workspace Architecture (`app.py`)
All 5 workspaces in ModelShield are completely independent, mutually exclusive top-level routes:
- **`Test Your Own Image`**: Classify arbitrary user images and inspect confidence distributions.
- **`Create an Adversarial Image`**: Synthesize single-instance FGSM/PGD perturbations and inspect noise.
- **`Test CIFAR-10 Dataset`**: Multi-sample batch evaluation with worst-case attack analysis.
- **`🛡️ Defense Verification Lab`**: Independent workspace for empirical defense experimentation (input quantization levels, clean accuracy vs. adversarial recovery trade-offs).
- **`⚔️ Adaptive Red-Team Mission Lab`**: Independent gamified AI security investigation workspace inspired by TryHackMe.

### 2. Adaptive Red-Team Mission Lab (`orchestrator/ui.py`)
An interactive cybersecurity investigation experience:
- **Mission Briefing:** Operational directive, target asset profile (`Carmon2019Unlabeled`), $L_\infty$ perturbation envelope, quota meters, and prominent launch action.
- **Clickable Investigation Map:** Interactive checkpoint nodes (`01 RECON` → `02 BASELINE` → `03 FGSM PROBE` → `04 PGD ESCALATION` → `05 DEFENSE VERIFICATION` → `06 DEBRIEF`). Selecting any checkpoint opens its deep technical dossier.
- **Technical Stage Dossier:** Exact parameters applied, measured metrics, latency, trust checks, and the exact policy condition that triggered the next step.
- **Adaptive Policy Decision Console:** Monospace terminal displaying the audit feed of backend decisions, triggering thresholds, and remaining budget.
- **Mission Milestones & Discoveries:** Real discoveries unlocked strictly when confirmed by empirical measurements:
  - 🎯 *Milestone: Clean Reference Anchored*
  - 💥 *Discovery: First-Order Gradient Leakage Observed*
  - ⚡ *Discovery: Iterative Non-Linear Boundary Exploitation Confirmed*
  - 🛡️ *Discovery: Perturbation Envelope Compliant*
  - 🔒 *Discovery: Defensive Mitigation via Spatial Discretization*
  - 🏁 *Milestone: Mission Protocol Debriefed*
- **Interactive Evidence & Prediction Inspector:** Inspects per-instance classifications (Ground Truth, Clean Pred, FGSM Pred, PGD Pred) and identifies which samples were flipped by the adversarial attack.
- **Mission History & Replay:** Revisit and audit previous mission runs from `results/runs/*.json` with timeline replay.
- **Final Technical Debrief & Defense Lab Handoff:** Executive security verdict, verified findings, scientific limitations, and a direct handoff callout to the independent Defense Verification Lab.

### 3. Adaptive Decision Policy (`orchestrator/policy.py`)
Autonomous, evidence-driven decision layer:
- **Preflight Check:** Validates input tensor dimensions `(N, C, H, W)`, bounds `[0, 1]`, and finite model outputs. Halts with `HALT_INTEGRITY_FAILURE` if checks fail.
- **Clean Baseline:** Benchmark non-adversarial accuracy and prediction distribution.
- **FGSM Probe:** Evaluates single-step linear loss gradient sensitivity.
- **PGD Escalation:** When FGSM reveals a measurable vulnerability (Attack Success Rate $\ge 15\%$), the engine automatically escalates to multi-step iterative PGD to evaluate non-linear gradient robustness.
- **Defense Verification:** Tests whether input quantization mitigates observed vulnerabilities without destroying clean accuracy.
- **Deterministic Budget Enforcement:** Strictly decrements and respects a user-configured evaluation budget ($N$ steps), stopping cleanly with full audit records when the budget is reached.

### 4. Interactive AI Red-Team Copilot (`orchestrator/copilot.py`)
An interactive technical assistant grounded in real run measurements:
- **Mission Briefing:** Explains threat model, scope, capabilities, and scientific boundaries.
- **Decision Explainer:** Explains why each attack or defense was triggered, citing exact numerical evidence.
- **Mathematical Clarifications:** Clearly explains the difference between **Adversarial Accuracy** (overall accuracy on adversarial images) and **Attack Success Rate (ASR)** (percentage of *initially correct* images that were fooled by perturbation).
- **Scientific Honesty:** Transparently flags when data is missing and highlights what attack vectors remain untested (AutoAttack, C&W L2, black-box decision attacks, patch attacks).

---

## 🔬 Scientific Limitations & Threat Model

1. **Threat Model:** $L_\infty$ norm-bounded adversarial perturbations with configurable budget $\epsilon$ (default $\epsilon = 8/255 \approx 0.0314$).
2. **Robustness Scope:** Passing FGSM and PGD demonstrates resistance to first-order gradient attacks under the tested perturbation budget. It does **not** certify complete security against adaptive attackers, AutoAttack, or decision-based black-box attacks.
3. **No Gradient Masking Assumptions:** The orchestrator does not assume defense validity without verifying loss trends. Input quantization can potentially be bypassed by Backward Pass Differentiable Approximation (BPDA).

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch & Torchvision
- Streamlit
- RobustBench: `pip install git+https://github.com/RobustBench/robustbench.git`

### Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

Launch the Streamlit dashboard:
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`. In the left sidebar, navigate to:
**`⚔️ Adaptive Red-Team Mission Lab`**

---

## 🧪 Running the Test Suites

### 1. Orchestrator & Copilot Unit Tests
Runs the test suite for preflight integrity, adaptive decision policy, budget exhaustion, full engine execution, defense quantization, and copilot inquiries:
```powershell
python -m unittest tests/test_orchestrator.py
```

### 2. Teammate Trust Checks & Flag Regression Tests
```powershell
python -m evaluation.test_trust_checks
python -m evaluation.test_trust_flag
python checksetup.py
```

---

## 📂 Repository Structure

```text
ModelShield/
├── app.py                         # Streamlit dashboard entry point (5 independent workspaces)
├── config.py                      # Global evaluation configuration
├── requirements.txt               # Project dependencies
├── attacks/
│   ├── fgsm.py                    # Fast Gradient Sign Method implementation
│   ├── pgd.py                     # Projected Gradient Descent implementation
│   └── ensemble.py                # Multi-attack ensemble evaluation
├── defenses/
│   └── defense_methods.py         # Identity & input quantization defenses
├── evaluation/
│   ├── defense_evaluation.py      # Defense comparison and attack success rate calculation
│   ├── metrics.py                 # Clean accuracy evaluation
│   ├── trust_checks.py            # Perturbation, pixel range, and output finite checks
│   ├── trust_flag.py              # PASS / WARNING / FAIL classification
│   ├── test_trust_checks.py       # Trust checks unit test
│   └── test_trust_flag.py         # Trust flag unit test
├── model/
│   ├── dataset.py                 # CIFAR-10 data loader with transforms
│   ├── load_model.py              # Model loader with verified checkpoint verification
│   ├── test_fgsm.py               # FGSM evaluation test
│   ├── test_pgd.py                # PGD evaluation test
│   └── test_model.py              # Model inference test
├── orchestrator/
│   ├── __init__.py                # Package exports
│   ├── types.py                   # Data classes, enums, and schemas
│   ├── preflight.py               # Preflight tensor, pixel, and output validator
│   ├── policy.py                  # Deterministic adaptive decision policy
│   ├── engine.py                  # Orchestrator coordinator & audit persistence
│   ├── copilot.py                 # Grounded AI investigation copilot
│   ├── ui.py                      # Adaptive Red-Team Mission Lab UI
│   └── api.py                     # Optional FastAPI REST endpoints
└── tests/
    ├── __init__.py                # Tests package
    └── test_orchestrator.py       # Full unit test suite (12 tests)
```
