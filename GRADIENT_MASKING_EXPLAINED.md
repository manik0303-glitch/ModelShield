# ModelShield — Gradient Masking Architecture, Theory & Mathematical Guide

**Document:** `GRADIENT_MASKING_EXPLAINED.md`  
**Purpose:** Comprehensive mathematical, algorithmic, and architectural guide to all gradient-masking components in the ModelShield project.

---

## Table of Contents
1. [Executive Overview: What is Gradient Masking?](#1-executive-overview-what-is-gradient-masking)
2. [The Two Locations in ModelShield](#2-the-two-locations-in-modelshield)
   - [Location 1: Gradient-Masking Diagnostic Suite (Test CIFAR-10 Dataset)](#location-1-gradient-masking-diagnostic-suite)
   - [Location 2: Gradient Masking Lab (Validation Suite with BPDA)](#location-2-gradient-masking-lab)
3. [Core Mathematical Formulations & Working Principles](#3-core-mathematical-formulations--working-principles)
   - [3.1 Clean Accuracy & Attack Success Rate (ASR)](#31-clean-accuracy--attack-success-rate-asr)
   - [3.2 Signal 1: Input Gradient Health & Numerical Sanity](#32-signal-1-input-gradient-health--numerical-sanity)
   - [3.3 Signal 2: 1D Loss Landscape Directional Monotonicity](#33-signal-2-1d-loss-landscape-directional-monotonicity)
   - [3.4 Signal 3: PGD Random Restart Sensitivity](#34-signal-3-pgd-random-restart-sensitivity)
   - [3.5 Signal 4: Attack Strength Consistency (Steps & Budget Scaling)](#35-signal-4-attack-strength-consistency-steps--budget-scaling)
   - [3.6 Signal 5: FGSM vs. Iterative PGD Discrepancy](#36-signal-5-fgsm-vs-iterative-pgd-discrepancy)
   - [3.7 Signal 6: Zero-Order (Gradient-Free) Random Search Baseline](#37-signal-6-zero-order-gradient-free-random-search-baseline)
4. [The Mathematics of Broken Defenses: Input Quantization](#4-the-mathematics-of-broken-defenses-input-quantization)
5. [The Mathematics of Circumvention: BPDA](#5-the-mathematics-of-circumvention-bpda)
6. [Codebase Map & Module Responsibilities](#6-codebase-map--module-responsibilities)
7. [Step-by-Step Execution Walkthrough](#7-step-by-step-execution-walkthrough)
8. [Scientific Interpretation & Limitations](#8-scientific-interpretation--limitations)

---

## 1. Executive Overview: What is Gradient Masking?

In adversarial machine learning, **gradient masking** (also termed *obfuscated gradients*; Athalye, Carlini & Wagner, ICML 2018) is a failure mode in which standard first-order adversarial attacks (such as FGSM and PGD) appear ineffective **not because the model is genuinely robust, but because the model's loss gradients are broken, uninformative, or misleading**.

### The Three Types of Obfuscated Gradients (Athalye et al., 2018):
1. **Shattered Gradients:** Non-differentiable operations (e.g., input rounding, quantization, JPEG compression) cause the gradient to either not exist or be zero almost everywhere ($\nabla_x L \approx 0$).
2. **Stochastic / Randomized Gradients:** Random transformations at inference time cause the gradient computed at step $t$ to point in a different direction than the gradient at step $t+1$.
3. **Vanishing / Exploding Gradients:** Extremely deep computation or saturating operations cause gradients to vanish to zero or explode to infinity ($\text{NaN} / \pm\infty$).

ModelShield provides both an **observational diagnostic engine** to detect these symptoms on any model and a **controlled simulation lab** that artificially creates shattered gradients to empirically prove the detector works.

---

## 2. The Two Locations in ModelShield

In the ModelShield dashboard, gradient masking is integrated into two distinct workspaces:

```text
ModelShield Dashboard
├── Option 3: "Test CIFAR-10 Dataset"
│   └── Section: 🔬 Gradient-Masking Diagnostic Suite
│       └── Goal: Passively diagnose any model under test (e.g. Carmon2019Unlabeled)
│                 to check if its reported robustness is genuine or an artifact of masking.
│
└── Option 4: "Gradient Masking Lab"
    └── Workspace: 4. 🔬 Gradient Masking Lab (Validation Suite)
        └── Goal: Actively run a controlled validation experiment. Prepends hard input
                  quantization to shatter gradients, runs standard attacks vs. BPDA-assisted PGD,
                  and validates that ModelShield correctly catches the masking.
```

---

### Location 1: Gradient-Masking Diagnostic Suite
* **Where to find it:** Sidebar $\to$ **`Test CIFAR-10 Dataset`** $\to$ Run base evaluation $\to$ Scroll to bottom section **"🔬 Gradient-Masking Diagnostic Suite"**.
* **Purpose:** Runs 6 mathematical diagnostic tests on the evaluated model and CIFAR-10 samples without modifying the model.
* **Expected Output on Genuine Adversarially Trained Model (`Carmon2019Unlabeled`):**
  - **Status:** `STATUS: NO_STRONG_SIGNAL` (Green banner)
  - **Median Input Grad L2:** $\sim 0.025$ (finite, healthy magnitude)
  - **Loss Ray Slope ($\Delta L$):** $+0.3658$ (positive, loss increases along gradient ray)
  - **PGD Restart Spread:** Small ($< 20\%$)
  - **FGSM vs PGD Delta:** Negative (Iterative PGD is stronger than one-step FGSM)
  - **Verdict Meaning:** The model does **not** rely on gradient masking; its robustness is legitimate.

---

### Location 2: Gradient Masking Lab
* **Where to find it:** Sidebar $\to$ **`Gradient Masking Lab`** (Workspace Option 4).
* **Purpose:** Acts as a **testbed and scientific proof**. It wraps the model in a temporary non-differentiable quantization transform:
  $$x_{\text{quantized}} = \frac{\text{round}(x \cdot (L-1))}{L-1}$$
  and pits standard attacks against **BPDA-assisted PGD** (*Backward Pass Differentiable Approximation*).
* **Expected Output on Quantized Pipeline:**
  - **Panel 1 (Simulation Validation):** `VALIDATED` (Confirms hard quantization was active, ordinary gradient was zeroed out, and BPDA bypassed it).
  - **Panel 2 (Detector Verdict):** `WARNING` (Independently catches the zeroed gradients without knowing it was in simulation mode!).
  - **Panel 4 (Attack Comparison):**
    - Standard FGSM ASR: $\approx 0\%$
    - Standard PGD ASR: $\approx 6.25\%$
    - BPDA-Assisted PGD ASR: $\approx 43.75\%$ (Exposes true vulnerability with $+37.50\%$ advantage!).

---

## 3. Core Mathematical Formulations & Working Principles

Below are the exact mathematical definitions implemented in [`evaluation/gradient_masking.py`](file:///c:/main%20el%20sem%203/gradient%20masking%20detector/ModelShield-master/evaluation/gradient_masking.py).

### 3.1 Clean Accuracy & Attack Success Rate (ASR)

Let $X = \{x_1, x_2, \dots, x_N\}$ be the evaluation batch with true labels $Y = \{y_1, y_2, \dots, y_N\}$, and let $f(x)$ be the model prediction $\arg\max_k f_k(x)$.

#### Clean Accuracy:
$$\text{Clean Accuracy} = 100\% \times \frac{1}{N} \sum_{i=1}^N \mathbb{I}(f(x_i) = y_i)$$

#### Eligible Denominator ($S_{\text{clean\_correct}}$):
An adversarial attack can only succeed on an image that the model **correctly classified in the first place**:
$$S_{\text{clean\_correct}} = \{i \in \{1, \dots, N\} \mid f(x_i) = y_i\}$$
$$N_{\text{eligible}} = |S_{\text{clean\_correct}}|$$

#### Attack Success Rate (ASR):
Given adversarial images $x^{\text{adv}}$, an attack is successful if the prediction on $x^{\text{adv}}$ is incorrect:
$$\text{ASR} = 100\% \times \frac{\sum_{i \in S_{\text{clean\_correct}}} \mathbb{I}(f(x_i^{\text{adv}}) \ne y_i)}{N_{\text{eligible}}}$$

> **Important Edge Case:** If $N_{\text{eligible}} = 0$, $\text{ASR}$ is mathematically undefined ($0/0$). ModelShield returns `None` and marks status `INCONCLUSIVE`, preventing false $0\%$ reporting or division-by-zero crashes.

---

### 3.2 Signal 1: Input Gradient Health & Numerical Sanity
*Function: `compute_gradient_health(model, images, labels)`*

Calculates the gradient of cross-entropy loss $\mathcal{L}_{\text{CE}}$ with respect to the input pixels $x$:
$$g_i = \nabla_{x_i} \mathcal{L}_{\text{CE}}(f(x_i), y_i) \in \mathbb{R}^{C \times H \times W}$$

For each sample $i \in \{1, \dots, N\}$:
1. **$L_2$ Norm:**
   $$\|g_i\|_2 = \sqrt{\sum_{c, h, w} (g_{i, c, h, w})^2}$$
2. **Mean Absolute Magnitude:**
   $$\text{MeanAbs}(g_i) = \frac{1}{D} \sum_{j=1}^D |g_{i, j}| \quad \text{where } D = 3 \times 32 \times 32 = 3072$$
3. **Near-Zero Fraction:**
   $$\text{ZeroFrac}(g_i) = \frac{1}{D} \sum_{j=1}^D \mathbb{I}(|g_{i, j}| < 10^{-7})$$
4. **Non-Finite Detection:**
   $$\text{NonFinite}(g_i) = \exists j \text{ s.t. } g_{i, j} \in \{\text{NaN}, +\infty, -\infty\}$$

#### Decision Rules:
- If $\text{NonFinite} == \text{True} \implies \text{Status} = \mathbf{ERROR}$
- If $\text{Median}(\|g_i\|_2) < 10^{-5}$ OR $\text{Median}(\text{ZeroFrac}) > 90\% \implies \text{Status} = \mathbf{SUSPICIOUS}$
- Else $\implies \text{Status} = \mathbf{OK}$

---

### 3.3 Signal 2: 1D Loss Landscape Directional Monotonicity
*Function: `evaluate_loss_landscape_ray(model, images, labels, clean_correct_mask, epsilon)`*

By first-order Taylor expansion, for a small step $\alpha > 0$ along the gradient ascent direction $v = \text{sign}(\nabla_x \mathcal{L})$, the loss **must increase**:
$$\mathcal{L}(x + \alpha \cdot \text{sign}(\nabla_x \mathcal{L})) \approx \mathcal{L}(x) + \alpha \langle \nabla_x \mathcal{L}, \text{sign}(\nabla_x \mathcal{L}) \rangle = \mathcal{L}(x) + \alpha \|\nabla_x \mathcal{L}\|_1 > \mathcal{L}(x)$$

ModelShield samples the loss at three interpolation points using **zero backward passes** (forward-only):
$$\alpha \in \{0.0, 0.5\epsilon, 1.0\epsilon\}$$
$$x(\alpha) = \text{clamp}(x + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}), 0, 1)$$

Let $L_0 = \mathcal{L}(x(0))$, $L_{0.5} = \mathcal{L}(x(0.5\epsilon))$, and $L_{1.0} = \mathcal{L}(x(\epsilon))$.
$$\Delta L = L_{1.0} - L_0$$

#### Decision Rules:
- If $\Delta L < -10^{-4}$ (loss decreases along gradient direction): $\implies \mathbf{SUSPICIOUS}$ (Shattered / misleading gradient).
- If $|\Delta L| < 10^{-6}$ and $L_0 > 0.05$ (loss curve is completely flat): $\implies \mathbf{SUSPICIOUS}$ (Vanishing gradient / plateau).
- Else $\implies \mathbf{OK}$ (Directional monotonicity holds).

---

### 3.4 Signal 3: PGD Random Restart Sensitivity
*Function: `evaluate_pgd_restarts(model, images, labels, clean_correct_mask, ...)`*

PGD starts from a random point in the $\epsilon$-ball: $x^{(0)} = x + \delta, \; \delta \sim \mathcal{U}(-\epsilon, \epsilon)$. In a shattered loss landscape with deceptive local minima, different random starting points yield vastly different attack results.

ModelShield measures ASR across $R$ independent restarts:
$$S = [\text{ASR}_1, \text{ASR}_2, \dots, \text{ASR}_R]$$
$$\text{Spread} = \max(S) - \min(S)$$

#### Optimization:
Baseline PGD from the main evaluation is reused as $\text{ASR}_1$; only $R-1$ additional runs are executed, saving 33% CPU time.

#### Decision Rules:
- If $\text{Spread} \ge 20.0\% \implies \mathbf{SUSPICIOUS}$ (High sensitivity to initialization indicates a rugged, shattered surface).
- Else $\implies \mathbf{OK}$.

---

### 3.5 Signal 4: Attack Strength Consistency (Steps & Budget Scaling)
*Function: `evaluate_attack_strength_consistency(...)`*

A fundamental theorem from Athalye et al. (2018) states:
> *"Unbounded adversarial examples must reach 100% attack success. If an attack does not approach 100% success as perturbation budget $\epsilon$ grows, the defense is obfuscating gradients."*

ModelShield tests monotonicity across two axes:

#### Axis A: Step Trajectory Snapshotting
Runs a single 20-step PGD trajectory and evaluates predictions at steps $k \in \{5, 10, 20\}$ without re-running from scratch:
$$\Delta_{\text{step}} = \text{ASR}_{k-1} - \text{ASR}_k$$
- If $\Delta_{\text{step}} > 5.0\% \implies$ Flag non-monotonic drop.

#### Axis B: Epsilon Budget Scaling
Tests PGD at three budgets:
$$\epsilon \in \{0.5\epsilon_{\text{base}}, 1.0\epsilon_{\text{base}}, 2.0\epsilon_{\text{base}}\}$$
$$\Delta_{\epsilon} = \text{ASR}_{\epsilon_{t-1}} - \text{ASR}_{\epsilon_t}$$
- If $\Delta_{\epsilon} > 5.0\% \implies$ Flag non-monotonic drop.

---

### 3.6 Signal 5: FGSM vs. Iterative PGD Discrepancy
*Function: `evaluate_fgsm_vs_pgd(fgsm_asr, pgd_asr)`*

A one-step attack (FGSM) takes a single large jump:
$$x^{\text{FGSM}} = \text{clamp}(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}), 0, 1)$$
An iterative attack (PGD) takes $K$ small steps with step size $\alpha < \epsilon$, projecting back to the $\epsilon$-ball:
$$x^{(t+1)} = \Pi_{x, \epsilon} \left( x^{(t)} + \alpha \cdot \text{sign}(\nabla_{x^{(t)}} \mathcal{L}) \right)$$

Mathematically, iterative optimization over the same space must match or outperform single-step linear approximation:
$$\text{ASR}_{\text{PGD}} \ge \text{ASR}_{\text{FGSM}} - \text{tolerance}$$

#### Decision Rule:
$$\text{Diff} = \text{ASR}_{\text{FGSM}} - \text{ASR}_{\text{PGD}}$$
- If $\text{Diff} > 5.0\% \implies \mathbf{SUSPICIOUS}$ (One-step better than iterative is classic masking).
- Else $\implies \mathbf{OK}$.

---

### 3.7 Signal 6: Zero-Order (Gradient-Free) Random Search Baseline
*Function: `evaluate_random_search_baseline(...)`*

If gradient descent is guided by real gradients, it must outperform pure random chance. ModelShield samples $K = 25$ uniform random perturbations per sample:
$$\delta_k \sim \mathcal{U}(-\epsilon, \epsilon), \quad x_k^{\text{rand}} = \text{clamp}(x + \delta_k, 0, 1)$$
$$\text{ASR}_{\text{rand}} = 100\% \times \frac{1}{N_{\text{eligible}}} \sum_{i \in S} \mathbb{I}\left( \exists k \in \{1,\dots,K\} \text{ s.t. } f(x_{i, k}^{\text{rand}}) \ne y_i \right)$$

#### Decision Rule:
- If $\text{ASR}_{\text{rand}} > \text{ASR}_{\text{PGD}} + 5.0\% \implies \mathbf{SUSPICIOUS}$ (White-box gradients are less informative than blind random noise).
- Else $\implies \mathbf{OK}$.

---

## 4. The Mathematics of Broken Defenses: Input Quantization

Implemented in [`defenses/defense_methods.py`](file:///c:/main%20el%20sem%203/gradient%20masking%20detector/ModelShield-master/defenses/defense_methods.py):
$$q(x) = \frac{\text{round}(x \cdot (L - 1))}{L - 1}, \quad \text{for } x \in [0, 1]$$

Where $L$ is the number of discrete quantization levels (e.g., $L = 16$).

### The Gradient Failure:
The mathematical derivative of the step/rounding function is:
$$\frac{d}{dx} \text{round}(x) = 0 \quad \forall x \notin \mathbb{Z} + 0.5$$
Because the derivative is zero almost everywhere, the chain rule yields:
$$\nabla_x \mathcal{L}(f(q(x)), y) = \left( \frac{\partial q}{\partial x} \right)^T \nabla_q \mathcal{L}(f(q), y) = 0 \cdot \nabla_q \mathcal{L} = \mathbf{0}$$

Standard PGD receives $\mathbf{0}$ for every gradient step, freezing the perturbation at $x^{(t+1)} = x^{(t)}$. Attackers fail, not because the model is robust, but because the gradient signal is wiped out.

---

## 5. The Mathematics of Circumvention: BPDA

Implemented in [`attacks/bpda_pgd.py`](file:///c:/main%20el%20sem%203/gradient%20masking%20detector/ModelShield-master/attacks/bpda_pgd.py) via a custom `torch.autograd.Function`:

```text
Forward Pass:   x ───► [ Hard Rounding q(x) ] ───► Model f(q) ───► Loss L
                       (Exact real quantization)

Backward Pass:  dL/dx ◄── [ Identity Estimator ] ◄── dL/dq ◄────── dL/df
                       (Approximate gradient = 1.0)
```

### Mathematical Formulation:
1. **Forward Rule (Identical to Hard Defense):**
   $$q_{\text{forward}}(x) = \frac{\text{round}(x \cdot (L - 1))}{L - 1}$$
2. **Backward Approximation (Straight-Through Estimator):**
   $$\left. \frac{\partial q}{\partial x} \right|_{\text{backward}} \approx \mathbf{I} \quad \text{(Identity Matrix)}$$
   $$\nabla_x \mathcal{L} \approx \mathbf{I} \cdot \nabla_{q} \mathcal{L} = \nabla_{q} \mathcal{L}$$

By bypassing the zero-derivative of the rounding step, BPDA feeds informative gradients to PGD while testing on the real quantized model, jumping ASR from $6.25\%$ to $43.75\%$.

---

## 6. Codebase Map & Module Responsibilities

| File | Primary Functions | Role in Gradient Masking |
|---|---|---|
| [`evaluation/gradient_masking.py`](file:///c:/main%20el%20sem%203/gradient%20masking%20detector/ModelShield-master/evaluation/gradient_masking.py) | `compute_gradient_health`<br>`evaluate_loss_landscape_ray`<br>`evaluate_pgd_restarts`<br>`evaluate_attack_strength_consistency`<br>`evaluate_fgsm_vs_pgd`<br>`evaluate_random_search_baseline`<br>`analyze_gradient_masking` | **The Core Detector:** Pure PyTorch implementation of the 6 diagnostic signals. Completely independent of the UI and simulation mode. |
| [`attacks/bpda_pgd.py`](file:///c:/main%20el%20sem%203/gradient%20masking%20detector/ModelShield-master/attacks/bpda_pgd.py) | `_StraightThroughQuantize`<br>`straight_through_quantize`<br>`bpda_pgd_attack` | **Circumvention Engine:** Implements BPDA straight-through estimator to attack non-differentiable pipelines. |
| [`evaluation/masking_simulation.py`](file:///c:/main%20el%20sem%203/gradient%20masking%20detector/ModelShield-master/evaluation/masking_simulation.py) | `_HardQuantizedPipeline`<br>`_smoke_check_pipeline`<br>`_run_simulation_validator`<br>`run_masking_simulation` | **Validation Lab Runner:** Creates temporary quantized model wrapper, evaluates standard vs. BPDA attacks, and performs the 5 validation checks. |
| [`app.py`](file:///c:/main%20el%20sem%203/gradient%20masking detector/ModelShield-master/app.py) | Lines 450–625 (`Test CIFAR-10 Dataset`)<br>Lines 880–1200 (`Gradient Masking Lab`) | **Streamlit UI:** Renders the dark cyber-defense cards, Plotly comparison bar charts, signal breakdown expanders, and CSV/JSON downloads. |
| [`tests/test_masking_simulation.py`](file:///c:/main%20el%20sem%203/gradient%20masking%20detector/ModelShield-master/tests/test_masking_simulation.py) | 25 unit tests | Validates quantization invariants, ordinary gradient vanishing, BPDA gradient usability, denominator consistency, and detector independence. |
| [`evaluation/test_gradient_masking.py`](file:///c:/main%20el%20sem%203/gradient%20masking%20detector/ModelShield-master/evaluation/test_gradient_masking.py) | 10 unit tests | Tests the 6 detector signals using synthetic PyTorch models without downloading weights. |

---

## 7. Step-by-Step Execution Walkthrough

```text
                                [ User Clicks "Run" ]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
        [ Location 1: Test Dataset ]                 [ Location 2: Validation Lab ]
                   │                                             │
      Loads Pretrained Carmon2019                    Builds _HardQuantizedPipeline
                   │                                             │
      Computes Standard FGSM & PGD                   Runs Smoke Checks on Pipeline
                   │                                             │
      Runs analyze_gradient_masking()                Evaluates Clean Accuracy Impact
                   │                                             │
       ┌───────────┴───────────┐                     Runs FGSM, PGD, and BPDA PGD
       │ Evaluates 6 Signals:  │                                 │
       │ 1. Gradient Health    │                     Runs analyze_gradient_masking()
       │ 2. 1D Loss Ray        │                     (Detector receives NO sim flag)
       │ 3. PGD Restarts       │                                 │
       │ 4. Strength Scaling   │                     Runs _run_simulation_validator()
       │ 5. FGSM vs PGD        │                     (Checks 5 empirical criteria)
       │ 6. Random Search      │                                 │
       └───────────┬───────────┘                     Renders 7 Panels in Dashboard:
                   │                                 - Simulation Validation Badge
       Renders in Dashboard:                         - Detector Verdict Badge
       - Status Banner (NO_STRONG_SIGNAL)            - Clean Accuracy Impact
       - Metric Cards & Line Charts                  - Plotly Attack ASR Chart
       - 6 Signal Breakdown Expanders                - Gradient Comparison Cards
       - Download JSON / CSV                         - Download JSON / CSV
```

---

## 8. Scientific Interpretation & Limitations

1. **A Warning is Not Formal Mathematical Proof:**
   A `WARNING` status indicates strong empirical evidence of obfuscated gradients (e.g., zero gradients, flat loss rays, or random search beating PGD). It highlights that the reported robustness cannot be trusted.
2. **`NO_STRONG_SIGNAL` is Not a Proof of Security:**
   Passing all 6 checks means standard gradient attacks do not appear obfuscated on the evaluated samples. It does not certify that the model is immune to adaptive attacks or larger perturbation budgets.
3. **BPDA is a Heuristic Approximation:**
   The straight-through estimator ($\frac{\partial q}{\partial x} \approx 1$) is an approximation of the backward pass, not the exact derivative of rounding.
4. **Threat Model Bounded:**
   All diagnostic measurements apply strictly to the evaluated perturbation norm ($L_\infty$), budget ($\epsilon = 8/255$), and sample batch.
