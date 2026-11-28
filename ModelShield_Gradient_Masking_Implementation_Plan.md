# ModelShield — Gradient-Masking Diagnostic Suite
## Enhanced Implementation Plan (v2.0)

**Audience:** AI coding agents and engineers implementing gradient-masking diagnostics in the ModelShield repository.  
**Objective:** Add a scientifically rigorous, computationally optimized gradient-masking *diagnostic and warning suite* to ModelShield that integrates cleanly with the existing attack pipeline and Streamlit dashboard without breaking existing features.

---

## 1. Executive Summary & Core Changes in v2.0

Gradient masking (also known as obfuscated gradients; Athalye et al., ICML 2018) is a failure mode where gradient-based optimization fails because gradients are vanished, shattered, stochastic, or misleading, creating a false sense of security while the underlying model remains vulnerable.

The original implementation plan provided solid principles (cautious language, multi-signal evidence, decoupling from Streamlit). However, detailed codebase and performance analysis revealed critical deficiencies that would lead to poor dashboard performance, UI state resets, missing literature-standard checks, and dead-end placeholders.

### Key Enhancements Introduced in v2.0 for "Better Output":

1. **Streamlit Session State Lifecycle Fix (`app.py`)**:
   - *Issue Identified:* In `app.py`, the CIFAR-10 evaluation currently executes inside `if st.button("Run CIFAR-10 Evaluation")` without storing `results`, `images`, or `labels` in `st.session_state`. Any widget interaction (e.g., image slider or diagnostic button) causes a Streamlit script rerun, erasing all computed results.
   - *v2.0 Solution:* Refactor the CIFAR-10 workflow to persist evaluation tensors and metrics in `st.session_state`. The diagnostic runner reuses these cached tensors seamlessly without forcing the user to re-download or re-evaluate CIFAR-10.

2. **1D Loss Landscape Ray Search (Directional Monotonicity)**:
   - *Scientific Grounding:* Athalye et al. (2018) show that if the loss does not increase along the gradient direction $\text{sign}(\nabla_x L)$, the gradient is obfuscated.
   - *Efficiency:* Requires **only forward passes** (0 backprop passes). It evaluates loss at $\alpha \in \{0, 0.5\epsilon, 1.0\epsilon\}$ along $x + \alpha \cdot \text{sign}(\nabla_x L)$. If $L(x + \epsilon \cdot \text{sign}(\nabla_x L)) \le L(x)$, gradients are demonstrably false.
   - *UI Output:* Renders an interactive 1D loss landscape curve directly in Streamlit.

3. **Trajectory Snapshotting & Runtime Optimization (40–50% Faster on CPU)**:
   - *Issue Identified:* The original plan ran separate PGD-5, PGD-10, PGD-20 attacks plus 3 PGD restarts, requiring over 65 backprop iterations. On CPU with WideResNet-28-10 (`Carmon2019Unlabeled`), this takes 2–3+ minutes, causing Streamlit to lag or freeze.
   - *v2.0 Solution:*
     - **Reuse Base PGD:** Base evaluation PGD-10 is reused as Restart #1; only $(R - 1)$ additional restarts are run.
     - **Trajectory Snapshotting:** For step consistency, run a single 20-step PGD attack and snapshot predictions at steps 5, 10, and 20. This slashes backward passes from 35 to 20.
     - **Diagnostic Subsample:** Add an optional/default sample limit (e.g., max 16 clean-correct images for CPU diagnostics) with clear user control.

4. **Epsilon Scaling Monotonicity Check**:
   - Athalye et al. emphasize that unbounded adversarial examples must achieve 100% attack success. If increasing perturbation budget $\epsilon$ ($0.5\epsilon, 1.0\epsilon, 2.0\epsilon$) does not increase attack success or plateaus far below 100%, gradients are masked. v2.0 integrates this alongside step consistency.

5. **Lightweight Zero-Order (Gradient-Free) Random Search Baseline**:
   - Replaces the dead-end `NOT_RUN` placeholder of Check E with a fast, zero-order random perturbation search (20–50 uniform random samples in $[-\epsilon, \epsilon]$).
   - If pure random noise achieves higher attack success than PGD or FGSM, that is an unambiguous red flag for gradient masking. Requires 0 backward passes.

6. **Actionable Remediation Guidance**:
   - When warnings are triggered, the UI provides concrete next steps (e.g., BPDA for non-differentiable preprocessing, AutoAttack / Square Attack for gradient-free evaluation).

7. **Extensibility to Defenses Lab (`quantization_defense`)**:
   - Prepares the diagnostic module so it can optionally inspect the defended model in the Defense Verification Lab, where input quantization naturally creates shattered gradients.

---

## 2. Existing Codebase Analysis & Integration Touchpoints

| File | Current Role | Integration Details & Fixes in v2.0 |
|---|---|---|
| `attacks/fgsm.py` | Generates FGSM perturbations in $[0, 1]$ | Reused directly. Catch potential `AttributeError` if input gradients are `None` (non-differentiable layers). |
| `attacks/pgd.py` | Iterative PGD with random start | Reused directly. Ensure `model.zero_grad(set_to_none=True)` and detach operations prevent PyTorch computation graph leaks. |
| `attacks/ensemble.py` | Evaluates clean, FGSM, and PGD; calculates ASR | Baseline source. Its outputs are passed into the diagnostic to avoid redundant computation. |
| `evaluation/trust_checks.py` | Perturbation, pixel range, output shape, metric checks | Kept intact. These remain numerical validity checks. |
| `evaluation/trust_flag.py` | Aggregates checks into `PASS`/`WARNING`/`FAIL` | Left independent. Gradient masking is a separate diagnostic, not a hard pipeline failure. |
| `app.py` | Streamlit dashboard | **Requires state management refactoring:** Save `cifar_results`, `images`, and `labels` in `st.session_state` so the new `Run Gradient-Masking Diagnostics` button does not destroy base evaluation results. |
| `model/load_model.py` | Loads RobustBench `Carmon2019Unlabeled` | Unchanged. |
| `defenses/defense_methods.py` | Implements `quantization_defense` | Reference for shattered gradients. The diagnostic architecture accepts any `nn.Module`. |

---

## 3. Architecture & File Layout

Add two new files to `evaluation/`:

```text
evaluation/
    gradient_masking.py       # Core diagnostic engine (pure PyTorch, NO Streamlit imports)
    test_gradient_masking.py  # Fast unit tests using synthetic models (0 external downloads)
```

### Module Responsibilities:
- `evaluation/gradient_masking.py` must be completely decoupled from the UI. It accepts standard PyTorch tensors and returns a standard JSON-serializable Python dictionary (`dict`).
- `app.py` imports `analyze_gradient_masking` and handles formatting, charts, and user interactions.

### Public API Specification:

```python
def analyze_gradient_masking(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    pgd_steps: int = 10,
    pgd_restarts: int = 2,
    baseline_pgd_asr: Optional[float] = None,
    baseline_fgsm_asr: Optional[float] = None,
    run_random_search: bool = True,
    max_diagnostic_samples: int = 16,
) -> dict:
    """
    Evaluates heuristic indicators of gradient masking / obfuscated gradients.
    
    Returns a JSON-serializable dictionary with diagnostic measurements,
    individual signal verdicts, overall status, and actionable remediation notes.
    """
```

---

## 4. The 6 Diagnostic Signals (Scientific Design & Formulas)

To avoid false positives while providing actionable insight, the diagnostic evaluates 6 distinct signals grounded in the literature (Athalye et al., ICML 2018):

```
                        ┌──────────────────────────────────────┐
                        │ analyze_gradient_masking()           │
                        └──────────────────┬───────────────────┘
                                           │
         ┌───────────────┬─────────────────┼─────────────────┬────────────────┐
         ▼               ▼                 ▼                 ▼                ▼
    [Signal 1]      [Signal 2]        [Signal 3]        [Signal 4]       [Signal 5]     [Signal 6]
   Input-Gradient  1D Loss Ray     PGD Restart       Attack-Strength    FGSM vs PGD    Zero-Order
      Health       Monotonicity     Sensitivity        Consistency      Discrepancy   Random Search
  (Norms, NaNs,    (0 backprop,    (Spread across    (Step Trajectory   (One-step vs  (Gradient-free
   Shattering)     forward only)     restarts)       & Epsilon Sweep)    Iterative)    baseline)
```

---

### Signal 1: Input-Gradient Health & Numerical Sanity
- **Scientific Rationale:** Gradient masking often manifests as vanishing gradients ($\nabla_x L \approx 0$), exploding gradients, or disconnected non-differentiable graph nodes.
- **Method:**
  1. Set `model.eval()`. Clone input images with `x_in = images.clone().detach().requires_grad_(True)`.
  2. Compute $L = \text{CrossEntropy}(f(x_{in}), y)$.
  3. Compute $g = \nabla_{x_{in}} L$ via `torch.autograd.grad(loss, x_in)[0]`.
  4. Per example $i$, record:
     - $L_2$ norm: $\|g_i\|_2$
     - Mean absolute gradient: $\frac{1}{D} \|g_i\|_1$
     - Near-zero fraction: $\frac{1}{D} \sum_j \mathbb{I}(|g_{i, j}| < 10^{-7})$
     - Non-finite check: `torch.isnan(g_i).any() or torch.isinf(g_i).any()`
- **Thresholds & Status:**
  - `ERROR`: Any non-finite values present.
  - `SUSPICIOUS`: Median $\|g_i\|_2 < 10^{-5}$ OR median near-zero fraction $> 90\%$.
  - `OK`: Finite gradients with usable magnitude.

---

### Signal 2: 1D Loss Landscape Ray Search (Directional Monotonicity)
- **Scientific Rationale:** If gradients are informative, the loss along the first-order ascent ray $x(\alpha) = \text{clamp}(x + \alpha \cdot \text{sign}(\nabla_x L), 0, 1)$ must be non-decreasing for small $\alpha \in [0, \epsilon]$. If the loss decreases or is completely flat, the gradient vector points in a misleading direction.
- **Method (Zero Backprop Passes):**
  1. Evaluate model loss at 3 interpolation points: $\alpha \in \{0.0, 0.5\epsilon, 1.0\epsilon\}$ using forward passes only.
  2. Record average loss across eligible clean-correct samples: $L_0, L_{0.5}, L_{1.0}$.
  3. Calculate slope: $\Delta L = L_{1.0} - L_0$.
- **Thresholds & Status:**
  - `SUSPICIOUS`: $L_{1.0} \le L_0 - 1e-4$ (loss decreases along the gradient direction) OR $|L_{1.0} - L_0| < 1e-6$ with clean loss $> 0.1$ (completely flat plateau).
  - `OK`: $L_{1.0} > L_0$.

---

### Signal 3: PGD Random Restart Sensitivity
- **Scientific Rationale:** In a shattered or non-smooth landscape, local optima trap gradient descent depending on initial random initialization.
- **Method (CPU Optimized):**
  1. Reuse the already computed baseline PGD ASR as Restart #1.
  2. Execute only $(R - 1)$ additional independent PGD runs (default $R = 2$ or $3$, 10 steps).
  3. Record ASR for each restart: $[S_1, S_2, \dots, S_R]$.
  4. Spread: $\text{Spread} = \max(S) - \min(S)$.
- **Thresholds & Status:**
  - `SUSPICIOUS`: $\text{Spread} \ge 20.0$ percentage points.
  - `OK`: $\text{Spread} < 20.0$ percentage points.

---

### Signal 4: Attack-Strength Consistency (Epsilon Schedule & Step Trajectory)
- **Scientific Rationale:** 
  - (Athalye et al., 2018): Attacks must not become significantly weaker when perturbation budget $\epsilon$ is increased.
  - Increasing PGD iterations should either improve or maintain ASR, not drop sharply.
- **Method (CPU Optimized via Trajectory Snapshotting):**
  1. **Epsilon Sweep:** Evaluate PGD at $\epsilon_{half} = 0.5\epsilon$ and $\epsilon_{double} = 2.0\epsilon$ (or $16/255$).
  2. **Step Snapshotting:** Rather than running separate PGD-5, PGD-10, PGD-20 attacks, run a single 20-step PGD attack and evaluate predictions at intermediate steps 5, 10, and 20.
  3. Record ASR at each setting.
- **Thresholds & Status:**
  - `SUSPICIOUS`: Drop $> 5.0$ percentage points when moving to a stronger budget ($\epsilon_{double} < \epsilon_{base} - 5\%$) or higher step count ($S_{20} < S_{10} - 5\%$).
  - `OK`: Monotonically non-decreasing within tolerance.

---

### Signal 5: FGSM vs Iterative PGD Discrepancy
- **Scientific Rationale:** A one-step attack (FGSM) should virtually never outperform a properly tuned iterative attack (PGD) bounded by the same $\epsilon$. If $\text{ASR}_{\text{FGSM}} > \text{ASR}_{\text{PGD}}$, gradients are likely misleading local linear approximations.
- **Method:**
  - Compare baseline `fgsm_attack_success_rate` and `pgd_attack_success_rate` directly from `evaluate_attack_ensemble`. Zero extra computation.
- **Thresholds & Status:**
  - `SUSPICIOUS`: $\text{ASR}_{\text{FGSM}} > \text{ASR}_{\text{PGD}} + 5.0$ percentage points.
  - `OK`: $\text{ASR}_{\text{PGD}} \ge \text{ASR}_{\text{FGSM}} - 5.0$ percentage points.

---

### Signal 6: Zero-Order (Gradient-Free) Random Search Check
- **Scientific Rationale:** If purely random noise fools the model more effectively than gradient-directed perturbations, gradient information is obfuscated.
- **Method (Fast Vectorized Forward Passes):**
  1. Sample $K = 25$ uniform perturbations $\delta_k \sim \mathcal{U}(-\epsilon, \epsilon)$, clamp to $[0, 1]$.
  2. Evaluate clean-correct samples on random noise.
  3. Record Random Search ASR ($\text{ASR}_{\text{rand}}$).
- **Thresholds & Status:**
  - `SUSPICIOUS`: $\text{ASR}_{\text{rand}} > \text{ASR}_{\text{PGD}} + 5.0$ percentage points.
  - `OK`: $\text{ASR}_{\text{PGD}} \ge \text{ASR}_{\text{rand}}$.
  - `NOT_RUN`: If disabled by user settings.

---

## 5. Result Schema & Status Semantics

The public function returns a pure Python dictionary containing only primitives, lists, and dicts (no PyTorch tensors):

```python
{
    "status": "WARNING",  # "WARNING", "NO_STRONG_SIGNAL", or "INCONCLUSIVE"
    "summary": "Potential gradient masking: 2 suspicious signals observed. Gradient-based evaluations may be unreliable.",
    "warning_reasons": [
        "1D Loss Landscape: Loss decreases along the gradient direction (shattered gradient signal).",
        "FGSM vs PGD Discrepancy: One-step FGSM achieved higher success than iterative PGD."
    ],
    "actionable_guidance": [
        "Evaluate using gradient-free attacks such as Square Attack or RayS.",
        "If using input discretization or preprocessing defenses, evaluate with BPDA (Backward Pass Differentiable Approximation)."
    ],
    "signals": {
        "gradient_health": {
            "status": "OK",  # "OK", "SUSPICIOUS", "ERROR", "NOT_AVAILABLE"
            "median_l2_norm": 0.0842,
            "median_mean_abs": 0.0012,
            "median_near_zero_fraction": 0.12,
            "non_finite": False,
            "details": "Input gradients have healthy non-zero magnitude and finite values."
        },
        "loss_landscape_ray": {
            "status": "SUSPICIOUS",
            "alphas": [0.0, 0.0157, 0.0314],
            "losses": [0.452, 0.410, 0.380],
            "slope": -0.072,
            "details": "Loss decreases along gradient ascent ray."
        },
        "pgd_restart_sensitivity": {
            "status": "OK",
            "restart_success_rates": [62.5, 68.75],
            "spread_percentage_points": 6.25,
            "details": "Restart spread is within normal range."
        },
        "attack_strength_consistency": {
            "status": "OK",
            "step_sweep": {
                "steps": [5, 10, 20],
                "success_rates": [50.0, 62.5, 68.75]
            },
            "epsilon_sweep": {
                "epsilons": [0.0157, 0.0314, 0.0627],
                "success_rates": [31.25, 62.5, 87.5]
            },
            "non_monotonic_drops": [],
            "details": "Attack success increases consistently with optimization effort and budget."
        },
        "fgsm_vs_pgd": {
            "status": "SUSPICIOUS",
            "fgsm_success_rate": 75.0,
            "pgd_success_rate": 62.5,
            "details": "One-step FGSM outperforms 10-step PGD by 12.5%."
        },
        "gradient_free_baseline": {
            "status": "OK",
            "random_search_success_rate": 12.5,
            "pgd_success_rate": 62.5,
            "details": "Gradient-based PGD substantially outperforms random perturbations."
        }
    },
    "metadata": {
        "sample_count": 16,
        "clean_correct_count": 14,
        "clean_accuracy": 87.5,
        "epsilon": 0.031372549,
        "pgd_steps": 10,
        "device": "cpu"
    },
    "limitations": [
        "Heuristic diagnostic: absence of warnings does not prove true adversarial robustness.",
        "Evaluations apply only to the tested threat model (Linf) and sample subset."
    ]
}
```

### Overall Status Rules:
- `INCONCLUSIVE`: Clean-correct count $N_{\text{clean\_correct}} = 0$, input validation error, or non-finite gradient crash.
- `WARNING`: At least 1 signal is `SUSPICIOUS` or `ERROR`.
- `NO_STRONG_SIGNAL`: All executed signals report `OK`.

---

## 6. Denominator Consistency & Safe Metric Math

To prevent invalid metrics or division-by-zero crashes:
1. **ASR Denominator:** Always conditioned exclusively on the subset of images that the model classifies correctly on clean inputs:
   $$\text{ASR} = 100 \times \frac{\sum \mathbb{I}(f(x) = y \land f(x_{\text{adv}}) \ne y)}{\sum \mathbb{I}(f(x) = y)}$$
2. **Zero Clean-Correct Guard:** If $\sum \mathbb{I}(f(x) = y) == 0$:
   - Set all ASR metrics to `None` (or `0.0` with explicit flag).
   - Return status `INCONCLUSIVE` with message: *"No clean samples were correctly classified; attack success rate is undefined."*
   - UI formatters must check `if val is not None` before executing `f"{val:.2f}%"` to avoid `TypeError`.

---

## 7. Streamlit Architecture & State Management (`app.py`)

### The State Management Solution
To ensure interactive sliders and separate buttons do not wipe out results, implement the following pattern in `app.py`:

```python
# Initialize session state stores if absent
if "cifar_eval_results" not in st.session_state:
    st.session_state["cifar_eval_results"] = None
if "cifar_eval_data" not in st.session_state:
    st.session_state["cifar_eval_data"] = None
if "gm_diag_results" not in st.session_state:
    st.session_state["gm_diag_results"] = None

# Step 1: Run CIFAR-10 Evaluation Button
if st.button("Run CIFAR-10 Evaluation", type="primary"):
    with st.spinner("Evaluating baseline attacks..."):
        # run evaluate_attack_ensemble
        st.session_state["cifar_eval_results"] = results
        st.session_state["cifar_eval_data"] = {"images": images, "labels": labels, "epsilon": dataset_epsilon}
        # Invalidate previous diagnostics when base evaluation changes
        st.session_state["gm_diag_results"] = None

# Step 2: Render base results if present in session state
if st.session_state["cifar_eval_results"] is not None:
    results = st.session_state["cifar_eval_results"]
    cached_data = st.session_state["cifar_eval_data"]
    
    # ... Render Performance, Worst-Case Analysis, Trust Flag, and Sample Inspector ...
    
    st.divider()
    st.subheader("🔬 Gradient-Masking Diagnostic Suite")
    st.write("Diagnose whether evaluation gradients are reliable or masked.")
    
    # Optional diagnostic configuration
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        diag_samples = st.selectbox("Diagnostic sample cap", [8, 16, 32], index=1)
    with col_d2:
        include_rand = st.checkbox("Include Random Search check", value=True)
        
    if st.button("Run Gradient-Masking Diagnostics", type="secondary"):
        with st.spinner("Running diagnostic signals (loss ray, restarts, strength sweep)..."):
            diag_results = analyze_gradient_masking(
                model=model,
                images=cached_data["images"],
                labels=cached_data["labels"],
                epsilon=cached_data["epsilon"],
                pgd_steps=dataset_pgd_steps,
                max_diagnostic_samples=diag_samples,
                baseline_pgd_asr=results["pgd_attack_success_rate"],
                baseline_fgsm_asr=results["fgsm_attack_success_rate"],
                run_random_search=include_rand,
            )
            st.session_state["gm_diag_results"] = diag_results

# Step 3: Render diagnostic results if present in session state
if st.session_state.get("gm_diag_results") is not None:
    render_gradient_masking_ui(st.session_state["gm_diag_results"])
```

### Visual UI Components in `render_gradient_masking_ui`:
1. **Status Banner:**
   - `WARNING` $\to$ `st.warning("⚠️ Potential Gradient Masking Detected")` with bulleted reasons.
   - `NO_STRONG_SIGNAL` $\to$ `st.success("✅ No Strong Masking Signal Found")`.
   - `INCONCLUSIVE` $\to$ `st.info("ℹ️ Evaluation Inconclusive")`.
2. **Key Metrics Row:**
   - 4 columns: Median $\|\nabla_x L\|_2$, Loss Ray Slope, PGD Restart Spread, FGSM vs PGD delta.
3. **Interactive Charts:**
   - 1D Loss Landscape Curve (`st.line_chart`).
   - Strength Trajectory Curve (ASR vs Steps & Epsilon).
4. **Signal Detail Expanders:**
   - Detailed expanders explaining each signal's measurements and scientific relevance.
5. **Actionable Remediation Box:**
   - Specific recommendations for the practitioner.
6. **Export Options:**
   - Download complete diagnostic results as JSON and CSV alongside existing evaluation exports.

---

## 8. Defenses Verification Lab Integration (Quantization Bridge)

ModelShield already includes an input quantization defense (`quantization_defense` in `defenses/defense_methods.py`). Input discretization shatters gradients because $\frac{\partial}{\partial x}\text{round}(x) = 0$ almost everywhere.

### Design Compatibility:
`analyze_gradient_masking` accepts any standard PyTorch `nn.Module`. In a future phase or advanced option, wrap the defended pipeline in a thin module:

```python
class DefendedModelWrapper(torch.nn.Module):
    def __init__(self, base_model, defense_fn):
        super().__init__()
        self.base_model = base_model
        self.defense_fn = defense_fn
        
    def forward(self, x):
        return self.base_model(self.defense_fn(x))
```

Passing `DefendedModelWrapper` into `analyze_gradient_masking` will immediately trigger `Signal 1 (Vanishing/Zero Norms)` and `Signal 2 (Flat Loss Ray)`, providing an empirical live demonstration of gradient masking in action!

---

## 9. Unit Testing Plan (`evaluation/test_gradient_masking.py`)

Tests must run in **under 5 seconds on CPU** and require **no network access, no RobustBench download, and no CIFAR-10 download**.

### Synthetic Test Fixtures:
1. **Linear Differentiable Model:** $f(x) = Wx + b$. Produces healthy, finite gradients with smooth loss curves. Verifies `NO_STRONG_SIGNAL` path.
2. **Zero-Gradient (Masked) Model:** $f(x) = W \cdot \text{detach}(x) + b$. Gradients are strictly zero. Verifies that Signal 1 and Signal 2 flag `SUSPICIOUS` and overall status is `WARNING`.
3. **Non-Finite Gradient Model:** Returns loss that creates `NaN` gradients. Verifies `ERROR` handling and non-crashing behavior.
4. **Non-Monotonic / Oscillating Model:** Simulates inverted gradients where loss decreases along the gradient ray. Verifies Signal 2.
5. **Zero Clean-Correct Fixture:** Labels rigged so clean accuracy is 0%. Verifies that ASR returns `None` without division-by-zero crashes, and status is `INCONCLUSIVE`.
6. **Output Purity Check:** Asserts that all returned dictionary values are standard Python types (`float`, `int`, `str`, `list`, `dict`, `bool`) and contains zero PyTorch tensors.

---

## 10. Implementation Sequence (Step-by-Step)

```text
  ┌────────────────────────────────────────────────────────┐
  │ Step 1: Pre-Flight Baseline & Verification             │
  │ - Verify Python environment and existing tests pass    │
  └──────────────────────────┬─────────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────────┐
  │ Step 2: Implement evaluation/gradient_masking.py       │
  │ - Build modular signal functions and orchestrator      │
  │ - Implement trajectory snapshotting & ray search       │
  └──────────────────────────┬─────────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────────┐
  │ Step 3: Implement evaluation/test_gradient_masking.py  │
  │ - Run pytest on synthetic models                       │
  │ - Test all 6 edge cases (finite, zero, NaN, empty)     │
  └──────────────────────────┬─────────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────────┐
  │ Step 4: Refactor app.py Session State Lifecycle        │
  │ - Store CIFAR-10 evaluation data in st.session_state   │
  │ - Add "Run Gradient-Masking Diagnostics" section       │
  └──────────────────────────┬─────────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────────┐
  │ Step 5: Build Streamlit Diagnostic UI & Visualization  │
  │ - Render status badges, metric columns, charts         │
  │ - Add JSON/CSV diagnostic download buttons             │
  └──────────────────────────┬─────────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────────┐
  │ Step 6: Regression Testing & Final Validation          │
  │ - Run full test suite & Streamlit smoke test           │
  └────────────────────────────────────────────────────────┘
```

---

## 11. Definition of Done Checklist

The gradient-masking diagnostic feature is considered complete when:

- [ ] `evaluation/gradient_masking.py` is implemented and independent of Streamlit.
- [ ] Incorporates 6 distinct signals: Gradient Health, 1D Loss Landscape Ray, PGD Restarts, Strength Consistency, FGSM vs PGD, and Zero-Order Random Search.
- [ ] Employs trajectory snapshotting and baseline PGD reuse to keep CPU runtime fast and interactive.
- [ ] Accurately computes ASR conditioned on clean-correct samples and safely handles zero clean-correct edge cases.
- [ ] Output schema is 100% JSON-serializable (no PyTorch tensors).
- [ ] Automated tests in `evaluation/test_gradient_masking.py` pass cleanly in $< 5$ seconds without external downloads.
- [ ] `app.py` preserves evaluation state across user interactions via `st.session_state`.
- [ ] Streamlit UI displays colored status banners, metric cards, 1D loss landscape chart, trajectory curves, and actionable guidance.
- [ ] Result export includes diagnostic metrics in CSV and JSON formats.
- [ ] Existing workspaces (Test Own Image, Create Adversarial Image, Defense Lab) remain fully operational with zero regressions.
- [ ] UI language remains scientifically calibrated (warns of potential masking without claiming absolute mathematical proof).
