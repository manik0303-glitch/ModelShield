# ModelShield --- Controlled Gradient-Masking Simulation Mode

## Implementation Plan for an AI Coding Agent

**Document type:** Implementation specification\
**Purpose:** Extend the already implemented ModelShield gradient-masking
diagnostic suite with an explicit, reproducible simulation mode that
deliberately introduces a gradient-obscuring condition and then runs the
existing detector against the resulting evidence.

------------------------------------------------------------------------

## 1. Objective

Add a dashboard option named **"Simulated Gradient Masking (Validation
Lab)"**. It must let a user deliberately run a controlled experiment
designed to make ordinary gradient-based attacks ineffective or
misleading, then evaluate whether ModelShield's existing
gradient-masking detector identifies the resulting evidence.

This is a validation feature, not a defense and not a mode that forces a
warning.

### Required end-to-end behavior

1.  The user runs or loads the normal CIFAR-10 evaluation.
2.  The user selects **Normal Evaluation** or **Simulated Gradient
    Masking**.
3.  In simulation mode, ModelShield builds a temporary experimental
    model pipeline that applies a non-differentiable input
    transformation, initially quantization.
4.  Standard FGSM/PGD are evaluated against that pipeline.
5.  BPDA-assisted PGD is evaluated against the same forward pipeline as
    a comparison attack. BPDA changes the backward gradient
    approximation only; the forward prediction must still use the actual
    quantization transformation.
6.  The same existing diagnostic decision logic is run on observed
    metrics. It must not receive the selected mode or expected label as
    a reason to choose its verdict.
7.  ModelShield displays measured metrics, detector verdict, and a
    separate validation assessment comparing the verdict with the
    experiment's independently known setup.
8.  The original model, model weights, CIFAR-10 data, labels, and normal
    evaluation results remain unchanged.

**Never hardcode `WARNING` merely because simulation mode is enabled.**
The simulation configuration and its expected test condition must be
separate from the detector's inputs.

------------------------------------------------------------------------

## 2. Scope and non-goals

### In scope

-   A new, isolated simulation module.
-   A BPDA-assisted PGD implementation, if one does not already exist.
-   A UI mode selector and separate "Run Simulation" action.
-   Reuse of the existing gradient-masking detector and attack
    infrastructure where compatible.
-   Comparison of standard gradient attacks with BPDA-assisted PGD.
-   Clean-accuracy and pipeline-integrity checks.
-   Unit tests that do not download CIFAR-10 or RobustBench weights.
-   A clear "inconclusive" result when the experiment does not establish
    the intended behavior.

### Out of scope

-   Replacing or retraining the production/evaluation model.
-   Editing CIFAR-10 images, labels, or dataset configuration to
    manufacture a result.
-   Changing the original model weights.
-   Making the detector always return a warning.
-   Claiming that one simulated case proves detection of every
    real-world gradient-masking mechanism.
-   Adding C&W, AutoAttack, or another large attack suite unless the
    existing project already supports it and it is required for a later
    extension.

------------------------------------------------------------------------

## 3. Mandatory pre-flight repository inspection

Before editing, inspect the current repository. The earlier diagnostic
plan may already have been implemented, so do not recreate or overwrite
existing functionality.

Inspect at minimum:

-   `app.py`
-   `evaluation/gradient_masking.py`
-   existing `attacks/fgsm.py`, `attacks/pgd.py`, and
    `attacks/ensemble.py`
-   `defenses/defense_methods.py` and the existing
    `quantization_defense`, if present
-   existing tests and dependency files (`requirements.txt`,
    `pyproject.toml`, etc.)

Record the actual function signatures and tensor conventions before
coding:

-   expected input range and normalization
-   model output format (logits versus probabilities)
-   device placement
-   epsilon/alpha conventions
-   ASR definition and denominator
-   whether model wrappers and attack functions preserve gradients
-   how diagnostic results are stored in `st.session_state`

Do not assume that filenames or signatures in this document exactly
match the current repository. Adapt to the real code while preserving
the behavior and public interfaces of already implemented features.

Run the current test suite before changes and record the baseline. If a
pre-existing test fails, identify it separately rather than attributing
it to this feature.

------------------------------------------------------------------------

## 4. Recommended design

Add a new module:

``` text
evaluation/
    masking_simulation.py
attacks/
    bpda_pgd.py             # only if equivalent implementation does not exist
tests/
    test_masking_simulation.py
```

Modify `app.py` only to expose the new mode and render its results.
Update the existing diagnostic module only where necessary to support a
clean, reusable analysis interface.

### Separation of responsibilities

-   `evaluation/masking_simulation.py`: constructs and runs the
    controlled experiment; owns the simulation settings and simulation
    metadata.
-   `attacks/bpda_pgd.py`: implements BPDA-assisted PGD, or place the
    implementation alongside existing attack abstractions if that better
    matches the repository.
-   Existing `evaluation/gradient_masking.py`: analyzes measured metrics
    and produces its ordinary diagnostic status and signal explanations.
-   `app.py`: handles mode selection, progress, session state, charts,
    and report display. Do not place attack mathematics or diagnostic
    thresholds in the UI layer.
-   `tests/test_masking_simulation.py`: validates simulation behavior,
    attack comparison, isolation, and failure handling.

Use the project's established naming and architecture if it differs. Do
not create duplicate attack or diagnostic implementations.

------------------------------------------------------------------------

## 5. Simulation mechanism: quantization in a temporary pipeline

### 5.1 Why quantization is the first experiment

Use a temporary input quantization operation. Quantization maps a
continuous pixel value to one of a finite number of levels. A hard
rounding operation has zero or unusable ordinary derivative almost
everywhere, so backpropagation through it can provide an uninformative
gradient even when the forward pipeline still produces predictions.

This is a plausible controlled test of gradient obfuscation, but it is
not automatically proof of a successful masking experiment. The
experiment must verify the actual gradient behavior and attack outcomes.

### 5.2 Isolation rule

The simulation must create a separate wrapper or callable pipeline
around the existing model. It must not mutate the original model, its
weights, the original dataset, or the normal evaluation configuration.

Conceptual structure:

``` text
x
└── temporary quantization transform
    └── existing model (same weights)
        └── logits
```

Implement the transform using the project's expected input range. If the
model expects normalized inputs, place quantization at the correct point
in the preprocessing pipeline and document it. Do not quantize
normalized tensors as if they were raw `[0, 1]` pixels unless that is
correct for the existing preprocessing.

Provide a configurable number of quantization levels, with a
conservative default chosen after inspecting the current input pipeline.
Do not assume one level count will preserve clean accuracy for every
model.

### 5.3 Forward-pass correctness

The forward path used by both standard attacks and BPDA-assisted PGD
must apply the exact same quantization operation and produce the same
predictions for the same inputs. BPDA must not bypass quantization
during forward inference.

Use a differentiable approximation only for the backward pass. One
acceptable pattern is a custom autograd function whose forward computes
hard quantization and whose backward passes the upstream gradient
through unchanged (straight-through BPDA approximation). Keep this
implementation local to the simulation/attack code; do not globally
replace PyTorch gradients or monkey-patch model behavior.

Important: the straight-through estimator is an approximation, not the
true derivative of rounding. Label the experiment as
BPDA/straight-through gradient approximation and state the limitation in
the report.

### 5.4 Validate the transformation

Before running full attacks, run a small smoke check on a small batch:

-   transformed tensor has the expected shape, dtype, device, and finite
    values
-   values remain within the expected range
-   hard quantization actually changes some non-grid-aligned inputs
-   standard autograd through hard quantization yields the expected
    missing/near-zero gradient behavior for the tested path
-   BPDA backward approximation yields finite, usable input gradients
-   the model still produces correctly shaped finite logits

If these checks fail, stop the experiment and return `INCONCLUSIVE` with
a useful error message. Do not proceed and then label the experiment
successful.

------------------------------------------------------------------------

## 6. BPDA-assisted PGD requirements

Implement BPDA-assisted PGD only if the project does not already contain
a compatible version.

### Requirements

-   Follow the same threat model as standard PGD: same selected images,
    labels, norm (`L∞` if that is the existing project convention),
    epsilon, step size, clipping range, and number of steps.
-   Use the same quantized forward pipeline as the standard attack.
-   Use the BPDA approximation only in backward gradient calculation.
-   Project every update back into the epsilon ball around the original
    input and clamp to the valid input range.
-   Detach iterative attack state appropriately to avoid retaining
    computation graphs.
-   Do not update model parameters; only calculate gradients with
    respect to the input.
-   Use `model.eval()` during the experiment and restore any prior
    training/evaluation state if the project expects that state to be
    preserved.
-   Use `torch.autograd.grad` or the repository's established safe
    input-gradient pattern; avoid relying on accumulated parameter
    `.grad` values.
-   Handle missing/non-finite gradients without crashing. Record the
    failure and return an inconclusive experiment if the comparison
    cannot be trusted.
-   Track the strongest successful adversarial candidate encountered,
    rather than only the last PGD iterate, where compatible with the
    project's attack design.

### Metric consistency

Compute attack success rate (ASR) on the same denominator for all
compared attacks: images that the evaluated pipeline classifies
correctly on clean inputs. Report the denominator count explicitly.

Use consistent attack budgets and sample identities. Do not compare a
standard PGD ASR on one sample subset against BPDA ASR on a different
subset.

If the quantized pipeline has zero clean-correct samples, ASR is
undefined and the experiment must be `INCONCLUSIVE`. Do not present zero
as a valid ASR in this case.

------------------------------------------------------------------------

## 7. Experiment protocol and success criteria

Run the following sequence in this order.

### Phase A --- Capture normal baseline

Use the existing evaluation's cached images and labels where possible.
Capture the standard model's clean predictions and the existing baseline
attack metrics. Do not silently rerun or replace the existing normal
evaluation.

### Phase B --- Build and validate simulated pipeline

Construct the quantized wrapper or pipeline without mutating the
original model. Run all smoke checks from Section 5.4.

### Phase C --- Measure clean behavior

Evaluate the simulated pipeline on the selected sample subset before
attacks. Report:

-   clean accuracy for the original pipeline on the selected samples
-   clean accuracy for the quantized pipeline on the same samples
-   absolute clean-accuracy change in percentage points
-   number of clean-correct samples eligible for ASR

Make the clean-accuracy impact visible. If the transformation destroys
useful classification behavior or leaves too few clean-correct examples,
return `INCONCLUSIVE` rather than calling the result a successful
demonstration.

Choose a configurable minimum eligible sample count. A small demo can
use a low sample cap for speed, but the UI must show the sample count
and warn that small samples produce noisy estimates. The minimum should
be explicit and tested, not hidden.

### Phase D --- Run attacks

On the same eligible images and with identical budgets, run:

1.  Standard FGSM through the hard-quantized forward pipeline.
2.  Standard PGD through the hard-quantized forward pipeline.
3.  BPDA-assisted PGD through the same hard-quantized forward pipeline
    with the approximate backward pass.

Reuse compatible existing FGSM/PGD implementations. If they cannot
accept a wrapped model without changes, add the smallest
backwards-compatible extension or a simulation-specific adapter. Do not
break existing attack calls.

Record per attack:

-   ASR percentage or `None` if undefined
-   successful-example count and eligible denominator
-   epsilon, alpha, steps, norm and clipping bounds
-   gradient validity and any attack errors
-   elapsed time, if runtime measurement is already supported or can be
    added cleanly

### Phase E --- Run the existing detector

Feed the measured simulation results into the existing gradient-masking
detector using the same decision rules used for normal evaluation.

The detector must not receive `simulation_expected_to_warn=True`, a
hardcoded expected status, or the simulation mode name as an input to
its verdict logic. Simulation metadata may be included in the final
report, but must not affect detector thresholds or status.

If the existing detector API only accepts a model and inputs, call it on
the simulated wrapper and sample batch using its ordinary interface. If
necessary, add a narrow, backward-compatible optional parameter or
result adapter to pass already measured attack results and avoid
duplicating expensive work. Document that change and preserve normal
callers.

Do not claim that the detector necessarily must raise `WARNING`.
Instead, use the actual result. If the configured detector is not
designed to compare standard PGD with BPDA-assisted PGD, report that the
comparison provides evidence but the detector has not yet consumed that
signal; only add a new detector signal if it can be defined and tested
rigorously.

### Phase F --- Independent validation verdict

Keep a separate **simulation validation** result distinct from the
detector verdict.

This validator can use the experiment's known setup and observed
measurements to answer: "Did this run successfully establish the
intended controlled condition?" It should check evidence such as:

-   hard quantization was active in the forward path
-   ordinary gradients were demonstrably uninformative on eligible
    inputs
-   BPDA gradients were finite and usable
-   clean accuracy/sample count remained sufficient
-   BPDA-assisted PGD found more adversarial examples than standard PGD
    by a configurable minimum margin, or another predeclared empirical
    criterion was met

Do not require the BPDA attack to win on every run or every sample.
Define a configurable success margin before the run; do not tune it
after seeing results to force a pass. If the expected difference is not
observed, mark the simulation `INCONCLUSIVE` or
`CONDITION_NOT_DEMONSTRATED`, preserve the raw results, and explain why.

The validator is allowed to know the experiment configuration because
its purpose is to validate the test fixture. The detector is not allowed
to use that expected outcome to decide its own status.

------------------------------------------------------------------------

## 8. Result contract

Return JSON-serializable primitives only from the simulation runner.
Never store raw tensors in the result dictionary.

Suggested result shape (adapt to existing project conventions):

``` python
{
    "mode": "simulated_gradient_masking",
    "simulation_validation": {
        "status": "VALIDATED",  # VALIDATED, INCONCLUSIVE, CONDITION_NOT_DEMONSTRATED, ERROR
        "checks": [
            {
                "name": "hard_quantization_active",
                "passed": True,
                "details": "..."
            },
            {
                "name": "ordinary_gradient_obscured",
                "passed": True,
                "details": "..."
            },
            {
                "name": "bpda_gradient_usable",
                "passed": True,
                "details": "..."
            },
            {
                "name": "clean_accuracy_acceptable",
                "passed": True,
                "details": "..."
            },
            {
                "name": "bpda_attack_advantage",
                "passed": False,
                "details": "..."
            }
        ],
        "summary": "..."
    },
    "detector_result": {
        # Existing detector schema, unchanged where possible.
        "status": "WARNING",
        "warning_reasons": [],
        "signals": {}
    },
    "measurements": {
        "sample_count": 16,
        "eligible_clean_correct_count": 12,
        "original_clean_accuracy": None,
        "simulated_clean_accuracy": None,
        "clean_accuracy_drop_percentage_points": None,
        "attacks": {
            "fgsm": {"asr": None, "success_count": 0, "denominator": 12},
            "pgd": {"asr": None, "success_count": 0, "denominator": 12},
            "bpda_pgd": {"asr": None, "success_count": 0, "denominator": 12}
        },
        "gradient_checks": {
            "ordinary_gradient_norm": None,
            "ordinary_near_zero_fraction": None,
            "bpda_gradient_norm": None,
            "bpda_gradient_finite": True
        }
    },
    "configuration": {
        "quantization_levels": 16,
        "epsilon": None,
        "alpha": None,
        "pgd_steps": None,
        "sample_count_requested": 16,
        "random_seed": 123
    },
    "limitations": [
        "This is a controlled quantization-based experiment, not proof of detection for every masking mechanism."
    ]
}
```

Do not copy illustrative numerical values into runtime output. All
metric values must come from actual measurements. Ensure `None` is
handled safely by the UI and serialized as JSON `null`.

Keep the meanings separate:

-   **Detector verdict:** `WARNING`, `NO_STRONG_SIGNAL`, or
    `INCONCLUSIVE` according to the existing detector's evidence.
-   **Simulation validation:** `VALIDATED`, `INCONCLUSIVE`,
    `CONDITION_NOT_DEMONSTRATED`, or `ERROR` according to the experiment
    checks.

It is possible for the simulation to be validated while the detector
returns `NO_STRONG_SIGNAL`. That is an important detector test
failure/coverage finding, not a reason to overwrite the detector's
verdict. Report it honestly.

------------------------------------------------------------------------

## 9. Streamlit UI requirements

Add a dedicated section or tab named **Gradient Masking Lab** without
replacing the current diagnostic section.

### Controls

-   Mode selector: `Normal Evaluation` / `Simulated Gradient Masking`.
-   Quantization levels: use a short, documented set of allowed values
    and a sensible default after inspecting the input pipeline.
-   Diagnostic sample cap: reuse existing diagnostic settings if
    present.
-   Random seed: default to a fixed seed for reproducibility; allow
    changing it if useful.
-   "Run Simulated Gradient-Masking Test" button.
-   "Clear Simulation Results" button only if consistent with existing
    UI conventions.

Do not offer controls that imply the experiment is scientifically valid
under arbitrary settings. Display epsilon, step size, steps, norm and
input range, and keep them aligned with existing attack settings by
default.

### Results presentation

Display these as distinct panels:

1.  **Simulation validation status** --- whether the controlled
    condition was established.
2.  **Detector verdict** --- the result from the existing detector,
    unmodified.
3.  **Clean accuracy impact** --- original versus simulated pipeline.
4.  **Attack comparison** --- FGSM, standard PGD and BPDA-assisted PGD
    ASR with the same denominator.
5.  **Gradient comparison** --- ordinary versus BPDA gradient
    norm/finite status.
6.  **Why the detector reached its verdict** --- existing signal details
    and warning reasons.
7.  **Interpretation and limitations** --- clearly label the result as
    simulated.

Use neutral, precise language. Suggested copy:

-   `VALIDATED`: "The configured quantization experiment produced the
    expected gradient-obscuring behavior under the checks shown below."
-   `CONDITION_NOT_DEMONSTRATED`: "The configured experiment did not
    establish the expected masking evidence. Review the measurements; no
    detector result has been forced."
-   Detector `WARNING`: "The detector found suspicious evidence in this
    run."
-   Detector `NO_STRONG_SIGNAL`: "The detector did not find sufficient
    evidence to trigger a warning in this run. This does not prove that
    masking is absent."
-   Detector `INCONCLUSIVE`: "The detector could not assess the
    configured checks reliably."

Never merge the simulation validation badge and detector verdict into a
single status. Do not display example or hardcoded metrics as though
they were measured.

### Session state

Follow the existing session-state approach. Store simulation output
separately, e.g. `st.session_state["gm_simulation_results"]`.

-   A new simulation run replaces only the prior simulation result.
-   Changing a UI widget must not erase normal evaluation results.
-   Re-running the normal evaluation should invalidate any simulation
    result whose cached images/model settings are now stale.
-   Avoid storing unnecessary GPU tensors in session state. Prefer
    reusing the existing cached dataset tensors and store only
    serializable result data after the run.
-   Handle missing cached baseline data with a clear instruction to run
    the base evaluation first, or explicitly load a supported dataset
    subset. Do not silently start a separate download.

------------------------------------------------------------------------

## 10. Test plan

Tests must run offline and should use tiny synthetic PyTorch models and
small in-memory tensors. Do not download CIFAR-10 or RobustBench
weights.

### Required tests

1.  **Hard quantization forward test**
    -   Shape, dtype, device and output range are preserved.
    -   Same input gives deterministic output for a fixed configuration.
    -   Original model parameters remain unchanged.
2.  **Gradient-obscuring test**
    -   Standard backpropagation through hard quantization yields the
        expected zero/missing gradient for the tested path.
    -   The test must distinguish a genuinely zero gradient from an
        exception or disconnected computation graph.
3.  **BPDA backward test**
    -   BPDA forward output equals hard-quantization forward output for
        the same input.
    -   BPDA backward produces finite input gradients.
    -   No model parameters are updated.
4.  **Attack constraints**
    -   Standard and BPDA PGD outputs remain within the configured
        epsilon ball and valid input range.
    -   Attack functions do not mutate the original images.
    -   Identical settings and seed are reproducible where deterministic
        operations allow.
5.  **Metric consistency**
    -   All ASRs use the same eligible clean-correct denominator.
    -   Zero eligible samples returns undefined ASR and an inconclusive
        result, not a divide-by-zero or fabricated zero.
6.  **Detector independence**
    -   The detector receives measured inputs/results only.
    -   Changing simulation metadata or the validator's expected outcome
        does not change the detector verdict for identical evidence.
    -   There is no `if simulation_mode: status = "WARNING"` shortcut.
7.  **Simulation validator**
    -   A successful synthetic fixture can validate the expected
        gradient condition.
    -   A fixture where BPDA provides no advantage is not automatically
        marked as validated if attack advantage is a required criterion.
    -   Failure to meet a criterion returns `CONDITION_NOT_DEMONSTRATED`
        or `INCONCLUSIVE` with a reason.
8.  **State and regression**
    -   Simulation runs do not alter normal evaluation results, model
        weights, labels or dataset configuration.
    -   Existing diagnostic tests pass.
    -   Existing Streamlit sections continue to render and run.
    -   Clearing simulation results does not clear normal results.
9.  **Result serialization**
    -   The output can be JSON-serialized without tensors, NumPy
        scalars, device objects or non-finite numbers.
    -   Missing values are represented consistently as `None`/JSON
        `null`.

Avoid brittle tests that assert a particular ASR on random synthetic
data. Test invariants and controlled behaviors instead.

------------------------------------------------------------------------

## 11. Runtime and resource safeguards

-   Start with a small user-visible sample cap, consistent with the
    existing diagnostic's CPU constraints.
-   Run under `torch.no_grad()` for forward-only metric calculations,
    but enable gradients for the attack steps.
-   Do not wrap gradient-based attack sections in `torch.no_grad()`.
-   Use batches and device placement consistent with the existing code.
-   Release temporary tensors/references after the run and avoid caching
    attack graphs.
-   Show a Streamlit spinner or progress message during attack
    execution.
-   Catch exceptions at the UI boundary, log useful developer details,
    and show a concise user-facing error.
-   Do not silently fall back from BPDA to standard PGD while labeling
    the result BPDA.
-   If an attack fails, report it as unavailable/failed and mark the
    relevant validation inconclusive.

------------------------------------------------------------------------

## 12. Implementation sequence

Follow this order and finish each stage before proceeding.

### Step 1 --- Inspect and baseline

-   Inspect actual repository files and current implementation.
-   Run existing tests.
-   Record current attack APIs, input range, preprocessing,
    session-state keys and result schema.
-   Identify whether quantization and BPDA utilities already exist.

### Step 2 --- Implement isolated quantization pipeline

-   Add a temporary wrapper/transform without changing existing
    defaults.
-   Ensure the forward path is identical for standard attacks and
    BPDA-assisted PGD.
-   Add smoke tests for range, shape, finite logits and gradients.

### Step 3 --- Implement or adapt BPDA-assisted PGD

-   Match existing PGD constraints and API conventions.
-   Verify forward parity and usable backward gradients.
-   Add unit tests for epsilon projection, clipping and no parameter
    mutation.

### Step 4 --- Implement experiment runner

-   Add a pure Python runner in `evaluation/masking_simulation.py`.
-   Measure clean accuracy, gradient health, attack success and BPDA
    comparison.
-   Keep detector output and simulation validation separate.
-   Return only serializable data.

### Step 5 --- Connect existing detector

-   Invoke the existing detector on the simulated pipeline/evidence.
-   Reuse already computed results where possible.
-   Add a detector signal only if necessary, clearly defined and tested.
-   Do not hardcode a status based on the selected mode.

### Step 6 --- Add UI and session state

-   Add the Gradient Masking Lab controls and report.
-   Preserve normal evaluation state.
-   Display validation and detector status separately.
-   Label every result as simulated and show the experiment settings.

### Step 7 --- Test and regression-check

-   Run the new offline tests.
-   Run the existing full test suite.
-   Smoke-test normal mode and simulation mode in Streamlit.
-   Verify original model parameters and cached normal results are
    unchanged.
-   Fix regressions before marking complete.

### Step 8 --- Document findings

-   Document the simulation mechanism, settings, metrics, known
    limitations and how to reproduce it.
-   Report any criterion that could not be validated; do not conceal
    inconclusive results.

------------------------------------------------------------------------

## 13. Definition of done

-   [ ] Existing code was inspected before edits; no existing feature
    was unnecessarily duplicated.
-   [ ] A dedicated simulation mode is available in the Streamlit UI.
-   [ ] The simulation introduces a real, controlled quantization-based
    gradient-obscuring condition in a separate pipeline.
-   [ ] Original CIFAR-10 data, labels, model weights and normal
    configuration remain unchanged.
-   [ ] Standard FGSM/PGD and BPDA-assisted PGD use the same samples,
    forward pipeline and threat model.
-   [ ] BPDA uses an approximate backward gradient while preserving the
    true quantized forward pass.
-   [ ] Clean accuracy impact and eligible denominator are shown.
-   [ ] The existing detector runs using evidence, not a hardcoded
    simulation-mode rule.
-   [ ] Detector verdict and simulation validation status are shown
    separately.
-   [ ] Failure to demonstrate masking is reported honestly as
    inconclusive/not demonstrated.
-   [ ] Unit tests cover forward behavior, gradients, attack
    constraints, denominator consistency, independence and regression.
-   [ ] New and existing tests pass, or any pre-existing failures are
    documented.
-   [ ] The normal ModelShield workflow continues to work.

------------------------------------------------------------------------

## 14. Scientific interpretation and limitations

This lab validates detection behavior against one controlled
quantization-based gradient-obscuring mechanism. It does not prove that
the detector detects every kind of gradient masking, and it does not
prove that the original CIFAR-10 model is masked.

A warning is not proof of masking. A lack of warning is not proof of
robustness. BPDA-assisted PGD is a useful comparison method for certain
non-differentiable transformations, but it is not a universal
ground-truth attack.

The quality of the demonstration depends on reporting all relevant
observations honestly: clean accuracy, gradient statistics, attack
budgets, ASRs, sample counts, validation criteria, detector verdict and
limitations. The goal is a reproducible experiment that can reveal a
detector coverage gap---not a guaranteed warning.
