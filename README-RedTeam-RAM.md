# ModelShield - RAM++ Adaptive Red-Team Orchestrator

## Overview

The Adaptive Red-Team Orchestrator is a component of
the ModelShield semester project, supporting robustness
evaluation of RAM++ Swin-L image-tagging models.

## Contributions

- Interactive Streamlit investigation dashboard
- RAM++ multi-label model integration
- Adaptive evaluation orchestration
- Preflight validation and integrity checks
- Neural network activation visualizations
- Tag preservation and score comparison
- Investigation Copilot
- Audit reporting and replay

## Runtime Requirements

- Python 3.11
- PyTorch and torchvision
- Streamlit
- Plotly
- NumPy, pandas and Pillow
- Compatible RAM++ upstream installation
- RAM++ pretrained checkpoint

Model weights and virtual environments are maintained
separately and are not committed to this branch.

## Verification Status

- Source-code integration completed.
- Dashboard rendering tested locally.
- RAM++ baseline image recognition tested successfully.
- Fresh installation and full end-to-end reproducibility
  remain to be verified.

## Scientific Limitations

Single-image evaluation results must not be interpreted
as dataset-wide accuracy or certified model robustness.

## Integration

This contribution is maintained on an isolated feature
branch. Shared modules and existing teammate branches
have not been modified by the publishing workflow.
