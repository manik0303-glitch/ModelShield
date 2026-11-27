# RAM++ Red-Team Orchestrator

## Contribution

- RAM++ multi-label recognition interface
- Neural network activation visualizations
- Tag retention and tag-score reporting
- Interactive investigation dashboard
- Mission audit export and replay
- Investigation Copilot

## Runtime requirements

- Python 3.11
- PyTorch and torchvision
- Streamlit
- Plotly
- pandas, NumPy and Pillow
- Compatible RAM++ installation and pretrained checkpoint

## Integration status

The frontend and reporting components are published.
Mission backend and shared RAM++ dependency integration
require compatibility review before standalone execution.

## Scientific limitations

Results describe individual investigated images and recorded
evaluation conditions, not dataset-wide accuracy or certified robustness.

## Remaining backend dependencies

- orchestrator/engine.py
- orchestrator/ram_attacks.py
- model/ram_adapter.py
- evaluation/ram_metrics.py

The locally tested implementation uses these modules.
RAM++ runtime infrastructure and pretrained weights
must be configured separately.

Existing shared modules and teammate branches must
remain unchanged during integration.
