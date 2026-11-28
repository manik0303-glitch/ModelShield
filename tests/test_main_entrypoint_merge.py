import contextlib
import io
from pathlib import Path
import runpy
import sys
import types
from unittest.mock import patch


def test_main_uses_ram_plus_loader_contract():
    class FakeModel:
        pass

    model_package = types.ModuleType("model")
    model_package.__path__ = []
    loader_module = types.ModuleType("model.load_model")
    loader_module.load_ram_plus_model = lambda: (
        FakeModel(),
        "cpu",
        ["tag"],
        [0.5],
        object(),
    )
    output = io.StringIO()
    entrypoint = Path(__file__).resolve().parents[1] / "main.py"

    with patch.dict(
        sys.modules,
        {"model": model_package, "model.load_model": loader_module},
    ):
        namespace = runpy.run_path(str(entrypoint), run_name="modelshield_test")
        with contextlib.redirect_stdout(output):
            namespace["main"]()

    assert "Model loaded successfully!" in output.getvalue()
    assert "Model: FakeModel" in output.getvalue()
    assert "Device: cpu" in output.getvalue()
    assert "Tag vocabulary size: 1" in output.getvalue()
