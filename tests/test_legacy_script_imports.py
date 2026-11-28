import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from model import dataset, load_model


@pytest.mark.parametrize(
    "filename",
    [
        "test_defense_evaluation.py",
        "test_ensemble.py",
        "test_fgsm.py",
        "test_model.py",
        "test_pgd.py",
        "testdataset.py",
    ],
)
def test_legacy_script_import_does_not_load_cifar(filename, monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("CIFAR resources must not be loaded on import")

    monkeypatch.setitem(sys.modules, "robustbench", None)
    monkeypatch.setattr(load_model, "load_cifar10_model", unexpected_load)
    monkeypatch.setattr(dataset, "load_cifar10_dataset", unexpected_load)
    path = Path(__file__).resolve().parents[1] / "model" / filename
    spec = importlib.util.spec_from_file_location("legacy_script_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)

    robustbench = ModuleType("robustbench")
    robustbench_data = ModuleType("robustbench.data")
    robustbench_data.load_cifar10 = unexpected_load
    monkeypatch.setitem(sys.modules, "robustbench", robustbench)
    monkeypatch.setitem(sys.modules, "robustbench.data", robustbench_data)
    with pytest.raises(AssertionError, match="CIFAR resources"):
        module.main()
