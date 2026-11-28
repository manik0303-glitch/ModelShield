import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import evaluate_clean_accuracy


class TestCleanAccuracy(unittest.TestCase):
    def test_counts_all_batches(self):
        outputs = torch.tensor([[2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 3.0]])
        labels = torch.tensor([0, 1, 1, 1])
        loader = DataLoader(TensorDataset(outputs, labels), batch_size=2)

        self.assertEqual(
            evaluate_clean_accuracy(torch.nn.Identity(), loader, torch.device("cpu")),
            75.0,
        )

    def test_rejects_empty_dataset(self):
        loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))

        with self.assertRaisesRegex(ValueError, "evaluation dataset is empty"):
            evaluate_clean_accuracy(torch.nn.Identity(), loader, torch.device("cpu"))
