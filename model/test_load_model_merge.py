import unittest
from unittest.mock import patch

import torch
from PIL import Image

from model.ram_wrapper import RAMPlusWrapper

class LoadModelMergeTests(unittest.TestCase):
    def test_load_model_uses_ram_plus_as_primary(self):
        import model.load_model as loader

        result = (object(), "cpu", ["tag"], torch.tensor([0.5]), object())
        with patch.object(loader, "load_ram_plus_model", return_value=result) as ram_loader:
            self.assertIs(loader.load_model(), result)
            ram_loader.assert_called_once_with()

    def test_ram_plus_loader_wraps_shared_model_and_returns_metadata(self):
        import model.load_model as loader

        class FakeRAM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tag_list = ["tag"]
                self.class_threshold = torch.tensor([0.5])
                self.num_class = 1

        base_model = FakeRAM()
        device = torch.device("cpu")
        settings = {"image_size": 384}

        with (
            patch.object(loader, "get_ram_plus_config", return_value=settings),
            patch.object(loader, "is_ram_available", return_value=(True, "")),
            patch.object(loader, "get_ram_model", return_value=(base_model, device)),
        ):
            model, returned_device, tags, thresholds, returned_transform = loader.load_ram_plus_model()

        self.assertIs(model.ram_model, base_model)
        self.assertEqual(returned_device, device)
        self.assertEqual(tags, ["tag"])
        self.assertIs(thresholds, base_model.class_threshold)
        transformed = returned_transform(Image.new("RGB", (4, 4), (255, 255, 255)))
        torch.testing.assert_close(transformed, torch.ones(3, 384, 384))

    def test_ram_plus_loader_fails_when_runtime_is_unavailable(self):
        import model.load_model as loader

        with (
            patch.object(loader, "get_ram_plus_config", return_value={"image_size": 384}),
            patch.object(loader, "is_ram_available", return_value=(False, "missing checkpoint")),
            patch.object(loader, "get_ram_model") as get_ram_model,
        ):
            with self.assertRaisesRegex(loader.ModelConfigurationError, "missing checkpoint"):
                loader.load_ram_plus_model()

        get_ram_model.assert_not_called()

    def test_local_cifar_checkpoint_requires_explicit_trust(self):
        import model.load_model as loader

        with patch.object(loader, "_try_load_cifar_checkpoint") as try_checkpoint:
            with self.assertRaisesRegex(
                loader.ModelConfigurationError,
                "trust_local_checkpoint=True",
            ):
                loader.load_cifar10_model(checkpoint_path="unverified.pt")

        try_checkpoint.assert_not_called()

    def test_ram_wrapper_normalizes_pixel_input_once_and_keeps_gradients(self):
        class CaptureEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_input = None

            def forward(self, image):
                self.last_input = image
                features = image.mean(dim=(1, 2, 3)).view(-1, 1, 1)
                return features.expand(-1, 1, 512)

        class TaggingHead(torch.nn.Module):
            def forward(self, encoder_embeds, **kwargs):
                return (encoder_embeds,)

        class FakeRAM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_class = 1
                self.visual_encoder = CaptureEncoder()
                self.image_proj = torch.nn.Identity()
                self.label_embed = torch.nn.Parameter(torch.ones(1, 512))
                self.reweight_scale = torch.nn.Parameter(torch.tensor(0.0))
                self.wordvec_proj = torch.nn.Identity()
                self.tagging_head = TaggingHead()
                self.fc = torch.nn.Linear(512, 1)

        base_model = FakeRAM()
        wrapper = RAMPlusWrapper(base_model)
        pixels = torch.rand(2, 3, 8, 8, requires_grad=True)

        logits = wrapper(pixels)
        logits.sum().backward()

        mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        torch.testing.assert_close(
            base_model.visual_encoder.last_input,
            (pixels.detach() - mean) / std,
        )
        self.assertIsNotNone(pixels.grad)


if __name__ == "__main__":
    unittest.main()
