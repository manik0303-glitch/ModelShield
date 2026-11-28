import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List

from model.ram_adapter import normalize_pixels


class RAMPlusWrapper(nn.Module):
    """
    Standard PyTorch wrapper for RAM++ (Recognize Anything Plus Model).

    Bypasses string generation to output raw (B, num_tags) logits directly,
    ensuring standard PyTorch autograd gradients flow properly back to the
    input image for adversarial attacks and gradient masking diagnostics.

    Forward:
        Input:  raw pixel tensor of shape (B, 3, H, W) in [0, 1]
        Output: logits tensor of shape (B, num_tags)
    """

    def __init__(self, ram_model: nn.Module, image_size: int = 384):
        super().__init__()
        self.ram_model = ram_model
        self.image_size = image_size
        self.num_class = getattr(ram_model, "num_class", None)
        self.tag_list = getattr(ram_model, "tag_list", None)
        self.class_threshold = getattr(ram_model, "class_threshold", None)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Differentiable forward pass returning raw logits (B, num_tags).
        """
        image = normalize_pixels(image)
        model = self.ram_model
        image_embeds = model.image_proj(model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)

        image_cls_embeds = image_embeds[:, 0, :]
        bs = image_embeds.shape[0]
        des_per_class = int(model.label_embed.shape[0] / model.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
        reweight_scale = model.reweight_scale.exp()
        logits_per_image = reweight_scale * image_cls_embeds @ model.label_embed.t()
        logits_per_image = logits_per_image.view(bs, -1, des_per_class)

        weight_normalized = F.softmax(logits_per_image, dim=2)
        reshaped_value = model.label_embed.view(-1, des_per_class, 512)

        # Vectorized batch computation without Python loop
        # weight_normalized: (bs, num_class, des_per_class)
        # reshaped_value: (num_class, des_per_class, 512)
        # einsum b c d, c d f -> b c f
        label_embed_reweight = torch.einsum("bcd,cdf->bcf", weight_normalized, reshaped_value)
        label_embed = F.relu(model.wordvec_proj(label_embed_reweight))

        tagging_embed = model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )

        logits = model.fc(tagging_embed[0]).squeeze(-1)
        return logits
