"""models.py

Model builder for the three mandated architectures (protocol section 11.8),
via timm, with a single-logit head (BCEWithLogitsLoss) and two-stage
transfer-learning freeze control (protocol 11.1).

Architectures (config key -> timm name):
    densenet121     -> densenet121
    efficientnet_b0 -> efficientnet_b0
    swin_tiny       -> swin_tiny_patch4_window7_224
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


def build_model(architecture_key: str, cfg: dict, seed: int) -> nn.Module:
    """Create a pretrained-ImageNet model with a fresh 1-logit classifier.

    The classification head is (re)initialised by timm at create time; we seed
    torch immediately before creation so head init is reproducible per run
    (protocol 11.8: "classification head diinisialisasi berdasarkan seed run").
    """
    arch_map = cfg["architectures"]
    if architecture_key not in arch_map:
        raise ValueError(
            f"Unknown architecture '{architecture_key}'. "
            f"Valid: {list(arch_map)}"
        )
    timm_name = arch_map[architecture_key]

    torch.manual_seed(seed)  # reproducible head initialisation
    model = timm.create_model(timm_name, pretrained=True, num_classes=1)
    return model


def classifier_parameter_ids(model: nn.Module) -> set[int]:
    """IDs of parameters belonging to the final classifier head."""
    head = model.get_classifier()
    return {id(p) for p in head.parameters()}


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Stage 1 (frozen=True): only the classifier head trains.
    Stage 2 (frozen=False): the whole backbone is unfrozen.
    """
    head_ids = classifier_parameter_ids(model)
    for p in model.parameters():
        if id(p) in head_ids:
            p.requires_grad = True
        else:
            p.requires_grad = not frozen


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
