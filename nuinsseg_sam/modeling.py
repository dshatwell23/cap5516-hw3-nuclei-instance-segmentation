from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .runtime import import_sam_components


TASK_HEAD_PREFIXES = (
    "mask_decoder.mask_tokens",
    "mask_decoder.iou_token",
    "mask_decoder.output_hypernetworks_mlps",
    "mask_decoder.iou_prediction_head",
)


def build_finetune_namespace(config: ExperimentConfig) -> SimpleNamespace:
    return SimpleNamespace(
        image_size=config.image_size,
        devices=[0, 1],
        if_encoder_lora_layer=config.enable_encoder_lora,
        if_decoder_lora_layer=config.enable_decoder_lora,
        encoder_lora_layer=config.encoder_lora_layers,
    )


def _is_trainable_parameter(name: str) -> bool:
    lora_markers = ("linear_a_", "linear_b_", ".w_a.", ".w_b.")
    if any(marker in name for marker in lora_markers):
        return True
    return name.startswith(TASK_HEAD_PREFIXES)


def freeze_for_lora_finetuning(model: torch.nn.Module) -> None:
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if _is_trainable_parameter(name):
            parameter.requires_grad = True


def count_parameters(model: torch.nn.Module) -> Dict[str, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": float((trainable / total) * 100.0 if total else 0.0),
    }


def build_model(config: ExperimentConfig, device: torch.device) -> tuple[torch.nn.Module, Dict[str, float]]:
    checkpoint_path = config.resolved_ckpt()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "MobileSAM checkpoint not found. Provide --mobile-sam-ckpt pointing to mobile_sam.pt."
        )

    sam_model_registry, LoRA_Sam = import_sam_components()
    sam_args = build_finetune_namespace(config)
    model = sam_model_registry[config.arch](sam_args, checkpoint=str(checkpoint_path), num_classes=1)
    model = LoRA_Sam(sam_args, model, r=config.lora_rank).sam
    freeze_for_lora_finetuning(model)
    summary = count_parameters(model)
    model.to(device)
    return model, summary


def forward_logits(
    model: torch.nn.Module, images: torch.Tensor, target_hw: Tuple[int, int]
) -> torch.Tensor:
    preprocessed = torch.stack([model.preprocess(image) for image in images], dim=0)
    image_embeddings = model.image_encoder(preprocessed)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
    )
    batch_size = images.shape[0]
    sparse_embeddings = sparse_embeddings.expand(batch_size, -1, -1)
    dense_embeddings = dense_embeddings.expand(batch_size, -1, -1, -1)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return F.interpolate(low_res_logits, size=target_hw, mode="bilinear", align_corners=False)
