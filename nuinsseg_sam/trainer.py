from __future__ import annotations

import csv
import json
import math
import random
import shutil
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig, save_config
from .dataset import NuInsSegDataset
from .metrics import aggregate_jaccard_index, dice_score, panoptic_quality, summarize_metric_rows
from .modeling import build_model, forward_logits
from .postprocess import probability_to_instances
from .runtime import PROJECT_ROOT, project_relative
from .visualization import generate_qualitative_figures


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, ignore_mask: torch.Tensor
) -> torch.Tensor:
    valid = (1.0 - ignore_mask).float()
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / denom


def masked_soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    valid = (1.0 - ignore_mask).float()
    probabilities = torch.sigmoid(logits) * valid
    targets = targets * valid
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denom = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def build_dataloaders(
    config: ExperimentConfig, split_dir: Path
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = NuInsSegDataset(
        split_dir / "train.csv",
        image_size=config.image_size,
        mode="train",
        enable_augmentation=True,
    )
    val_dataset = NuInsSegDataset(
        split_dir / "val.csv",
        image_size=config.image_size,
        mode="val",
        enable_augmentation=False,
    )
    test_dataset = NuInsSegDataset(
        split_dir / "test.csv",
        image_size=config.image_size,
        mode="test",
        enable_augmentation=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def _snapshot_split_files(split_dir: Path, output_dir: Path) -> None:
    snapshot_dir = output_dir / "split_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "val.csv", "test.csv"):
        shutil.copy2(split_dir / name, snapshot_dir / name)
    manifest_path = split_dir.parent / "manifest.csv"
    summary_path = split_dir.parent / "summary.json"
    if manifest_path.exists():
        shutil.copy2(manifest_path, snapshot_dir / "manifest.csv")
    if summary_path.exists():
        shutil.copy2(summary_path, snapshot_dir / "splits_summary.json")


def append_csv_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(jsonl_path: Path, row: Dict[str, object]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _save_prediction_arrays(
    output_dir: Path,
    split_name: str,
    sample_id: str,
    pred_binary: np.ndarray,
    pred_instances: np.ndarray,
) -> tuple[str, str]:
    binary_dir = output_dir / f"{split_name}_predictions" / "binary"
    instance_dir = output_dir / f"{split_name}_predictions" / "instances"
    binary_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)
    binary_path = binary_dir / f"{sample_id}.png"
    instance_path = instance_dir / f"{sample_id}.tif"
    Image.fromarray((pred_binary.astype(np.uint8) * 255)).save(binary_path)
    Image.fromarray(pred_instances.astype(np.uint16)).save(instance_path)
    return project_relative(binary_path), project_relative(instance_path)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    split_name: str,
    save_predictions: bool = False,
) -> tuple[Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        binary_targets = batch["binary_mask"].to(device)
        ignore_masks = batch["ignore_mask"].to(device)
        logits = forward_logits(model, images, target_hw=binary_targets.shape[-2:])
        loss = masked_bce_with_logits(logits, binary_targets, ignore_masks) + masked_soft_dice_loss(
            logits, binary_targets, ignore_masks
        )
        total_loss += float(loss.item())

        probabilities = torch.sigmoid(logits).cpu().numpy()
        binary_targets_np = batch["binary_mask"].cpu().numpy()
        ignore_masks_np = batch["ignore_mask"].cpu().numpy()
        instance_targets_np = batch["instance_mask"].cpu().numpy()

        for sample_index in range(images.shape[0]):
            sample_probability = probabilities[sample_index, 0]
            sample_ignore = ignore_masks_np[sample_index, 0] > 0
            sample_gt_binary = binary_targets_np[sample_index, 0] > 0.5
            sample_gt_instances = np.array(instance_targets_np[sample_index], copy=True)
            sample_gt_instances[sample_ignore] = 0

            sample_pred_binary = sample_probability >= config.probability_threshold
            sample_pred_binary[sample_ignore] = False
            sample_pred_instances = probability_to_instances(
                sample_probability,
                threshold=config.probability_threshold,
                min_object_size=config.min_object_size,
                peak_min_distance=config.peak_min_distance,
                peak_threshold_abs=config.peak_threshold_abs,
                ignore_mask=sample_ignore,
            )
            metrics_row: Dict[str, object] = {
                "split": split_name,
                "sample_id": batch["sample_id"][sample_index],
                "organ": batch["organ"][sample_index],
                "image_path": batch["image_path"][sample_index],
                "instance_mask_path": batch["instance_mask_path"][sample_index],
                "binary_mask_path": batch["binary_mask_path"][sample_index]
                if "binary_mask_path" in batch
                else batch["instance_mask_path"][sample_index],
                "ignore_mask_path": batch["ignore_mask_path"][sample_index],
                "dice": dice_score(sample_pred_binary, sample_gt_binary, sample_ignore),
                "aji": aggregate_jaccard_index(sample_gt_instances, sample_pred_instances, sample_ignore),
                "pq": panoptic_quality(sample_gt_instances, sample_pred_instances, sample_ignore),
            }
            if save_predictions:
                prediction_binary_path, prediction_instance_path = _save_prediction_arrays(
                    output_dir,
                    split_name=split_name,
                    sample_id=batch["sample_id"][sample_index],
                    pred_binary=sample_pred_binary,
                    pred_instances=sample_pred_instances,
                )
                metrics_row["prediction_binary_path"] = prediction_binary_path
                metrics_row["prediction_instance_path"] = prediction_instance_path
            rows.append(metrics_row)

    summary = summarize_metric_rows(rows, metric_names=("dice", "aji", "pq"))
    summary["loss_mean"] = total_loss / max(len(dataloader), 1)
    summary["split"] = split_name
    return summary, rows


def save_metric_rows(csv_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_best_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)


def evaluate_checkpoint(
    config: ExperimentConfig,
    split_csv: Path,
    checkpoint_path: Path,
    output_dir: Path,
    split_name: str,
    save_predictions: bool = True,
) -> Dict[str, object]:
    set_deterministic_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NuInsSegDataset(
        split_csv,
        image_size=config.image_size,
        mode=split_name,
        enable_augmentation=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model, parameter_summary = build_model(config, device=device)
    del parameter_summary
    _load_best_checkpoint(model, checkpoint_path, device)
    summary, rows = evaluate_loader(
        model=model,
        dataloader=dataloader,
        config=config,
        device=device,
        output_dir=output_dir,
        split_name=split_name,
        save_predictions=save_predictions,
    )
    save_metric_rows(output_dir / f"{split_name}_metrics.csv", rows)
    (output_dir / f"{split_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "rows": rows}


def train_fold(config: ExperimentConfig, fold: int) -> Dict[str, object]:
    set_deterministic_seed(config.random_seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = config.resolved_split_root() / f"fold_{fold}"
    output_dir = config.resolved_output_root() / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(replace(config, fold=fold), output_dir)
    _snapshot_split_files(split_dir, output_dir)

    train_loader, val_loader, test_loader = build_dataloaders(config, split_dir)
    model, parameter_summary = build_model(config, device=device)
    (output_dir / "trainable_parameters.json").write_text(json.dumps(parameter_summary, indent=2))

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")

    best_aji = -math.inf
    best_step = 0
    no_improve_evals = 0
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)

    for step in range(1, config.max_steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch["image"].to(device)
        binary_targets = batch["binary_mask"].to(device)
        ignore_masks = batch["ignore_mask"].to(device)

        context = autocast if scaler.is_enabled() else nullcontext
        with context():
            logits = forward_logits(model, images, target_hw=binary_targets.shape[-2:])
            bce_loss = masked_bce_with_logits(logits, binary_targets, ignore_masks)
            dice_loss = masked_soft_dice_loss(logits, binary_targets, ignore_masks)
            loss = bce_loss + dice_loss
            scaled_loss = loss / config.grad_accum_steps

        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if step % config.grad_accum_steps == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % config.log_every == 0:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "bce_loss": float(bce_loss.item()),
                "dice_loss": float(dice_loss.item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            append_csv_row(output_dir / "train_scalars.csv", row)
            append_jsonl(output_dir / "train_scalars.jsonl", row)

        if step % config.save_every == 0:
            torch.save(model.state_dict(), output_dir / "checkpoint_last.pth")

        if step % config.eval_every == 0 or step == config.max_steps:
            val_summary, val_rows = evaluate_loader(
                model=model,
                dataloader=val_loader,
                config=config,
                device=device,
                output_dir=output_dir,
                split_name="val",
                save_predictions=False,
            )
            val_summary["step"] = step
            append_csv_row(output_dir / "val_history.csv", val_summary)
            append_jsonl(output_dir / "val_history.jsonl", val_summary)
            save_metric_rows(output_dir / f"val_metrics_step_{step}.csv", val_rows)

            if val_summary["aji_mean"] > best_aji:
                best_aji = float(val_summary["aji_mean"])
                best_step = step
                no_improve_evals = 0
                torch.save(model.state_dict(), output_dir / "checkpoint_best.pth")
            else:
                no_improve_evals += 1

            if config.early_stop_patience > 0 and no_improve_evals >= config.early_stop_patience:
                break

    torch.save(model.state_dict(), output_dir / "checkpoint_last.pth")
    best_checkpoint = output_dir / "checkpoint_best.pth"
    if not best_checkpoint.exists():
        torch.save(model.state_dict(), best_checkpoint)
    _load_best_checkpoint(model, best_checkpoint, device)

    test_summary, test_rows = evaluate_loader(
        model=model,
        dataloader=test_loader,
        config=config,
        device=device,
        output_dir=output_dir,
        split_name="test",
        save_predictions=config.save_test_predictions,
    )
    save_metric_rows(output_dir / "test_metrics.csv", test_rows)
    (output_dir / "test_summary.json").write_text(json.dumps(test_summary, indent=2))

    result = {
        "fold": fold,
        "best_step": best_step,
        "best_val_aji": best_aji,
        "test_dice": test_summary["dice_mean"],
        "test_aji": test_summary["aji_mean"],
        "test_pq": test_summary["pq_mean"],
        "output_dir": project_relative(output_dir),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def aggregate_fold_results(output_root: Path, folds: Iterable[int]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for fold in folds:
        result_path = output_root / f"fold_{fold}" / "result.json"
        rows.append(json.loads(result_path.read_text()))
    metrics = {
        "dice": [float(row["test_dice"]) for row in rows],
        "aji": [float(row["test_aji"]) for row in rows],
        "pq": [float(row["test_pq"]) for row in rows],
    }
    summary: Dict[str, object] = {"folds": rows}
    for metric_name, values in metrics.items():
        values_np = np.array(values, dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values_np.mean())
        summary[f"{metric_name}_std"] = float(values_np.std(ddof=0))
    return summary


def collect_test_rows(output_root: Path, folds: Iterable[int]) -> List[Dict[str, str]]:
    merged_rows: List[Dict[str, str]] = []
    for fold in folds:
        test_metrics_path = output_root / f"fold_{fold}" / "test_metrics.csv"
        if not test_metrics_path.exists():
            continue
        with test_metrics_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                row["fold"] = str(fold)
                merged_rows.append(row)
    merged_rows.sort(key=lambda row: (row["organ"], row["sample_id"]))
    return merged_rows


def run_cross_validation(config: ExperimentConfig, folds: Iterable[int]) -> Dict[str, object]:
    folds = list(folds)
    output_root = config.resolved_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    results = [train_fold(config, fold=fold) for fold in folds]
    summary = aggregate_fold_results(output_root, folds=folds)
    (output_root / "cross_validation_summary.json").write_text(json.dumps(summary, indent=2))

    aggregate_rows = collect_test_rows(output_root, folds=folds)
    aggregate_metrics_csv = output_root / "aggregate_test_metrics.csv"
    save_metric_rows(aggregate_metrics_csv, aggregate_rows)
    generate_qualitative_figures(
        aggregate_metrics_csv=aggregate_metrics_csv,
        output_dir=output_root / "qualitative_figures",
        max_examples_per_organ=config.max_examples_per_organ,
    )
    return {"results": results, "summary": summary}
