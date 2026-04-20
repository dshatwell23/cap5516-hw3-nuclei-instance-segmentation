from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop

from .runtime import PROJECT_ROOT


class NuInsSegDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        image_size: int = 1024,
        mode: str = "train",
        enable_augmentation: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.image_size = image_size
        self.mode = mode
        self.enable_augmentation = enable_augmentation and mode == "train"
        self.color_jitter = ColorJitter(
            brightness=0.35,
            contrast=0.35,
            saturation=0.35,
            hue=0.08,
        )
        with self.split_csv.open("r", newline="") as handle:
            self.rows: List[Dict[str, str]] = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def _load_rgb(self, relative_path: str) -> Image.Image:
        return Image.open(PROJECT_ROOT / relative_path).convert("RGB")

    def _load_array(self, relative_path: str) -> np.ndarray:
        return np.array(Image.open(PROJECT_ROOT / relative_path))

    def _apply_shared_transforms(
        self,
        image: Image.Image,
        instance_mask: np.ndarray,
        ignore_mask: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        if not self.enable_augmentation:
            return image, instance_mask, ignore_mask

        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            instance_mask = np.fliplr(instance_mask).copy()
            ignore_mask = np.fliplr(ignore_mask).copy()
        if torch.rand(1).item() < 0.5:
            image = TF.vflip(image)
            instance_mask = np.flipud(instance_mask).copy()
            ignore_mask = np.flipud(ignore_mask).copy()

        rotation_k = int(torch.randint(low=0, high=4, size=(1,)).item())
        if rotation_k:
            image = image.rotate(90 * rotation_k)
            instance_mask = np.rot90(instance_mask, rotation_k).copy()
            ignore_mask = np.rot90(ignore_mask, rotation_k).copy()

        if torch.rand(1).item() < 0.7:
            image, instance_mask, ignore_mask = self._apply_random_resized_crop(
                image, instance_mask, ignore_mask
            )

        if torch.rand(1).item() < 0.9:
            image = self.color_jitter(image)

        if torch.rand(1).item() < 0.3:
            sigma = 0.1 + 1.9 * torch.rand(1).item()
            image = TF.gaussian_blur(image, kernel_size=5, sigma=sigma)

        return image, instance_mask, ignore_mask

    def _apply_random_resized_crop(
        self,
        image: Image.Image,
        instance_mask: np.ndarray,
        ignore_mask: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        original_height, original_width = instance_mask.shape
        top, left, height, width = RandomResizedCrop.get_params(
            image,
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
        )
        image = TF.resized_crop(
            image,
            top=top,
            left=left,
            height=height,
            width=width,
            size=[original_height, original_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        instance_mask_image = Image.fromarray(instance_mask.astype(np.int32), mode="I")
        ignore_mask_image = Image.fromarray(ignore_mask.astype(np.uint8), mode="L")
        instance_mask = np.array(
            TF.resized_crop(
                instance_mask_image,
                top=top,
                left=left,
                height=height,
                width=width,
                size=[original_height, original_width],
                interpolation=InterpolationMode.NEAREST,
            ),
            dtype=np.int32,
        )
        ignore_mask = np.array(
            TF.resized_crop(
                ignore_mask_image,
                top=top,
                left=left,
                height=height,
                width=width,
                size=[original_height, original_width],
                interpolation=InterpolationMode.NEAREST,
            ),
            dtype=np.uint8,
        )
        return image, instance_mask, ignore_mask

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        image = self._load_rgb(row["image_path"])
        instance_mask = self._load_array(row["instance_mask_path"]).astype(np.int32)
        ignore_mask = self._load_array(row["ignore_mask_path"]).astype(np.uint8)
        original_hw = tuple(instance_mask.shape)

        image, instance_mask, ignore_mask = self._apply_shared_transforms(
            image, instance_mask, ignore_mask
        )

        resized_image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(np.array(resized_image, dtype=np.float32).transpose(2, 0, 1))
        )

        binary_mask = (instance_mask > 0).astype(np.float32)
        ignore_mask = (ignore_mask > 0).astype(np.float32)

        return {
            "image": image_tensor,
            "binary_mask": torch.from_numpy(binary_mask).unsqueeze(0),
            "instance_mask": torch.from_numpy(instance_mask.astype(np.int64)),
            "ignore_mask": torch.from_numpy(ignore_mask).unsqueeze(0),
            "organ": row["organ"],
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "instance_mask_path": row["instance_mask_path"],
            "binary_mask_path": row["binary_mask_path"],
            "ignore_mask_path": row["ignore_mask_path"],
            "original_hw": original_hw,
        }
