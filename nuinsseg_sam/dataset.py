from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

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

        if torch.rand(1).item() < 0.8:
            brightness = 0.9 + 0.2 * torch.rand(1).item()
            contrast = 0.9 + 0.2 * torch.rand(1).item()
            saturation = 0.9 + 0.2 * torch.rand(1).item()
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
            image = ImageEnhance.Color(image).enhance(saturation)

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
