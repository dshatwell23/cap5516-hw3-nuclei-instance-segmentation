from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
FINETUNE_SAM_ROOT = PROJECT_ROOT / "finetune-SAM"


def ensure_finetune_sam_on_path() -> None:
    finetune_path = str(FINETUNE_SAM_ROOT)
    if finetune_path not in sys.path:
        sys.path.insert(0, finetune_path)


def import_sam_components() -> Tuple[object, object]:
    ensure_finetune_sam_on_path()
    from models.sam import sam_model_registry
    from models.sam_LoRa import LoRA_Sam

    return sam_model_registry, LoRA_Sam


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
