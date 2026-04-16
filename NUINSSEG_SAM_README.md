# NuInsSeg MobileSAM LoRA Pipeline

This workspace now contains a repo-local training and evaluation pipeline for fine-tuning MobileSAM with LoRA on NuInsSeg.

## Design Choices
- Training target: binary nuclei foreground derived from `label masks modify > 0`.
- Instance evaluation target: `archive/**/label masks modify/*.tif`.
- Ignore-mask policy: ambiguous pixels from `archive/**/vague areas/mask binary/*.png` are excluded from both loss and metrics.
- Outer split: deterministic 5-fold split with `shuffle=True` and `random_state=19`.
- Inner validation: deterministic organ-stratified holdout carved from the outer-train set with `val_fraction=0.125`.
- Best checkpoint: selected by validation AJI.
- Default model: MobileSAM `vit_t` with encoder and decoder LoRA enabled, rank 4.

## Added Files
- `nuinsseg_sam/`: dataset indexing, dataloaders, MobileSAM+LoRA wrapper, trainer, metrics, post-processing, visualization.
- `scripts/prepare_nuinsseg_splits.py`: build manifest and deterministic fold CSVs.
- `scripts/train_nuinsseg_fold.py`: train one fold and run final test evaluation.
- `scripts/run_nuinsseg_cv.py`: run the full cross-validation workflow and aggregate results.
- `scripts/evaluate_nuinsseg_checkpoint.py`: standalone evaluation for an existing checkpoint.
- `tests/`: manifest/split, metrics, post-processing, and dataset smoke tests.
- `requirements-nuinsseg.txt`: package list for the new pipeline.

## Environment Notes
The active environment in this workspace is not ready to run training yet. The missing or inconsistent pieces observed during implementation were:
- `torch`
- `torchvision`
- `monai`
- `tensorboardX`
- `segment-anything`
- a NumPy 2.x environment that conflicts with older compiled packages

Use `numpy==1.26.4` for this pipeline. The pinned list is in `requirements-nuinsseg.txt`.

You also need a MobileSAM checkpoint file, typically `mobile_sam.pt`. The scripts fail fast if that checkpoint path is missing.

## How To Run
Run all commands from the project root:

### 1. Prepare manifest and folds
```bash
python scripts/prepare_nuinsseg_splits.py \
  --dataset-root archive \
  --output-root runs/nuinsseg_mobilesam_lora
```

This writes:
- `runs/nuinsseg_mobilesam_lora/splits/manifest.csv`
- `runs/nuinsseg_mobilesam_lora/splits/fold_0/{train,val,test}.csv`
- ...
- `runs/nuinsseg_mobilesam_lora/splits/summary.json`

### 2. Train one fold
```bash
python scripts/train_nuinsseg_fold.py \
  --dataset-root archive \
  --output-root runs/nuinsseg_mobilesam_lora \
  --mobile-sam-ckpt /absolute/path/to/mobile_sam.pt \
  --fold 0 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --max-steps 4000 \
  --eval-every 500 \
  --save-every 500
```

### 3. Run the full 5-fold pipeline
```bash
python scripts/run_nuinsseg_cv.py \
  --dataset-root archive \
  --output-root runs/nuinsseg_mobilesam_lora \
  --mobile-sam-ckpt /absolute/path/to/mobile_sam.pt \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --max-steps 4000 \
  --eval-every 500
```

### 4. Re-evaluate an existing checkpoint
```bash
python scripts/evaluate_nuinsseg_checkpoint.py \
  --run-dir runs/nuinsseg_mobilesam_lora/fold_0
```

You can override the checkpoint or split CSV:
```bash
python scripts/evaluate_nuinsseg_checkpoint.py \
  --run-dir runs/nuinsseg_mobilesam_lora/fold_0 \
  --checkpoint-path /absolute/path/to/checkpoint_best.pth \
  --split-csv /absolute/path/to/custom.csv \
  --split-name test
```

## Output Layout
Each fold writes to `runs/nuinsseg_mobilesam_lora/fold_<k>/`:
- `config.json`, `config.yaml`
- `trainable_parameters.json`
- `split_snapshot/`
- `train_scalars.csv`, `train_scalars.jsonl`
- `val_history.csv`, `val_history.jsonl`
- `val_metrics_step_<step>.csv`
- `checkpoint_best.pth`, `checkpoint_last.pth`
- `test_metrics.csv`, `test_summary.json`
- `test_predictions/binary/`
- `test_predictions/instances/`
- `result.json`

The full 5-fold run also writes:
- `cross_validation_summary.json`
- `aggregate_test_metrics.csv`
- `qualitative_figures/<organ>/<sample_id>.png`

## Metrics And Post-processing
- Binary overlap metric: Dice on predicted nuclei foreground after masking ignored pixels.
- Instance recovery: probability threshold -> remove tiny objects -> distance transform -> local maxima seeds -> watershed -> sequential relabeling.
- Instance metrics:
  - AJI
  - PQ with IoU threshold 0.5

## Testing
Recommended checks after dependencies are installed:
```bash
python -m unittest tests.test_manifest_and_splits tests.test_metrics tests.test_postprocess tests.test_dataset_loader
```

For a minimal end-to-end smoke run:
```bash
python scripts/train_nuinsseg_fold.py \
  --dataset-root archive \
  --output-root runs/nuinsseg_smoke \
  --mobile-sam-ckpt /absolute/path/to/mobile_sam.pt \
  --fold 0 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 10 \
  --eval-every 5 \
  --save-every 5
```

## Notes
- The new pipeline is intentionally isolated from the original demo trainers in `finetune-SAM`.
- Training uses 512x512 supervision and upsamples logits back to that size before loss and metrics.
- Prompting is disabled; the pipeline uses the automatic no-prompt segmentation path.
