# NuInsSeg MobileSAM LoRA Pipeline

This repo contains the training and evaluation pipeline for fine-tuning MobileSAM with LoRA on NuInsSeg.

## Design Choices
- Training target: binary nuclei foreground derived from `label masks modify > 0`.
- Instance evaluation target: `archive/**/label masks modify/*.tif`.
- Ignore-mask policy: ambiguous pixels from `archive/**/vague areas/mask binary/*.png` are excluded from both loss and metrics.
- Outer split: deterministic 5-fold split with `shuffle=True` and `random_state=19`.
- Inner validation: deterministic organ-stratified holdout carved from the outer-train set with `val_fraction=0.125`.
- Best checkpoint: selected by validation AJI.
- Default model: MobileSAM `vit_t` with encoder and decoder LoRA enabled, rank 4.

## How To Run
Run all commands from the project root:

### 1. Prepare folds
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
  --output-root runs/nuinsseg_mobilesam_lora_v2 \
  --mobile-sam-ckpt /media/dshatwell/SharedData/courses/cap5516_medical_imaging_computing/cap5516-hw3-nuclei-instance-segmentation/checkpoints/mobile_sam.pt \
  --fold 0 \
  --batch-size 16 \
  --num-workers 8 \
  --grad-accum-steps 1 \
  --max-steps 8000 \
  --eval-every 500 \
  --log-every 100 \
  --save-every 100 \
  --probability-threshold 0.5 \
  --min-object-size 15 \
  --peak-min-distance 7 \
  --peak-threshold-abs 0.086
```

### 3. Train full 5-folds
```bash
python scripts/run_nuinsseg_cv.py \
  --dataset-root archive \
  --output-root runs/nuinsseg_mobilesam_lora_final \
  --mobile-sam-ckpt /media/dshatwell/SharedData/courses/cap5516_medical_imaging_computing/cap5516-hw3-nuclei-instance-segmentation/checkpoints/mobile_sam.pt \
  --batch-size 16 \
  --num-workers 8 \
  --grad-accum-steps 1 \
  --max-steps 10000 \
  --eval-every 500 \
  --log-every 100 \
  --save-every 100 \
  --probability-threshold 0.5 \
  --min-object-size 15 \
  --peak-min-distance 7 \
  --peak-threshold-abs 0.086
```

### 4. Eval existing checkpoint
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

### 5. Sweep post-processing on an existing checkpoint
Use this to search watershed and threshold parameters on a trained run without retraining the model. This is most useful on the validation split, then you can reuse the best settings for test evaluation.

```bash
python scripts/sweep_nuinsseg_postprocess.py \
  --run-dir runs/nuinsseg_mobilesam_lora/fold_0 \
  --split-name val \
  --metric aji_mean \
  --probability-thresholds 0.35 0.4 0.45 0.5 0.55 \
  --min-object-sizes 5 10 15 20 \
  --peak-min-distances 2 3 4 5 6 \
  --peak-threshold-abs-values 0.05 0.1 0.15 0.2
```

You can also provide the grid as JSON:

```bash
python scripts/sweep_nuinsseg_postprocess.py \
  --run-dir runs/nuinsseg_mobilesam_lora/fold_0 \
  --split-name val \
  --metric pq_mean \
  --grid-json '{"probability_threshold":[0.4,0.45,0.5],"min_object_size":[5,10,15],"peak_min_distance":[3,4,5],"peak_threshold_abs":[0.05,0.1,0.15]}'
```

For random search, the script samples uniformly from the min and max implied by the values you provide for each parameter. Integer-valued parameters are sampled as integers.

```bash
python scripts/sweep_nuinsseg_postprocess.py \
  --run-dir runs/nuinsseg_mobilesam_lora/fold_0 \
  --split-name val \
  --metric aji_mean \
  --search-mode random \
  --num-trials 50 \
  --eval-batch-size 8 \
  --num-workers 8 \
  --use-autocast true \
  --random-seed 19 \
  --probability-thresholds 0.45 0.55 \
  --min-object-sizes 12 16 \
  --peak-min-distances 5 7 \
  --peak-threshold-abs-values 0.08 0.1
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

Each post-processing sweep writes to `runs/.../fold_<k>/postprocess_sweep_<split>/`:
- `grid.json`
- `search_config.json`
- `sweep_results.csv`
- `sweep_results.json`
- `best_result.json`
- `combo_*/` reevaluation outputs for each parameter combination

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