# vibes_pipe — CLAUDE.md

## What this project does
End-to-end deep learning pipeline for **3-D MRE (Magnetic Resonance Elastography) brain segmentation**.
The model learns from both human GT masks and NLI-derived pseudo-labels, with scanner-aware noise augmentation for GE and Siemens scanners.

## Architecture at a glance

| Component | Key file(s) |
|-----------|-------------|
| Model | `src/vibes_pipe/models/prob_unet.py` — `ProbUNet3D` (probabilistic + deterministic) |
| Prior / Posterior | `src/vibes_pipe/models/components/prior.py`, `posterior.py` |
| Latent combiner | `src/vibes_pipe/models/components/fcomb.py` |
| Training engine | `src/vibes_pipe/training/engine.py` — `TrainEngine.fit()` |
| Loss | `src/vibes_pipe/losses/recon_combo_loss.py` — `ProbUNetLoss` + `FocalDiceComboLoss` |
| Dataset | `src/vibes_pipe/data/dataset.py` — `ManifestDataset` |
| Augmentation | `src/vibes_pipe/augmentation/augment_pipeline.py` — `MREAugmentation` |
| Inference / MC | `src/vibes_pipe/infer/monte_carlo_sampling.py` — `monte_carlo_sampling()` |
| Metrics | `src/vibes_pipe/metric/eval_metrics.py` — Dice, IoU, HD95, SurfaceDice |
| CLI | `src/vibes_pipe/cli/pipeline_cli.py` |
| Config loader | `src/vibes_pipe/utils/config.py` |

## Data layout
```
experiments/<exp_name>/
  raw_data/<subject_id>/          ← original .mat / .nii files
  workspace_root/
    manifest.json                 ← single source of truth for all splits + file paths
    train|val|test/<subject_id>/
      t2stack.mat                 ← image (input)
      GT(human).mat               ← ground truth segmentation
      NLI_output.mat              ← NLI-derived pseudo label (optional)
      subject_noise.mat           ← scanner noise profile (optional)
```
`manifest.json` contains `pairs[]` with fields: `id`, `split`, `scanner_type` (GE/SIEMENS), `files{}`.

## YAML config structure
All runs are driven by a YAML config. Key top-level keys:
```yaml
seed, device
model:   {class_name, kwargs}    # ProbUNet3D kwargs
loss:    {class_name, kwargs}    # ProbUNetLoss → recon_loss → FocalDiceComboLoss
optimizer: {class_name, kwargs}
trainer: {num_epochs, grad_clip, fast_val, save_path, final_save_path, history_save_path}
data:    {manifest, workspace_root, split, batch_size, ...}
```
Deterministic mode: set `model.kwargs.inject_latent: false`.

## Model modes
- **Probabilistic** (`inject_latent: true`): prior + posterior at train time; samples from prior at inference. Returns `(logits, (mu_p, logvar_p), (mu_q, logvar_q))` during training, just `logits` at eval.
- **Deterministic** (`inject_latent: false`): vanilla U-Net backbone + Fcomb conv head.

## Loss
`ProbUNetLoss = recon_loss + beta * KL`
- `beta` linearly anneals from 0 → `beta_final` over `beta_warmup` epochs.
- `recon_loss = FocalDiceComboLoss = alpha_combo * Focal + (1-alpha_combo) * Dice`.
- Set `alpha_combo: 0.0` to use pure Dice.

## Augmentation pipeline
`MREAugmentation`:
1. Spatial augmentation on (image, label) — records transform params.
2. Apply same spatial transform to subject noise field.
3. Add spatially-transformed noise to augmented image.

## Inference / uncertainty
`monte_carlo_sampling(model, image, n_samples)` — full-volume inference with auto-padding to multiples of 16.
Returns: `{mean, samples, uncertainty: {predictive_entropy, std, ged}, latent}`.

## Common builder pattern
Each subsystem has a `builders.py` that reads YAML config dict and returns instantiated objects. Don't instantiate model/loss/optimizer/dataset directly — use the builders.

## Checkpoints
Saved as `{epoch, model_state_dict, optimizer_state_dict, val_dice}` at `save_path` (best) and `final_save_path` (last epoch).

## Key conventions
- All `.mat` files loaded via `src/vibes_pipe/data/io_mat.py` (`load_mat_dict` + `find_primary_array`).
- Images returned as `[B, C, D, H, W]` float32 tensors with channel dim added by dataset.
- `scanner_type` normalized to uppercase (GE / SIEMENS) in dataset `_index_manifest`.
- Tests live in `tests/`.
