# CorrFlow image generation

Jointly fine-tune an ImageNet DiT and a velocity-residual refiner on frozen
RAEv2 latents, then sample with or without the refiner using the same DiT.
Run commands in this directory.

## Code map

| Path | Purpose |
| --- | --- |
| `src/train.py`, `src/stage2/engine.py` | Joint DiT/refiner training, separate optimizers and EMAs |
| `src/stage2/refiner/` | `Refiner`, hidden-state capture, residual loss, sampling correction |
| `src/stage2/transport/` | Flow-matching objectives and ODE sampling |
| `src/offline_eval.py` | Distributed generation, A/B evaluation, saved-NPZ evaluation |
| `src/stage1/`, `src/encoders/`, `src/data/` | RAE, decoder, encoder, and data dependencies |
| `configs/stage2/` | Refiner training/sampling recipes and shared baseline configurations |
| `scripts/eval/` | FDD/FDr, CMMD, precision/recall, and Vendi evaluation |

Shared RAEv2 modules retain their original conditioning and baseline support.

## Environment and assets

The dependency versions in `pyproject.toml` follow the source research environment
(Python 3.10–3.13, PyTorch 2.10/CUDA 12.8). Git dependencies use HTTPS.

```bash
uv sync
export IMAGENET_DATA_DIR=/path/to/imagenet-256
export CORRFLOW_TMPDIR="$PWD/tmp"
```

Without these variables, configurations use `data/imagenet-256` and `tmp`.
The data loader expects the RAEv2 preprocessed ImageNet Arrow layout, including
`imagenet-256-train/` and `imagenet-256-val/`. Evaluation uses the validation
split's labels; complete paper evaluations use all 50,000 validation examples.

Place the pretrained RAEv2 assets at:

```text
pretrained_models/
  encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  stage1/imagenet/dinov3l-k7/decoder.pt
  stage1/imagenet/dinov3l-k7/stats.pt
  stage2/imagenet/dinov3l-k7/checkpoint.pt
```

See the upstream [RAEv2 repository](https://github.com/nanovisionx/RAEv2) for
asset preparation. The DINOv3 encoder definition is loaded through `torch.hub`;
offline nodes need its repository and weights cached in advance.

## Training

The standard recipe is
`configs/stage2/training/imagenet-dinov3l-k7-ft-refiner.yaml`.
It warm-starts the DiT, initializes the refiner from scratch, and keeps stage 1
frozen. Refiner training and sampling use the full `[0, 1]` interval.

```bash
export EXPERIMENT_NAME=in256-k7-ft-refiner
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 src/train.py \
  --config configs/stage2/training/imagenet-dinov3l-k7-ft-refiner.yaml \
  --results-dir ckpts/stage2 --precision bf16
```

Set GPU count and `training.grad_accum_steps` together:
`micro_batch = global_batch_size / (world_size * grad_accum_steps)`.
The standard recipe uses a global batch of 1024 and accumulation of 2 for
eight large-memory GPUs. For one GPU, accumulation of 128 gives micro-batch 8;
adjust the configuration to the available memory before running.

Training automatically resumes the highest-epoch checkpoint in the experiment
directory. `--ckpt` adds an explicit resume candidate. `--wandb` enables logging
with `WANDB_PROJECT` and optional `WANDB_ENTITY`. The optional
`--sync-checkpoints` integration requires the source environment's
`~/setup/ssd-symlinks.sh`; leave it disabled unless you provide that helper.

Outputs:

```text
ckpts/stage2/in256-k7-ft-refiner/checkpoints/
  ep-0000080.pt
  refiner/refiner_ep-0000080.pt
```

The combined checkpoint contains `model`, `ema`, `optimizer`, `scheduler`,
`refiner`, `refiner_ema`, `refiner_optimizer`, and `refiner_scheduler`.
The separate refiner file is a flat EMA state dictionary.

## Sampling and evaluation

The sampling YAML is a template: replace `@EPOCH@` before running. This example
creates a small generation configuration with metrics disabled:

```bash
uv run python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
template = Path("configs/stage2/sampling/imagenet-dinov3l-k7-ft-refiner.yaml")
cfg = OmegaConf.create(template.read_text().replace("@EPOCH@", "0000080"))
cfg.training.global_batch_size = 4
cfg.eval.datasets.imagenet.num_samples = 4
cfg.eval.datasets.imagenet.metrics = []
cfg.refiner.eval_ab = True
Path("results").mkdir(exist_ok=True)
OmegaConf.save(cfg, "results/sampling.yaml")
PY
EXPERIMENT_NAME=sample-ep80 SAVE_GEN_DIR=results/samples \
  uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/offline_eval.py --config results/sampling.yaml --precision bf16
```

This writes `sample-ep80_ema.npz` and `sample-ep80_ema_refiner.npz`, containing
uint8 RGB arrays with shape `(N, 256, 256, 3)`. Set `refiner.eval_ab: false` for
only the refined branch. `--refiner-scale 0` disables corrections;
`--refiner-scale 1` uses the configured correction without additional guidance
amplification. `stage_2.ckpt_key` selects the DiT `ema` or `model` weights;
`refiner.model.ckpt_key: refiner_ema` selects refiner EMA weights from a combined
checkpoint, while `null` loads a flat refiner state dictionary.

For paper evaluation, retain the template's 100 steps, 50,000 examples, IG=1.78,
and metric list. The small example above is only a generation check. Saved NPZs
can also be evaluated without loading a generator:

```bash
EXPERIMENT_NAME=saved-samples uv run python src/offline_eval.py \
  --config results/sampling.yaml --npz results/samples/sample-ep80_ema_refiner.npz \
  --tag ema_refiner
```


