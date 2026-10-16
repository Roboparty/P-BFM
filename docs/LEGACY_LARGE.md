# Historical large-network reproduction

This release preserves the Oct 5 implementation used by the Oct 14, 2026
large-network retraining run. It is a historical baseline, not the later
RP1-sized, shared-critic or corrected-QV implementation.

## Configuration

| Setting | Value |
| --- | --- |
| Actor / F | 2048 × 6 |
| QD / QAux | 1024 × 4 / 2048 × 4 |
| QH / QV | 1024 × 3 / 1024 × 3 |
| Q organization | Four independent critics, two Q estimates per critic |
| D loss / reward / learning rate | BCE / log-odds / 5e-6 |
| Actor coefficients D / Aux / QH / QV | .01 / .02 / .002 / .002 |
| Aux normalization | Dynamic; no freezing |
| Depth | 8 × 18 × 32, first CNN stride 1, frame feature 256, GRU feature 256 |
| Velocity estimator | 721 → 512 → 256 → 3, yaw-frame velocity, depth supervision |
| Slippage / stumble | -4 / -2 |
| Precision / replay | FP32 / GPU |
| Data | LaFAN-only; dataset must be supplied separately |
| Training | 8 ranks, 1024 environments/rank, seed 4831, 192020480 global steps |
| Curriculum | Starts at 0 for scratch, interval 19200000 global steps |
| Terrain ranges | Stairs 10–20 cm, rough 2–6 cm, boxes 5–15 cm high |

`pbfm/configs/legacy_large.json` is the unaltered historical `run_config.json`.
`pbfm/configs/depth_contract.json` retains the recorded depth contract. The older
source config and job records remain in the separate historical archive, not the
runtime tree. The template retains its old resume clock
(192020480 → 384040960). `scripts/train_legacy_scratch.py` uses the normal builder
and changes only the scratch schedule and its identifying tags; the standard CLI
still resolves paths, devices, ranks and logging for each host.
It refuses a work directory containing a checkpoint to avoid accidental resume.

## Launch a new 192M run

From the repository root, install Python 3.10 dependencies using `uv sync`.
Use a machine with eight free GPUs and enough GPU memory for replay. This command
does not stop jobs or wait for cards. Replace the dataset and output paths below
with your own. Host-specific launchers are not included in this release.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MUJOCO_GL=egl NCCL_NVLS_ENABLE=0
export UFO_CACHE_DIR=/path/to/writable-disk/pbfm-cache

uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_legacy_scratch.py \
  --gpu-ids single --terrain-mode rp1_simple --seed 4831 \
  --num-envs 1024 --num-env-steps 192020480 \
  --data-path /path/to/lafan_29dof_10s-clipped.pkl \
  --work-dir /path/to/new-run
```

Optional W&B: authenticate separately and append
`--use-wandb --wandb-project PBFM --wandb-run-name legacy-large`.
For a smoke run, add `--smoke` and use a **different new work directory**; this
entry point runs 32768 global steps so all eight ranks can perform updates.
Weights and replay are not uploaded. Inference and ONNX entry points remain
those documented in [INFERENCE.md](INFERENCE.md).

## Recorded run and limitations

Historical run identifier:
`PBFM_legacy20261005_AF2048_D001_QV0002_H2096p2_8gpu_20261014`.
The completed checkpoint reports 192020480 global transitions. The last full
flat LaFAN tracking evaluation, at 192004096, recorded mean EMD
0.8005109468192975 over 862 clips; the best recorded mean was approximately
0.796447 at 182403072. These are results of one historical seed, **not** a new
evaluation performed by publishing this branch and not a terrain-success metric.

The pre-cleanup source was compared with the original training queue snapshot.
This cleanup removes archived jobs and relocates resources/diagnostics. Model,
loss and replay implementations are retained; configuration file contents are
unchanged. Upstream UFO attribution and license are retained.

Historical QV artificial-window truncation and replay/reset boundary behavior
remain intentionally unchanged. This branch does not contain later boundary
fixes. CPU checks cannot establish deployment safety, new training convergence
or bitwise reproducibility across CUDA/hardware versions.

## Release checks (2026-10-16)

The cleaned Git-tree export passed **508 tests**, with two explicit skips:
the full external-LaFAN coverage test and CUDA edge-query parity. Packaged-wheel
checks confirmed that replay implementations, the default configuration,
simulator YAMLs and G1 XML resources remain available outside the source checkout.

CPU regression checks cover the scratch entry point, historical network/config
defaults, resource relocation, checkpoint bookkeeping, QV/heading contexts,
depth/velocity supervision, Aux normalization, DR, curriculum and ONNX export.
ONNX checks use reduced-size CPU test networks, including nonzero uint8 depth
normalization and saved velocity estimates. They do not load the full 192M model.
An exact reduced-model comparison also checks one optimizer update, all critic
outputs, command contexts, all three intent encodings, optimizer state and loading
the pre-cleanup checkpoint (361 state tensors, zero numerical tolerance).
No new eight-GPU training or real-checkpoint rollout was launched for publication.

Unchanged configuration SHA-256 values at their new paths:

```text
6ed7c384dafdad7ee7ec415baa0ab230d5c3616151f6eff9bc70774ca5f1cd97  pbfm/configs/legacy_large.json
ea6c9ebd23ffb08080bad1112498451912c36a67daa7b17639ef38a8a593d1c5  pbfm/configs/depth_contract.json
```
