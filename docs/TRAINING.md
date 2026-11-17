# Training

## Robot configuration

This release supports G1. Use `pbfm/configs/robots/g1_29dof.yaml` and
G1-retargeted reference motions. A run saves its robot calibration;
inference and export retain that calibration.

## G1 default configuration

| Setting | Value |
| --- | --- |
| Actor / F | 2048 × 6 |
| QD / QAux | 1024 × 4 / 2048 × 4 |
| QH / QV | 1024 × 3 / 1024 × 3 |
| Critics | Four independent critics, two Q estimates per critic |
| D loss / reward / learning rate | BCE / log-odds / 5e-6 |
| Actor coefficients D / Aux / QH / QV | .01 / .02 / .002 / .002 |
| Aux normalization | Dynamic |
| Depth | 8 × 18 × 32, first CNN stride 1, frame feature 256, GRU feature 256 |
| Velocity estimator | 721 → 512 → 256 → 3, yaw-frame velocity, depth supervision |
| Slippage / stumble | -4 / -2 |
| Precision / replay | FP32 / CPU |
| Data | LaFAN-only; supplied separately |
| Training | 8 ranks, 1024 environments/rank, seed 4831, 192020480 global steps |
| Curriculum | Starts at 0 for scratch, interval 19200000 global steps |
| Terrain ranges | Stairs 10–20 cm, rough 2–6 cm, boxes 5–15 cm high |

The training template is `pbfm/configs/legacy_large.json`.
`pbfm/configs/depth_contract.json` specifies the depth input contract.

## Motion data

`bash scripts/download_data.sh` downloads the G1 LaFAN PKLs. A motion PKL maps
motion names to clip dictionaries containing `fps`, `root_trans_offset` and
axis-angle `pose_aa`, with the G1 skeleton and joint/body ordering.
Use `--data-path` to select a different G1-retargeted PKL. Motion IDs index the
selected dataset; goal definitions must refer to that same dataset.

## Terrains

The seven-family terrain preset is named `training_mix` and is independent of
the robot model. Archived configs using `rp1_simple` are accepted as an alias.
The historical `legacy_rp1` depth identifier, terrain aliases and source-provenance
metadata are retained for archived G1 checkpoints. They do not enable another
robot; the supported robot specification is G1 only.

## Train from scratch

Install dependencies from the repository root with `uv sync`. Use eight available
GPUs and edit the dataset, cache and output paths below for your machine.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MUJOCO_GL=egl NCCL_NVLS_ENABLE=0
export UFO_CACHE_DIR=/path/to/writable-disk/pbfm-cache

uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_legacy_scratch.py \
  --gpu-ids single --terrain-mode training_mix --seed 4831 \
  --robot-config pbfm/configs/robots/g1_29dof.yaml \
  --num-envs 1024 --num-env-steps 192020480 \
  --data-path /path/to/lafan_29dof_10s-clipped.pkl \
  --work-dir /path/to/new-run
```

The default and scratch entry points both run 192020480 global steps with the
terrain curriculum starting at step 0. The scratch entry point additionally
refuses a work directory that already contains a checkpoint.

For W&B, authenticate separately and append
`--use-wandb --wandb-project PBFM --wandb-run-name pbfm`.

For a short test, add `--smoke` and choose a separate output directory. This runs
32768 global steps with fewer environments and a smaller replay; evaluation and
replay checkpoint writing are disabled. Check GPU memory and disk space for the
full replay capacity before starting a longer run.

## Checkpoints and storage

Model/optimizer checkpoints and replay shards serve different purposes. Tracking,
goal inference and ONNX export need model checkpoints, not replay. Computing a new
reward-inference latent through the current CLI also requires replay samples;
executing an already-exported context does not. Training continuation depends on
replay, optimizer state and the depth observation contract. Keep the run's
`config.json` alongside the selected model checkpoint for robot calibration.

The default main replay stores the current collector row and obtains the
successor from the following row; `truncated` marks trajectory boundaries. Each
stored depth observation keeps one uint8 frame, and temporal stacks are
reconstructed when sampling. Use a new work directory when changing replay
formats or collection semantics.

Replay is placed on CPU by default. Model checkpoints and ONNX interfaces are
unchanged. Allow host memory for replay, GPU memory for model/optimizer state and
training batches, and disk space for each rank's replay checkpoint. A small-replay
smoke test is not a full-capacity memory check.

## Tests and validation

```bash
PYTHONPATH=. uv run python -m pytest tests
```

Tests that require external data or a GPU are skipped when those resources are
unavailable. Run the following only on an available GPU. For G1:

```bash
PBFM_TEST_RESET_GPU=1 PBFM_TEST_RESET_DATA=/path/to/lafan_29dof_10s-clipped.pkl \
  PYTHONPATH=. MUJOCO_GL=egl uv run python -m pytest \
  tests/test_reset_transitions.py -k real_mjlab
```

To compare G1's effective configuration against a separate published checkout:

```bash
PBFM_G1_BASELINE=/path/to/published-checkout PYTHONPATH=. uv run python -m pytest \
  tests/test_g1_migration_parity.py
```

These checks verify implementation behavior, not policy convergence or real-robot
safety. See [Inference and deployment export](INFERENCE.md) for deployment input
and calibration requirements.
