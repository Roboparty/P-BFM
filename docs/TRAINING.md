# Training

## Configuration

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
| Precision / replay | FP32 / GPU |
| Data | LaFAN-only; supplied separately |
| Training | 8 ranks, 1024 environments/rank, seed 4831, 192020480 global steps |
| Curriculum | Starts at 0 for scratch, interval 19200000 global steps |
| Terrain ranges | Stairs 10–20 cm, rough 2–6 cm, boxes 5–15 cm high |

The training template is `pbfm/configs/legacy_large.json`.
`pbfm/configs/depth_contract.json` specifies the depth input contract.

## Train from scratch

Install dependencies from the repository root with `uv sync`. Use eight available
GPUs and edit the dataset, cache and output paths below for your machine.

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

The scratch entry point starts the terrain curriculum at step 0 and refuses a work
directory that already contains a checkpoint. The underlying template uses a
192020480 → 384040960 continuation schedule; a direct `python -m pbfm.train`
invocation does not apply the scratch schedule automatically.

For W&B, authenticate separately and append
`--use-wandb --wandb-project PBFM --wandb-run-name pbfm`.

For a short test, add `--smoke` and choose a separate output directory. This runs
32768 global steps with fewer environments and a smaller replay; evaluation and
replay checkpoint writing are disabled. Check GPU memory and disk space for the
full replay capacity before starting a longer run.

## Checkpoints and storage

Model/optimizer checkpoints and replay shards serve different purposes. Inference
and ONNX export require the model checkpoint, not replay. Training continuation
also depends on the replay format and depth observation contract.

Training replay must record `explicit_next=true`. If a buffer does not have this
field, keep it separate and load only the model into a new training directory,
rebuilding replay with a collection warmup before updates. Do not change the
metadata to force an incompatible buffer to load.

New replay uses `sparse_next=true`: ordinary successors reuse the following
collector row, while reset boundaries store their real pre-reset observations
separately. The newest successor is retained until another row arrives. Heading
and velocity command fields remain explicit because collection may replace them
before the next action. Each stored depth observation keeps one uint8 frame;
temporal stacks are reconstructed when sampling. Successor flags, qpos and qvel
remain explicit, including for reward inference. Sparse data is saved inside the
same `buffer.hdf5` as the replay; do not copy only selected datasets.

Existing `explicit_next=true` replay remains loadable in its original storage
format; it is not silently converted or deleted. Model checkpoints and ONNX
interfaces are unchanged. Savings depend on episode lengths and reset frequency.
Allow GPU memory for replay, model/optimizer state and training batches, and disk
space for each rank's replay checkpoint. A small-replay smoke test is not a
full-capacity memory check.

## Tests

```bash
PYTHONPATH=. uv run python -m pytest tests
```

Tests that require external data or a GPU are skipped when those resources are
unavailable. To run the real-simulator transition test on an available GPU:

```bash
PBFM_TEST_RESET_GPU=1 PBFM_TEST_RESET_DATA=/path/to/lafan_29dof_10s-clipped.pkl \
  PYTHONPATH=. MUJOCO_GL=egl uv run python -m pytest \
  tests/test_reset_transitions.py -k real_mjlab
```

These checks verify implementation behavior, not policy convergence or real-robot
safety. See [Inference and deployment export](INFERENCE.md) for deployment input
and calibration requirements.
