# Training

## Robot selection

G1 and RP1 share the PBFM learning algorithm and MJLab backend, not motion data
or checkpoints. Select the robot explicitly when creating a run:

| Robot | `--robot-config` | Motion data |
| --- | --- | --- |
| G1 (default) | `pbfm/configs/robots/g1_29dof.yaml` | G1-retargeted LaFAN |
| RP1 | `pbfm/configs/robots/rp1_24dof.yaml` | RP1-retargeted PKL, supplied separately |

Both use the same training entry point. Joint/body dimensions, control, camera,
physical rewards and DR are selected by robot configuration. A run saves its
robot calibration; inference and export retain that calibration.

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

### RP1-specific settings

RP1 retains the shared network widths, independent double critics and Actor
coefficients above, with input/output dimensions adapted to its 24 joints.
It does **not** reuse G1's physical reward or DR preset:

| Setting | RP1 |
| --- | --- |
| Depth | 8 × 18 × 36; normalization over 0.1–2.5 m; its own camera mount and image noise |
| Aux rewards | Nine terms selected by the RP1 YAML |
| PD gain randomization | 0.7–1.3 × nominal, at reset |
| Link mass/inertia | 0.8–1.2 × nominal, at startup |
| Torso COM | ±3 cm per axis, at startup |
| Sliding friction | 0.3–1.7, at startup |
| Motor delay | 0/1/2 physics ticks (0/5/10 ms), shared per episode |
| Torque noise | ±10% peak torque, before motor-envelope clipping |
| Push | Every 1–3 s; overwrite XY/angular velocity within ±1 m/s and ±1 rad/s respectively, vertical velocity zero |
| Camera installation DR | ±2 cm and ±[3,10,3]° roll/pitch/yaw, at startup |
| Forced lying reset | Disabled |

The nine Aux terms are action rate, contact-gated foot vertical motion, joint
acceleration, joint velocity, joint limits, body impact, slippage, ankle roll
and stumble. Their weights are defined in
[`rp1_24dof.yaml`](../pbfm/configs/robots/rp1_24dof.yaml), not inferred from a
checkpoint name. Details of contact calculations, image processing and the pinned
source are in [RP1 provenance](../pbfm/data/robots/rp1/SOURCE.txt).

This is RP1 training within PBFM/MJLab, not a reproduction of the source Newton
networks, dataset or physics. Original Newton RP1 weights are not directly loadable.

## Motion data

`bash scripts/download_data.sh` downloads G1 LaFAN data only. It does not retarget
G1 motions to RP1 or download an RP1 training dataset.

An RP1 PKL maps motion names to clip dictionaries. Each clip must use the RP1
skeleton and include `robot_name: rp1_24dof`, `joint_names`, and `body_names`
matching the selected robot specification, together with `fps`,
`root_trans_offset` (T × 3), and axis-angle `pose_aa` (T × 25 × 3).
Joint-angle data, when supplied as `dof_pos`, uses T × 24 and the same joint order.
Do not rename a G1 file or pass a source `.npz` directly as a PBFM training PKL.
Geometry/axis conversion and retargeting must precede training.

Always supply RP1 data with `--data-path /path/to/rp1_motions.pkl`. The loader
rejects mismatched robot/joint/body metadata. A training clip index is specific
to its dataset; the same numeric motion ID need not name the same motion in
G1 and RP1 data.

## Terrains

The seven-family terrain preset is named `training_mix` and is independent of
the robot model. Archived configs using `rp1_simple` are accepted as an alias.
Depth processing profile identifiers and source-provenance metadata are retained
so existing checkpoints keep the same input contract.

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

For RP1, replace the robot configuration with
`pbfm/configs/robots/rp1_24dof.yaml` and the data path with its retargeted PKL;
use a separate new work directory. Complete commands for both robots are in
the [README](../README.md#train).

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

For RP1, supply its own retargeted motion PKL. This covers real GPU rollout and
partial reset, with DR both disabled and enabled:

```bash
PBFM_TEST_RP1_GPU=1 PBFM_TEST_RP1_DATA=/path/to/rp1_motions.pkl \
  PYTHONPATH=. MUJOCO_GL=egl uv run python -m pytest \
  tests/test_rp1.py -k gpu_rollout_and_partial_reset
```

To compare G1's effective configuration against a separate pre-integration checkout:

```bash
PBFM_G1_BASELINE=/path/to/g1-only-checkout PYTHONPATH=. uv run python -m pytest \
  tests/test_g1_migration_parity.py
```

Integration validation recorded on 2026-10-22:

| Check | Scope |
| --- | --- |
| G1 configuration regression | Effective training, task, action, DR, actuator and depth configurations matched the pre-integration checkout |
| RP1 CPU tests | Interfaces, physics compilation, rewards/DR, a small-model update, checkpoint reload and ONNX runtime parity |
| RP1 GPU rollout | Actual simulation and reset-successor checks, DR off/on |
| RP1 single-GPU smoke | Full network, training terrain mix, 2,048 global steps and one optimizer update |
| G1 and RP1 eight-GPU smoke | Each used the full network and training terrain mix for 2,048 global steps and three optimizer updates; finite checked losses, zero rank parameter/buffer difference, checkpoint and all eight replay shards saved |
| Both robots: tracking CLI | Saved-checkpoint reload, reference encoding, heading/velocity context export and a 16-step stair rollout/video |
| Both robots: goal CLI | Three target frames encoded and switched at steps 0/4/8 during a 12-step rollout/video |
| Both robots: reward CLI | Locomotion reward recomputed from 32 saved replay samples, latent inference and a 12-step rollout/video |
| Both robots: production ONNX | Policy (including depth encoder and velocity estimator) and backward exports passed runtime parity; policy tests covered float and nonzero uint8 source depth at batch sizes 1/2/3 |

The latest eight-GPU checks ran on H20 hardware with four environments per rank,
batch size 32 per rank and replay capacity 4,096 per rank, using two real motion
clips per robot. Training kept robot-specific rewards and DR; inference disabled
DR and observation noise. All six videos decoded successfully. Maximum absolute
policy ONNX error was 3.88e-6 for G1 and 4.45e-6 for RP1; backward errors were
1.91e-6 and 2.63e-6, respectively (rounded upward).

These tests exercised the production configuration builder and training workspace
with bounded test overrides, not a full run of the default CLI schedule. They
are not completed RP1 policy training, an eight-GPU terrain benchmark, or a
full-capacity memory test. The standard `--smoke` command above runs a separate
32,768-step check. Videos generated with an original Newton RP1 checkpoint and
its original inference implementation are not validation of this migrated code.

These checks verify implementation behavior, not policy convergence or real-robot
safety. See [Inference and deployment export](INFERENCE.md) for deployment input
and calibration requirements.
