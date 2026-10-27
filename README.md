<h1 align="center">
  P-BFM: Perceptive Behavioral Foundation Models for Humanoid Control via Unsupervised Reinforcement Learning
</h1>

<p align="center">
  <a href="https://roboparty.github.io/P-BFM/"><img alt="Website" src="https://img.shields.io/badge/Website-Project_Page-2563eb?style=for-the-badge" /></a>
  <a href="https://roboparty.github.io/P-BFM/#overview"><img alt="Video" src="https://img.shields.io/badge/Video-Overview-7c3aed?style=for-the-badge" /></a>
  <a href="https://arxiv.org/abs/2610.15833"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2610.15833-b31b1b?style=for-the-badge" /></a>
  <a href="https://roboparty.github.io/P-BFM/interactive/"><img alt="Interactive demo" src="https://img.shields.io/badge/Demo-Try_the_Policy-0f766e?style=for-the-badge" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-CN.md">中文</a>
</p>

<p align="center">
  <img src="docs/assets/pbfm-wordmark.svg" alt="P-BFM" height="64" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://lab.roboparty.com/"><img src="docs/assets/robopartylab.png" alt="RoboPartyLab" height="64" /></a>
</p>

P-BFM maps **motion references, target body states, and reward functions** into a shared behavioral intent space. A policy combines that intent with depth perception and explicit heading and velocity instructions to adapt whole-body behavior to terrain.

The paper's G1 experiments use **40 flat-ground LaFAN motion sequences and online terrain interaction**, without terrain-specific motion demonstrations. This repository also integrates RP1 into the same PBFM algorithm and MJLab backend, with robot-specific assets, control, rewards and domain randomization (DR).

[Training and configuration](docs/TRAINING.md) · [Inference and deployment](docs/INFERENCE.md) · [Code structure](docs/ARCHITECTURE.md)

## Supported robots

**One algorithm, separate robot configurations, datasets and policies.** G1 remains the default; selecting RP1 does not change G1 settings.

| | Unitree G1 | RoboParty RP1 |
| --- | --- | --- |
| Controlled joints | 29 | 24 |
| Robot configuration | [g1_29dof.yaml](pbfm/configs/robots/g1_29dof.yaml) | [rp1_24dof.yaml](pbfm/configs/robots/rp1_24dof.yaml) |
| Depth stack, K × H × W | 8 × 18 × 32 | 8 × 18 × 36 |
| Reference motions | G1-retargeted LaFAN | RP1-retargeted motions, supplied separately |
| Validation scope | Platform used in the paper; current integration also smoke-tested | GPU rollout/reset, eight-rank training, three inference entry points and ONNX parity; long-run training and hardware validation pending |

G1 and RP1 weights and joint trajectories are **not interchangeable**. The RP1 integration uses assets and calibration from `zerojuhao/roboparty_train`; it does not reproduce the Newton backend or directly load that repository's trained policies. See [RP1 provenance and backend differences](pbfm/data/robots/rp1/SOURCE.txt).

## Installation

Requirements: Python 3.10, MJLab 1.4, MuJoCo 3.8 and PyTorch 2.7. Run commands from the repository root:

```bash
uv sync --extra test
```

Video export also requires `ffmpeg` on `PATH`; check with `ffmpeg -version`.

For G1, the optional downloader supplies the two LaFAN PKLs under `pbfm/data/`:

```bash
bash scripts/download_data.sh
```

It does **not** supply RP1 data. RP1 training requires a retargeted motion PKL with robot and joint/body ordering metadata; see [motion data requirements](docs/TRAINING.md#motion-data).

## Train

Use eight available GPUs and replace the cache/data/output paths for your machine:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MUJOCO_GL=egl NCCL_NVLS_ENABLE=0
export UFO_CACHE_DIR=/path/to/writable-disk/pbfm-cache
```

### G1

```bash
uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_legacy_scratch.py --gpu-ids single --terrain-mode training_mix \
  --robot-config pbfm/configs/robots/g1_29dof.yaml \
  --num-envs 1024 --num-env-steps 192020480 --seed 4831 \
  --data-path pbfm/data/lafan_29dof_10s-clipped.pkl \
  --work-dir /path/to/new-g1-run
```

### RP1

```bash
uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_legacy_scratch.py --gpu-ids single --terrain-mode training_mix \
  --robot-config pbfm/configs/robots/rp1_24dof.yaml \
  --num-envs 1024 --num-env-steps 192020480 --seed 4831 \
  --data-path /path/to/rp1_motions.pkl \
  --work-dir /path/to/new-rp1-run
```

Both commands start from scratch with 192,020,480 total global steps and the terrain curriculum starting at step 0. Replay is stored in CPU memory; check host memory and disk capacity before launching. For a short test, append `--smoke` and use a separate new output directory (32,768 global steps, reduced environments and replay). These commands do not wait for idle GPUs or stop existing jobs.

For W&B, authenticate separately and append `--use-wandb`. Network sizes, robot-specific rewards/DR, schedules and checkpoint requirements are in the [training guide](docs/TRAINING.md).

## Inference and deployment

The same three entry points select the robot from the saved run configuration:

```bash
uv run python -m pbfm.inference.tracking --help
uv run python -m pbfm.inference.goal --help
uv run python -m pbfm.inference.reward --help
```

Tracking accepts reference motions, terrain and camera settings. RP1 goal inference requires its own `--goal-json`; RP1 reward inference currently supports locomotion rewards. Computing a new reward latent requires replay samples, unlike tracking, goal inference and model export.

```bash
uv run python -m scripts.export_depth_policy_onnx \
  --checkpoint /path/to/run/checkpoint --output /path/to/export/policy.onnx
uv run python -m scripts.export_backward_encoder_onnx \
  --model-folder /path/to/run --output /path/to/export/backward_encoder.onnx --device cpu
```

Here, `policy.onnx` includes the **depth encoder, velocity estimator, observation normalization and Actor**; it is not an actor-only split graph. Deploy with the matching robot calibration and complete context, including heading/velocity commands and their validity flags. See the [inference guide](docs/INFERENCE.md) for examples, depth preprocessing and verification.

## Development and validation

```text
pbfm/    Shared algorithm, training, simulation, robot configs/assets and inference
scripts/ Training and export commands
tests/   Regression tests and small fixtures
docs/    Configuration, architecture and deployment guides
```

```bash
PYTHONPATH=. uv run python -m pytest tests
```

Tests needing external data or GPUs skip when those inputs are unavailable. Both robots have passed full-network eight-rank short training, followed by tracking/goal/reward rollouts from the saved checkpoints, video decoding and policy/backward ONNX parity checks. RP1 also has CPU interface/reward/DR and real GPU reset tests. These checks do not establish convergence, production-scale memory capacity or hardware safety. Test sizes and validation scope are in [Training](docs/TRAINING.md#tests-and-validation).

## Acknowledgments

PBFM builds on [UFO](https://github.com/Roboparty/UFO), retaining its notices and CC BY-NC 4.0 [license](LICENSE). RP1 assets retain their [source license](pbfm/data/robots/rp1/SOURCE_LICENSE); see the provenance file for asset redistribution considerations.

## Project and community

- Project website and demos: https://roboparty.github.io/P-BFM/
- Paper: https://arxiv.org/abs/2610.15833
- Team: [RoboPartyLab](https://lab.roboparty.com/)
- Contact: [xuewangusst@gmail.com](mailto:xuewangusst@gmail.com)
- To join our discussion group, add **15251239451** on WeChat.

## Repository branches

- `main`: PBFM runtime and robot configurations.
- `pages`: project website, paper, figures, and demonstration videos. GitHub Pages publishes from the root of this branch.

Website updates should be committed to `pages`, independently of code changes on `main`.
