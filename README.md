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

P-BFM is a perceptive behavioral foundation model for terrain-adaptive humanoid control. It maps **motion references, target body states, and reward functions** into a shared intent space. A single frozen policy combines these intents with depth perception and explicit heading and velocity instructions to execute whole-body behaviors across terrains.

Training uses **40 flat-ground LaFAN motion sequences and online interaction with simulated terrains**, without terrain-specific motion demonstrations. The policy transfers zero-shot to a **Unitree G1 with onboard depth sensing**, supporting motion tracking, goal reaching, reward inference, teleoperation, and terrain-aware recovery.

[Architecture](docs/ARCHITECTURE.md) · [Training](docs/TRAINING.md) · [Inference / export](docs/INFERENCE.md)

## Layout

```text
P-BFM/
├── pbfm/                  Runtime package
│   ├── train.py           Training CLI
│   ├── configs/           One home for model, simulator, robot and depth settings
│   ├── agents/            Actor, FB, discriminator, Q critics and replay
│   ├── training/          Collection, updates, checkpointing and training evaluation
│   ├── envs/              MJLab bridge, observations, resets and physical rewards
│   ├── perception/        Depth processing and camera utilities
│   ├── terrains/          Terrain generation and curriculum
│   ├── motion/            Motion loading, conversion and reference observations
│   ├── robots/            G1 asset parsing and validation
│   ├── inference/         Tracking / goal / reward; shared runtime and ONNX export
│   ├── evaluation/        Offline metrics, terrain comparisons and diagnostics
│   ├── data/              Bundled G1 assets and a small CSV example
│   └── utils/             Shared numerical and resource helpers
├── scripts/               Launch, download, export and verification commands
├── tests/                 Regression tests and their small fixtures
└── docs/                  Usage and architecture
```

Configuration files are bundled under `pbfm/configs/`, including in
the installed package; downloaded motions, weights, replay and videos stay external.

## Model and settings

| Component | Hidden width × layers |
| --- | --- |
| Actor / Forward F | 2048 × 6 |
| QD | 1024 × 4 |
| QAux | 2048 × 4 |
| QH / QV | 1024 × 3 |

The four critics are independent; each uses two Q estimates. The training preset
is `fb_depth`: BCE/log-odds, D=.01, Aux=.02, QH=QV=.002, dynamic Aux normalization,
LaFAN-only, slippage=-4, stumble=-2. Depth is Crop18 (`8×18×32`), CNN stride 1,
frame feature 256 and GRU feature 256. The velocity estimator is depth-supervised.

## Install and train

Python 3.10, MJLab 1.4, MuJoCo 3.8, PyTorch 2.7. Install from the repository root:

```bash
uv sync --extra test
# Optional: download the separately hosted LaFAN files.
bash scripts/download_data.sh
```

Fresh 192M training (eight free GPUs, GPU replay; edit the data/output paths):

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MUJOCO_GL=egl NCCL_NVLS_ENABLE=0
export UFO_CACHE_DIR=/path/to/writable-disk/pbfm-cache

uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_legacy_scratch.py --gpu-ids single --terrain-mode training_mix \
  --num-envs 1024 --num-env-steps 192020480 --seed 4831 \
  --data-path /path/to/lafan_29dof_10s-clipped.pkl \
  --work-dir /path/to/new-run
```

Add `--use-wandb` after authenticating separately. Add `--smoke` with a different
new work directory for a 32768-step short test. These commands do not stop other
users' jobs or wait for free GPUs.

Use the scratch entry point above for training from step 0: it sets 192020480
total global steps and starts the terrain curriculum at step 0.
See [Training](docs/TRAINING.md) for schedule and checkpoint requirements.

## Inference and export

```bash
uv run python -m pbfm.inference.tracking --help
uv run python -m pbfm.inference.goal --help
uv run python -m pbfm.inference.reward --help
uv run python -m pbfm.evaluation.terrain --help
uv run python -m scripts.export_depth_policy_onnx --help
uv run python -m scripts.export_backward_encoder_onnx --help
```

Tracking selects terrain and camera settings in the same entry point. The policy ONNX includes depth
encoding and velocity estimation; the backward encoder is exported separately.
Input normalization, context and hardware control requirements are described in
[the inference guide](docs/INFERENCE.md).

## Tests

```bash
PYTHONPATH=. uv run python -m pytest tests
```

External-data/hardware checks are explicitly skipped when their inputs are
unavailable. CPU and small-model ONNX tests are not full training or real-robot
validation.

PBFM builds on [UFO](https://github.com/Roboparty/UFO). Original notices and the
upstream CC BY-NC 4.0 [license](LICENSE) are retained.


## Project and community

- Project website and demos: https://roboparty.github.io/P-BFM/
- Paper: https://arxiv.org/abs/2610.15833
- Team: [RoboPartyLab](https://lab.roboparty.com/)
- Contact: [xuewangusst@gmail.com](mailto:xuewangusst@gmail.com)
- To join our discussion group, add **15251239451** on WeChat.

## Repository branches

- `main`: PBFM training, evaluation and inference for Unitree G1.
- `pages`: project website, paper, figures, and demonstration videos. GitHub Pages publishes from the root of this branch.

Website updates should be committed to `pages`, independently of code changes on `main`.
