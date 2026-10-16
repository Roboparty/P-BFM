# P-BFM

**Perceptive Behavioral Foundation Models for Humanoid Control via Unsupervised Reinforcement Learning**

- Project website: https://roboparty.github.io/P-BFM/
- Affiliation: RoboPartyLab
- Contact: xuewangusst@gmail.com

## Repository branches

- `main`: G1 training and inference code, based on the historical large-network PBFM implementation.
- `pages`: project website, paper, figures, and demonstration videos. GitHub Pages publishes from the root of this branch.

Website updates should be committed to `pages`, independently of code changes on `main`.

[中文说明](README_zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Inference/export](docs/INFERENCE.md) · [Baseline and limitations](docs/LEGACY_LARGE.md)

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
└── docs/                  Usage, architecture and provenance
```

There are no server watchers, historical experiment launchers or run outputs in
this release. Configuration files are bundled under `pbfm/configs/`, including in
the installed package; downloaded motions, weights, replay and videos stay external.

## Preserved model and settings

| Component | Hidden width × layers |
| --- | --- |
| Actor / Forward F | 2048 × 6 |
| QD | 1024 × 4 |
| QAux | 2048 × 4 |
| QH / QV | 1024 × 3 |

The four critics are independent; each retains two Q estimates. The sole preset
is `fb_depth`: BCE/log-odds, D=.01, Aux=.02, QH=QV=.002, dynamic Aux normalization,
LaFAN-only, slippage=-4, stumble=-2. Depth is Crop18 (`8×18×32`), CNN stride 1,
frame feature 256 and GRU feature 256. The velocity estimator is depth-supervised.
Network mathematics, rewards and DR are unchanged by the directory cleanup.

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
  scripts/train_legacy_scratch.py --gpu-ids single --terrain-mode rp1_simple \
  --num-envs 1024 --num-env-steps 192020480 --seed 4831 \
  --data-path /path/to/lafan_29dof_10s-clipped.pkl \
  --work-dir /path/to/new-run
```

Add `--use-wandb` after authenticating separately. Add `--smoke` with a different
new work directory for a 32768-step short test. These commands do not stop other
users' jobs or wait for free GPUs.

**Schedule warning:** `pbfm/configs/legacy_large.json` preserves the original
384M resume template. The scratch entry point above rebases the curriculum to
step 0 and defaults to 192020480 total global steps. A plain `python -m pbfm.train`
retains the archived schedule; it is not the same scratch experiment.

## Inference and export

```bash
uv run python -m pbfm.inference.tracking --help
uv run python -m pbfm.inference.goal --help
uv run python -m pbfm.inference.reward --help
uv run python -m pbfm.evaluation.terrain --help
uv run python -m scripts.export_depth_policy_onnx --help
uv run python -m scripts.export_backward_encoder_onnx --help
```

Tracking selects terrain and camera settings in the same entry point. Export
retains the historical interface: the production policy ONNX includes depth
encoding and velocity estimation; the backward encoder is exported separately.
Input normalization, context and hardware control requirements are described in
[the inference guide](docs/INFERENCE.md).

## Verification and scope

```bash
PYTHONPATH=. uv run python -m pytest tests
```

External-data/hardware checks are explicitly skipped when their inputs are
unavailable. CPU and small-model ONNX tests are not full training or real-robot
validation. The network and QV-window objective remain historical. The replay/reset
path now preserves pre-reset successors; old replay cannot be resumed as new-format
training data. Explicit successor storage increases replay memory use.
See [the baseline record](docs/LEGACY_LARGE.md) before comparing experiments.

PBFM builds on [UFO](https://github.com/Roboparty/UFO). Original notices and the
upstream CC BY-NC 4.0 [license](LICENSE) are retained.
