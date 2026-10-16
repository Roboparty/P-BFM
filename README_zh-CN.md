# P-BFM

PBFM 面向 Unitree G1，结合深度感知、行为表示与显式方向和速度指令，
支持训练、评估以及 tracking / goal / reward 三种推理。代码位于 `main`，网站由 `pages` 发布。

## 从哪里开始读

| 要看什么 | 位置 |
| --- | --- |
| 训练怎么启动 | `scripts/train_legacy_scratch.py` → `pbfm/train.py` |
| 用了什么配置 | `pbfm/configs/legacy_large.json` |
| 网络、loss、Actor 更新 | `pbfm/agents/fb_cpr_aux/{config,model,agent}.py` |
| rollout、更新、存盘 | `pbfm/training/workspace.py` |
| 观测、reset、物理奖励 | `pbfm/envs/mjlab/` |
| 深度图、地形 | `pbfm/perception/`、`pbfm/terrains/` |
| 三种推理 | `pbfm/inference/{tracking,goal,reward}.py` |
| ONNX 实现 / 命令 | `pbfm/inference/export/` / `scripts/` |
| 指标和诊断 | `pbfm/evaluation/` |

顶层只有四个职责目录：`pbfm/`、`scripts/`、`tests/`、`docs/`。
配置统一放进包内 `pbfm/configs/`，确保安装后也能找到资源。
动作数据、模型、replay 和视频独立保存，不包含在代码仓库中。

## 训练设置

Actor/F=2048×6；QD=1024×4，QAux=2048×4，QH/QV=1024×3。
四项独立 Critic，每项保留双 Q。BCE/log-odds，D=.01、Aux=.02、QH/QV=.002；
动态 Aux、LaFAN-only、slippage=-4、stumble=-2、Crop18、速度估计与深度监督。

从头训练命令见 [README](README.md)。使用专用 scratch 入口，课程从 0 开始、总计
192020480 步。课程设置、续训和存储要求见 [训练指南](docs/TRAINING.md)。

推理和导出见 [INFERENCE](docs/INFERENCE.md)。完整 policy ONNX 内含
深度编码器与速度估计器；Backward 单独导出，不应误当成只含动作网络的 split ONNX。

PBFM 基于 [UFO](https://github.com/Roboparty/UFO)，保留上游版权声明及
CC BY-NC 4.0 [许可证](LICENSE)。
