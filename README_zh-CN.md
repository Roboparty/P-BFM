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

P-BFM 面向地形自适应的人形机器人全身控制，将**动作参考、目标身体状态与奖励函数**映射到统一意图空间。单一冻结策略结合深度感知与显式航向、速度指令，生成适应不同地形的运动执行。

训练以 **40 条平地 LaFAN 动作序列和在线仿真地形交互**为基础，无需地形专属的动作示范。策略零样本迁移到搭载机载深度相机的 **Unitree G1**，支持动作跟踪、目标到达、奖励推断、遥操作与地形相关的扰动恢复。

代码位于 `main`，项目网站由 `pages` 发布。

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


## 项目与交流

- 项目主页与演示：https://roboparty.github.io/P-BFM/
- 论文：https://arxiv.org/abs/2610.15833
- 团队：[RoboPartyLab](https://lab.roboparty.com/)
- 联系邮箱：[xuewangusst@gmail.com](mailto:xuewangusst@gmail.com)
- 如需加入交流群，请添加微信 **15251239451**。
