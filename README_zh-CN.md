# P-BFM

旧大网络 PBFM 的精简发布版。只整理职责、资源路径和入口，不改网络数学、奖励、
DR、深度处理或 checkpoint 参数键。正式代码位于 `main`，网站仍由 `pages` 发布。

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
旧 experiments、服务器 watcher、迁移记录和重复入口不进入此版本；原始归档没有删除。

## 保留的训练设置

Actor/F=2048×6；QD=1024×4，QAux=2048×4，QH/QV=1024×3。
四项独立 Critic，每项保留双 Q。BCE/log-odds，D=.01、Aux=.02、QH/QV=.002；
动态 Aux、LaFAN-only、slippage=-4、stumble=-2、Crop18、速度估计与深度监督不变。

从头训练命令见 [README](README.md)。使用专用 scratch 入口，课程从 0 开始、总计
192020480 步。不要直接把保留的 384M resume 模板当作 192M 从头训练配置。

推理和导出见 [INFERENCE](docs/INFERENCE.md)。本历史版的完整 policy ONNX 内含
深度编码器与速度估计器；Backward 单独导出，不应误当成只含动作网络的 split ONNX。

本次整理不加入后来 QV 人工窗口、replay/reset 边界修复，不启动或修改正在运行的训练。
验证范围和历史结果见 [LEGACY_LARGE](docs/LEGACY_LARGE.md)。
