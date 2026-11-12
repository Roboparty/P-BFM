# 部署版本验证范围

当前代码在 Linux x86_64、Python 3.10.20 的全新虚拟环境安装验证。所有直接 Python 依赖已固定；`requirements/constraints.txt` 记录此次解析出的完整依赖版本。环境未继承原部署虚拟环境的 site-packages。

已执行的验证：

- 模型与 context SHA256；拆分网络与原始完整 ONNX 的 32 组输入对照。
- A417 仿真记录的前 256 帧回放，跨越上楼开始阶段：重建 low-state 数据包、观测及历史、heading、QV、z、Depth Encoder、Actor。参考动作由原始完整 ONNX 生成。
- 合成原始深度经训练射线采样、裁剪、Gaussian、时间历史和 Encoder 进入实际 policy 控制循环；深度与 low-state 过期检查。
- 真实 z-server 子进程接收合成重定向数据，检查暂停、跟随、参考重置、断流停发及恢复。进程只绑定 loopback，不创建电机接口。
- C++ 指令看门狗的超时边界、持续更新、迟到更新和锁定；本机编译、链接和导入完整 `g1_interface`，未构造 `G1Interface`。
- 固定版本的重定向网页依赖实际初始化，未连接 Pico。
- 精简后复验全部 18 项任务；检查键盘与遥控器共用的停止锁定、reward 站姿、goal 切换与重置，以及手动恢复拒绝未来或缺失时间戳。
- 用模拟的电机接口检查关节重排、PD、dry-run 不发送和关闭后拒绝发送；用合成 SDK 数据检查 low-state 重复帧、IMU 无效值和控制器时钟重启。未创建真实电机接口。

测试命令：

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -q tests
bash run.sh verify
```

编译看门狗测试需要系统 C++17 编译器。本机原生模块使用 Unitree SDK2 2.0.0 的既有源码快照和 x86_64 库；快照无可确认的 Git revision，因此不把它写成可复现的上游版本锁。该模块未打包分发，目标机按 README 自行构建。详细版本、快照指纹和结果记录在 `validation_results.json`。

尚未完成的部分：

- Jetson ARM64 的干净安装、原生模块构建、TensorRT engine 构建与运行。
- 目标 Jetson 的 JetPack / CUDA / TensorRT / RealSense / Unitree SDK2 / XRoboToolkit 精确版本组合及来源 revision。当前脚本针对 TensorRT 8.5；本机 Python 依赖锁不证明 ARM64 原生兼容。
- 实际相机内参、安装姿态、曝光及 USB 出帧；Pico SDK 和人体重定向链路。
- 包含相机原始帧、采集时间、真实 low-state、原始人体数据的同一时钟真机录制回放。
- 固件在 DDS 停发后的响应、失去支撑的风险和真机动作。100 ms 原生看门狗只保证本进程不继续发送过期目标，不能保证机器人稳定。

以上缺口保留为当前发布版本的限制。目标机验证记录补齐前，不应宣称真机端到端复现已通过。
