# P-BFM G1 Deployment

P-BFM 在 Unitree G1 29-DoF 上的真机部署代码。A417 tracking、实时 teleop、reward 和 goal 共用同一个 policy，通过不同 context 切换行为。

包含相机采集、深度编码、Actor 推理、Pico 重定向、实时 context 和网页可视化。统一入口为 `bash run.sh`。

文件任务统一使用 NPZ context，实时 teleop 使用 ZMQ；reward/goal 也是 NPZ 任务。`ctx_path` 为仓库根目录的相对路径或绝对路径。不再支持旧的 pickle / joblib / torch context，不依赖 PyTorch 或 joblib。

电机运行只支持 `ROBOT_TYPE=g1_real`，policy 和 Encoder 从独立进程的 ZMQ 通道接收带时间戳的 low-state。旧的仿真指令、物体动捕订阅及 policy 内直接读取 SDK 状态的路径已移除；离线 dry-run 和回放验证仍保留。

## 运行结构

```text
RealSense → 训练射线采样 → 裁剪 / Gaussian → 深度历史 → Depth Encoder ─┐
G1 low-state → 本体观测 / 观测历史 ───────────────────────────────────┤
文件 context，或 Pico → 重定向 → Backward Encoder → z / heading / v ─┤
                                                                  ↓
                                                          Actor → G1 电机指令
```

相机、low-state、Depth Encoder、Actor、重定向和实时 context 是独立进程，通过 ZMQ 通信。包含时间戳的内部数据流应在同一台机器人计算机上运行。

| 组件 | 默认设置 |
|---|---|
| 相机 | RealSense，60 Hz |
| Low-state | Unitree SDK2，200 Hz |
| Depth Encoder | ONNX Runtime CPU，50 Hz |
| Actor | TensorRT FP32，50 Hz |
| 实时 context | Backward Encoder + heading + velocity，50 Hz |
| 深度网页 / 重定向网页 | 8090 / 8080 |

## 1. 准备环境

Python 3.10。先取得 `deploy` 分支，以下命令在仓库根目录执行：

```bash
git clone --branch deploy --single-branch --depth 1 https://github.com/Roboparty/P-BFM.git
cd P-BFM
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# 使用 Pico teleop 时安装：
python -m pip install -r requirements/teleop.txt
```

直接依赖及其依赖版本已固定在 `requirements/`。这套版本已在隔离的 Linux x86_64 / Python 3.10 环境安装并测试；Jetson ARM64 的安装与硬件链路尚未复验。详细验证范围见 [验证记录](docs/validation.md)。

目标机器还需要以下原生依赖，与 Jetson 系统和当前 Python ABI 匹配：

| 依赖 | 用途 |
|---|---|
| RealSense SDK、`pyrealsense2` | 深度采集 |
| Unitree SDK2 C++、CycloneDDS | G1 电机接口 |
| `unitree_sdk2py` | 独立 low-state 接收 |
| TensorRT Python、`trtexec` | 构建和运行 Actor engine |
| XRoboToolkit PC Service、`xrobotoolkit_sdk` | Pico 数据，仅 teleop 需要 |

原生依赖不包含在通用 pip 依赖列表中。现有 TensorRT 构建脚本使用 8.5 接口，其他版本尚未验证。使用虚拟环境时，确保上述 Python 模块能被该环境导入。

电机控制使用本仓库的 `g1_interface` 扩展。安装好 Unitree SDK2 的 CMake package 后构建：

```bash
python -m pip install pybind11==3.1.0
cmake -S native/g1_interface -B build/g1_interface \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/unitree_sdk2 \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build build/g1_interface -j2
```

SDK 安装路径不同则修改 `CMAKE_PREFIX_PATH`。缺少共享库时，检查 `CYCLONEDDS_HOME`、`LD_LIBRARY_PATH` 和 SDK 路径。普通 `unitree_sdk2py` 不替代这个电机接口。

## 2. 检查模型并构建 Actor

每个终端都进入仓库根目录、激活环境，并设置本机网卡和扩展路径：

```bash
source .venv/bin/activate
export G1_INTERFACE=eth0
export UNITREE_SDK_LIB="$PWD/build/g1_interface"
```

`G1_INTERFACE` 是机器人内部低层通信网卡名称，应按实际机器填写。

模型文件位于 `model/g1_policy/onnx/`：

Git 仓库不包含 ONNX。配套模型位于 [xuewang/P-BFM](https://huggingface.co/xuewang/P-BFM)。在仓库根目录下载固定版本的三个文件：

```bash
MODEL_REVISION=738380cebfc26d0a93613ffbef394d061f84f936
mkdir -p model/g1_policy/onnx
for name in actor depth_encoder backward_encoder; do
  curl --fail --location --retry 3 \
    "https://huggingface.co/xuewang/P-BFM/resolve/${MODEL_REVISION}/${name}.onnx" \
    --output "model/g1_policy/onnx/${name}.onnx" || break
done
bash run.sh verify
```

`verify` 会检查三个模型和所有 context 的 SHA256。下载失败或文件不完整时不会通过校验，请重新下载后再继续。下载版本也记录在 `model/g1_policy/manifest.json`；离线压缩包的打包和解压方法见文末。

| 文件 | 输入 | 输出 |
|---|---|---|
| `actor.onnx` | state 64、last_action 29、history_actor 372、heading 2、velocity_command 3、depth_latent 256、z 256 | action 29 |
| `depth_encoder.onnx` | float32 `[1,8,18,32]`，值域 `[0,1]` | depth_latent 256 |
| `backward_encoder.onnx` | 重定向生成的 mimic 观测 | z 256 |

```bash
bash run.sh verify
bash run.sh list
bash run.sh build
```

`verify` 校验模型/context 哈希、实际选中的配置及 ONNX 输出。`POLICY_CONFIG` / `ROBOT_CONFIG` 可指定本机配置；观测顺序、PD、默认关节姿态、action scale、深度预处理等必须符合 `model/g1_policy/deployment_contract.json` 中的审核基线。`build` 使用选中配置的 engine 路径，默认生成 `model/g1_policy/tensorrt/actor_fp32.engine`，随后与 ONNX 对照。启动 real 会再次校验实际使用的 engine、配置和任务。构建不会启动电机控制；不要直接复制另一台机器的 TensorRT engine。

## 3. 启动相机、Encoder 和深度网页

在三个已配置环境的终端中分别执行：

```bash
# 终端 1：相机与独立 low-state 接收
bash run.sh camera
```

```bash
# 终端 2：CPU Depth Encoder
bash run.sh encoder
```

```bash
# 终端 3：可选的深度网页
bash run.sh depth
```

相机默认自动选择唯一的 RealSense；存在多台设备时用 `CAMERA_SERIAL=序列号 bash run.sh camera` 指定。

打开 **`http://机器人IP:8090`**，查看同一相机帧对应的原始深度、采样裁剪结果和 Gaussian 结果。网页默认 10 Hz，可用 `bash run.sh depth --fps 5` 降低开销。它订阅已有数据并重算显示阶段，右侧不是 Actor 历史帧/latent 的直接抓取。

确认相机、Encoder 正常持续出帧后，再启动对应任务。

## 4. Tracking / reward / goal

以 A417 为例：

```bash
UFO_REAL_ROBOT_OK=1 bash run.sh real a417
```

| G1 遥控器按键 | 行为 |
|---|---|
| A | 初始化姿态 |
| R1 | 启用 policy |
| B | 开始播放 |
| R2 | 锁定停止；停止后需先松开按键，再用 R1+B 重新启用 |

A417 共 1015 帧、20.3 秒，播放一次后保持第一帧。启动进程不会自动执行上述按键。

其他任务通过 `bash run.sh list` 查看，使用同一个入口：

```bash
UFO_REAL_ROBOT_OK=1 bash run.sh real 03_forward08_arms_spread
UFO_REAL_ROBOT_OK=1 bash run.sh real 01_goal21
UFO_REAL_ROBOT_OK=1 bash run.sh real 16_crouch40_forward1p0
```

这些命令是可选示例，一次运行一个任务。切换任务前先停止并退出当前 policy。

- `01_goal21`：R1 选择默认站姿，B 开始第一个 goal，X 下一个，Y 重置到站姿。
- `16_crouch40_forward1p0`：R1 选择默认站姿，B 播放，X 返回站姿。
- 其他 reward 文件保留各自的速度目标、有效标志和结束行为，见 `config/tasks/`。

## 5. Pico teleop

先启动已安装的 XRoboToolkit 服务：

```bash
bash /opt/apps/roboticsservice/runService.sh
```

Pico 连接机器人运行该服务的 IP。保持相机和 Encoder 运行，再分别启动：

```bash
# 重定向，按实际人体身高设置
ACTUAL_HUMAN_HEIGHT=1.75 bash run.sh retarget
```

```bash
# 实时 z、heading、velocity
bash run.sh z
```

```bash
# 电机 policy
UFO_REAL_ROBOT_OK=1 bash run.sh real teleop
```

重定向页面：**`http://机器人IP:8080`**。

| Pico 按键 | 行为 |
|---|---|
| A | 初始化 |
| A+B | 启用 / 开始 |
| B | 锁定停止 |
| X | 暂停 / 跟随 |

Teleop 根据人体位移估计速度，与 z、heading 一起发布。`velocity_command` 为机器人当前 yaw 坐标系中的 `[vx, vy, valid]`。heading 输入为 `[1−cos(误差), sin(误差)]`。

每维 z 更新限制为 `max_z_delta=0.75`。文件 context 的 window=1，没有额外的时间平均。Teleop 短暂断流时保留缓存深度并过渡到编码站姿；连续 0.1 秒输入新鲜且参考重置确认后允许自动恢复。长断流、生产者重启、主动停止需要重新启用；low-state 故障不使用这一回退。

X 暂停／恢复使用同一套姿态过渡，按源数据时间推进，默认 300 ms：`POSE_TRANSITION_MS=300 bash run.sh z`。过渡期间 heading / velocity 暂时无效，恢复逻辑会等待参考就绪；这与主动按 X 暂停不同。旧的 `PAUSE_RAMP_MS` / `RESUME_RAMP_MS` 已移除，设置它们会报错。没有在姿态过渡之后再叠加另一层 z 过渡；`max_z_delta` 限幅和断流站姿回退仍保留。

旧的 `POSE_BUFFER_*` / `--pose-buffer-*` 接收时间插值路径已删除；当前姿态过渡按源时间戳处理。

## 深度与配置

深度链路固定使用 `optical_z_maxcrop18_reset_v2`：

1. 按训练相机的射线和真实相机内参做 nearest 采样，保持 36×64 传输尺寸。
2. 按 optical-Z 截断到 2.5 m；无效值和视场外位置置为 2.5 m。
3. 左右各裁 16 列、上方裁 18 行，得到 18×32；不做上下 flip。
4. 使用训练对应的 3×3 Gaussian，转换为 uint8，再除以 255 输入 Encoder。
5. 按采集时间组成 8 帧历史，从旧到新覆盖 700～0 ms，间隔 100 ms。

| 设置 | 位置 | 默认值 |
|---|---|---|
| Encoder 频率 | `model/g1_policy/policy.yaml` → `depth_encoder_rate_hz` | 50 Hz |
| Gaussian | 同文件 → `direct_depth_preprocess.gaussian_enabled` | 开启 |
| Action 滤波 | 同文件 → `action_filter_alpha` | 1.0，关闭 |
| 重力投影偏置 | 同文件 → `projected_gravity_xy_bias` | `[0,0]` |
| Heading gain | 同文件 → `tracking_heading_yaw_gain` | 1.0 |
| Spatial / Temporal | `config/robot/g1_real.yaml` | 关闭 |
| 关节顺序、PD、action scale | `config/policy/g1_policy.yaml` | 部署基准配置 |
| Teleop 自动恢复 | `model/g1_policy/teleop_policy.yaml` | 开启，新鲜窗口 0.1 秒 |

频率也可临时覆盖：`bash run.sh encoder --rate-hz 60`，历史时间跨度保持不变。当前 `verify` 会拒绝关闭 Gaussian 或改变预处理 profile 的配置；要做这类实验，需要另外验证输入契约。

代码未强制写入相机曝光、增益和激光功率，使用设备当前设置。

## 停止与排查

先通过遥控器停止 policy，再在 policy 终端按 Ctrl+C；随后退出 Encoder、相机和网页。R2 是软件锁定停止，不等同于物理急停或固件超时保证。当前固件命令超时契约未完成真机验证。

启用终端键盘输入时，`i` / `]` / `[` / `o` 分别调用与遥控器 A / R1 / B / R2 相同的处理；`p` 回到任务的停止参考，goal 模式会重置整个序列，`n` 切到下一个 goal。`o` 会锁定停止，单按 `]` 或 `[` 无法解除，需通过遥控器重新启用或重启进程。

原生 `g1_interface` 使用单调时钟检查指令更新：发送首条指令后，连续 100 ms 没有新指令就停止 DDS 下发并锁定；迟到的新指令不会解除锁定，必须重启 policy。启动会拒绝缺少此功能的旧扩展，需要重新编译。这个保护防止 Python 卡住后持续重发旧目标，**不保证停发后机器人仍能站立**；固件的停发反应必须在目标机器上验证。

| 现象 | 检查位置 |
|---|---|
| `g1_interface` / 共享库导入失败 | Python ABI、`UNITREE_SDK_LIB`、SDK / DDS 库路径 |
| 收不到 low-state、A 无反应 | 内部网卡、机器人状态、遥控器是否配对正确 |
| 深度 stale / 无 latent | 相机终端、Encoder 终端、USB 枚举、模型输入尺寸 |
| Pico 有连接但无动作 | XRoboToolkit 服务、重定向终端、实时 z 终端、X 状态 |
| 自动回到站姿 | policy 日志中的 `STANDING FALLBACK` 及各输入的新鲜度 |
| 网页不刷新 | 对应网页进程、网络可达性、上游帧是否持续更新 |

## 目录与验证

```text
run.sh                       统一命令入口
config/                      机器人、观测、任务和重定向配置
model/g1_policy/             ONNX、context、站立 z、溯源与哈希
rl_policy/                   Actor 控制循环和观测
utils/                       消息、时间同步、heading/QV、回退
scripts/onboard/             相机、low-state、Encoder、TensorRT 构建
scripts/teleop/               Pico 重定向及其必要资源
scripts/realtime/             实时 context
scripts/vis/                  深度网页
native/g1_interface/          电机接口源码
data/robots/g1/               FK / 网页需要的 XML 和 mesh
tests/                       部署回归测试
```

完整文件列表见 `FILES.txt`。模型哈希和来源见 `model/g1_policy/manifest.json`。部署会调用校验脚本，因此模型参考样例和校验文件保留在仓库中。

机器人 XML 共用 `data/robots/g1/meshes/`，不同几何版本的 mesh 分别保留。基础观测与 PD 在 `config/policy/g1_policy.yaml`，运行配置在 `model/g1_policy/policy.yaml`，teleop 只覆盖需要变更的参数。

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -q tests
bash run.sh verify
```

离线验证不启动电机。已检查拆分网络与原始完整 ONNX 的 32 组参考输出；目标 Jetson 的编译、相机/Pico 链路和真机行为仍需复验。结果见 `validation_results.json`。

新增回放包括原始 A417 仿真记录的 256 帧：从重建的关节／IMU 数据包进入部署观测链，比较历史、heading、z 和原始完整 ONNX 的动作输出；另有合成原始深度到 Actor 的测试，以及实际 z-server 进程的 X 切换、断流、恢复测试。仿真记录不包含原生相机和 Pico 数据，不能替代完整真机录制回放。

## 打包发布

```bash
python scripts/package_release.py --output-dir ../PBFM-deploy-release
```

输出源码包、三个 ONNX 的模型包和 `SHA256SUMS`，可用于离线搬运。把三个文件放在同一目录，先运行 `sha256sum -c SHA256SUMS`，再依次解压两个包到同一目录。打包脚本本身不会发布到外网。`FILES.txt` 是明确的打包清单；新增需要发布的文件时同步更新它。engine、原生 `.so`、本地配置和运行日志不进入发布包。

## 许可证

保留根目录 `LICENSE`（CC BY-NC 4.0）以及第三方目录中的许可证和来源说明。ONNX 托管在 [Hugging Face](https://huggingface.co/xuewang/P-BFM)，不加入 Git；TensorRT engine 在目标机生成。
