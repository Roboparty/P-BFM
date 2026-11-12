# P-BFM G1 Deployment

**English** | [简体中文](README_zh-CN.md)

Real-robot deployment of P-BFM on the 29-DoF Unitree G1. A417 tracking, live teleoperation, reward and goal tasks share one policy, with different contexts selecting the behavior.

The repository includes camera acquisition, depth encoding, Actor inference, Pico retargeting, live context generation and web visualization. The common entry point is `bash run.sh`.

File-based tasks use NPZ contexts; live teleoperation uses ZMQ. Reward and goal tasks also use NPZ files. `ctx_path` can be absolute or relative to the repository root. Legacy pickle / joblib / torch contexts are no longer supported; PyTorch and joblib are not required.

Motor control supports only `ROBOT_TYPE=g1_real`. The policy and Encoder receive timestamped low-state data over ZMQ from a separate process. Legacy simulation commands, object mocap subscriptions and direct SDK state reads inside the policy have been removed. Offline dry runs and replay validation remain available.

## Runtime architecture

```text
RealSense → Training-ray sampling → Crop / Gaussian → Depth history → Depth Encoder ─┐
G1 low-state → Proprioceptive observations / Observation history ────────────────────┤
File context, or Pico → Retargeting → Backward Encoder → z / heading / v ────────────┤
                                                                                   ↓
                                                                        Actor → G1 motor commands
```

Camera acquisition, low-state reception, Depth Encoder, Actor, retargeting and live context generation run as separate processes connected by ZMQ. Internal streams carrying timestamps should run on the same robot computer.

| Component | Default |
|---|---|
| Camera | RealSense, 60 Hz |
| Low-state | Unitree SDK2, 200 Hz |
| Depth Encoder | ONNX Runtime CPU, 50 Hz |
| Actor | TensorRT FP32, 50 Hz |
| Live context | Backward Encoder + heading + velocity, 50 Hz |
| Depth viewer / Retargeting viewer | 8090 / 8080 |

## 1. Set up the environment

Use Python 3.10. Clone the `deploy` branch first, then run the following commands from the repository root:

```bash
git clone --branch deploy --single-branch --depth 1 https://github.com/Roboparty/P-BFM.git
cd P-BFM
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Also install these dependencies for Pico teleoperation:
python -m pip install -r requirements/teleop.txt
```

Direct and transitive Python dependencies are pinned in `requirements/`. These versions have been installed and tested in an isolated Linux x86_64 / Python 3.10 environment. Installation on Jetson ARM64 and the live hardware pipeline still need validation. See the [validation record (Chinese)](docs/validation.md) for the exact scope.

The target machine also needs the following native dependencies, matching its Jetson system and Python ABI:

| Dependency | Purpose |
|---|---|
| RealSense SDK, `pyrealsense2` | Depth acquisition |
| Unitree SDK2 C++, CycloneDDS | G1 motor interface |
| `unitree_sdk2py` | Independent low-state reception |
| TensorRT Python, `trtexec` | Build and run the Actor engine |
| XRoboToolkit PC Service, `xrobotoolkit_sdk` | Pico data; needed only for teleoperation |

Native dependencies are not included in the general pip requirements. The TensorRT build scripts target the 8.5 API; other versions have not been validated. When using a virtual environment, ensure that it can import the native Python modules listed above.

Motor control uses the `g1_interface` extension in this repository. Once the Unitree SDK2 CMake package is installed, build it with:

```bash
python -m pip install pybind11==3.1.0
cmake -S native/g1_interface -B build/g1_interface \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/unitree_sdk2 \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build build/g1_interface -j2
```

Adjust `CMAKE_PREFIX_PATH` if the SDK is installed elsewhere. For missing shared libraries, check `CYCLONEDDS_HOME`, `LD_LIBRARY_PATH` and the SDK paths. The standard `unitree_sdk2py` package does not replace this motor interface.

## 2. Verify the models and build the Actor

In each terminal, enter the repository root, activate the environment, and set the local network interface and extension path:

```bash
source .venv/bin/activate
export G1_INTERFACE=eth0
export UNITREE_SDK_LIB="$PWD/build/g1_interface"
```

`G1_INTERFACE` is the robot's internal network interface for low-level communication. Set it to the correct interface on your machine.

Model files belong in `model/g1_policy/onnx/`.

ONNX files are hosted at [xuewang/P-BFM](https://huggingface.co/xuewang/P-BFM) and are not included in Git. Download all three files at the pinned revision from the repository root:

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

`verify` checks the SHA256 hashes of all three models and every context. Failed or incomplete downloads will not pass verification; download them again before continuing. The pinned revision is also recorded in `model/g1_policy/manifest.json`. Offline archive packaging and extraction are described at the end of this README.

| File | Input | Output |
|---|---|---|
| `actor.onnx` | state 64, last_action 29, history_actor 372, heading 2, velocity_command 3, depth_latent 256, z 256 | action 29 |
| `depth_encoder.onnx` | float32 `[1,8,18,32]`, range `[0,1]` | depth_latent 256 |
| `backward_encoder.onnx` | Mimic observations produced by retargeting | z 256 |

```bash
bash run.sh verify
bash run.sh list
bash run.sh build
```

`verify` checks model/context hashes, the selected configuration and ONNX outputs. Use `POLICY_CONFIG` / `ROBOT_CONFIG` to select local configuration files. Observation ordering, PD gains, default joint positions, action scale and depth preprocessing must match the reviewed baseline in `model/g1_policy/deployment_contract.json`. `build` uses the engine path in the selected configuration, defaulting to `model/g1_policy/tensorrt/actor_fp32.engine`, and then compares its outputs with ONNX. Starting a real task checks the actual engine, configuration and task again. Building does not start motor control. Build the TensorRT engine on the target machine instead of copying one from another machine.

## 3. Start the camera, Encoder and depth viewer

Run these commands in three terminals with the environment configured:

```bash
# Terminal 1: camera and independent low-state reception
bash run.sh camera
```

```bash
# Terminal 2: CPU Depth Encoder
bash run.sh encoder
```

```bash
# Terminal 3: optional depth viewer
bash run.sh depth
```

The camera process automatically selects the only connected RealSense by default. If multiple devices are connected, select one with `CAMERA_SERIAL=SERIAL_NUMBER bash run.sh camera`.

Open **`http://ROBOT_IP:8090`** to view raw depth, sampled/cropped depth and the Gaussian result for the same camera frame. The viewer defaults to 10 Hz; use `bash run.sh depth --fps 5` to reduce overhead. It subscribes to existing data and recomputes the displayed processing stages. The right panel is not a direct capture of the Actor's depth history or latent.

Confirm that the camera and Encoder are producing a continuous stream before starting a task.

## 4. Tracking / reward / goal

For A417:

```bash
UFO_REAL_ROBOT_OK=1 bash run.sh real a417
```

| G1 remote button | Behavior |
|---|---|
| A | Initialize the pose |
| R1 | Enable the policy |
| B | Start playback |
| R2 | Latch the stop; release the buttons before using R1+B to re-enable |

A417 contains 1015 frames and lasts 20.3 seconds. It plays once, then holds the first frame. Starting the process does not automatically trigger these button actions.

List other tasks with `bash run.sh list` and use the same entry point:

```bash
UFO_REAL_ROBOT_OK=1 bash run.sh real 03_forward08_arms_spread
UFO_REAL_ROBOT_OK=1 bash run.sh real 01_goal21
UFO_REAL_ROBOT_OK=1 bash run.sh real 16_crouch40_forward1p0
```

These are alternative examples: run one task at a time. Stop and exit the current policy before switching tasks.

- `01_goal21`: R1 selects the default standing pose, B starts the first goal, X selects the next goal, and Y resets to standing.
- `16_crouch40_forward1p0`: R1 selects the default standing pose, B starts playback, and X returns to standing.
- Other reward files retain their own velocity targets, validity flags and end behavior. See `config/tasks/`.

## 5. Pico teleoperation

Start the installed XRoboToolkit service:

```bash
bash /opt/apps/roboticsservice/runService.sh
```

Connect Pico to the IP address of the robot running this service. Keep the camera and Encoder running, then start the following in separate terminals:

```bash
# Retargeting: set the operator's actual height in meters
ACTUAL_HUMAN_HEIGHT=1.75 bash run.sh retarget
```

```bash
# Live z, heading and velocity
bash run.sh z
```

```bash
# Motor policy
UFO_REAL_ROBOT_OK=1 bash run.sh real teleop
```

Retargeting viewer: **`http://ROBOT_IP:8080`**.

| Pico button | Behavior |
|---|---|
| A | Initialize |
| A+B | Enable / start |
| B | Latch the stop |
| X | Pause / follow |

Teleoperation estimates velocity from human displacement and publishes it together with z and heading. `velocity_command` is `[vx, vy, valid]` in the robot's current yaw frame. The heading observation is `[1−cos(error), sin(error)]`.

Each z dimension is limited to `max_z_delta=0.75` per update. File contexts use window=1, with no additional temporal averaging. During brief teleoperation stream outages, cached depth is retained while the reference transitions to the encoded standing pose. Automatic recovery is allowed after inputs remain fresh for 0.1 seconds and the reference reset is acknowledged. Long outages, producer restarts and explicit stops require re-enabling. Low-state failures do not use this fallback.

Pausing and resuming with X share the same pose transition, advanced using source timestamps. The default is 300 ms: `POSE_TRANSITION_MS=300 bash run.sh z`. Heading and velocity are temporarily invalid during the transition, and recovery waits for the reference to be ready; this recovery gating is distinct from deliberately pausing with X. The old `PAUSE_RAMP_MS` / `RESUME_RAMP_MS` options have been removed and cause an error if set. No additional z transition is applied after the pose transition. The `max_z_delta` limit and standing fallback on stream loss remain active.

The old `POSE_BUFFER_*` / `--pose-buffer-*` receive-time interpolation path has been removed. Pose transitions now use source timestamps.

## Depth processing and configuration

The depth pipeline uses `optical_z_maxcrop18_reset_v2`:

1. Use nearest sampling based on the training camera's rays and the real camera's intrinsics, retaining a 36×64 transport resolution.
2. Clip optical-Z depth to 2.5 m; set invalid values and positions outside the field of view to 2.5 m.
3. Crop 16 columns from each side and 18 rows from the top to obtain 18×32. Do not flip vertically.
4. Apply the training 3×3 Gaussian filter, convert to uint8, then divide by 255 for Encoder input.
5. Build an eight-frame history using capture timestamps, ordered from oldest to newest, covering 700–0 ms at 100 ms intervals.

| Setting | Location | Default |
|---|---|---|
| Encoder rate | `model/g1_policy/policy.yaml` → `depth_encoder_rate_hz` | 50 Hz |
| Gaussian | Same file → `direct_depth_preprocess.gaussian_enabled` | Enabled |
| Action filtering | Same file → `action_filter_alpha` | 1.0, disabled |
| Projected gravity bias | Same file → `projected_gravity_xy_bias` | `[0,0]` |
| Heading gain | Same file → `tracking_heading_yaw_gain` | 1.0 |
| Spatial / Temporal | `config/robot/g1_real.yaml` | Disabled |
| Joint order, PD, action scale | `config/policy/g1_policy.yaml` | Deployment baseline |
| Automatic teleoperation recovery | `model/g1_policy/teleop_policy.yaml` | Enabled, 0.1-second freshness window |

Override the Encoder rate temporarily with `bash run.sh encoder --rate-hz 60`; the history's time span remains unchanged. The current `verify` command rejects configurations that disable Gaussian filtering or change the preprocessing profile. Experiments with these settings require separate input-contract validation.

The code does not force camera exposure, gain or laser power; it uses the device's current settings.

## Stopping and troubleshooting

Stop the policy through the remote first, then press Ctrl+C in the policy terminal. Exit the Encoder, camera and viewers afterward. R2 is a software stop latch, not a physical emergency stop or a guarantee of firmware timeout behavior. The firmware command-timeout contract has not been validated on hardware.

When terminal keyboard input is enabled, `i` / `]` / `[` / `o` invoke the same handlers as remote buttons A / R1 / B / R2. `p` returns to the task's stop reference and resets the entire sequence in goal mode; `n` selects the next goal. `o` latches the stop. Pressing `]` or `[` alone cannot clear it; re-enable through the remote or restart the process.

The native `g1_interface` checks command updates using a monotonic clock. After the first command, a 100 ms gap without a new command stops DDS publication and latches the watchdog. Late commands do not clear the latch; the policy must be restarted. Startup rejects older extensions without this feature, so rebuild the extension. This prevents indefinitely resending stale targets when Python stalls, but **does not guarantee that the robot will remain standing after publication stops**. Verify the firmware's response on the target robot.

| Symptom | Check |
|---|---|
| Cannot import `g1_interface` or a shared library | Python ABI, `UNITREE_SDK_LIB`, SDK / DDS library paths |
| No low-state data or no response to A | Internal network interface, robot state, correct remote pairing |
| Stale depth or no latent | Camera and Encoder terminals, USB enumeration, model input dimensions |
| Pico connects but no motion follows | XRoboToolkit service, retargeting terminal, live z terminal, X state |
| Automatic return to standing | `STANDING FALLBACK` in policy logs and freshness of each input |
| Viewer does not refresh | Viewer process, network connectivity, continuous upstream frames |

## Repository layout and validation

```text
run.sh                       Common command entry point
config/                      Robot, observation, task and retargeting configuration
model/g1_policy/             ONNX, contexts, standing z, provenance and hashes
rl_policy/                   Actor control loop and observations
utils/                       Messages, clock synchronization, heading/QV, fallback
scripts/onboard/             Camera, low-state, Encoder, TensorRT build
scripts/teleop/              Pico retargeting and required assets
scripts/realtime/            Live context generation
scripts/vis/                 Depth viewer
native/g1_interface/         Motor interface source
data/robots/g1/              XML and meshes for FK and visualization
tests/                      Deployment regression tests
```

See `FILES.txt` for the complete file list and `model/g1_policy/manifest.json` for model hashes and provenance. Deployment invokes the verification script, so model reference cases and verification files are included in the repository.

Robot XML files share `data/robots/g1/meshes/`; meshes with different geometry revisions are retained separately. Base observations and PD gains live in `config/policy/g1_policy.yaml`. Runtime configuration lives in `model/g1_policy/policy.yaml`, and teleoperation overrides only the parameters it needs to change.

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -q tests
bash run.sh verify
```

Offline validation does not start the motors. The split networks have been checked against 32 reference cases from the original complete ONNX. Compilation on the target Jetson, live camera/Pico integration and physical robot behavior still need validation. Results are recorded in `validation_results.json`.

Replay tests include 256 frames from the original A417 simulation recording. Reconstructed joint/IMU packets enter the deployment observation pipeline to compare history, heading, z and actions against the original complete ONNX. Other tests cover synthetic raw depth through to the Actor, and X switching, stream loss and recovery in an actual z-server process. Simulation recordings do not include native camera or Pico data and cannot replace synchronized real-sensor replay.

## Release packaging

```bash
python scripts/package_release.py --output-dir ../PBFM-deploy-release
```

This produces a source archive, a model archive containing the three ONNX files, and `SHA256SUMS` for offline transfer. Put all three files in one directory, run `sha256sum -c SHA256SUMS`, then extract both archives into the same directory. The packaging script does not publish anything online. `FILES.txt` is the explicit packaging inventory; update it when adding release files. Engines, native `.so` files, local configuration and runtime logs are excluded from the archives.

## License

Retain the root `LICENSE` (CC BY-NC 4.0) and the licenses and source notices in third-party directories. ONNX files are hosted on [Hugging Face](https://huggingface.co/xuewang/P-BFM) and excluded from Git. TensorRT engines are built on the target machine.
