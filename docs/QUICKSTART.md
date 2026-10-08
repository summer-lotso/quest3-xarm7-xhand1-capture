# Quest 3 + xArm7 + XHand1 完整工程使用说明

本工程覆盖 Quest USB WebXR 接入、左手 XHand1 厂商重定向、xArm7 相对 TCP 控制、三路 RGB 采集、片段确认和 LeRobot Dataset v2.1 保存。使用四个受统一入口管理的进程；控制和录制仍在各自的循环中运行。

当前支持的是已有现场的 **左手 XHand1、RS485、Python 3.10、Linux x86_64**。机械臂的 home、TCP 和相机序列号都由现场配置提供。动作字段为 TCP 6 维加手关节 12 维，总共 18 维。详细数据流和端口见 [链路说明](QUEST3_TELEOP_CHAIN.md)。

## 文件结构

```text
configs/
  hardware.example.yaml       # 机器人、home、TCP、相机配置模板
  runtime.example.yaml        # 环境路径、任务、输出、控制参数模板
  vendor_usb.example.yaml     # 厂商单手 USB 配置模板
scripts/
  start_quest3_capture.py      # 统一入口和进程生命周期
  check_arm_ready.py          # 只读机械臂检查
  usb_webxr_relay.py          # Quest USB 接入和片段状态机
  run_official_vr_xarm_bridge.py
  run_official_xhand_with_telemetry.py
  record_vr_episode_stream.py
  package_capture.py          # 生成可交付代码包
src/xarm7_xhand1/              # 机器人适配、坐标映射、相机、保存和配置
tests/                        # 无实机回归检查
docs/                         # 安装、运行和链路说明
```

## 两个运行环境与外部依赖

| 组件 | 环境 | 必需依赖 |
|---|---|---|
| 启动器、xArm 桥接器、预检 | 控制环境 | `environment.yml` 中的固定依赖 |
| Quest 页面、中继、XHand | 厂商环境 | RobotEra `xhand_tele_ops`、`webxr`、`retargetx` 及配套原生库、授权文件 |
| v2.1 录制器 | 厂商环境 | PyAV、datasets、Torch 等已验证的数据集依赖；本地 LeRobot v2.1 源码 |

厂商程序和授权文件由 RobotEra 交付，本代码包提供其调用代码及配置模板。LeRobot 源码目录必须包含 `src/lerobot/datasets/lerobot_dataset.py` 且 `CODEBASE_VERSION` 为 `v2.1`；启动器会检查这一值。当前现场使用本地 0.3.4 源码树，包版本与 Dataset 格式版本分别检查。

控制环境可按以下方式创建。厂商环境按厂商交付的软件包说明安装，并在 runtime 配置里填写该环境的 Python 路径。

```bash
conda env create -f environment.yml
conda activate xarm7-xhand1-deploy
python -m pip install -e . --no-deps
```

如需用控制环境直接诊断 XHand，另安装 RobotEra 的 Python 3.10 `xhand_controller` wheel；统一遥操的手部路径使用厂商环境里的 `XHandTeleOps`。

## 配置一次，之后统一启动

```bash
cp configs/hardware.example.yaml configs/hardware.local.yaml
cp configs/runtime.example.yaml configs/runtime.local.yaml
```

填写硬件配置中的 xArm IP、已实测 home、TCP 偏移、手的可达 home、串口、厂商 torque 字段和相机序列号。runtime 配置填写厂商目录、厂商 Python、LeRobot 源码、任务描述与输出目录。YAML 中的相对路径以 YAML 所在目录为基准，支持 `~`；命令行参数可覆盖 YAML。

将 `configs/vendor_usb.example.yaml` 复制到已授权厂商目录作为 `config_meta_quest_usb.yaml`。核对串口与 hardware 配置指向同一设备，`start_web` 保持 `false`，授权文件按厂商配置放置。

先执行只读检查，再启动。统一入口会自动激活厂商子进程环境。

```bash
conda activate xarm7-xhand1-deploy
python scripts/start_quest3_capture.py \
  --runtime-config configs/runtime.local.yaml --check-only

python scripts/start_quest3_capture.py \
  --runtime-config configs/runtime.local.yaml \
  --execute --confirm I_HAVE_CLEARED_THE_WORKSPACE
```

`--check-only` 检查依赖、端口、Quest ADB、串口、相机、磁盘、机械臂静止状态、TCP 配置和控制器安全边界；检查完成后退出。运动许可参数只能在命令行显式传入。

## 在 Quest 上控制片段

Quest 用 USB 连电脑并允许 ADB。Browser 打开 `https://localhost:8010`，接受本地证书并进入 AR。左手追踪有效后，厂商重定向先完成预热。

1. 音量 **+**：左手静止校准约 3 秒，然后开始跟随和录制。
2. 音量 **−**：结束本条并等待确认。
3. 音量 **+** 保存，或音量 **−** 丢弃；臂和手回配置的 home。
4. 归位和录制器均就绪后开始下一条。

`Ctrl+C` 结束统一启动器，由它停止子进程。机械臂连接后即发送空闲心跳，开始录制仍需要有效的 Quest 左手追踪和片段开始事件。`--no-wear` 仅用于已经固定在支架上的 Quest，会在退出时恢复 proximity 设置。

## 输出与故障定位

默认录制 20 Hz、三路 640×480 RGB，保存时生成：

```text
dataset/
  meta/info.json                    # codebase_version=v2.1
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.<view>/episode_000000.mp4
```

同一 v2.1 目录可以续采，启动时核对 FPS 和字段。NPZ 备用保存方式通过 `--format npz` 选择，使用独立输出目录。已有数据目录和授权文件都不会进入代码交付包。

日志位于 `logs/quest3_capture_*.log`，每行标明 relay、xhand、xarm 或 recorder。`CHECK ARM` 查看机械臂心跳/控制器故障；`CHECK TRACKING` 查看 Quest AR 与左手数据；录制或预热未就绪时按日志提示等待或检查相应进程。端口被占用时应退出上一套流程。

## 验证与交付

```bash
python -m pytest -q
python scripts/package_capture.py
```

交付 ZIP 包包括应用源码、测试、配置模板、安装说明和逐文件 SHA256 清单。厂商 wheel、认证密钥、现场配置、日志和已采数据保留在各自部署机器上。
