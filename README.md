# RMVtraining：视觉组装甲板自瞄大作业

本仓库是新赛季视觉组个人大作业的起始工程。作业目标是把电控发送的云台状态、相机图像和装甲板观测串成一条可以现场运行的装甲板自瞄链路：

```text
串口控制器帧
    │
    ▼
读线程保存最新云台状态
    │
    ▼
操作线程：取流 → 检测/分类 → PnP → EKF 预测 → 64 字节回包
    │                                      │
    └──────────── Web Debug 数值观测 ◄──────┘
```

仓库已经提供串口、双线程主循环、通信结构体、Web Debug 和通用 EKF 框架；学生需要完成或按自己的方案改进相机取流、装甲板识别、位姿解算、旋转车体跟踪和最终调参。编译通过或能播放一段视频，都不能单独证明现场功能已经完成。

## 文档导航

- [作业任务书](docs/大作业任务书.md)：要求、固定接口和提交物。
- [验收与评分](docs/验收与评分.md)：100 分评分表和必验场景。
- [通信结构体](通信结构体.md)：64 字节字段、偏移、单位和安全回包。
- [轨迹预测说明](轨迹预测说明.md)：旋转车体 EKF 的最低要求。
- [环境依赖](环境依赖.md)：基础依赖和现场前检查。
- [相机到云台外参](云台外参.md)：外参定义、坐标变换和标定材料要求。
- [装甲板尺寸](装甲板尺寸.txt)：小装甲板和大装甲板的物理尺寸。

## 1. 作业边界

### 已提供的基础设施

- `serialPort/`：Linux 串口打开、8N1 配置、超时、短读缓存和完整写入。
- `serialPort/include/Communication.hpp`：固定的 `MessData_AutoAim` 和 `Translator`，线上帧长必须保持 64 字节。
- `main.cpp`：首帧等待、读线程、操作线程、共享数据保护、结果合法性检查和资源退出。
- `web_debug/`：本机数值曲线 Web 服务，`WEB_LOG(key, value)` 是唯一需要使用的记录接口。
- `armor/include/KalmanFilter.hpp` 和 `armor/src/KalmanFilter.cpp`：可复用的通用扩展卡尔曼滤波器。
- `CMakeLists.txt`：最小可编译的工程和测试目标。

### 学生需要负责的内容

- 编译器、CMake、OpenCV、Eigen、HIKROBOT MVS SDK 和推理运行时的安装与路径配置；
- 相机枚举、打开、参数设置、开始取流、像素格式转换、超时和关闭；
- 相机内参、畸变参数、相机到云台外参和误差验证；
- 颜色/亮度处理、灯条匹配、角点生成、大小分类和数字分类；
- 按小/大装甲板尺寸进行 PnP，计算距离、角度和重投影误差；
- 旋转车体 EKF：初始化、预测、关联、切换、离群拒绝、丢失外推和重置；
- 将内部 SI 单位结果映射回原通信结构体，并在不稳定或异常时禁止开火；
- 实验记录、参数来源、测试结果、已知问题和答辩说明。

禁止修改通信结构体的字段、类型、顺序、对齐方式或 64 字节总长度。`crc` 在本作业中是 EKF 状态字段，不是 CRC 校验和。

## 2. 工程结构和代码入口

```text
.
├── main.cpp                         # 程序入口、双线程和完整处理顺序
├── CMakeLists.txt                   # CMake 目标、OpenCV/Eigen/MVS 依赖
├── include/                         # AutoAimTypes、AppSupport 和协议映射
├── src/                             # YAML 读取、结果合法性和安全回包
├── student/include/                 # 四个任务及检测、PnP、MVS、EKF 接口
├── student/src/                     # 当前取流、检测、分类、PnP 和 EKF 实现
├── student/model/                   # mlp.onnx 和 label.txt
├── armor/                           # 通用 EKF、弹道和规划参考代码
├── config/camera.yaml               # 当前本地相机标定文件
├── tools/mock_serial.py             # PTY 模拟电控板
├── tests/test_mock_serial.py        # 协议和分片解析测试
└── docs/                            # 任务书和验收标准
```

四个固定入口位于 `student/include/StudentTasks.hpp`：

```cpp
bool get_pic(cv::Mat& pic);

std::vector<ArmorDetection> armor_detect(
    const cv::Mat& pic, TeamColor enemy_color);

std::vector<ArmorPose> armor_solve(
    const std::vector<ArmorDetection>& detections,
    const CameraParameters& camera,
    const GimbalState& gimbal);

PredictionResult ekf_predict(
    const std::vector<ArmorPose>& observations,
    const GimbalState& gimbal,
    double timestamp_seconds);
```

入口函数之间的职责要保持清楚：`get_pic` 只取一张 BGR 图像，`armor_detect` 不做 PnP，`armor_solve` 不维护跨帧状态，`ekf_predict` 不访问相机也不修改原图。

## 3. 依赖和环境

- Linux；
- 支持 C++17 的编译器；
- CMake 3.16 或更高版本；
- OpenCV 4，至少需要 `core imgproc imgcodecs calib3d dnn highgui`；
- Eigen3 和 POSIX Threads；
- HIKROBOT MVS SDK，默认查找 `/opt/MVS/include` 和 `/opt/MVS/lib/64`。

当前数字分类器通过 OpenCV DNN 加载 `student/model/mlp.onnx`，模型和标签必须在现场离线可用。SDK 在其他目录时配置：

```bash
cmake -S . -B build-rmvtraining \
  -DMVS_ROOT=/path/to/MVS \
  -DENABLE_WEB_DEBUG=ON \
  -DBUILD_TESTING=ON
```

如果运行时报告找不到 `libMvCameraControl.so`，确认系统动态库配置，或在当前终端临时设置：

```bash
export LD_LIBRARY_PATH=/opt/MVS/lib/64:$LD_LIBRARY_PATH
```

## 4. 配置、构建和检查

### 4.1 相机 YAML

`loadCameraParameters()` 会检查：`calibrated=1`、图像宽高为正、相机矩阵为有限的 `3×3` 双精度矩阵、畸变系数个数为 OpenCV 支持的 4/5/8/12/14、旋转矩阵正交且行列式接近 `+1`、平移是有限的 `3×1` 米制向量，以及非负的重投影误差。

仓库当前只有 `config/camera.yaml`，没有通用的 `camera.yaml.example`。建议为自己的相机复制一份配置：

```bash
cp config/camera.yaml config/camera.local.yaml
```

这个文件中的分辨率、内参和外参只代表当前工作区的标定记录，不能直接套用到另一台相机或另一种安装方式。不要把全零旋转矩阵当作单位矩阵；缺失或未确认的旋转应保持无效并让程序拒绝启动。

### 4.2 干净目录构建

```bash
cmake -S . -B build-rmvtraining \
  -DMVS_ROOT=/opt/MVS \
  -DENABLE_WEB_DEBUG=ON \
  -DBUILD_TESTING=ON
cmake --build build-rmvtraining -j"$(nproc)"
```

可选检查：

```bash
ctest --test-dir build-rmvtraining --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

这些检查覆盖协议布局、PTY 分片解析、CLI 帮助和安全空实现，不覆盖真实相机、分类精度、PnP 标定质量、EKF 收敛或开火效果。`ENABLE_WEB_DEBUG=OFF` 可关闭 Web Debug，`BUILD_TESTING=OFF` 可不生成测试目标。

### 4.3 运行参数

```bash
./build-rmvtraining/rmv_training --help
```

参数顺序固定为：

```text
rmv_training <串口设备> <波特率> <相机标定 YAML> [Web 端口]
```

串口通常使用 `115200`、8N1，实际波特率以电控配置为准。

## 5. 启动顺序和运行方式

### 5.1 真机

```bash
./build-rmvtraining/rmv_training /dev/ttyACM0 115200 config/camera.local.yaml 8080
```

浏览器打开 `http://127.0.0.1:8080`。启动顺序是：读取并校验 YAML，打开串口，尝试读取最多 5 次合法首帧，创建读线程和操作线程，取流并执行检测/PnP/EKF，写回完整帧，退出时释放相机、Web、线程和串口。

正常操作依赖有效控制器帧。只有收到过且仍在 500 ms 内、`status` 为 `0` 或 `5`、姿态和 `bias` 为有限值的帧，操作线程才会进入 `get_pic()`。只连接相机而没有电控数据时，不会进入完整视觉处理链路。

### 5.2 伪串口

```bash
python3 tools/mock_serial.py \
  --app ./build-rmvtraining/rmv_training \
  --camera-config config/camera.local.yaml \
  --status 0 --rate-hz 50 --duration 10
```

蓝方和预测偏置示例：

```bash
python3 tools/mock_serial.py \
  --app ./build-rmvtraining/rmv_training \
  --camera-config config/camera.local.yaml \
  --status 5 --yaw 0.2 --pitch -0.1 --bias 0.04 --duration 10
```

工具只创建 PTY，不会打开实体串口。它能检查帧边界、字段编码、分片解析和程序回包；没有可用相机时，不能把它当作取流或跟踪验证。当前相机连续取流失败超过 3 次后，操作线程会发送安全帧并退出。

## 6. 通信协议和单位

### 6.1 64 字节布局

`MessData_AutoAim` 使用 1 字节对齐，帧头为 `0x71`，帧尾为 `0x4C`，`static_assert` 强制总长为 64 字节。

| 偏移 | 字段 | 类型 | 方向 | 单位/含义 |
| ---: | --- | --- | --- | --- |
| 0 | `head` | `uint8_t` | 双方 | `0x71` |
| 1/5/9 | `yaw/pitch/roll` | `float` | 电控 → 视觉 | 云台角度，rad |
| 13 | `status` | `uint8_t` | 电控 → 视觉 | `0` 红方，`5` 蓝方 |
| 14 | `is_far` | `uint8_t` | 电控 → 视觉 | 历史字段，当前不使用 |
| 15 | `armor_flag` | `uint8_t` | 视觉 → 电控 | 目标编号，`0` 无目标 |
| 16 | `latency` | `float` | 视觉 → 电控 | 处理耗时，ms |
| 20 | `bias` | `float` | 电控 → 视觉 | 预测时间偏置，s |
| 24 | `distance` | `float` | 视觉 → 电控 | 距离，m |
| 28 | `pitch_offset` | `float` | 保留 | 当前写 0 |
| 32/36 | `coo_x/coo_y` | `float` | 视觉 → 电控 | EKF 中心，mm |
| 40 | `wheel_w` | `float` | 保留 | 当前写 0 |
| 44 | `fire_allowance` | `uint8_t` | 视觉 → 电控 | `0` 禁止，`1` 允许 |
| 45/49 | `target_yaw/target_pitch` | `float` | 视觉 → 电控 | 目标角度，rad |
| 53/57 | `yaw_vel/pitch_vel` | `float` | 视觉 → 电控 | 目标角速度，rad/s |
| 61 | `crc` | `uint16_t` | 视觉 → 电控 | EKF 状态，不是校验和 |
| 63 | `tail` | `uint8_t` | 双方 | `0x4C` |

### 6.2 状态和安全回包

| 状态 | 含义 | 开火 |
| ---: | --- | --- |
| `crc=0` | `NoTarget`，无可用目标 | 禁止 |
| `crc=1` | `Stable`，跟踪稳定 | 还需 `fire_allowed=true` |
| `crc=2` | `Unstable`，已有目标但未稳定/正在外推 | 禁止 |

无目标、PnP 失败、控制器帧过期、结果含 `NaN/Inf` 或 EKF 失效时，至少清零 `armor_flag`、`fire_allowance`、`distance`、`coo_x`、`coo_y`、`yaw_vel`、`pitch_vel`，并写 `crc=0`。`applyPrediction()` 还会检查米到毫米转换后的浮点值。

## 7. 处理线程的完整数据流

```text
复制最新控制器帧
  ├─ 帧过期/状态非法/姿态非有限 → 安全回包
  ▼
get_pic(cv::Mat&)
  ├─ 失败：前几次用黑帧保持顺序，最终强制安全回包
  └─ 成功：记录 camera_fps
  ▼
armor_detect(image, enemy_color)
  ▼
armor_solve(detections, camera, gimbal)
  ├─ 空观测也继续向下传递
  ▼
ekf_predict(observations, gimbal, timestamp)
  ├─ 有观测：关联并更新
  └─ 无观测：短时纯预测，直到丢失策略触发
  ▼
applyPrediction(translator) → SerialPort::Write(64 bytes)
```

读线程只负责读取完整帧、检查 `0x71 ... 0x4C` 边界并更新共享 `temp`；两个线程通过互斥锁复制快照。不要无保护地同时读写同一个 `Translator`。

## 8. 相机取流：`get_pic()`

接口约定：成功时返回一张非空 `CV_8UC3`、BGR 排列的图像；超时、断开或转换失败时返回 `false`，不要在函数内部直接终止进程。

当前 `student/src/MVS.cpp` 的实现要点如下：

- 初始化 MVS SDK，只枚举 `MV_USB_DEVICE`；
- 通过序列号 `00D36741056` 选择相机；换相机时必须修改选择策略；
- 默认设置 `1440×1080`、曝光 `50000 us`、增益 `0`、Gamma `1.0`；
- 关闭触发模式后开始连续取流，单帧缓冲等待超时为 400 ms；
- 使用 `frame.stFrameInfo.enPixelType` 和 `nFrameLen` 作为源格式元数据，调用 MVS SDK 转成 `PixelType_Gvsp_BGR8_Packed`；
- 复制到 OpenCV 图像后当前代码执行一次 180° 翻转；相机安装方向改变时应重新确认；
- 无论转换成功与否，都要调用 `MV_CC_FreeImageBuffer()`；退出时由 `MVS::shutdown()` 停止取流、关闭设备、销毁句柄和反初始化 SDK。

这些值是当前实现的本地配置，不是作业固定答案。现场更换相机、分辨率或安装方向时，要同步检查内参、外参、曝光、图像方向和检测颜色阈值。

## 9. 装甲板检测和数字分类

### 9.1 统一输出

每个 `ArmorDetection` 必须包含：

- 四个像素角点，顺序为左上、右上、右下、左下；
- `ArmorSize::Small` 或 `ArmorSize::Large`；
- `target_id`，`0` 保留给未知/无效编号；
- `[0, 1]` 的 `confidence`。

检测器必须允许一帧没有目标，不能假定每帧都能找到装甲板。

### 9.2 当前传统视觉路线

`student/src/ArmorDetector.cpp` 当前实现采用：

1. 根据敌方颜色做 BGR 通道差分；
2. 灰度亮度阈值 `150` 与颜色阈值 `40` 相与；
3. 使用 `3×3` 矩形核做闭运算；
4. 提取外轮廓，用 `RotatedRect` 筛选灯条；
5. 按灯条中心 x 排序并两两匹配；
6. 依据长度比、中心距离和高度差过滤错误配对；
7. 按灯条间距/平均长度大于 `3.0` 判断大装甲板，否则为小装甲板；
8. 调用数字分类器，只有分类成功的结果才加入输出。

这些阈值可以根据镜头、曝光和场地调整，但报告中要记录调整依据，不要把一次场景的经验值写成通用标定常量。

### 9.3 当前 ONNX 分类器

`student/model/label.txt` 的类别顺序为：

```text
1, 2, 3, 4, 5, outpost, guard, base, negative
```

当前分类器将数字区域透视到小装甲板 `32×28` 或大装甲板 `54×28`，取中央 `20×28` 区域，转灰度、Otsu 二值化并归一化到 `[0, 1]`。OpenCV DNN 接收的张量形状为 `[1, 1, 28, 20]`。置信度低于 `0.5`、类别为 `negative` 或无法映射到编号时，分类结果被拒绝。

模型路径由 CMake 的 `RMVTRAINING_MODEL_DIR` 指向源码树中的 `student/model`。如果把可执行文件单独复制到别处，模型路径仍可能指向原构建时的源码目录；提交时应说明这一点，或改为可配置的运行时路径。

当前检测器还会调用 `cv::imshow("Armor Detection", ...)` 和 `cv::waitKey(1)` 显示本地窗口。无图形桌面或远程无 DISPLAY 时，这部分可能导致运行问题；验收前应在目标运行环境验证，必要时改成可关闭的调试选项。

## 10. PnP 位姿解算和坐标系

### 10.1 装甲板物点

`装甲板尺寸.txt` 的尺寸单位是 mm：

| 类型 | 宽 × 高 |
| --- | --- |
| 小装甲板 | `135 × 57 mm` |
| 大装甲板 | `230 × 57 mm` |

构造 `solvePnP` 物点时必须先转换为 m。当前 `ArmorSolver.cpp` 使用半宽 `0.0675 m`/`0.115 m` 和半高 `0.0285 m`，四点位于同一平面。

### 10.2 当前解算路径

对每个检测结果，当前实现会：

1. 根据大小选择物点；
2. 调用 OpenCV `solvePnP(..., SOLVEPNP_IPPE)`；
3. 用 `Rodrigues()` 得到装甲板姿态矩阵；
4. 保存相机坐标系位置 `position_camera_m`；
5. 应用固定外参 `p_gimbal_local = R_camera_to_gimbal * p_camera + t_camera_to_gimbal`；
6. 根据当前云台 yaw/pitch/roll 将局部云台坐标变到固定参考坐标；
7. 用姿态法向的水平投影计算 `armor_yaw`；
8. 用 `projectPoints()` 计算像素重投影误差。

`ArmorPose::position_gimbal_m` 是历史字段名；当前代码写入的是经过固定外参与当前云台姿态变换后的固定参考坐标。报告必须明确坐标轴方向、正负号、原点和单位，不能只依据字段名猜测坐标含义。

内部位置、外参平移、半径和速度统一使用 m 或 m/s；通信字段 `coo_x`、`coo_y` 再由 `applyPrediction()` 转为 mm。目标 yaw 已经在固定参考系中计算时，不要在 EKF 输出后再次叠加当前云台 yaw，否则会重复补偿。

### 10.3 观测拒绝

作业要求拒绝以下观测：

- `solvePnP` 失败；
- 深度或位置不是有限值；
- 目标在不可能的坐标范围；
- 重投影误差过大；
- 角点顺序、图像范围或目标编号非法。

当前 EKF 的 `validObservation()` 还会检查位置有限、`x > 0`、距离在 `0.05～30 m`、编号在 `1～255`，以及重投影误差小于 `20 px`（非正误差视为未提供）。如果把检测器或 PnP 换成自己的实现，建议在 `armor_solve()` 内尽早拒绝异常结果，减少错误观测进入关联器的机会。

## 11. 旋转车体 EKF

### 11.1 当前状态模型

当前 `student/src/KalmanFilter.cpp` 使用 11 维状态：

```text
[xc, vxc, yc, vyc, z_even, z_odd,
 re, ro, body_yaw, yaw_rate, vz]
```

- `xc, yc`：车体旋转中心位置，m；
- `vxc, vyc`：中心平移速度，m/s；
- `z_even, z_odd`：两组对置装甲板高度，m；
- `re, ro`：偶数/奇数槽位的旋转半径，m；
- `body_yaw`：车体 yaw，rad；
- `yaw_rate`：车体角速度，rad/s；
- `vz`：中心高度速度，m/s。

四个装甲板槽位的理论角度为：

```text
slot_yaw = body_yaw + slot * pi / 2
```

位置模型为：

```text
xa = xc - radius(slot) * cos(slot_yaw)
ya = yc - radius(slot) * sin(slot_yaw)
za = z_even 或 z_odd
```

四个槽位各提供 `(x, y, z, armor_yaw)`，因此观测向量为 16 维。通用 `ExtendedKalmanFilter` 负责矩阵计算；学生模型在 `student/src/KalmanFilter.cpp` 中提供过程模型 `f`、观测模型 `h`、两个雅可比矩阵、`Q`、`R` 和角度残差归一化。

### 11.2 当前跟踪策略

当前实现的主要策略如下：

- 初始化时从合法观测中选择重投影误差最小的一块装甲板；
- 初始旋转半径为 `0.25 m`，运行中限制在 `0.12～0.40 m`；
- 已有滤波器时先预测，再对四个槽位做一对一组合关联；
- 只把相同 `target_id` 的观测喂给当前滤波器，编号切换需要经过丢失后重新初始化；
- 位置关联门限为 `0.15 m`，yaw 关联门限为 `1.0 rad`；
- 缺失槽位使用预测值填充，并把该槽位测量噪声放大 100 倍；
- yaw 残差跨越 `-pi/pi` 时归一化，单帧 yaw 修正量和 yaw rate 变化也有限幅；
- 连续成功匹配 5 帧后进入 `Stable`；短时丢失输出 `Unstable` 并继续预测；
- 短时丢失超过约 5 帧进入 Lost，累计丢失超过 `50` 帧时重置滤波器；
- 预测时间由控制器 `bias` 加实际处理延迟得到，限制在 `0～0.2 s`，并对帧间变化做平滑；
- 由未来目标装甲板位置计算距离、target yaw/pitch 和角速度。

这些常量是当前代码的起点，不是无需验证的最优参数。调参时要区分检测抖动、PnP 误差、数据关联错误、EKF 噪声和预测延迟；不要仅凭编译成功或 Web 曲线有数据就判断跟踪质量。

### 11.3 输出和开火条件

`PredictionResult` 输出车体中心、速度、yaw、yaw rate、两种半径、目标角度和角速度。`applyPrediction()` 只有在结果完整且有限时才写回；`fire_allowance` 还要求 `TrackingState::Stable` 且 `fire_allowed` 为真。任何 `crc != 1` 的状态都必须禁止开火。

如果要加入弹道飞行时间或重力补偿，应说明子弹速度、模型、迭代方法、坐标轴和时间单位，并把它与 EKF 的处理延迟/bias 分开记录，避免重复加预测时间。

## 12. Web Debug

Web 服务默认只监听 `127.0.0.1`，端口默认为 `8080`，也可以通过第四个命令行参数修改。当前主循环记录的关键量包括：

| 类别 | key |
| --- | --- |
| 取流/耗时 | `empty_frame`, `camera_fps`, `frame_dt_ms`, `process_time_ms` |
| 检测 | `armor_count`, `detection_id`, `detection_confidence`, `target_x_px`, `target_y_px` |
| PnP | `pnp_reprojection_error`, `pnp_distance_m` |
| EKF 状态 | `target_found`, `ekf_state`, `target_id`, `ekf_x_m`, `ekf_y_m`, `ekf_vx_m_s`, `ekf_vy_m_s`, `ekf_yaw_rad`, `ekf_yaw_rate_rad_s`, `ekf_radius_1_m`, `ekf_radius_2_m` |
| 输出指令 | `target_yaw`, `target_pitch`, `target_distance` |

Web Debug 是数值遥测，不是视频流，也不提供在线调参。判断问题时建议按以下顺序查看：

1. `camera_fps`、`empty_frame`、`process_time_ms`；
2. `armor_count`、编号和置信度；
3. PnP 距离和重投影误差；
4. EKF 状态、yaw rate、半径和 `target_found`；
5. 最后再判断 `fire_allowance` 和控制器响应。

## 13. 验收和提交清单

### 13.1 现场必验

- 从新的构建目录完成 CMake 配置和编译；
- 串口持续收发，确认帧头、帧尾、字段映射和 EKF 状态；
- 相机启动、连续取流，并演示超时或断流后的安全处理；
- 静态小/大装甲板检测、数字分类和 PnP；
- 目标平移、旋转和装甲板切换时的 EKF 跟踪；
- 短时遮挡、完全消失和重新出现后的状态恢复；
- Web 页面连续显示关键曲线，刷新页面不阻塞视觉线程；
- 正常退出后读线程、操作线程、串口、相机和 Web 服务都释放。

### 13.2 应提交的材料

- 可以从干净目录构建的完整源码；
- 环境、版本、CMake 参数、运行命令和资源路径说明；
- 相机内参、畸变、外参、标定图片和重投影误差；
- 模型文件、标签文件、预处理流程和模型来源；
- 坐标系示意图、单位约定和通信字段映射；
- EKF 状态定义、过程/观测模型、Q/R/P0、关联门限和重置策略；
- 静态、平移、旋转、遮挡、丢失恢复等测试记录；
- Web Debug 截图或导出的曲线、已知问题和后续改进计划；
- 个人答辩材料，能够解释每个输入、输出和关键参数。

不要提交 `build/`、IDE 缓存、临时日志或与运行无关的大文件。引用开源代码或模型时注明来源，并能解释实际改动。

### 13.3 评分概览

| 项目 | 分值 | 核心证据 |
| --- | ---: | --- |
| 环境配置与工程 | 10 | 干净构建、依赖复现、资源释放 |
| 串口与双线程 | 10 | 64 字节协议、共享数据安全、正常退出 |
| 相机取流与标定 | 15 | 稳定取流、异常处理、内外参和误差 |
| 装甲板识别 | 25 | 角点、大小、数字、丢失处理和稳定性 |
| 位姿解算 | 15 | PnP、坐标变换、角度距离和异常拒绝 |
| EKF 跟踪与预测 | 20 | 旋转模型、关联切换、收敛和遮挡恢复 |
| Web Debug 与答辩 | 5 | 关键数据可观测、代码和设计可解释 |

## 14. 常见问题排查

### CMake 找不到 MVS

确认头文件和库文件存在：

```bash
ls /opt/MVS/include/MvCameraControl.h
ls /opt/MVS/lib/64/libMvCameraControl.so
```

不在默认目录时重新配置 `-DMVS_ROOT=...`，不要只修改 IDE 的私有环境变量。

### 程序提示相机取流失败

按顺序检查：USB 权限和设备枚举、序列号选择、分辨率是否被相机接受、MVS 动态库、像素格式转换返回值和相机是否被其他程序占用。当前代码只选择一个写死的 USB 序列号，并在操作线程中连续失败超过 3 次后退出。

### 只连接相机但程序没有取流

先确认是否有合法的控制器首帧。程序正常工作前需要串口输入 `status=0` 或 `status=5` 且四个控制浮点数有限；相机连接本身不会触发 `get_pic()`。

### YAML 被拒绝

检查 `calibrated`、矩阵维度和类型、旋转矩阵正交性/行列式、平移单位、畸变系数个数和 `mean_reprojection_error_px`。不要用全零矩阵表示“暂时未知的旋转”。

### 画面偏色或红蓝判断反了

确认 MVS 源像素类型是否交给 SDK 转换，不能无条件把所有输入当作 `BayerRG`。然后确认电控 `status` 到敌方颜色的映射：我方红（`0`）检测蓝色，我方蓝（`5`）检测红色，并重新检查曝光、增益和阈值。

### Web 页面没有目标曲线

先确认操作线程已经拿到有效控制器帧、相机没有连续失败、Web 端口没有被占用。`ENABLE_WEB_DEBUG=OFF` 时所有 `WEB_LOG` 都不会产生数据；Web 页面本身也不能证明检测和 EKF 正确。

### EKF 有输出但不能开火

查看 `ekf_state` 是否为 `1`。初始化阶段、短时丢失、观测关联失败、结果过期、出现非有限值或控制器状态非法都会让输出保持 `crc=0/2`，这是安全策略。确认检测连续性、PnP 重投影误差、编号稳定和预测延迟后，再调整 Q/R 或门限。

## 15. 代码修改原则

1. 先确认输入、输出、坐标系和单位，再改算法；
2. 保持 `StudentTasks.cpp` 只做接口转发，把模型和辅助逻辑放到对应的 student 文件；
3. 不在操作线程加入阻塞式打印、等待或无限重试；
4. 对相机、串口、模型、矩阵和浮点结果逐层检查，并在失败时回安全帧；
5. 将检测、PnP、关联、EKF 和硬件响应分开记录，避免把单一曲线当成整条链路的证据；
6. 每次改动后从新构建目录编译，并保留现场可复现命令；
7. 任何参数调整都记录场景、旧值、新值、指标和已知副作用。

完成这份作业的标准是：能够在目标硬件上稳定取流，识别和定位装甲板，解释 EKF 在旋转、切换和遮挡场景中的行为，并在结果不可信时始终输出安全状态。
