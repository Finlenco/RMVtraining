# RMVtraining：视觉组装甲板自瞄大作业

本仓库是新赛季视觉组个人大作业。作业目标是把电控发送的云台状态、相机图像和装甲板观测串成一条可以现场运行的装甲板自瞄链路：

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



## 文档导航

- [作业任务书](docs/大作业任务书.md)：要求、固定接口和提交物。
- [验收与评分](docs/验收与评分.md)：100 分评分表和必验场景。
- [通信结构体](通信结构体.md)：64 字节字段、偏移、单位和安全回包。
- [轨迹预测说明](轨迹预测说明.md)：旋转车体 EKF 的最低要求。
- [环境依赖](环境依赖.md)：基础依赖和现场前检查。
- [相机到云台外参](云台外参.md)：外参定义、坐标变换和标定材料要求。
- [装甲板尺寸](装甲板尺寸.txt)：小装甲板和大装甲板的物理尺寸。


## 工程结构和代码入口

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
├── tests/                           # 测试
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



## 构建方式

```
cmake -S . -B build-rmvtraining 
cmake --build build-rmvtraining -j
```

## 运行

```
./build-rmvtraining/rmv_training /dev/ttyACM0 115200 config/camera.yaml 8080
```

## 已实现的功能

- 相机 SDK 配置与内参标定

- 相机初始化、取流、像素格式转换及异常处理

- 装甲板检测、数字分类
- PnP 位姿解算
- EKF预测
## 参考项目
https://github.com/chenjunnn/rm_auto_aim

https://github.com/sanlanx/RMVtraining
