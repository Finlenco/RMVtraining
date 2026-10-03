#include "StudentTasks.hpp"

#include "ArmorDetector.hpp"
#include "ArmorSolver.hpp"
#include "EkfTracker.hpp"
#include "MVS.hpp"

bool get_pic(cv::Mat& pic) {
    // TODO(student)：在这里完成相机的一次性初始化/打开/启动，随后获取一帧并转换为 BGR。
    // 无论成功还是失败，都要释放 SDK 缓冲区
    return MVS::getframe(pic);
}

std::vector<ArmorDetection> armor_detect(const cv::Mat& image, TeamColor enemy_color) {
    // TODO(student)：实现传统灯条匹配与数字分类，或使用离线神经网络检测器。
    // 拒绝格式错误或超出图像范围的角点集合。
    return armor_detector::detect(image, enemy_color);
}

std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection>& detections,
                                  const CameraParameters& camera, const GimbalState& gimbal) {
    // TODO(student)：按装甲板尺寸建立物点（毫米转换为米），调用 solvePnP，
    // 拒绝深度或重投影误差异常的结果，再应用已标定的刚体变换。
    return armor_solver::solve(detections, camera, gimbal);
}

PredictionResult ekf_predict(const std::vector<ArmorPose>& observations,
                             const GimbalState& gimbal,double timestamp_seconds) {
    
    return ekf_tracker::update(observations, gimbal, timestamp_seconds);
}
