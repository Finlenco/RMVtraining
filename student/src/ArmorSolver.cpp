#include "ArmorSolver.hpp"

#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace {

// 将随云台转动的局部坐标系变换到固定参考坐标系。

cv::Mat gimbalToReferenceRotation(const GimbalState& gimbal) {
    const double cy = std::cos(static_cast<double>(gimbal.yaw));
    const double sy = std::sin(static_cast<double>(gimbal.yaw));
    const double cp = std::cos(static_cast<double>(gimbal.pitch));
    const double sp = std::sin(static_cast<double>(gimbal.pitch));
    const double cr = std::cos(static_cast<double>(gimbal.roll));
    const double sr = std::sin(static_cast<double>(gimbal.roll));

    const cv::Mat yaw = (cv::Mat_<double>(3, 3) <<
        cy, -sy, 0.0,
        sy,  cy, 0.0,
        0.0, 0.0, 1.0);
    const cv::Mat pitch = (cv::Mat_<double>(3, 3) <<
        cp, 0.0, -sp,
        0.0, 1.0, 0.0,
        sp, 0.0,  cp);
    const cv::Mat roll = (cv::Mat_<double>(3, 3) <<
        1.0, 0.0, 0.0,
        0.0,  cr, -sr,
        0.0,  sr,  cr);

    return yaw * pitch * roll;
}

std::vector<cv::Point3f> modelPoints(ArmorSize size) {
    // 装甲板尺寸：小 135x57 mm，大 230x57 mm。
    const float half_width = size == ArmorSize::Large ? 0.115F : 0.0675F;
    const float half_height = 0.0285F;

    return {
        {-half_width, -half_height, 0.0F},
        { half_width, -half_height, 0.0F},
        { half_width,  half_height, 0.0F},
        {-half_width,  half_height, 0.0F}
    };
}

//检查 PnP 解得准不准
double reprojectionError(const std::vector<cv::Point3f>& object_points,
                         const std::vector<cv::Point2f>& image_points,
                         const cv::Mat& rvec,
                         const cv::Mat& tvec,
                         const CameraParameters& camera) {

    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, camera.camera_matrix, camera.distortion_coefficients, projected);

    double squared_error = 0.0;
    for (std::size_t i = 0; i < image_points.size(); ++i) {
        squared_error += cv::norm(projected[i] - image_points[i]) *
                         cv::norm(projected[i] - image_points[i]);
    }
    return std::sqrt(squared_error / image_points.size());
}

} 

namespace armor_solver {

std::vector<ArmorPose> solve(const std::vector<ArmorDetection>& detections,const CameraParameters& camera,const GimbalState& gimbal) {
    
    std::vector<ArmorPose> poses;

    const cv::Mat gimbal_to_reference = gimbalToReferenceRotation(gimbal);

    for (const ArmorDetection& detection : detections) {
        const std::vector<cv::Point3f> object_points = modelPoints(detection.size);
        const std::vector<cv::Point2f> image_points(detection.corners.begin(),detection.corners.end());

        cv::Mat rvec;
        cv::Mat tvec;

        const bool solved = cv::solvePnP(object_points, image_points, camera.camera_matrix,camera.distortion_coefficients, rvec, tvec, false,cv::SOLVEPNP_IPPE);
        
        if (!solved) continue;

        cv::Mat rotation;
        cv::Rodrigues(rvec, rotation);//转换为旋转矩阵

        const cv::Vec3d position_camera(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        cv::Vec3d position_gimbal = position_camera;
        if (camera.rotation_camera_to_gimbal.rows == 3 && camera.translation_camera_to_gimbal.rows == 3) {
            cv::Mat camera_point = (cv::Mat_<double>(3, 1)<< position_camera[0], position_camera[1],position_camera[2]);
            const cv::Mat gimbal_point = camera.rotation_camera_to_gimbal * camera_point +
                                         camera.translation_camera_to_gimbal;
            // 固定外参先转换到云台局部坐标，再用当前云台姿态转换到固定参考系。
            const cv::Mat reference_point = gimbal_to_reference * gimbal_point;
            position_gimbal = {reference_point.at<double>(0),
                               reference_point.at<double>(1),
                               reference_point.at<double>(2)};
        }

        // PnP 的旋转是装甲板坐标系到相机坐标系
        ArmorPose pose;
        pose.detection = detection;
        pose.position_camera_m = position_camera;
        pose.position_gimbal_m = position_gimbal;
        
        // 当前角点物点顺序下，+Z 法向指向车体中心
        // 转到固定参考系后，法向水平投影的角度作为 EKF 的 yaw 观测。
        const cv::Mat normal_camera = (cv::Mat_<double>(3, 1) << rotation.at<double>(0, 2), rotation.at<double>(1, 2),rotation.at<double>(2, 2));
        const cv::Mat normal_gimbal = camera.rotation_camera_to_gimbal * normal_camera;
        const cv::Mat normal_reference = gimbal_to_reference * normal_gimbal;
        pose.armor_yaw = std::atan2(normal_reference.at<double>(1, 0),
                                    normal_reference.at<double>(0, 0));

        pose.reprojection_error = reprojectionError(object_points, image_points, rvec, tvec, camera);
        poses.push_back(pose);
    }
    return poses;
}

}  
