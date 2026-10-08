#include "ArmorSolver.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
    // 装甲板：小 135x57 ，大 230x57
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

std::vector<ArmorPose> solve(const std::vector<ArmorDetection>& detections,
                             const CameraParameters& camera,
                             const GimbalState& gimbal,
                             std::vector<cv::Vec3d>* rotation_vectors) {
    
    std::vector<ArmorPose> poses;
    if (rotation_vectors != nullptr) rotation_vectors->clear();

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
        if (rotation_vectors != nullptr) {
            rotation_vectors->emplace_back(rvec.at<double>(0), rvec.at<double>(1),
                                           rvec.at<double>(2));
        }
    }
    return poses;
}

void displayResults(const cv::Mat& image,
                    const std::vector<ArmorDetection>& detections,
                    const std::vector<ArmorPose>& poses,
                    const std::vector<cv::Vec3d>& rotation_vectors,
                    const CameraParameters& camera) {
    if (image.empty()) return;

    cv::Mat display = image.clone();
    constexpr int kLineHeight = 21;
    constexpr int kPanelPadding = 9;
    constexpr float kAxisLengthM = 0.10F;
    const int max_rows = std::max(1, (display.rows - 2 * kPanelPadding) / kLineHeight);
    const std::size_t displayed_count = std::min(
        poses.size(), static_cast<std::size_t>(max_rows / 3));

    for (const ArmorDetection& detection : detections) {
        for (int corner = 0; corner < 4; ++corner) {
            cv::line(display, detection.corners[corner],
                     detection.corners[(corner + 1) % 4],
                     cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
        const cv::Point2f center =
            (detection.corners[0] + detection.corners[1] +
             detection.corners[2] + detection.corners[3]) * 0.25F;
        cv::putText(display, "ID " + std::to_string(detection.target_id),
                    center, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    for (std::size_t index = 0; index < poses.size(); ++index) {
        const ArmorPose& pose = poses[index];
        if (index >= rotation_vectors.size()) continue;

        cv::Mat rotation_vector = (cv::Mat_<double>(3, 1)
            << rotation_vectors[index][0], rotation_vectors[index][1],
               rotation_vectors[index][2]);
        cv::Mat translation_vector = (cv::Mat_<double>(3, 1)
            << pose.position_camera_m[0], pose.position_camera_m[1],
               pose.position_camera_m[2]);
        // OpenCV draws X/Y/Z axes in red/green/blue using the PnP pose
        // (armor frame expressed in camera coordinates).
        cv::drawFrameAxes(display, camera.camera_matrix,
                          camera.distortion_coefficients, rotation_vector,
                          translation_vector, kAxisLengthM, 2);
    }

    if (displayed_count > 0) {
        std::vector<std::string> lines;
        lines.reserve(displayed_count * 3);
        for (std::size_t index = 0; index < displayed_count; ++index) {
            const ArmorPose& pose = poses[index];
            const cv::Vec3d& tvec = pose.position_camera_m;
            if (index >= rotation_vectors.size()) continue;
            const cv::Vec3d& rvec = rotation_vectors[index];
            const cv::Vec3d& p_ref = pose.position_gimbal_m;
            const std::string prefix = "ID " + std::to_string(pose.detection.target_id) + " ";
            lines.push_back(prefix + "tvec[m] = (" + cv::format("%.3f, %.3f, %.3f",
                tvec[0], tvec[1], tvec[2]) + ")");
            lines.push_back(prefix + "rvec[rad] = (" + cv::format("%.3f, %.3f, %.3f",
                rvec[0], rvec[1], rvec[2]) + ")");
            lines.push_back(prefix + "p_ref[m] = (" + cv::format("%.3f, %.3f, %.3f",
                p_ref[0], p_ref[1], p_ref[2]) + ")");
        }

        int text_width = 0;
        constexpr double kFontScale = 0.48;
        for (const std::string& line : lines) {
            int baseline = 0;
            text_width = std::max(text_width, cv::getTextSize(
                line, cv::FONT_HERSHEY_SIMPLEX, kFontScale, 1, &baseline).width);
        }
        const int panel_width = std::min(display.cols, text_width + 2 * kPanelPadding);
        const int panel_height = std::min(display.rows,
            static_cast<int>(lines.size()) * kLineHeight + 2 * kPanelPadding);
        const int panel_x = std::max(0, display.cols - panel_width - 10);
        const int panel_y = 10;
        cv::rectangle(display,
                      cv::Rect(panel_x, panel_y, panel_width, panel_height),
                      cv::Scalar(20, 20, 20), cv::FILLED);
        for (std::size_t index = 0; index < lines.size(); ++index) {
            const cv::Scalar line_color = index % 3 == 0
                ? cv::Scalar(235, 235, 235)
                : (index % 3 == 1 ? cv::Scalar(180, 255, 180)
                                  : cv::Scalar(255, 210, 150));
            cv::putText(display, lines[index],
                        {panel_x + kPanelPadding,
                         panel_y + kPanelPadding + static_cast<int>(index + 1) * kLineHeight - 4},
                        cv::FONT_HERSHEY_SIMPLEX, kFontScale, line_color, 1,
                        cv::LINE_AA);
        }
    }

    cv::imshow("Armor Detection", display);
    cv::waitKey(1);
}

}  
