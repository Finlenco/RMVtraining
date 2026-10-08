#pragma once

#include "AutoAimTypes.hpp"

#include <vector>

namespace armor_solver {

std::vector<ArmorPose> solve(
    const std::vector<ArmorDetection>& detections,
    const CameraParameters& camera,
    const GimbalState& gimbal,
    std::vector<cv::Vec3d>* rotation_vectors = nullptr);

void displayResults(const cv::Mat& image,
                    const std::vector<ArmorDetection>& detections,
                    const std::vector<ArmorPose>& poses,
                    const std::vector<cv::Vec3d>& rotation_vectors,
                    const CameraParameters& camera);

}  
