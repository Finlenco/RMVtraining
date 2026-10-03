#pragma once

#include "AutoAimTypes.hpp"

#include <vector>

namespace armor_solver {

std::vector<ArmorPose> solve(
    const std::vector<ArmorDetection>& detections,
    const CameraParameters& camera,
    const GimbalState& gimbal);

}  
