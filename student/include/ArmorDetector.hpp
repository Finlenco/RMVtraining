#pragma once

#include "AutoAimTypes.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace armor_detector {

std::vector<ArmorDetection> detect(
    const cv::Mat& image,
    TeamColor enemy_color);

}  
