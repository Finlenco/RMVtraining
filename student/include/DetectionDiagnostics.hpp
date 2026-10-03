#pragma once

#include "AutoAimTypes.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace detection_diagnostics {
cv::Mat makeImage(const cv::Mat& image, TeamColor enemy_color,
                  const std::vector<ArmorDetection>& detections);
}
