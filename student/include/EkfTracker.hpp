#pragma once

#include "AutoAimTypes.hpp"

#include <vector>

namespace ekf_tracker {

PredictionResult update(const std::vector<ArmorPose>& observations,
                        const GimbalState& gimbal,
                        double timestamp_seconds);

void reset();

}  
