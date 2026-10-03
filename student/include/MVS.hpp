#pragma once

#include <opencv2/core.hpp>

namespace MVS{

bool getframe(cv::Mat& frame);

void shutdown();

bool getframe(cv::Mat& frame);

bool setExposure(float exposure_us);

bool setGain(float gain);

bool setGamma(float gamma);

void shutdown();

}  