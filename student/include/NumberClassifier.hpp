#pragma once

#include "AutoAimTypes.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

namespace number_classifier {

// 负责装甲板数字区域的预处理和 ONNX 分类。
class NumberClassifier {
public:
    NumberClassifier();

    bool classify(const cv::Mat& source, ArmorDetection& detection);
    bool ready() const noexcept;

private:
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    bool ready_{false};
};

} 
