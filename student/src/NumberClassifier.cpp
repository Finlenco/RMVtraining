#include "NumberClassifier.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace number_classifier {

namespace {

const int kNumberImageHeight = 28;
const int kNumberRoiWidth = 20;
//20 * 28 框出来的数字区域

const int kSmallWidth = 32;
const int kLargeWidth = 54;
const int kLightLength = 12;
const float kThreshold = 0.5F;

int targetIdFromLabel(std::string label) {

    if (label == "1") return 1;
    if (label == "2") return 2;
    if (label == "3") return 3;
    if (label == "4") return 4;
    if (label == "5") return 5;

    if (label == "outpost") return 6;
    if (label == "guard") return 7;
    if (label == "base") return 8;

    return 0;
}

bool extractNumberImage(const cv::Mat& source,const ArmorDetection& detection,cv::Mat& number_image) {
    if (source.empty() || source.channels() != 3) {
        return false;
    }

    const int warp_width = detection.size == ArmorSize::Large ? kLargeWidth: kSmallWidth;

    // 把灯条放到 32/54x28 的固定位置，再取中央 20x28 数字区域。
    const std::vector<cv::Point2f> source_vertices = {detection.corners[3], detection.corners[0],detection.corners[1], detection.corners[2]};
    
    // 透视变换，从左下角开始顺时针
    const int top_light_y = (kNumberImageHeight - kLightLength) / 2 - 1;

    const int bottom_light_y = top_light_y + kLightLength;

    const std::vector<cv::Point2f> target_vertices = {
        {0.0F, static_cast<float>(bottom_light_y)},
        {0.0F, static_cast<float>(top_light_y)},
        {static_cast<float>(warp_width - 1), static_cast<float>(top_light_y)},
        {static_cast<float>(warp_width - 1), static_cast<float>(bottom_light_y)}};

    const cv::Mat transform = cv::getPerspectiveTransform(source_vertices, target_vertices); //透视变换矩阵
    
    cv::Mat warped;
    cv::warpPerspective(source, warped, transform, cv::Size(warp_width, kNumberImageHeight));
    
    if (warped.empty() || warped.cols < kNumberRoiWidth || warped.rows < kNumberImageHeight) {
        return false;
    }

    const int roi_x = (warp_width - kNumberRoiWidth) / 2;
    number_image = warped(cv::Rect(roi_x, 0, kNumberRoiWidth, kNumberImageHeight)).clone();//防止动原图

    cv::cvtColor(number_image, number_image, cv::COLOR_BGR2GRAY);
    cv::threshold(number_image, number_image, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return true;
}

} 

NumberClassifier::NumberClassifier() {
    const std::string model_path = std::string(RMVTRAINING_MODEL_DIR) + "/" +"mlp.onnx";
    const std::string label_path = std::string(RMVTRAINING_MODEL_DIR) + "/" +"label.txt";

    net_ = cv::dnn::readNetFromONNX(model_path);

    std::ifstream label_file(label_path);
    std::string line;

    while (std::getline(label_file, line)) {
        class_names_.push_back(line);
    }
    ready_ = true;//分类器准备好了
}

bool NumberClassifier::ready() const noexcept {
    return ready_;
}

bool NumberClassifier::classify(const cv::Mat& source, ArmorDetection& detection) {
    detection.target_id = 0;
    detection.confidence = 0.0F;

    cv::Mat number_image;
    if (!ready_ || !extractNumberImage(source, detection, number_image)) {
        return false;
    }

    cv::Mat normalized;
    number_image.convertTo(normalized, CV_32F, 1.0 / 255.0);

    const cv::Mat blob = cv::dnn::blobFromImage(normalized, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);

    cv::Mat output;

    net_.setInput(blob);
    output = net_.forward();

    if (output.empty() || output.total() != class_names_.size()) {
        std::cerr << "数字分类模型输出类别与 label.txt 不一致\n";
        ready_ = false;
        return false;
    }

    const cv::Mat logits = output.reshape(1, 1);
    double max_logit = 0.0;
    cv::minMaxLoc(logits, nullptr, &max_logit, nullptr, nullptr);

    //softmax
    cv::Mat probabilities;
    cv::exp(logits - static_cast<float>(max_logit), probabilities);
    const double sum = cv::sum(probabilities)[0];
    if (!std::isfinite(sum) || sum <= 0.0) {
        return false;
    }
    probabilities /= sum;

    cv::Point best;
    double best_probability = 0.0;
    cv::minMaxLoc(probabilities, nullptr, &best_probability, nullptr, &best);

    detection.confidence = static_cast<float>(best_probability);
    int target_id = targetIdFromLabel(class_names_[best.x]);

    if (target_id == 0 || detection.confidence < kThreshold) {
        detection.target_id = 0;
        return false;
    }

    detection.target_id = target_id;
    return true;
}

}
