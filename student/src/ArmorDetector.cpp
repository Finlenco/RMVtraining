#include "ArmorDetector.hpp"
#include "NumberClassifier.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace armor_detector {

namespace {

struct LightBar {
    cv::RotatedRect rect;
    float length = 0.0F;
    float width = 0.0F;
};

cv::Mat makeColorMask(const cv::Mat& image, TeamColor enemy_color) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::Mat color_difference;

    if (enemy_color == TeamColor::Red) {
        cv::subtract(channels[2], channels[0], color_difference);
    } else {
        cv::subtract(channels[0], channels[2], color_difference);
    }

    cv::Mat gray;
    cv::Mat bright_mask;
    cv::Mat color_mask;

    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bright_mask, 150, 255, cv::THRESH_BINARY);
    cv::threshold(color_difference, color_mask, 40, 255, cv::THRESH_BINARY);

    cv::Mat mask;
    cv::bitwise_and(bright_mask, color_mask, mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    //先膨胀后腐蚀
    return mask;
}

bool makeLightBar(const std::vector<cv::Point>& contour, LightBar& light) {
    if (cv::contourArea(contour) < 10.0) {
        return false;
    }

    light.rect = cv::minAreaRect(contour);
    const float width = light.rect.size.width;
    const float height = light.rect.size.height;
    light.length = std::max(width, height);
    light.width = std::min(width, height);

    if (light.length < 8.0F || light.width <= 0.0F) {
        return false;
    }
    const float ratio = light.length / light.width;
    return ratio >= 1.5F && ratio <= 20.0F;
}

//找角点
cv::Point2f topPoint(const cv::RotatedRect& rect) {
    cv::Point2f points[4];
    rect.points(points);

    cv::Point2f top = points[0];
    for(int i = 0; i < 4 ; i++){
        if(points[i].y < top.y){
            top = points[i];
        }
    }
    return top;
}

cv::Point2f bottomPoint(const cv::RotatedRect& rect) {
    cv::Point2f points[4];
    rect.points(points);

    cv::Point2f top = points[0];
    for(int i = 0; i < 4 ; i++){
        if(points[i].y > top.y){
            top = points[i];
        }
    }
    return top;
}

bool canPair(const LightBar& left, const LightBar& right) {
   
    const float average_length = (left.length + right.length) * 0.5F;

    const float distance = static_cast<float>(
        cv::norm(left.rect.center - right.rect.center));
    const float length_ratio = std::min(left.length, right.length) /
                               std::max(left.length, right.length);
    const float height_difference =
        std::abs(left.rect.center.y - right.rect.center.y);

    return length_ratio >= 0.6F &&
           distance / average_length >= 1.0F &&
           distance / average_length <= 6.0F &&
           height_difference <= average_length * 0.8F;
}

}  
std::vector<ArmorDetection> detect(const cv::Mat& image, TeamColor enemy_color) {
    if (image.empty() || image.channels() != 3) return {};
    
    const cv::Mat mask = makeColorMask(image, enemy_color);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); //找轮廓

    std::vector<LightBar> lights;

    for (const auto& contour : contours) {
        LightBar light;
        if (makeLightBar(contour, light)) {
            lights.push_back(light);
        }
    }

    std::sort(lights.begin(), lights.end(), [](const LightBar& a, const LightBar& b) {
        return a.rect.center.x < b.rect.center.x;
    });//按 x 坐标排序

    static number_classifier::NumberClassifier classifier;
    std::vector<ArmorDetection> detections;
    for (std::size_t i = 0; i < lights.size(); ++i) {
        for (std::size_t j = i + 1; j < lights.size(); ++j) {
            if (!canPair(lights[i], lights[j])) {
                continue;
            }

            const float average_length = (lights[i].length + lights[j].length) * 0.5F;
            const float distance = static_cast<float>(
                cv::norm(lights[i].rect.center - lights[j].rect.center));

            ArmorDetection detection;
            detection.corners = {
                topPoint(lights[i].rect), topPoint(lights[j].rect),
                bottomPoint(lights[j].rect), bottomPoint(lights[i].rect)};
            detection.size = distance / average_length > 3.0F ? ArmorSize::Large : ArmorSize::Small;
            if (!classifier.classify(image, detection)) {
                continue;
            }

            detections.push_back(detection);
        }
    }
    return detections;
}

}  
