#include "DetectionDiagnostics.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <opencv2/imgproc.hpp>

namespace detection_diagnostics {
namespace {
struct LightBar { cv::RotatedRect rect; float length{}; float width{}; };
bool lightBar(const std::vector<cv::Point>& contour, LightBar& bar) {
    if (cv::contourArea(contour) < 10.0) return false;
    bar.rect = cv::minAreaRect(contour);
    bar.length = std::max(bar.rect.size.width, bar.rect.size.height);
    bar.width = std::min(bar.rect.size.width, bar.rect.size.height);
    if (bar.length < 8.0F || bar.width <= 0.0F) return false;
    const float ratio = bar.length / bar.width;
    return ratio >= 1.5F && ratio <= 20.0F;
}
bool pairable(const LightBar& a, const LightBar& b) {
    const float average = (a.length + b.length) * 0.5F;
    const float distance = static_cast<float>(cv::norm(a.rect.center - b.rect.center));
    const float ratio = std::min(a.length, b.length) / std::max(a.length, b.length);
    const float height = std::abs(a.rect.center.y - b.rect.center.y);
    return ratio >= 0.6F && distance / average >= 1.0F && distance / average <= 6.0F && height <= average * 0.8F;
}
cv::Mat titled(const cv::Mat& source, const std::string& title) {
    cv::Mat output;
    if (source.channels() == 1) cv::cvtColor(source, output, cv::COLOR_GRAY2BGR);
    else output = source.clone();
    cv::putText(output, title, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.75, {0, 255, 255}, 2, cv::LINE_AA);
    return output;
}
}
cv::Mat makeImage(const cv::Mat& image, TeamColor enemy_color, const std::vector<ArmorDetection>& detections) {
    if (image.empty() || image.channels() != 3) return {};
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Mat gray, bright, color, mask, difference;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bright, 150, 255, cv::THRESH_BINARY);
    if (enemy_color == TeamColor::Red) cv::subtract(channels[2], channels[0], difference);
    else cv::subtract(channels[0], channels[2], difference);
    cv::threshold(difference, color, 40, 255, cv::THRESH_BINARY);
    cv::bitwise_and(bright, color, mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<LightBar> bars;
    for (const auto& contour : contours) { LightBar bar; if (lightBar(contour, bar)) bars.push_back(bar); }
    std::sort(bars.begin(), bars.end(), [](const LightBar& a, const LightBar& b) { return a.rect.center.x < b.rect.center.x; });
    int pairs = 0;
    for (std::size_t i = 0; i < bars.size(); ++i) for (std::size_t j = i + 1; j < bars.size(); ++j) pairs += pairable(bars[i], bars[j]) ? 1 : 0;
    cv::Mat result = image.clone();
    for (const auto& bar : bars) { cv::Point2f points[4]; bar.rect.points(points); for (int i = 0; i < 4; ++i) cv::line(result, points[i], points[(i + 1) % 4], {0, 255, 0}, 2); }
    for (const auto& detection : detections) {
        for (int i = 0; i < 4; ++i) cv::line(result, detection.corners[i], detection.corners[(i + 1) % 4], {0, 0, 255}, 3);
        const cv::Point2f center = (detection.corners[0] + detection.corners[1] + detection.corners[2] + detection.corners[3]) * 0.25F;
        cv::putText(result, "id=" + std::to_string(detection.target_id), center, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 255}, 2, cv::LINE_AA);
    }
    cv::putText(result, "lights=" + std::to_string(bars.size()) + " pairs=" + std::to_string(pairs) + " detections=" + std::to_string(detections.size()), {12, 58}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 0}, 2, cv::LINE_AA);
    cv::Mat raw = titled(image, "raw BGR"), bright_view = titled(bright, "brightness > 150"), color_view = titled(color, "color difference > 40"), result_view = titled(result, "green=light bar red=accepted");
    const cv::Size tile(image.cols / 2, image.rows / 2);
    cv::resize(raw, raw, tile); cv::resize(bright_view, bright_view, tile); cv::resize(color_view, color_view, tile); cv::resize(result_view, result_view, tile);
    cv::Mat top, bottom, collage;
    cv::hconcat(raw, bright_view, top); cv::hconcat(color_view, result_view, bottom); cv::vconcat(top, bottom, collage);
    return collage;
}
}
