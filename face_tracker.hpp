#ifndef FACE_TRACKER_HPP
#define FACE_TRACKER_HPP

#include <vector>
#include "face_def.hpp"
#include "opencv2/opencv.hpp"

class Detector; // 前置声明，避免循环依赖

// 单条人脸轨迹，坐标系与 Detector 的 input_shape 一致（默认 160x120）
struct TrackedFace {
    int id = -1;
    cv::Rect2f bbox;              // 检测框，input_shape 坐标系
    Landmarks_5 landmarks;        // 五官关键点，input_shape 坐标系
    float score = 0.f;
    int lost_frames = 0;          // 连续未匹配帧数
    int max_lost = 15;            // 超过此帧数将被移除
    int hits = 0;                 // 连续匹配帧数
    bool confirmed = false;       // 是否已确认为稳定目标（避免闪烁）
    cv::Point2f velocity;         // 匀速预测的每帧位移，input_shape 坐标系
};

// 轻量级人脸跟踪器：基于 IoU 贪婪匹配 + 匀速预测
// 大部分帧不跑 DNN 检测，仅在 detect_every 帧或全丢失时触发全图检测。
class FaceTracker {
public:
    FaceTracker(float iou_threshold = 0.45f,
                int detect_every = 5,
                int max_lost = 15,
                int confirm_hits = 3);

    // 每帧调用。内部决定是否调用 detector.detect() 做全图检测。
    // 返回当前有效人脸（已关联 ID，坐标系同 detector.input_shape）。
    std::vector<Face>& update(cv::Mat& img, Detector& detector);

    const std::vector<Face>& faces() const { return current_faces_; }
    size_t size() const { return current_faces_.size(); }

    void set_detect_interval(int n);
    void set_max_lost(int n);

private:
    float iou(const cv::Rect2f& a, const cv::Rect2f& b) const;

    std::vector<TrackedFace> tracks_;
    std::vector<Face> current_faces_;
    int next_id_ = 0;
    int frame_count_ = 0;
    float iou_threshold_;
    int detect_every_;
    int max_lost_;
    int confirm_hits_;
};

#endif // FACE_TRACKER_HPP
