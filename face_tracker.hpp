#ifndef FACE_TRACKER_HPP
#define FACE_TRACKER_HPP

#include <vector>
#include "face_def.hpp"
#include "opencv2/opencv.hpp"

class Detector; // 前置声明，避免循环依赖

// 单条人脸轨迹，坐标系与 Detector 的 input_shape 一致（默认 160x120）
struct TrackedFace {
    int id = -1;
    cv::Rect2f bbox;                    // 检测框，input_shape 坐标系
    Landmarks_5 landmarks;              // 五官关键点，input_shape 坐标系
    float score = 0.f;
    int hits = 0;                       // 累计匹配到检测的次数
    int lost_frames = 0;                // 连续未匹配帧数
    int age_since_update = 0;           // 距上次匹配到检测的帧数（速度按帧折算用）
    cv::Point2f velocity;               // 每帧位移，input_shape 坐标系
    cv::Point2f last_det_center;        // 上次匹配到检测时的框中心
};

// 轻量级人脸跟踪器：恒定速度预测 + IoU 贪婪匹配
// 仅每 detect_every 帧（或全部丢失时）跑一次全图 DNN 检测，其余帧用预测填充，
// 使框连续跟随而不是每 N 帧跳一次。
class FaceTracker {
public:
    FaceTracker(float iou_threshold = 0.45f,
                int detect_every = 5,
                int max_lost = 15,
                int min_hits = 1);

    // 每帧调用。内部决定是否调用 detector.detect() 做全图检测。
    // 返回当前有效人脸（坐标系同 detector.input_shape）。
    std::vector<Face>& update(cv::Mat& img, Detector& detector);

    const std::vector<Face>& faces() const { return current_faces_; }
    size_t size() const { return current_faces_.size(); }

    void set_detect_interval(int n);
    void set_max_lost(int n);

private:
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);
    void advance(TrackedFace& t, const cv::Size& bounds);

    std::vector<TrackedFace> tracks_;
    std::vector<Face> current_faces_;
    int next_id_ = 0;
    int frame_count_ = 0;
    float iou_threshold_;
    int detect_every_;
    int max_lost_;
    int min_hits_;
};

#endif // FACE_TRACKER_HPP
