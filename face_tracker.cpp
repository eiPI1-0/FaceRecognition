#include "face_tracker.hpp"
#include "FaceRecognition.h"

#include <algorithm>

FaceTracker::FaceTracker(float iou_threshold, int detect_every, int max_lost, int min_hits)
    : iou_threshold_(iou_threshold), detect_every_(std::max(1, detect_every)),
      max_lost_(std::max(1, max_lost)), min_hits_(std::max(1, min_hits)) {}

float FaceTracker::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float ix1 = std::max(a.tl().x, b.tl().x);
    float iy1 = std::max(a.tl().y, b.tl().y);
    float ix2 = std::min(a.br().x, b.br().x);
    float iy2 = std::min(a.br().y, b.br().y);

    float iw = ix2 - ix1 + 1.f;
    float ih = iy2 - iy1 + 1.f;
    if (iw <= 0.f || ih <= 0.f) {
        return 0.f;
    }
    float inter = iw * ih;
    float area_a = (a.br().x - a.tl().x + 1.f) * (a.br().y - a.tl().y + 1.f);
    float area_b = (b.br().x - b.tl().x + 1.f) * (b.br().y - b.tl().y + 1.f);
    float denom = area_a + area_b - inter;
    if (denom <= 0.f) {
        return 0.f;
    }
    return inter / denom;
}

// 按恒定速度前推一帧，并把框夹在检测画面范围内
void FaceTracker::advance(TrackedFace& t, const cv::Size& bounds) {
    t.bbox.x += t.velocity.x;
    t.bbox.y += t.velocity.y;
    t.bbox.x = std::max(0.f, std::min(t.bbox.x, bounds.width - t.bbox.width));
    t.bbox.y = std::max(0.f, std::min(t.bbox.y, bounds.height - t.bbox.height));

    t.landmarks.right_eye += t.velocity;
    t.landmarks.left_eye += t.velocity;
    t.landmarks.nose_tip += t.velocity;
    t.landmarks.mouth_left += t.velocity;
    t.landmarks.mouth_right += t.velocity;

    // 注意：last_det_center 保持锚定在"上次匹配到的检测中心"，不随预测移动，
    // 否则速度估计退化成预测残差，会持续低估速度、导致检测间隔内框逐渐落后。
    t.age_since_update++;
    t.lost_frames++;
}

std::vector<Face>& FaceTracker::update(cv::Mat& img, Detector& detector) {
    frame_count_++;
    const cv::Size bounds = detector.shape();
    const bool force_detect = tracks_.empty() || (frame_count_ % detect_every_ == 0);

    // 所有帧（含检测帧）都先按速度前推一帧，保证轨迹与检测处于同一时刻
    for (auto& t : tracks_) {
        advance(t, bounds);
    }

    current_faces_.clear();

    if (force_detect) {
        std::vector<Face>& detections = detector.detect(img); // input_shape 坐标系

        std::vector<bool> track_matched(tracks_.size(), false);
        std::vector<bool> det_matched(detections.size(), false);

        for (size_t d = 0; d < detections.size(); ++d) {
            const Face& fd = detections[d];
            cv::Rect2f det_rect(fd.bbox.top_left.x, fd.bbox.top_left.y,
                                fd.bbox.bottom_right.x - fd.bbox.top_left.x,
                                fd.bbox.bottom_right.y - fd.bbox.top_left.y);

            int best_track = -1;
            float best_iou = iou_threshold_;
            for (size_t t = 0; t < tracks_.size(); ++t) {
                if (track_matched[t]) continue;
                float v = iou(det_rect, tracks_[t].bbox);
                if (v > best_iou) {
                    best_iou = v;
                    best_track = static_cast<int>(t);
                }
            }

            if (best_track >= 0) {
                TrackedFace& t = tracks_[best_track];

                // 速度 = 两次检测之间的位移 / 经过的帧数  => 单位是"每帧"，
                // 绝不能赋成"两次检测的总位移"，否则非检测帧会成倍过冲。
                const int dt = std::max(1, t.age_since_update);
                const cv::Point2f det_center = (fd.bbox.top_left + fd.bbox.bottom_right) * 0.5f;
                cv::Point2f v_est = (det_center - t.last_det_center) * (1.0f / dt);

                // 轻平滑抑制检测抖动；首次估计时没有历史可平滑，
                // 若仍按 alpha 混合会把速度砍半，导致第一个周期内框明显落后。
                if (t.hits <= 1) {   // 新建时 hits=1，故首次估计速度时 hits==1
                    t.velocity = v_est;
                } else {
                    const float alpha = 0.5f;
                    t.velocity = t.velocity * (1.0f - alpha) + v_est * alpha;
                }

                t.bbox = det_rect;
                t.landmarks = fd.landmarks;
                t.score = fd.score;
                t.last_det_center = det_center;
                t.age_since_update = 0;
                t.lost_frames = 0;
                t.hits++;
                track_matched[best_track] = true;
                det_matched[d] = true;
            }
        }

        for (size_t d = 0; d < detections.size(); ++d) {
            if (det_matched[d]) continue;
            const Face& fd = detections[d];
            TrackedFace t;
            t.id = next_id_++;
            t.bbox = cv::Rect2f(fd.bbox.top_left.x, fd.bbox.top_left.y,
                                fd.bbox.bottom_right.x - fd.bbox.top_left.x,
                                fd.bbox.bottom_right.y - fd.bbox.top_left.y);
            t.landmarks = fd.landmarks;
            t.score = fd.score;
            t.hits = 1;
            t.lost_frames = 0;
            t.age_since_update = 0;
            t.velocity = cv::Point2f(0.f, 0.f);
            t.last_det_center = (fd.bbox.top_left + fd.bbox.bottom_right) * 0.5f;
            tracks_.push_back(std::move(t));
        }
    }

    // 输出：已至少匹配过一次、且尚未超过容忍丢失帧数的轨迹
    for (const auto& t : tracks_) {
        if (t.hits < min_hits_ || t.lost_frames > max_lost_) {
            continue;
        }
        Face f;
        f.bbox = {{t.bbox.x, t.bbox.y},
                  {t.bbox.x + t.bbox.width, t.bbox.y + t.bbox.height}};
        f.landmarks = t.landmarks;
        f.score = t.score;
        current_faces_.push_back(f);
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [this](const TrackedFace& t) {
                                     return t.lost_frames > max_lost_;
                                 }),
                  tracks_.end());

    return current_faces_;
}

void FaceTracker::set_detect_interval(int n) {
    detect_every_ = std::max(1, n);
}

void FaceTracker::set_max_lost(int n) {
    max_lost_ = std::max(1, n);
}
