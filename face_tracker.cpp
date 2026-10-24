#include "face_tracker.hpp"
#include "FaceRecognition.h"

#include <algorithm>

FaceTracker::FaceTracker(float iou_threshold, int detect_every, int max_lost, int confirm_hits)
    : iou_threshold_(iou_threshold), detect_every_(std::max(1, detect_every)),
      max_lost_(std::max(1, max_lost)), confirm_hits_(std::max(1, confirm_hits)) {}

float FaceTracker::iou(const cv::Rect2f& a, const cv::Rect2f& b) const {
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

std::vector<Face>& FaceTracker::update(cv::Mat& img, Detector& detector) {
    frame_count_++;
    bool force_detect = tracks_.empty() || (frame_count_ % detect_every_ == 0);

    std::vector<Face> detections;
    if (force_detect) {
        detections = detector.detect(img); // 返回 input_shape 坐标系
    }

    current_faces_.clear();

    // 非检测帧：用匀速预测现有目标，不跑 DNN
    if (!force_detect) {
        for (auto& t : tracks_) {
            // 预测 bbox
            t.bbox.x += t.velocity.x;
            t.bbox.y += t.velocity.y;

            // 防止跑出 input_shape 边界
            const cv::Size& shape = detector.shape();
            t.bbox.x = std::max(0.f, std::min(t.bbox.x,
                                              static_cast<float>(shape.width) - t.bbox.width));
            t.bbox.y = std::max(0.f, std::min(t.bbox.y,
                                              static_cast<float>(shape.height) - t.bbox.height));

            // 预测 landmarks
            cv::Point2f v = t.velocity;
            t.landmarks.left_eye += v;
            t.landmarks.right_eye += v;
            t.landmarks.nose_tip += v;
            t.landmarks.mouth_left += v;
            t.landmarks.mouth_right += v;

            t.lost_frames++;

            // 未确认的目标在首帧丢失后直接隐藏，减少幽灵框闪烁
            if (t.confirmed || t.lost_frames == 1) {
                Face f;
                f.bbox = {{t.bbox.x, t.bbox.y},
                          {t.bbox.x + t.bbox.width, t.bbox.y + t.bbox.height}};
                f.landmarks = t.landmarks;
                f.score = t.score;
                current_faces_.push_back(f);
            }
        }
        return current_faces_;
    }

    // ---------- 有检测结果：贪婪 IoU 匹配 ----------
    std::vector<bool> track_matched(tracks_.size(), false);
    std::vector<bool> det_matched(detections.size(), false);

    // 对每个检测框，寻找 IoU 最高且超过阈值的未匹配轨迹
    for (size_t d = 0; d < detections.size(); ++d) {
        cv::Rect2f det_rect(detections[d].bbox.top_left.x,
                            detections[d].bbox.top_left.y,
                            detections[d].bbox.bottom_right.x - detections[d].bbox.top_left.x,
                            detections[d].bbox.bottom_right.y - detections[d].bbox.top_left.y);

        int best_track = -1;
        float best_iou = iou_threshold_;
        for (size_t t = 0; t < tracks_.size(); ++t) {
            if (track_matched[t]) continue;
            float iou_val = iou(det_rect, tracks_[t].bbox);
            if (iou_val > best_iou) {
                best_iou = iou_val;
                best_track = static_cast<int>(t);
            }
        }

        if (best_track >= 0) {
            TrackedFace& t = tracks_[best_track];

            // 更新速度（中心点位移）
            cv::Point2f center_new = (detections[d].bbox.top_left +
                                      detections[d].bbox.bottom_right) * 0.5f;
            cv::Point2f center_old = (t.bbox.tl() + t.bbox.br()) * 0.5f;
            t.velocity = center_new - center_old;

            t.bbox = cv::Rect2f(detections[d].bbox.top_left.x,
                                detections[d].bbox.top_left.y,
                                detections[d].bbox.bottom_right.x - detections[d].bbox.top_left.x,
                                detections[d].bbox.bottom_right.y - detections[d].bbox.top_left.y);
            t.landmarks = detections[d].landmarks;
            t.score = detections[d].score;
            t.lost_frames = 0;
            t.hits++;
            if (t.hits >= confirm_hits_) {
                t.confirmed = true;
            }
            track_matched[best_track] = true;
            det_matched[d] = true;
        }
    }

    // 未匹配的检测：新建轨迹
    for (size_t d = 0; d < detections.size(); ++d) {
        if (det_matched[d]) continue;
        TrackedFace t;
        t.id = next_id_++;
        t.bbox = cv::Rect2f(detections[d].bbox.top_left.x,
                            detections[d].bbox.top_left.y,
                            detections[d].bbox.bottom_right.x - detections[d].bbox.top_left.x,
                            detections[d].bbox.bottom_right.y - detections[d].bbox.top_left.y);
        t.landmarks = detections[d].landmarks;
        t.score = detections[d].score;
        t.lost_frames = 0;
        t.hits = 1;
        t.confirmed = (confirm_hits_ <= 1);
        t.velocity = cv::Point2f(0.f, 0.f);
        tracks_.push_back(std::move(t));
        track_matched.push_back(false); // 新轨迹本帧未匹配（它是新建的，无需再标记丢失）
    }

    // 未匹配的轨迹：丢失计数++
    for (size_t t = 0; t < tracks_.size(); ++t) {
        if (!track_matched[t]) {
            tracks_[t].lost_frames++;
        }
    }

    // 输出当前有效目标：已确认，或本帧刚检测到的
    for (const auto& t : tracks_) {
        if (t.confirmed || t.lost_frames == 0) {
            Face f;
            f.bbox = {{t.bbox.x, t.bbox.y},
                      {t.bbox.x + t.bbox.width, t.bbox.y + t.bbox.height}};
            f.landmarks = t.landmarks;
            f.score = t.score;
            current_faces_.push_back(f);
        }
    }

    // 清理长期丢失的轨迹
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
