#include "face_tracker.hpp"
#include "FaceRecognition.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>

FaceTracker::FaceTracker(const TrackerParams& params) : p_(params) {
    p_.detect_every = std::max(1, p_.detect_every);
    p_.max_lost     = std::max(1, p_.max_lost);
    p_.min_hits     = std::max(1, p_.min_hits);
}

float FaceTracker::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const cv::Rect2f inter = a & b;
    if (inter.area() <= 0.f) {
        return 0.f;
    }
    const float union_area = a.area() + b.area() - inter.area();
    if (union_area <= 0.f) {
        return 0.f;
    }
    return inter.area() / union_area;
}

// 恒定速度外推：框和关键点作为一个整体平移，绝不单独夹取边角。
// 旧实现在这里把 x 夹到 [0, bounds.width - width]，贴边时 width > bounds.width
// 会让上界变负、整框被推到 x=0 并压扁，下一帧关联更难，于是同一个脸又拆一条轨迹。
// 越出画面的轨迹交给 prune() 按"与画面无交集"清掉，几何始终忠实于运动模型。
void FaceTracker::advance(Track& t) {
    t.box += t.velocity;
    t.lm.right_eye    += t.velocity;
    t.lm.left_eye     += t.velocity;
    t.lm.nose_tip     += t.velocity;
    t.lm.mouth_left   += t.velocity;
    t.lm.mouth_right  += t.velocity;
    ++t.age;
}

// 关联：两条门（框 IoU / 归一化中心位移）取分，候选对全局按分数降序贪心接受。
//
// 为什么不是"按检测顺序给每条检测找最优轨迹"：那种贪婪写法依赖检测的排列顺序，
// 一条检测会抢走后面更需要的轨迹；为什么不是匈牙利最优分配：这里 n ≤ 几十，
// 全局贪心按分数排序后的解与 O(n³) 的匈牙利在分数分得很开的实际场景里等价，
// 少 80 行容易写错的代码。
//
// 距离门不是可有可无的加分项：贴边时同一个脸的框几何会在"整脸框"和"只剩可见部分的框"
// 之间跳（实测相邻帧 IoU 只有 0.37，低于旧的 0.45 匹配阈值），单靠 IoU 必然拆轨迹。
// 位移预算随"距上次实测的帧数"放宽，因为外推误差本身就随时间累积。
void FaceTracker::associate(const std::vector<Face>& dets) {
    struct Pair {
        float score;
        size_t d;
        size_t t;
    };
    std::vector<Pair> pairs;
    pairs.reserve(dets.size() * tracks_.size());

    for (size_t d = 0; d < dets.size(); ++d) {
        const cv::Rect2f db(dets[d].bbox.top_left.x, dets[d].bbox.top_left.y,
                            dets[d].bbox.bottom_right.x - dets[d].bbox.top_left.x,
                            dets[d].bbox.bottom_right.y - dets[d].bbox.top_left.y);
        const cv::Point2f cd = center(db);

        for (size_t t = 0; t < tracks_.size(); ++t) {
            const Track& tr = tracks_[t];
            const float i = iou(db, tr.box);
            const float size = 0.5f * (std::max(db.width, tr.box.width) +
                                       std::max(db.height, tr.box.height));
            const float allow = p_.dist_ratio * size +
                                0.5f * cv::norm(tr.velocity) * float(tr.age);
            const float dn = cv::norm(cd - center(tr.box));
            const float s_dist = (allow > 0.f && dn < allow) ? (1.f - dn / allow) : 0.f;

            if (i < p_.iou_threshold && s_dist <= 0.f) {
                continue;
            }
            pairs.push_back({std::max(i, s_dist), d, t});
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.d != b.d) return a.d < b.d;
        return a.t < b.t;
    });

    // 注意：下面会往 tracks_ 追加新轨迹，所以所有按轨迹下标索引的表都要按"进入本函数
    // 时的轨迹数"来定尺寸并只遍历这一段，否则新加的行会越界读。
    const size_t ntracks = tracks_.size();
    std::vector<char> det_used(dets.size(), 0), tr_used(ntracks, 0);
    std::vector<size_t> match(ntracks, SIZE_MAX);          // 轨迹 -> 检测
    for (const Pair& pr : pairs) {
        if (det_used[pr.d] || tr_used[pr.t]) {
            continue;
        }
        det_used[pr.d] = 1;
        tr_used[pr.t]  = 1;
        match[pr.t]    = pr.d;
    }

    // 命中：更新速度/几何。速度必须是"每帧"位移：外推是纯平移，所以上次实测的中心
    // 就是当前预测中心回退 velocity*age —— 不需要额外存 last_det_center，也就没有
    // "把它跟着预测往前推、导致速度退化成预测残差"的那类错误。
    for (size_t t = 0; t < ntracks; ++t) {
        if (match[t] == SIZE_MAX) {
            continue;
        }
        const Face& fd = dets[match[t]];
        Track& tr = tracks_[t];

        const cv::Rect2f db(fd.bbox.top_left.x, fd.bbox.top_left.y,
                            fd.bbox.bottom_right.x - fd.bbox.top_left.x,
                            fd.bbox.bottom_right.y - fd.bbox.top_left.y);
        const int dt = std::max(1, tr.age);
        const cv::Point2f last_center = center(tr.box) - tr.velocity * float(tr.age);
        const cv::Point2f v_est = (center(db) - last_center) * (1.0f / float(dt));
        if (tr.hits <= 1) {          // 新建时 hits=1：首次估计没有历史可平滑，按 alpha 混合会把速度砍半
            tr.velocity = v_est;
        } else {
            const float alpha = 0.5f;
            tr.velocity = tr.velocity * (1.0f - alpha) + v_est * alpha;
        }

        tr.box   = db;
        tr.lm    = fd.landmarks;
        tr.score = fd.score;
        tr.age   = 0;
        ++tr.hits;
    }

    // 未命中的检测：新轨迹（hits=1，达到 min_hits 之前不上屏）
    for (size_t d = 0; d < dets.size(); ++d) {
        bool used = false;
        for (size_t t = 0; t < ntracks; ++t) {
            if (match[t] == d) { used = true; break; }
        }
        if (used) {
            continue;
        }
        Track tr;
        tr.id      = next_id_++;
        tr.box     = cv::Rect2f(dets[d].bbox.top_left.x, dets[d].bbox.top_left.y,
                                dets[d].bbox.bottom_right.x - dets[d].bbox.top_left.x,
                                dets[d].bbox.bottom_right.y - dets[d].bbox.top_left.y);
        tr.lm      = dets[d].landmarks;
        tr.score   = dets[d].score;
        tr.hits    = 1;
        tr.age     = 0;
        tracks_.push_back(tr);
    }
}

// 去重：关联再好也挡不住偶发的拆分（一次严重的几何突变 + 一次漏检就够了），
// 这一步保证"一个脸至多一条轨迹"是硬性质而不是运气。保留证据更强的那条：
// 本帧有检测支撑优先，其次命中次数，再次分数。
void FaceTracker::dedup() {
    std::vector<char> dead(tracks_.size(), 0);
    for (size_t a = 0; a < tracks_.size(); ++a) {
        if (dead[a]) continue;
        for (size_t b = a + 1; b < tracks_.size(); ++b) {
            if (dead[b]) continue;
            const Track& A = tracks_[a];
            const Track& B = tracks_[b];
            const float i = iou(A.box, B.box);
            const float size = 0.5f * (std::max(A.box.width, B.box.width) +
                                       std::max(A.box.height, B.box.height));
            const float dn = cv::norm(center(A.box) - center(B.box));
            if (i <= 0.5f && dn >= 0.35f * size) {
                continue;                      // 距离近到这个量级只可能是同一个目标
            }
            const bool a_fresh = (A.age == 0), b_fresh = (B.age == 0);
            const bool keep_a = std::tie(a_fresh, A.hits, A.score) >
                                std::tie(b_fresh, B.hits, B.score);
            dead[keep_a ? b : a] = 1;
        }
    }
    for (size_t i = dead.size(); i-- > 0;) {          // 从后往前删，前面的下标保持有效
        if (dead[i]) {
            tracks_.erase(tracks_.begin() + long(i));
        }
    }
}

// 生命周期：未确认的候选下一帧没有再次命中就清掉（单帧误检永远不会上屏、也不会被
// 识别）；确认后的轨迹最多外推 max_lost 帧；已经与画面无交集的轨迹直接清掉
// （外推把目标带出画面时，与其在边上钉一个幽灵框，不如让它按运动模型离开）。
void FaceTracker::prune(const cv::Size& bounds) {
    const cv::Rect2f frame(0.f, 0.f, float(bounds.width), float(bounds.height));
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) {
                                     return t.age > p_.max_lost ||
                                            (t.hits < p_.min_hits && t.age > 1) ||
                                            (t.box & frame).area() <= 0.f;
                                 }),
                  tracks_.end());
}

void FaceTracker::build_views() {
    views_.clear();
    views_.reserve(tracks_.size());
    // min_hits 的"要被后续检测佐证"前提本身需要后续帧才成立：流的最初 min_hits 帧
    // 无从佐证，此时照常输出，否则单帧源（一张照片）永远不出结果。
    const bool corroborate = (frame_count_ > p_.min_hits);
    for (const auto& t : tracks_) {
        if ((corroborate && t.hits < p_.min_hits) || t.age > p_.max_lost) {
            continue;
        }
        TrackView v;
        v.face.bbox = {{t.box.x, t.box.y}, {t.box.x + t.box.width, t.box.y + t.box.height}};
        v.face.landmarks = t.lm;
        v.face.score     = t.score;
        v.id             = t.id;
        v.measured       = (t.age == 0);
        views_.push_back(v);
    }
}

void FaceTracker::update(cv::Mat& img, Detector& detector) {
    ++frame_count_;
    const cv::Size bounds = detector.shape();
    const bool detect_now = tracks_.empty() || (frame_count_ % p_.detect_every == 0);

    // 每一帧（含检测帧）都先外推一帧，让轨迹与检测处于同一时刻
    for (auto& t : tracks_) {
        advance(t);
    }

    if (detect_now) {
        associate(detector.detect(img));
    }

    dedup();
    prune(bounds);
    build_views();
}
