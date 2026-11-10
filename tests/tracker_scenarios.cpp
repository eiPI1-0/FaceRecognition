// FaceTracker 的脚本化回归测试（PLAN.md D6 第二条）。
//
// 这些序列是 AGENTS.md 里那套 A/B 配方的最小化复现，断言直接取自实测基线。它们锁住的
// 四条性质，每一条都对应过一次真实的用户可见 bug：
//   1. 贴边时框几何突变（整脸框 vs 只剩可见部分）**不能**拆成两条轨迹 —— "突然一堆框"；
//   2. 两个真实人脸相距 0.7 个脸宽时**永远**是两条轨迹 —— 去重不能把真的合并；
//   3. 目标消失后最多外推 max_lost 帧，且这些帧必须 measured=false —— 幽灵框被识别；
//   4. 单帧误检永远不上屏（min_hits），7 px/帧运动下 ID 稳定、框不滞后。
//
// 用纯跟踪入口（update(Size, dets)）驱动：不需要模型、不需要摄像头，0.0x 秒跑完。

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/face_def.hpp"
#include "core/face_tracker.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

const cv::Size kFrame(160, 120);

// 一个几何合理的检测：框 + 按比例摆放的 5 个关键点（跟踪器只搬运它们，不看语义）。
Face make_face(float cx, float cy, float w, float h, float score = 0.9f) {
    Face f;
    f.bbox.top_left = {cx - w / 2.f, cy - h / 2.f};
    f.bbox.bottom_right = {cx + w / 2.f, cy + h / 2.f};
    f.landmarks.right_eye = {cx - 0.2f * w, cy - 0.15f * h};
    f.landmarks.left_eye = {cx + 0.2f * w, cy - 0.15f * h};
    f.landmarks.nose_tip = {cx, cy + 0.05f * h};
    f.landmarks.mouth_left = {cx - 0.15f * w, cy + 0.25f * h};
    f.landmarks.mouth_right = {cx + 0.15f * w, cy + 0.25f * h};
    f.score = score;
    return f;
}

struct FrameResult {
    int n = 0;
    int measured = 0;
    std::vector<int> ids;
    std::vector<cv::Point2f> centers;
    std::vector<bool> was_measured;
};

FrameResult feed(FaceTracker& tr, const std::vector<Face>& dets) {
    tr.update(kFrame, dets);
    FrameResult r;
    for (const auto& v : tr.tracks()) {
        r.ids.push_back(v.id);
        r.centers.push_back((v.face.bbox.top_left + v.face.bbox.bottom_right) * 0.5f);
        r.was_measured.push_back(v.measured);
        if (v.measured) ++r.measured;
    }
    r.n = static_cast<int>(tr.tracks().size());
    return r;
}

// ---- 1. 贴边：框宽在 60/34 px 之间交替（整脸框 vs 只剩可见部分框）-------------
void edge_clipping_keeps_one_track() {
    FaceTracker tr;
    int doubled = 0, frames_with_track = 0, zero_frames = 0;
    std::set<int> ids;
    for (int i = 0; i < 40; ++i) {
        // 相邻帧中心差 13 px、平均框宽 ~47 px -> 位移门 0.28 < 0.6 通过；
        // 而 IoU 只有 0.57/0.34 之间抖动，旧版单门 0.45 会拆轨迹。
        const float w = (i % 2 == 0) ? 60.f : 34.f;
        const float cx = (i % 2 == 0) ? 100.f : 113.f;
        const FrameResult r = feed(tr, {make_face(cx, 60.f, w, 56.f)});
        if (r.n > 1) ++doubled;
        if (r.n >= 1) ++frames_with_track;
        if (r.n == 0) ++zero_frames;
        for (int id : r.ids) ids.insert(id);
    }
    check(doubled == 0, "edge-clipped face never produces two parallel tracks");
    check(ids.size() == 1, "the same face keeps one stable id for 40 frames");
    check(frames_with_track == 40 && zero_frames == 0,
          "the flip is measured as a match on all 40 frames (1 track each, 0 dropped)");
}

// ---- 2. 两张脸按 0.7 / 1.0 / 1.5 个脸宽分开：永远两条，且不会被合并 -----------
void two_faces_are_never_merged() {
    for (float gap : {0.7f, 1.0f, 1.5f}) {
        FaceTracker tr;
        const float w = 40.f;
        bool always_two = true;
        bool distinct_ids = true;
        for (int i = 0; i < 24; ++i) {
            const FrameResult r =
                feed(tr, {make_face(60.f, 60.f, w, 48.f), make_face(60.f + gap * w, 60.f, w, 48.f)});
            if (r.n != 2) always_two = false;
            if (r.n == 2 && r.ids[0] == r.ids[1]) distinct_ids = false;
        }
        char buf[96];
        std::snprintf(buf, sizeof(buf), "two faces %.1f face-widths apart -> 2 tracks, 2 ids", gap);
        check(always_two && distinct_ids, buf);
    }
}

// ---- 3. 目标消失：最多外推 max_lost 帧，且外推帧 measured=false ---------------
void vanished_target_is_carried_but_not_measured() {
    FaceTracker tr;
    int id = -1;
    for (int i = 0; i < 6; ++i) {
        const FrameResult r = feed(tr, {make_face(80.f, 60.f, 44.f, 50.f)});
        if (!r.ids.empty()) id = r.ids[0];
    }
    check(id >= 0, "the face is tracked before it vanishes");

    int carried = 0, carried_measured = 0;
    for (int i = 0; i < 20; ++i) {   // 之后一直没有任何检测
        const FrameResult r = feed(tr, {});
        if (r.n > 0) {
            ++carried;
            carried_measured += r.measured;
            if (r.ids[0] != id) check(false, "carried track must keep its id");
        }
    }
    check(carried == tr.params().max_lost,
          "a vanished face is carried exactly max_lost frames (" +
              std::to_string(tr.params().max_lost) + ")");
    // 这条是"幽灵框被识别"的直接防线：外推帧上不能有任何 measured=true。
    check(carried_measured == 0, "predicted-only frames are never marked measured (no embedding)");
}

// ---- 4. 单帧误检永不上屏 ------------------------------------------------------
void single_frame_false_positive_is_never_drawn() {
    FaceTracker tr;
    int fp_emitted = 0;
    for (int i = 0; i < 12; ++i) {
        std::vector<Face> dets;
        dets.push_back(make_face(80.f, 60.f, 44.f, 50.f));          // 真脸
        if (i == 4) dets.push_back(make_face(20.f, 100.f, 30.f, 30.f));  // 一帧误检
        const FrameResult r = feed(tr, dets);
        for (const auto& c : r.centers) {
            if (c.x < 40.f && c.y > 80.f) ++fp_emitted;
        }
    }
    check(fp_emitted == 0, "a one-frame false positive is never emitted (min_hits)");
}

// ---- 5. 7 px/帧运动：ID 稳定，框不滞后 ---------------------------------------
void constant_motion_has_no_lag() {
    FaceTracker tr;
    std::set<int> ids;
    double sum_err = 0.0, peak_err = 0.0;
    int counted = 0;
    cv::Point2f truth(30.f, 60.f);
    for (int i = 0; i < 30; ++i, truth.x += 7.f) {
        const FrameResult r = feed(tr, {make_face(truth.x, truth.y, 44.f, 50.f)});
        for (size_t k = 0; k < r.ids.size(); ++k) {
            ids.insert(r.ids[k]);
            const double err = cv::norm(r.centers[k] - truth);
            sum_err += err;
            peak_err = std::max(peak_err, err);
            ++counted;
        }
    }
    const double mean_err = counted ? sum_err / counted : 999.0;
    check(ids.size() == 1, "7 px/frame motion keeps a single id");
    check(mean_err <= 3.0, "mean box-centre error <= 3 px at 7 px/frame (baseline 1.70)");
    check(peak_err <= 6.0, "peak box-centre error <= 6 px at 7 px/frame (baseline 2.92)");
}

// ---- 6. 三帧漏检：轨迹不断，measured 只在真有检测的帧上为 true ----------------
void miss_burst_keeps_the_track() {
    FaceTracker tr;
    std::set<int> ids;
    std::vector<bool> measured_by_frame;
    for (int i = 0; i < 16; ++i) {
        const bool has_det = !(i >= 6 && i <= 8);   // 第 6-8 帧漏检
        const FrameResult r = feed(tr, has_det ? std::vector<Face>{make_face(80.f, 60.f, 44.f, 50.f)}
                                              : std::vector<Face>{});
        for (int id : r.ids) ids.insert(id);
        measured_by_frame.push_back(r.n == 1 && r.measured == 1);
    }
    check(ids.size() == 1, "a 3-frame miss burst does not split the track");
    bool gap_unmeasured = true, resumed_measured = true;
    for (int i = 6; i <= 8; ++i) {
        if (measured_by_frame[static_cast<size_t>(i)]) gap_unmeasured = false;
    }
    for (int i = 9; i < 16; ++i) {
        if (!measured_by_frame[static_cast<size_t>(i)]) resumed_measured = false;
    }
    check(gap_unmeasured, "frames inside the miss burst are measured=false");
    check(resumed_measured, "recognition resumes on the first frame with a detection again");
}

}  // namespace

int main() {
    std::printf("tracker scenarios (frame %dx%d, defaults: iou 0.30 / dist 0.60 / max_lost 5 / "
                "min_hits 2)\n", kFrame.width, kFrame.height);
    edge_clipping_keeps_one_track();
    two_faces_are_never_merged();
    vanished_target_is_carried_but_not_measured();
    single_frame_false_positive_is_never_drawn();
    constant_motion_has_no_lag();
    miss_burst_keeps_the_track();

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}