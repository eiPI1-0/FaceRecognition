#include "detector_head.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

// 融合头：单个 loc/conf/iou，先验网格由 PriorBox 生成。
class FusedHead final : public DetectorHead {
public:
    FusedHead(std::vector<cv::String> names, cv::Size input_shape)
        : names_(std::move(names)), pb_(input_shape, input_shape) {}

    const std::vector<cv::String>& outputNames() const override { return names_; }

    std::vector<Face> decode(const std::vector<cv::Mat>& outs, float score_thresh) const override {
        return pb_.decode(outs[0], outs[1], outs[2], score_thresh);
    }

    const char* name() const override { return "fused (loc/conf/iou)"; }

private:
    std::vector<cv::String> names_;
    PriorBox pb_;
};

// 按层头（YuNet-G 风格）：每层输出 cls/obj/bbox/kps，每格 1 个 anchor，stride 写在名字里。
// 解码公式取自 OpenCV 4.13 modules/objdetect/src/face_detect.cpp
// （FaceDetectorYNImpl::postProcess）：score = sqrt(clamp(cls) * clamp(obj))，
// 框 = ((c+dx)*s, (r+dy)*s, exp(dw)*s, exp(dh)*s)，关键点 = (kx+c)*s, (ky+r)*s。
class PerLevelHead final : public DetectorHead {
public:
    PerLevelHead(std::vector<int> strides, cv::Size input_shape)
        : strides_(std::move(strides)), in_(input_shape) {
        for (int s : strides_) {
            // OpenCV 的实现会把输入 pad 到 32 的倍数再前向；我们不做 pad，
            // 所以网格尺寸必须整除，否则这里算出的 idx 与模型的格子对不上。
            if (in_.width % s != 0 || in_.height % s != 0) {
                CV_Error(cv::Error::StsBadArg,
                         "per-level YuNet head: detector input " + std::to_string(in_.width) + "x" +
                         std::to_string(in_.height) + " must be a multiple of stride " + std::to_string(s));
            }
        }
        for (const char* kind : {"cls_", "obj_", "bbox_", "kps_"}) {
            for (int s : strides_) names_.push_back(cv::String(kind) + std::to_string(s));
        }
    }

    const std::vector<cv::String>& outputNames() const override { return names_; }

    std::vector<Face> decode(const std::vector<cv::Mat>& outs, float score_thresh) const override {
        const size_t levels = strides_.size();
        std::vector<Face> dets;
        for (size_t lvl = 0; lvl < levels; ++lvl) {
            const cv::Mat& cls = outs[lvl];
            const cv::Mat& obj = outs[lvl + levels];
            const cv::Mat& bbox = outs[lvl + 2 * levels];
            const cv::Mat& kps = outs[lvl + 3 * levels];

            const int cols = in_.width / strides_[lvl];
            const int rows = in_.height / strides_[lvl];
            const size_t cells = static_cast<size_t>(cols) * rows;

            // 与 PriorBox::decode 同样的理由：契约不符就停下，别让下面的循环越界读。
            if (!cls.isContinuous() || !obj.isContinuous() || !bbox.isContinuous() || !kps.isContinuous() ||
                cls.total() != cells || obj.total() != cells ||
                bbox.total() != cells * 4 || kps.total() != cells * 10) {
                CV_Error(cv::Error::StsBadArg,
                         "per-level YuNet head: level stride " + std::to_string(strides_[lvl]) +
                         " expects cls/obj=" + std::to_string(cells) + " bbox=" + std::to_string(cells * 4) +
                         " kps=" + std::to_string(cells * 10) + ", got cls=" + std::to_string(cls.total()) +
                         " obj=" + std::to_string(obj.total()) + " bbox=" + std::to_string(bbox.total()) +
                         " kps=" + std::to_string(kps.total()) + " for detector input " +
                         std::to_string(in_.width) + "x" + std::to_string(in_.height));
            }

            const float* cls_v = cls.ptr<float>(0);
            const float* obj_v = obj.ptr<float>(0);
            const float* bbox_v = bbox.ptr<float>(0);
            const float* kps_v = kps.ptr<float>(0);
            const float s = static_cast<float>(strides_[lvl]);

            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const size_t idx = static_cast<size_t>(r) * cols + c;
                    const float cls_score = std::min(std::max(cls_v[idx], 0.f), 1.f);
                    const float obj_score = std::min(std::max(obj_v[idx], 0.f), 1.f);
                    const float score = std::sqrt(cls_score * obj_score);
                    if (score < score_thresh) continue;

                    const float cx = (c + bbox_v[idx * 4 + 0]) * s;
                    const float cy = (r + bbox_v[idx * 4 + 1]) * s;
                    const float w = std::exp(bbox_v[idx * 4 + 2]) * s;
                    const float h = std::exp(bbox_v[idx * 4 + 3]) * s;

                    // 关键点顺序（OpenCV 注释）：右眼、左眼、鼻尖、右嘴角、左嘴角。
                    // Landmarks_5 的槽位顺序是 {右眼, 左眼, mouth_left, nose_tip, mouth_right}；
                    // 左右嘴角的命名在 YuNet 导出里本就不一致，align_detection 会用相似变换
                    // 拟合残差去试眼交换×嘴交换，所以命名反了也不会影响对齐质量。
                    const float* k = kps_v + idx * 10;
                    auto kp = [&](int n) { return cv::Point2f((k[2 * n] + c) * s, (k[2 * n + 1] + r) * s); };

                    Face face;
                    face.score = score;
                    face.bbox = {{cx - w / 2.f, cy - h / 2.f}, {cx + w / 2.f, cy + h / 2.f}};
                    face.landmarks = {kp(0), kp(1), kp(4), kp(2), kp(3)};
                    dets.push_back(face);
                }
            }
        }
        return dets;
    }

    const char* name() const override { return "per-level (cls/obj/bbox/kps)"; }

private:
    std::vector<int> strides_;
    std::vector<cv::String> names_;
    cv::Size in_;
};

bool has_all(const std::vector<cv::String>& layer_names, const std::vector<cv::String>& wanted) {
    for (const auto& w : wanted) {
        if (std::find(layer_names.begin(), layer_names.end(), w) == layer_names.end()) return false;
    }
    return true;
}

// 从层名里解析 `<kind>_<stride>` 四胞胎；返回 stride -> {出现的 kind}
std::map<int, std::set<std::string>> scan_per_level(const std::vector<cv::String>& layer_names) {
    static const std::string kinds[] = {"cls_", "obj_", "bbox_", "kps_"};
    std::map<int, std::set<std::string>> found;
    for (const auto& name : layer_names) {
        for (const auto& kind : kinds) {
            if (name.size() > kind.size() && name.compare(0, kind.size(), kind) == 0) {
                const std::string tail(name, kind.size());
                if (tail.find_first_not_of("0123456789") == std::string::npos) {
                    found[std::atoi(tail.c_str())].insert(kind);
                }
            }
        }
    }
    return found;
}

}  // namespace

std::unique_ptr<DetectorHead> make_detector_head(const std::vector<cv::String>& layer_names,
                                                 cv::Size input_shape,
                                                 const std::vector<cv::String>& fused_names) {
    if (has_all(layer_names, fused_names)) {
        return std::unique_ptr<DetectorHead>(new FusedHead(fused_names, input_shape));
    }

    auto found = scan_per_level(layer_names);
    std::vector<int> strides;
    for (const auto& kv : found) {
        if (kv.second.size() == 4) strides.push_back(kv.first);
    }
    if (!strides.empty()) {
        // 出现了 cls_8/obj_8/bbox_8 但缺 kps_8 这类半套输出：说清楚缺了什么，
        // 而不是悄悄只用其中一层。
        for (const auto& kv : found) {
            if (kv.second.size() != 4) {
                std::string missing;
                for (const char* kind : {"cls_", "obj_", "bbox_", "kps_"}) {
                    if (!kv.second.count(kind)) missing += std::string(" ") + kind;
                }
                CV_Error(cv::Error::StsBadArg,
                         "per-level YuNet head: stride " + std::to_string(kv.first) +
                         " is missing the output(s):" + missing);
            }
        }
        std::sort(strides.begin(), strides.end());
        return std::unique_ptr<DetectorHead>(new PerLevelHead(std::move(strides), input_shape));
    }

    std::string names;
    for (const auto& n : layer_names) {
        if (n.find("onnx_node!") == 0) continue;  // 中间层，噪音
        names += " " + std::string(n);
    }
    CV_Error(cv::Error::StsBadArg,
             "unrecognized detector head: expected fused outputs named"
             " " + fused_names[0] + "/" + fused_names[1] + "/" + fused_names[2] +
             ", or per-level cls_<s>/obj_<s>/bbox_<s>/kps_<s>; the model exposes:" + names);
}
