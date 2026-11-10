#include "app/api.hpp"

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>


#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app/evaluation.hpp"   // one_n_far_from_pair_rate：阈值自检的算术在这里，只写一次

namespace facerec {

namespace {

// 与 utils::letterbox 完全一致的缩放/补边参数，用于把检测坐标
// 从 detector 空间逆映射回原图像素坐标。
struct LetterboxMeta {
    double scale = 1.0;
    int pad_left = 0;
    int pad_top = 0;
    int canvas_w = 0;
    int canvas_h = 0;
};

LetterboxMeta letterbox_meta(const cv::Mat& img, const cv::Size& canvas) {
    LetterboxMeta m;
    m.canvas_w = canvas.width;
    m.canvas_h = canvas.height;
    if (img.empty() || canvas.width <= 0 || canvas.height <= 0) {
        return m;
    }
    m.scale = std::min(static_cast<double>(canvas.width) / img.cols,
                       static_cast<double>(canvas.height) / img.rows);
    const int new_w = std::max(1, static_cast<int>(std::lround(img.cols * m.scale)));
    const int new_h = std::max(1, static_cast<int>(std::lround(img.rows * m.scale)));
    m.pad_left = (canvas.width - new_w) / 2;
    m.pad_top = (canvas.height - new_h) / 2;
    return m;
}

// detector 空间 -> 画布空间 -> 原图像素
cv::Point2f det_to_original(const cv::Point2f& p_det, const LetterboxMeta& meta,
                            const cv::Size& det_shape) {
    const double x_canvas = p_det.x * (static_cast<double>(meta.canvas_w) / det_shape.width);
    const double y_canvas = p_det.y * (static_cast<double>(meta.canvas_h) / det_shape.height);
    const double x_orig = (x_canvas - meta.pad_left) / meta.scale;
    const double y_orig = (y_canvas - meta.pad_top) / meta.scale;
    return cv::Point2f(static_cast<float>(x_orig), static_cast<float>(y_orig));
}

Landmarks_5 landmarks_to_original(const Landmarks_5& lm, const LetterboxMeta& meta,
                                  const cv::Size& det_shape) {
    Landmarks_5 out;
    out.right_eye   = det_to_original(lm.right_eye, meta, det_shape);
    out.left_eye    = det_to_original(lm.left_eye, meta, det_shape);
    out.mouth_left  = det_to_original(lm.mouth_left, meta, det_shape);
    out.nose_tip    = det_to_original(lm.nose_tip, meta, det_shape);
    out.mouth_right = det_to_original(lm.mouth_right, meta, det_shape);
    return out;
}

std::string base64_encode(const std::vector<uchar>& buf) {
    static const char* table =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string b64;
    b64.reserve(buf.size() * 4 / 3 + 4);
    size_t k = 0;
    for (; k + 2 < buf.size(); k += 3) {
        b64 += table[(buf[k] >> 2) & 0x3F];
        b64 += table[((buf[k] << 4) | (buf[k + 1] >> 4)) & 0x3F];
        b64 += table[((buf[k + 1] << 2) | (buf[k + 2] >> 6)) & 0x3F];
        b64 += table[buf[k + 2] & 0x3F];
    }
    const size_t rem = buf.size() - k;
    if (rem == 1) {
        b64 += table[(buf[k] >> 2) & 0x3F];
        b64 += table[(buf[k] << 4) & 0x3F];
        b64 += "==";
    } else if (rem == 2) {
        b64 += table[(buf[k] >> 2) & 0x3F];
        b64 += table[((buf[k] << 4) | (buf[k + 1] >> 4)) & 0x3F];
        b64 += table[(buf[k + 1] << 2) & 0x3F];
        b64 += "=";
    }
    return b64;
}

}  // namespace

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

Pipeline::Pipeline(const PipelineConfig& cfg) : cfg_(cfg) {
    dnn_config_ = (cfg.backend == "cuda")        ? DNNConfig::cuda()
                  : (cfg.backend == "openvino")  ? DNNConfig::openvino()
                                                 : DNNConfig::cpu();

    det_shape_ = cfg.detector_input();
    display_shape_ = cfg.display_shape();
    output_names_ = {cfg.out_loc, cfg.out_conf, cfg.out_iou};

    // 默认尺寸的检测器先建好：gallery 构建要用它，第一个请求也不必等。
    Detector& default_detector = detector_for(det_shape_);

    extractor_ = std::make_unique<FeatureExtractor>(cfg.embedder_model, cfg.embedder_input(),
                                                    dnn_config_);
    if (cfg.batch_mode != 0) {
        extractor_->set_batch_mode(cfg.batch_mode > 0);
    }

    // align.border 必须在本行之前设好：注册就发生在 Gallery::load_or_build 里，
    // 注册与推理用了不同的填充，gallery 与查询特征就不可比（本项目的硬约束）。
    if (cfg.align_border == "reflect") border_mode_ = cv::BORDER_REFLECT;
    else if (cfg.align_border == "replicate") border_mode_ = cv::BORDER_REPLICATE;
    else border_mode_ = cv::BORDER_CONSTANT;
    extractor_->set_align_border(border_mode_);

    // 注册永远按配置的默认检测尺寸走（画布固定），否则换请求尺寸就会重算 gallery。
    // 有索引就只读索引（不解码照片），源文件/模型变了才重建。
    std::string gallery_err;
    gallery_ = Gallery::load_or_build(cfg, default_detector, *extractor_, display_shape_,
                                      cfg.rfactor, nullptr, &gallery_err);
    if (!gallery_err.empty()) {
        std::cerr << "gallery: " << gallery_err << std::endl;
    }
}

Detector& Pipeline::detector_for(cv::Size det_shape) {
    const std::pair<int, int> key{det_shape.width, det_shape.height};
    auto it = detectors_.find(key);
    if (it != detectors_.end()) {
        it->second.last_used = ++clock_;
        return *it->second.detector;
    }
    // 有界：尺寸来自请求，不设上限就是一条无界内存增长路径。淘汰最久未用的那个。
    if (detectors_.size() >= kMaxDetectorBuckets) {
        auto oldest = detectors_.begin();
        for (auto cur = detectors_.begin(); cur != detectors_.end(); ++cur) {
            if (cur->second.last_used < oldest->second.last_used) oldest = cur;
        }
        detectors_.erase(oldest);
    }
    std::vector<cv::String> names = output_names_;
    DetectorEntry entry;
    entry.detector = std::make_unique<Detector>(
        cfg_.detector_model, det_shape, cfg_.conf_thresh, cfg_.nms_thresh,
        cfg_.keep_top_k, dnn_config_, std::move(names));
    entry.last_used = ++clock_;
    auto [pos, ok] = detectors_.emplace(key, std::move(entry));
    (void)ok;
    return *pos->second.detector;
}

std::string Pipeline::fingerprint() const {
    // gallery 的内容摘要已经涵盖"模型 + 每张注册照片的大小/mtime + 照片列表"，
    // 再拼上画布尺寸（换画布等于换预处理，特征也会变）。
    std::ostringstream os;
    os << gallery_.fingerprint() << '|' << display_shape_.width << 'x'
       << display_shape_.height << '|' << gallery_.size();
    return os.str();
}

IdentifyResult Pipeline::identify(const cv::Mat& img, const IdentifyOptions& opt,
                                  const std::string& source) {
    IdentifyResult out;
    out.source = source;
    out.gallery_size = gallery_.size();
    out.detector_model = cfg_.detector_model;
    out.embedder_model = cfg_.embedder_model;
    if (img.empty()) {
        return out;
    }
    out.image_w = img.cols;
    out.image_h = img.rows;

    // 请求级检测尺寸：只换送进检测器的像素数。画布（display_shape_）保持配置值，
    // 因此对齐裁剪的来源分辨率与 gallery 一致 —— 换检测尺寸不会把特征推到另一个域。
    cv::Size det_shape = det_shape_;
    if (opt.detector_w > 0 && opt.detector_h > 0) {
        det_shape = cv::Size(opt.detector_w, opt.detector_h);
    }
    out.det_w = det_shape.width;
    out.det_h = det_shape.height;

    cv::Mat work = letterbox(img, display_shape_);
    const LetterboxMeta meta = letterbox_meta(img, display_shape_);

    const auto t0 = cv::getTickCount();
    std::vector<Face> dets = detector_for(det_shape).detect(work);
    const auto t1 = cv::getTickCount();
    std::vector<cv::Mat> features = extract_face_features(work, dets, *extractor_, det_shape);
    const auto t2 = cv::getTickCount();

    const double freq = cv::getTickFrequency();
    out.ms_detect = static_cast<double>(t1 - t0) / freq * 1000.0;
    out.ms_embed = static_cast<double>(t2 - t1) / freq * 1000.0;

    out.faces.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        const Face& f = dets[i];
        const cv::Mat& feat = (i < features.size()) ? features[i] : cv::Mat();

        FaceResult fr;
        fr.index = static_cast<int>(i);
        const cv::Point2f tl = det_to_original(f.bbox.top_left, meta, det_shape);
        const cv::Point2f br = det_to_original(f.bbox.bottom_right, meta, det_shape);
        fr.bbox = {tl.x, tl.y, br.x - tl.x, br.y - tl.y};
        fr.det_score = f.score;
        fr.aligned = !feat.empty();

        if (opt.include_landmarks) {
            fr.landmarks = landmarks_to_original(f.landmarks, meta, det_shape);
        }

        if (!feat.empty()) {
            float sim = -1.f;
            std::string name;
            if (gallery_.best_match(feat, cfg_.recogn_thresh, sim, name) &&
                sim >= opt.min_score) {
                fr.matched = FaceMatch{name, sim};
            }
            for (const auto& [idx, score] : gallery_.rank(feat, opt.top_k)) {
                if (score < opt.min_score) continue;
                fr.top_k.push_back(FaceMatch{gallery_.entries()[idx].person_id, score});
            }
        }

        if (opt.include_crops && !feat.empty()) {
            cv::Mat aligned = align_detection(work, f, det_shape, extractor_->inputSize(),
                                              border_mode_);
            if (!aligned.empty()) {
                std::vector<uchar> buf;
                if (cv::imencode(".png", aligned, buf)) {
                    fr.crop_png_b64 = base64_encode(buf);
                }
            }
        }

        if (opt.include_embedding && !feat.empty()) {
            const int dims = static_cast<int>(feat.total());
            fr.embedding.resize(dims);
            const float* ptr = feat.ptr<float>();
            std::copy(ptr, ptr + dims, fr.embedding.begin());
        }

        out.faces.push_back(std::move(fr));
    }
    out.ms_decode = 0.0;   // 解码发生在调用方（imread/imdecode），由外壳层填
    return out;
}

cv::Mat Pipeline::embed_image(const cv::Mat& img, int face_index) {
    if (img.empty()) {
        return {};
    }
    cv::Mat work = letterbox(img, display_shape_);
    std::vector<Face> dets = detector_for(det_shape_).detect(work);
    if (face_index < 0 || face_index >= static_cast<int>(dets.size())) {
        return {};
    }
    return extract_face_feature(work, dets[face_index], *extractor_, det_shape_);
}

ThresholdCheck check_recognize_threshold(const Gallery& gallery, const PipelineConfig& cfg) {
    ThresholdCheck c;
    c.threshold = cfg.recogn_thresh;
    c.est = gallery.self_estimate(static_cast<double>(cfg.recogn_thresh));
    c.estimable = c.est.gallery >= 2 && c.est.pairs > 0;
    if (c.est.pairs)
        // 与 est.far_1n 用同一个 n（=库大小），两个数才能直接比。
        c.far_1n_resolution = one_n_far_from_pair_rate(1.0 / static_cast<double>(c.est.pairs),
                                                       c.est.gallery);

    char buf[700];
    std::snprintf(buf, sizeof buf,
                  "\"thresh\":%.4f,\"gallery\":%zu,\"sampled\":%zu,\"pairs\":%zu,"
                  "\"above\":%zu,\"pair_rate\":%.3e,\"far_1n\":%.3e,"
                  "\"far_1n_resolution\":%.3e,\"estimable\":%s",
                  c.threshold, c.est.gallery, c.est.sampled, c.est.pairs, c.est.above,
                  c.est.pair_rate, c.est.far_1n, c.far_1n_resolution,
                  c.estimable ? "true" : "false");
    c.fields = buf;

    if (!c.estimable) {
        c.human = "threshold check: fewer than 2 enrolled people, so the 1:N "
                  "false-accept rate cannot be estimated yet";
        return c;
    }
    c.too_permissive = c.est.far_1n >= static_cast<double>(cfg.self_far_warn);
    if (c.too_permissive) {
        std::snprintf(buf, sizeof buf,
                      "recognize.thresh=%.2f against a %zu-person gallery admits "
                      "strangers: an unenrolled face is matched to somebody with "
                      "probability %.1f%% (per-pair rate %.3e over %zu gallery "
                      "cross-matches; a gallery-side estimate is good to the order of "
                      "magnitude, not the third digit - on LFW it ran 11 points high vs "
                      "held-out probes). Raise recognize.thresh, or calibrate on your "
                      "own population with './main evaluate <dataset>'.",
                      c.threshold, c.est.gallery, 100.0 * c.est.far_1n,
                      c.est.pair_rate, c.est.pairs);
    } else if (c.est.above == 0) {
        // "一个都没超"不是"零"：库内配对只有 pairs 个，能证的尾概率下限是
        // 1/pairs。不报这个下限，0.00 会被读成"已证明安全"——那和把 0.2 当真
        // 是同一个错，只是方向相反。
        const double floor_far = c.far_1n_resolution;
        std::snprintf(buf, sizeof buf,
                      "recognize.thresh=%.2f against a %zu-person gallery: no cross-person "
                      "pair among %zu exceeded it, so the expected false-accept rate is "
                      "below %.1e (this gallery's one-pair resolution). Enroll more people, "
                      "or run './main evaluate <dataset>' for a tighter number.",
                      c.threshold, c.est.gallery, c.est.pairs, floor_far);
    } else {
        std::snprintf(buf, sizeof buf,
                      "recognize.thresh=%.2f against a %zu-person gallery: an "
                      "unenrolled face is matched to somebody with probability %.2e "
                      "(per-pair rate %.3e over %zu gallery cross-matches; order of "
                      "magnitude, not the third digit).",
                      c.threshold, c.est.gallery, c.est.far_1n, c.est.pair_rate, c.est.pairs);
    }
    c.human = buf;
    return c;
}

// ---------------------------------------------------------------------------
// JSON 序列化
// ---------------------------------------------------------------------------

bool valid_detector_size(int w, int h, std::string* err) {
    auto fail = [&](const std::string& msg) {
        if (err) *err = msg;
        return false;
    };
    if (w <= 0 && h <= 0) return true;   // 未指定 = 用配置默认值
    if (w < 32 || h < 32 || w > 1024 || h > 1024) {
        return fail("detector_input must be within [32,1024], got " +
                    std::to_string(w) + "x" + std::to_string(h));
    }
    if (w % 8 != 0 || h % 8 != 0) {
        return fail("detector_input must be divisible by 8, got " +
                    std::to_string(w) + "x" + std::to_string(h));
    }
    return true;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string format_float(double v, int precision) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << v;
    return os.str();
}

std::string error_to_json(const std::string& message, const std::string& code) {
    return "{\"status\":\"error\",\"error\":{\"code\":\"" + json_escape(code) +
           "\",\"message\":\"" + json_escape(message) + "\"}}";
}

std::string identify_to_json(const IdentifyResult& r, const IdentifyOptions& opt) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(4);

    os << "{";
    os << "\"status\":\"ok\",";
    os << "\"source\":\"" << json_escape(r.source) << "\",";
    os << "\"image_size\":[" << r.image_w << "," << r.image_h << "],";
    os << "\"detector_input\":[" << r.det_w << "," << r.det_h << "],";
    os << "\"timing_ms\":{";
    os << "\"detect\":" << format_float(r.ms_detect) << ",";
    os << "\"embed\":" << format_float(r.ms_embed) << ",";
    os << "\"total\":" << format_float(r.ms_detect + r.ms_embed);
    os << "},";
    os << "\"model\":{";
    os << "\"detector_model\":\"" << json_escape(r.detector_model) << "\",";
    os << "\"embedder_model\":\"" << json_escape(r.embedder_model) << "\",";
    os << "\"gallery_size\":" << r.gallery_size;
    os << "},";

    os << "\"faces\":[";
    for (size_t i = 0; i < r.faces.size(); ++i) {
        if (i) os << ",";
        const FaceResult& f = r.faces[i];
        os << "{";
        os << "\"index\":" << f.index << ",";
        os << "\"bbox\":[" << format_float(f.bbox[0]) << "," << format_float(f.bbox[1])
           << "," << format_float(f.bbox[2]) << "," << format_float(f.bbox[3]) << "],";
        os << "\"det_score\":" << format_float(f.det_score) << ",";
        os << "\"aligned\":" << (f.aligned ? "true" : "false") << ",";

        if (f.landmarks) {
            const Landmarks_5& lm = *f.landmarks;
            auto pt = [](const cv::Point2f& p) {
                return "[" + format_float(p.x) + "," + format_float(p.y) + "]";
            };
            os << "\"landmarks\":{";
            os << "\"right_eye\":" << pt(lm.right_eye) << ",";
            os << "\"left_eye\":" << pt(lm.left_eye) << ",";
            os << "\"mouth_left\":" << pt(lm.mouth_left) << ",";
            os << "\"nose_tip\":" << pt(lm.nose_tip) << ",";
            os << "\"mouth_right\":" << pt(lm.mouth_right);
            os << "},";
        }

        os << "\"matched\":";
        if (f.matched) {
            os << "{\"person_id\":\"" << json_escape(f.matched->person_id) << "\","
               << "\"score\":" << format_float(f.matched->score) << "}";
        } else {
            os << "null";
        }
        os << ",";

        os << "\"top_k\":[";
        for (size_t k = 0; k < f.top_k.size(); ++k) {
            if (k) os << ",";
            os << "{\"person_id\":\"" << json_escape(f.top_k[k].person_id) << "\","
               << "\"score\":" << format_float(f.top_k[k].score) << "}";
        }
        os << "]";

        if (opt.include_crops && !f.crop_png_b64.empty()) {
            os << ",\"crop_png_b64\":\"" << f.crop_png_b64 << "\"";
        }
        if (opt.include_embedding && !f.embedding.empty()) {
            os << ",\"embedding\":[";
            for (size_t d = 0; d < f.embedding.size(); ++d) {
                if (d) os << ",";
                os << format_float(f.embedding[d], 6);
            }
            os << "]";
        }
        os << "}";
    }
    os << "]}";
    return os.str();
}

std::string people_to_json(const Pipeline& pipeline) {
    // 只暴露标识与来源信息，**永不**返回 embedding 或照片字节。
    std::ostringstream os;
    os << "{\"status\":\"ok\",\"items\":[";
    const auto& entries = pipeline.gallery().entries();
    bool first = true;
    for (const auto& e : entries) {
        if (!first) os << ",";
        first = false;
        os << "{\"person_id\":\"" << json_escape(e.person_id) << "\",\"n_photos\":"
           << e.n_photos << "}";
    }
    os << "],\"count\":" << entries.size() << ",\"fingerprint\":\""
       << json_escape(pipeline.gallery().fingerprint()) << "\"}";
    return os.str();
}

}  // namespace facerec