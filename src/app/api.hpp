#ifndef FACERECOGNITION_API_HPP
#define FACERECOGNITION_API_HPP

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

#include "app/config.hpp"
#include "core/engine.hpp"
#include "app/gallery.hpp"

namespace facerec {

// 单次识别请求的可调项。全部是请求级参数：改阈值不需要重启进程。
struct IdentifyOptions {
    int top_k = 5;
    float min_score = 0.0f;          // 0 = 只用 recognize.thresh 决定 matched
    bool include_landmarks = false;
    bool include_crops = false;      // 112x112 对齐图，PNG base64
    bool include_embedding = false;  // 512 维 float 数组
    // 请求级检测输入（0 = 用配置里的默认值）。量化到几档以减少 PriorBox 重建。
    int detector_w = 0;
    int detector_h = 0;
};

struct FaceMatch {
    std::string person_id;
    float score = 0.f;
};

// 一张人脸的识别结果。坐标一律是**原图像素**（已从 detector 空间逆映射回来）。
struct FaceResult {
    int index = 0;
    std::array<float, 4> bbox{};           // x, y, w, h
    float det_score = 0.f;
    std::optional<Landmarks_5> landmarks;  // 原图像素坐标
    std::optional<FaceMatch> matched;      // nullopt = 低于阈值（未知）
    std::vector<FaceMatch> top_k;
    std::string crop_png_b64;
    std::vector<float> embedding;
    bool aligned = false;                  // 对齐成功（否则特征为空，无法匹配）
};

struct IdentifyResult {
    std::string source;          // 图片来源（路径/URL/调用方标识）
    int image_w = 0, image_h = 0;
    int det_w = 0, det_h = 0;
    double ms_decode = 0, ms_detect = 0, ms_embed = 0;
    size_t gallery_size = 0;
    std::string detector_model, embedder_model;
    std::vector<FaceResult> faces;
};

// 应用级流水线：detector + extractor + gallery。
//
// **一个实例只能被一个线程使用**：cv::dnn::Net 的 setInput/forward 不是线程安全的，
// 检测与提特征都写 net 的内部状态。并发靠"每线程一个 Pipeline"，
// 见 server.cpp 的 PipelinePool。
class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& cfg);

    // 识别一张已解码的图。返回的 bbox/landmarks 在原图像素坐标系。
    //
    // 请求级检测输入（opt.detector_w/h）只改变**送进检测器的像素数**，不改变
    // letterbox 画布：画布始终是 cfg.display_shape()。这条边界很关键 —— 对齐裁剪取自
    // 画布，所以 gallery 与任何请求的对齐来源分辨率一致，换检测尺寸不会造成特征域偏移；
    // 而检测器从同一张更大的画布上重采样，等于用更多像素换召回（远景/合照）。
    // 传 0 表示用配置默认值。不同尺寸的 Detector 按需创建并缓存（见 detectors_）。
    //
    // 非 const：会为新的检测尺寸惰性建 Detector。一个实例仍只允许单线程使用。
    IdentifyResult identify(const cv::Mat& img, const IdentifyOptions& opt,
                            const std::string& source = std::string());

    // 单张图的对齐人脸特征（用于 /v1/embed）。失败返回空 Mat。
    cv::Mat embed_image(const cv::Mat& img, int face_index);

    size_t gallery_size() const { return gallery_.size(); }
    const PipelineConfig& config() const { return cfg_; }
    const Gallery& gallery() const { return gallery_; }

    // 单人匹配（demo 的按轨迹缓存标签用）。threshold 之下返回 false，但 score/name
    // 仍写最高分那位 —— 调用方要能解释"最像谁，但不够像"。
    bool best_match(const cv::Mat& feature, float threshold, float& score,
                    std::string& name) const {
        return gallery_.best_match(feature, threshold, score, name);
    }

    // 缓存键的一部分：模型 + gallery 内容摘要。换模型/重注册后旧缓存必须失效。
    std::string fingerprint() const;

private:
    // 按检测尺寸取（必要时创建）Detector。超过 kMaxDetectorBuckets 个尺寸时淘汰最久未用的。
    Detector& detector_for(cv::Size det_shape);

    static constexpr size_t kMaxDetectorBuckets = 8;

    PipelineConfig cfg_;
    cv::Size det_shape_;          // 配置里的默认检测尺寸
    cv::Size display_shape_;      // letterbox 画布（固定，与请求无关）
    int border_mode_ = cv::BORDER_CONSTANT;
    std::unique_ptr<FeatureExtractor> extractor_;
    Gallery gallery_;
    std::vector<cv::String> output_names_;
    DNNConfig dnn_config_;

    struct DetectorEntry {
        std::unique_ptr<Detector> detector;
        uint64_t last_used = 0;
    };
    std::map<std::pair<int, int>, DetectorEntry> detectors_;
    uint64_t clock_ = 0;
};

// ---------------------------------------------------------------------------
// 启动自检：阈值 + 库规模（见 Gallery::self_estimate）
// ---------------------------------------------------------------------------

// 把"当前阈值在当前库规模下意味着什么"算成一个数，几个外壳共用同一条判定与同一
// 句话（两份描述一定有一份会过时）。背景：knob 注释曾写 "cross-identity near 0"——
// 在 610 人库里 probe 对单张注册照的均值确实是 0.0051，但对 609 个对手取最大值后均值
// 变成 0.3125。1:1 与 1:N 不是同一个分布，所以那句话用在 identify 上是错的。
struct ThresholdCheck {
    Gallery::SelfEstimate est;
    float threshold = 0.0f;
    bool too_permissive = false;   // far_1n >= recognize.self_far_warn
    bool estimable = false;        // 库里至少两人，否则没有"不同人配对"
    // 这个库能证的**最小** 1:N 误接受率（把一个配对算进去时的值）。far_1n 低于它
    // 就只是"没观察到"，不是"已证明"。
    double far_1n_resolution = 0.0;
    // far_1n 是不是一个**测量值**：库里观察到过至少一个 >= 阈值的跨人配对才算。
    // above == 0 时 far_1n == 0 不代表"安全"，只代表 pairs 个配对里没观察到，能证的
    // 下限是 far_1n_resolution。打印方必须能把这两种 0 分开——0.000000 会被读成
    // "已证明安全"，那和把 0.2 当真是同一个错误，只是方向相反。
    bool certifiable = false;
    std::string human;             // 一行话（banner / stderr）
    std::string fields;            // log.event 的 fields（对象成员列表）
};

ThresholdCheck check_recognize_threshold(const Gallery& gallery, const PipelineConfig& cfg);

// "阈值 + 库规模"这组字段的唯一一种编码：/readyz、serve 的 startup 审计行、mcp 的
// startup 审计行共用它（三处手写列表的结局一定是其中一处过时，而 /readyz 是唯一
// 能在事后回答"当时那个库上这个阈值意味着什么"的地方）。
// far_1n 不可证时打 **null**，并附上分辨率与原始计数（above/pairs），让消费者自己
// 算上界；thresh_key 让 /readyz 的嵌套块与审计行的顶层键各自自然。
std::string recognize_audit_fields(const ThresholdCheck& c, const char* thresh_key = "recognize_thresh");

// ---------------------------------------------------------------------------
// JSON 序列化（stdout / HTTP 响应共用一份，避免两个外壳输出不一致）
// ---------------------------------------------------------------------------

// 完整的识别结果对象。status_ok=false 时输出 {"status":"error","error":...}。
std::string identify_to_json(const IdentifyResult& result, const IdentifyOptions& opt);
std::string error_to_json(const std::string& message, const std::string& code = "error");

// GET /v1/people 的响应（不含 embedding / 照片）。
std::string people_to_json(const Pipeline& pipeline);

// 请求级检测尺寸的合法性：必须能被 8 整除、在 [32,1024] 内。
// 两种检测头（融合头 strides 8/16/32/64、逐层头 input/stride）都要求这个；
// 越界或不合法的尺寸会让 PriorBox 的先验网格与模型对不上，那是一个 500 而不是用户错误。
bool valid_detector_size(int w, int h, std::string* err = nullptr);

// 通用小工具（CLI 与 HTTP 都要用）
std::string json_escape(const std::string& s);
std::string format_float(double v, int precision = 4);

}  // namespace facerec

#endif  // FACERECOGNITION_API_HPP