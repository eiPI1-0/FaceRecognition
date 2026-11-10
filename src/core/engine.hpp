#ifndef FACERECOGNITION_ENGINE_HPP
#define FACERECOGNITION_ENGINE_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/detector_head.hpp"
#include "core/face_def.hpp"
#include "core/utils.hpp"

#include "opencv2/opencv.hpp"

// ---------------------------------------------------------------------------
// 库层推理内核（engine）。无 GUI、无 config.hpp 依赖、无跨请求可变状态。
// 不同线程/进程应各自持有一份 Detector + FeatureExtractor 实例。
// ---------------------------------------------------------------------------

// DNN 推理后端配置，统一管理 Detector / FeatureExtractor 的硬件加速选项。
struct DNNConfig {
    cv::dnn::Backend backend = cv::dnn::DNN_BACKEND_OPENCV;
    cv::dnn::Target target = cv::dnn::DNN_TARGET_CPU;

    static DNNConfig cpu() { return {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU}; }
    static DNNConfig cuda() { return {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA}; }
    static DNNConfig openvino() { return {cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, cv::dnn::DNN_TARGET_MYRIAD}; }
};

// 人脸检测器。detect() 按值返回结果，不维护跨调用的可变缓存，
// 因而同一实例的连续调用在逻辑上是无状态的（OpenCV Net 内部的临时
// 状态通过 mutable 隔离，供单线程/单实例独占使用）。
class Detector {
public:
    // output_names: 融合头三个输出张量的名字（本仓库的 YuNet 导出是 loc/conf/iou），
    // 只对融合头生效。按层拆开的头（cls_{s}/obj_{s}/bbox_{s}/kps_{s}）由模型自己的层名
    // 推断并自动解码，不需要也不能用这三个名字描述。两种头都认不出来时构造即抛异常。
    Detector(const std::string& model_path, cv::Size input_shape,
             float conf_thresh = 0.6f, float nms_thresh = 0.3f, int keep_top_k = 750,
             const DNNConfig& config = DNNConfig::cpu(),
             std::vector<cv::String> output_names = {"loc", "conf", "iou"});

    // 检测：返回检测框（含 5 点关键点、置分），坐标系相对于 input_shape（默认 160x120）。
    // 按值返回，不暴露内部引用，多个调用方各自持有结果。
    std::vector<Face> detect(const cv::Mat& img) const;

    cv::Size shape() const { return input_shape; }

private:
    mutable cv::dnn::Net model;          // mutable：setInput/forward 在逻辑上不改“模型”
    cv::Size input_shape;
    float conf_thresh, nms_thresh;
    int keep_top_k;
    std::unique_ptr<DetectorHead> head;  // 解码方案，构造时从模型的输出层名推断
};

// 嵌入特征提取器。112x112 对齐人脸 -> 512-d（或模型定义的 dim）向量。
class FeatureExtractor {
public:
    FeatureExtractor(const std::string& model_path, cv::Size input_shape,
                     const DNNConfig& config = DNNConfig::cpu());

    // 提取单张对齐人脸的特征（不修改输入图）。
    cv::Mat extract(const cv::Mat& img) const;
    // 批量提取：一次 forward 处理多张对齐后的人脸（均应为 input_shape 尺寸）。
    std::vector<cv::Mat> extract_batch(const std::vector<cv::Mat>& imgs) const;

    void set_batch_mode(bool enabled) { batch_mode_ = enabled; }
    bool batch_mode() const { return batch_mode_; }
    cv::Size inputSize() const { return input_shape; }

    // 贴边人脸对齐时的边界填充（cv::BORDER_CONSTANT | _REFLECT | _REPLICATE）。
    void set_align_border(int border_mode) { align_border_ = border_mode; }
    int align_border() const { return align_border_; }

private:
    mutable cv::dnn::Net model;
    cv::Size input_shape;
    bool batch_mode_ = false;
    int align_border_ = cv::BORDER_CONSTANT;
};

// 相似度度量。CosSimilarity::similarity 假设 stored（库侧）特征已 unit-norm，
// 返回值是 ‖ref‖·cos(query, ref)，并非 [0,1] 归一化 cosine；两端必须一起改。
class Similarity {
public:
    virtual double similarity(const cv::Mat& feature, const cv::Mat& ref_feature) = 0;
    virtual bool feature_fusion(std::vector<cv::Mat>& features, cv::Mat& feature) = 0;
    virtual ~Similarity() = default;
};

class NormSimilarity : public Similarity {
public:
    double similarity(const cv::Mat& feature, const cv::Mat& ref_feature) override;
    bool feature_fusion(std::vector<cv::Mat>& features, cv::Mat& feature) override;
};

class CosSimilarity : public Similarity {
public:
    double similarity(const cv::Mat& feature, const cv::Mat& ref_feature) override;
    bool feature_fusion(std::vector<cv::Mat>& features, cv::Mat& feature) override;
};

// ---------------------------------------------------------------------------
// 纯函数：对齐 + 提取。不修改传入的图像。
// ---------------------------------------------------------------------------

// 用 5 点相似变换（旋转+均匀缩放+平移）把 face_roi 对齐到 output_size（默认 112x112）。
// lm 中的关键点为 face_roi 内坐标。几何退化时返回空 Mat。
cv::Mat align_face(const cv::Mat& face_roi,
                   const Landmarks_5& lm,
                   const cv::Size& output_size = cv::Size(112, 112));

// 根据原图 + 检测框，对齐单人脸到固定尺寸，不提取特征。
// det 的 bbox/landmarks 坐标系与 detection_shape 一致（默认 160x120），
// 内部自动映射到原图坐标系后再做裁剪与对齐。
// border_mode：贴边人脸时窗口超出图像边界的填充方式（默认 BORDER_CONSTANT，黑边）。
cv::Mat align_detection(const cv::Mat& img, const Face& det,
                        const cv::Size& detection_shape,
                        const cv::Size& output_size = cv::Size(112, 112),
                        int border_mode = cv::BORDER_CONSTANT);

// 根据原图 + 检测框，提取单人脸对齐后的特征。对齐尺寸取自 extractor.inputSize()。
cv::Mat extract_face_feature(const cv::Mat& img, const Face& det,
                             FeatureExtractor& extractor,
                             const cv::Size& detection_shape);

// 对齐并提取全部人脸特征。features[i] 与 dets[i] 一一对应，
// 对齐失败的人脸对应空 Mat。
std::vector<cv::Mat> extract_face_features(const cv::Mat& img,
                                           const std::vector<Face>& dets,
                                           FeatureExtractor& extractor,
                                           const cv::Size& detection_shape);

// 检测 + 逐人脸对齐 + 特征提取。内部 clone 输入图，不修改调用方传入的 img。
// 与主程序的 letterbox 预处理保持一致，注册/推理尺度才可比。
bool features_from_img(const cv::Mat& img, Detector& detector,
                       FeatureExtractor& extractor, std::vector<cv::Mat>& features,
                       const cv::Size& display_shape, int rfactor);

std::vector<std::string> list_person_dirs(const std::string& path);

#endif // FACERECOGNITION_ENGINE_HPP
