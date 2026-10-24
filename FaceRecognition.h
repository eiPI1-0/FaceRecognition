#ifndef FACERECOGNITION_FACERECOGNITION_H
#define FACERECOGNITION_FACERECOGNITION_H

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "priorbox.hpp"
#include "utils.hpp"

#include "opencv2/opencv.hpp"

class FeatureExtractor;

// DNN 推理后端配置，统一管理 Detector / FeatureExtractor 的硬件加速选项。
struct DNNConfig {
    cv::dnn::Backend backend = cv::dnn::DNN_BACKEND_OPENCV;
    cv::dnn::Target target = cv::dnn::DNN_TARGET_CPU;

    // 常用预设
    static DNNConfig cpu() { return {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU}; }
    static DNNConfig cuda() { return {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA}; }
    static DNNConfig openvino() { return {cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, cv::dnn::DNN_TARGET_MYRIAD}; }
};

class Detector {
public:
    Detector(const std::string& model_path, cv::Size input_shape,
             float conf_thresh = 0.6f, float nms_thresh = 0.3f, int keep_top_k = 750,
             const DNNConfig& config = DNNConfig::cpu());
    // 检测结果的 bbox/landmarks 坐标系相对于 input_shape（160x120）。
    std::vector<Face>& detect(cv::Mat& img);
    size_t size() const;
    std::vector<Face>& faces();
    cv::Size& shape();

private:
    cv::dnn::Net model;
    cv::Size input_shape;
    float conf_thresh, nms_thresh;
    int keep_top_k;
    PriorBox pb;
    std::vector<cv::String> output_names = {"loc", "conf", "iou"};
    std::vector<cv::Mat> output_blobs;
    std::vector<Face> dets;
    size_t detected_nums = 0;
};

// 对齐单张人脸 ROI 到固定尺寸（默认 112x112），供嵌入提取使用。
// left_eye / right_eye 均为 face_roi 内的坐标。
cv::Mat align_face(const cv::Mat& face_roi,
                   const cv::Point2f& left_eye,
                   const cv::Point2f& right_eye,
                   const cv::Size& output_size = cv::Size(112, 112));

// 纯函数：根据原图 + 检测框，对齐单人脸到固定尺寸（默认 112x112），不提取特征。
// det 的 bbox/landmarks 坐标系与 detection_shape 一致（默认 160x120），
// 内部自动映射到原图坐标系后再做裁剪与对齐。
cv::Mat align_detection(const cv::Mat& img, const Face& det,
                        const cv::Size& detection_shape,
                        const cv::Size& output_size = cv::Size(112, 112));

// 纯函数：根据原图 + 检测框，提取单人脸对齐后的特征，不修改输入图。
cv::Mat extract_face_feature(const cv::Mat& img, const Face& det,
                             FeatureExtractor& extractor,
                             const cv::Size& detection_shape,
                             const cv::Size& output_size = cv::Size(112, 112));

// 对齐并提取全部人脸特征。返回的 features[i] 与 dets[i] 一一对应，
// 对齐失败的人脸对应空 Mat。batch/逐张策略由 extractor.batch_mode() 决定。
std::vector<cv::Mat> extract_face_features(const cv::Mat& img,
                                           const std::vector<Face>& dets,
                                           FeatureExtractor& extractor,
                                           const cv::Size& detection_shape);

class FeatureExtractor {
public:
    FeatureExtractor(const std::string& model_path, cv::Size input_shape,
                     const DNNConfig& config = DNNConfig::cpu());
    cv::Mat extract(const cv::Mat& img);
    // 批量提取：一次 forward 处理多张对齐后的人脸（均应为 input_shape 尺寸）。
    // 返回与输入等长的特征向量，顺序一一对应。
    std::vector<cv::Mat> extract_batch(const std::vector<cv::Mat>& imgs);

    // 是否使用 batch 前向。仅对 GPU/NPU 等异步后端有收益；
    // OpenCV CPU 后端逐样本执行图，batch 反而因内部 blob 重建而变慢。
    void set_batch_mode(bool enabled) { batch_mode_ = enabled; }
    bool batch_mode() const { return batch_mode_; }
    cv::Size inputSize() const { return input_shape; }

private:
    cv::dnn::Net model;
    cv::Size input_shape;
    bool batch_mode_ = false;
};

// 检测 + 逐人脸对齐 + 特征提取。内部会 clone 输入图，不修改调用方传入的 img。
bool features_from_img(cv::Mat& img, Detector& detector, FeatureExtractor& extractor,
                       std::vector<cv::Mat>& features, const cv::Size& display_shape, int rfactor);

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

// 列出 data_path 下所有人脸注册目录（排除 . / .. 及非目录条目），结果已排序，确保确定性。
std::vector<std::string> list_person_dirs(const std::string& path);

template<typename T>
class FeatureDatabase {
public:
    FeatureDatabase(std::string data_path, Detector& detector,
                    FeatureExtractor& extractor, const cv::Size& display_shape,
                    int rfactor, float threshold = 0.8f)
        : path(std::move(data_path)), threshold(threshold),
          similar_calculator(std::make_unique<T>()) {
        load_database(detector, extractor, display_shape, rfactor);
    }

    ~FeatureDatabase() = default;

    bool select_max_similarity(const cv::Mat& feature, float& similarity,
                               std::string& name) {
        similarity = -1.0f;
        if (feature.empty()) {
            return false;
        }
        for (const auto& entry : feature_database) {
            const std::string& person_name = entry.first;
            const cv::Mat& ref_feature = entry.second;
            if (ref_feature.empty()) {
                continue;
            }
            double sim = similar_calculator->similarity(feature, ref_feature);
            if (similarity < sim) {
                similarity = static_cast<float>(sim);
                name = person_name;
            }
        }
        return similarity >= threshold;
    }

    size_t size() const { return feature_database.size(); }

private:
    void load_database(Detector& detector, FeatureExtractor& extractor,
                       const cv::Size& display_shape, int rfactor) {
        std::vector<std::string> person_dirs = list_person_dirs(path);

        std::vector<cv::String> photos;
        for (const auto& person : person_dirs) {
            std::vector<cv::Mat> person_features;
            std::string glob_pattern = path + "/" + person + "/*";
            cv::glob(glob_pattern, photos);
            for (const auto& photo : photos) {
                cv::Mat img = cv::imread(photo);
                if (img.empty()) {
                    continue;
                }
                std::vector<cv::Mat> feats;
                if (features_from_img(img, detector, extractor, feats,
                                      display_shape, rfactor) &&
                    !feats.empty() && !feats[0].empty()) {
                    person_features.push_back(feats[0].clone());
                }
            }
            photos.clear();

            if (!person_features.empty()) {
                cv::Mat fused;
                if (similar_calculator->feature_fusion(person_features, fused)) {
                    feature_database.emplace_back(person, std::move(fused));
                }
            }
        }
    }

    std::string path;
    std::unique_ptr<Similarity> similar_calculator;
    // 每人一份融合后的特征，按人名显式配对，不再依赖 readdir 顺序与 i=2 假设
    std::vector<std::pair<std::string, cv::Mat>> feature_database;
    float threshold;
};

#endif // FACERECOGNITION_FACERECOGNITION_H
