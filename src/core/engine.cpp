#include "core/engine.hpp"

#include <cstdlib>
#include <dirent.h>
#include <thread>
#include <sys/stat.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// 整个内核在一个命名空间里：`facerec::` 以上是库的边界，
// 也是 `api` 往上早就在用的那个名字（见 RFPLAN §6.1）。
namespace facerec {

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

Detector::Detector(const std::string& model_path, cv::Size input_shape,
                   float conf_thresh, float nms_thresh, int keep_top_k,
                   const DNNConfig& config, std::vector<cv::String> output_names) :
        model(cv::dnn::readNet(model_path)), input_shape(std::move(input_shape)),
        conf_thresh(conf_thresh), nms_thresh(nms_thresh), keep_top_k(keep_top_k) {
    if (output_names.size() != 3) {
        CV_Error(cv::Error::StsBadArg, "Detector: needs exactly 3 output names (loc, conf, iou)");
    }
    for (const auto& name : output_names) {
        if (name.empty()) {
            CV_Error(cv::Error::StsBadArg, "Detector: empty detector output name");
        }
    }
    head = make_detector_head(model.getLayerNames(), input_shape, output_names);
    model.setPreferableBackend(config.backend);
    model.setPreferableTarget(config.target);
}

std::vector<Face> Detector::detect(const cv::Mat& img) const {
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, input_shape, cv::Scalar());
    model.setInput(blob);
    // 输出收在局部向量里：不跨调用复用成员缓冲，detect() 里没有任何跨调用状态。
    // （模型本身仍是有状态的：setInput/forward 需要每个工作线程独占一条 pipeline，
    //   这就是 PipelinePool 存在的原因。）
    std::vector<cv::Mat> blobs;
    model.forward(blobs, head->outputNames());
    std::vector<Face> dets = head->decode(blobs, conf_thresh);
    if (!dets.empty()) {
        nms(dets, nms_thresh);
        if (dets.size() > static_cast<size_t>(keep_top_k)) {
            dets.erase(dets.begin() + keep_top_k, dets.end());
        }
    }
    return dets;
}

// ---------------------------------------------------------------------------
// 注册目录扫描
// ---------------------------------------------------------------------------

std::vector<std::string> list_person_dirs(const std::string& path) {
    std::vector<std::string> dirs;
    DIR* dirp = opendir(path.c_str());
    if (dirp == nullptr) {
        return dirs;
    }
    struct dirent* dp;
    while ((dp = readdir(dirp)) != nullptr) {
        std::string name = dp->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        std::string full_path = path + "/" + name;
        struct stat st;
        if (stat(full_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            dirs.push_back(name);
        }
    }
    closedir(dirp);
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

namespace {

// ArcFace 标准 5 点参考模板（112x112 帧），嵌入模型训练时使用的对齐约定。
// 槽位顺序：[0]=眼睛A, [1]=眼睛B, [2]=嘴A, [3]=鼻, [4]=嘴B
// 数值来源：insightface / ArcFace 默认 arcface_src。
const float ARC_TEMPLATE[5][2] = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {41.5493f, 92.3655f},
    {56.0252f, 71.7366f},
    {70.7299f, 92.5066f},
};

// 最小二乘拟合相似变换（旋转+均匀缩放+平移，4 自由度），把 src 映射到 dst。
// 闭式解：把 (a+bi) 当作复数缩放因子，a = Σdot/A，b = Σcross/A。
cv::Mat fit_similarity(const std::array<cv::Point2f, 5>& src,
                       const std::array<cv::Point2f, 5>& dst,
                       double* rms_out) {
    cv::Point2f sc(0.f, 0.f), dc(0.f, 0.f);
    for (int i = 0; i < 5; ++i) { sc += src[i]; dc += dst[i]; }
    sc *= 0.2f;
    dc *= 0.2f;

    double A = 0.0, C = 0.0, B = 0.0;
    for (int i = 0; i < 5; ++i) {
        double px = src[i].x - sc.x, py = src[i].y - sc.y;
        double qx = dst[i].x - dc.x, qy = dst[i].y - dc.y;
        A += px * px + py * py;
        C += px * qx + py * qy;
        B += px * qy - py * qx;
    }
    if (A < 1e-9) {
        return {};
    }

    const double a = C / A;
    const double b = B / A;
    const double tx = dc.x - (a * sc.x - b * sc.y);
    const double ty = dc.y - (b * sc.x + a * sc.y);

    if (rms_out) {
        double rss = 0.0;
        for (int i = 0; i < 5; ++i) {
            double x = a * src[i].x - b * src[i].y + tx;
            double y = b * src[i].x + a * src[i].y + ty;
            rss += (x - dst[i].x) * (x - dst[i].x) + (y - dst[i].y) * (y - dst[i].y);
        }
        *rms_out = std::sqrt(rss / 5.0);
    }

    return (cv::Mat_<double>(2, 3) << a, -b, tx, b, a, ty);
}

}  // namespace

// ---------------------------------------------------------------------------
// 对齐
// ---------------------------------------------------------------------------

cv::Mat align_face(const cv::Mat& face_roi,
                   const Landmarks_5& lm,
                   const cv::Size& output_size) {
    if (face_roi.empty() || output_size.width <= 0 || output_size.height <= 0) {
        return {};
    }

    const float fx = static_cast<float>(output_size.width) / 112.0f;
    const float fy = static_cast<float>(output_size.height) / 112.0f;
    std::array<cv::Point2f, 5> dst;
    for (int i = 0; i < 5; ++i) {
        dst[i] = cv::Point2f(ARC_TEMPLATE[i][0] * fx, ARC_TEMPLATE[i][1] * fy);
    }

    std::array<cv::Point2f, 5> base;
    base[0] = lm.right_eye;
    base[1] = lm.left_eye;
    base[2] = lm.mouth_left;
    base[3] = lm.nose_tip;
    base[4] = lm.mouth_right;

    // YuNet 的 left/right 指人物自身左右，模板槽位按图像方位定义，两者可能左右相反。
    // 枚举眼睛/嘴的两两交换，取拟合残差最小的配对。
    cv::Mat best;
    double best_rms = std::numeric_limits<double>::max();
    for (int swap_eyes = 0; swap_eyes < 2; ++swap_eyes) {
        for (int swap_mouth = 0; swap_mouth < 2; ++swap_mouth) {
            std::array<cv::Point2f, 5> src = base;
            if (swap_eyes) std::swap(src[0], src[1]);
            if (swap_mouth) std::swap(src[2], src[4]);

            double rms = 0.0;
            cv::Mat M = fit_similarity(src, dst, &rms);
            if (M.empty()) continue;
            if (rms < best_rms) {
                best_rms = rms;
                best = M;
            }
        }
    }
    if (best.empty()) {
        return {};
    }

    cv::Mat aligned;
    cv::warpAffine(face_roi, aligned, best, output_size);
    return aligned;
}

cv::Mat align_detection(const cv::Mat& img, const Face& det,
                        const cv::Size& detection_shape,
                        const cv::Size& output_size,
                        int border_mode) {
    if (img.empty() || detection_shape.width <= 0 || detection_shape.height <= 0) {
        return {};
    }

    const float sx = static_cast<float>(img.cols) / detection_shape.width;
    const float sy = static_cast<float>(img.rows) / detection_shape.height;

    Landmarks_5 lm;
    lm.right_eye    = cv::Point2f(det.landmarks.right_eye.x * sx,    det.landmarks.right_eye.y * sy);
    lm.left_eye     = cv::Point2f(det.landmarks.left_eye.x * sx,     det.landmarks.left_eye.y * sy);
    lm.mouth_left   = cv::Point2f(det.landmarks.mouth_left.x * sx,   det.landmarks.mouth_left.y * sy);
    lm.nose_tip     = cv::Point2f(det.landmarks.nose_tip.x * sx,     det.landmarks.nose_tip.y * sy);
    lm.mouth_right  = cv::Point2f(det.landmarks.mouth_right.x * sx,  det.landmarks.mouth_right.y * sy);

    const float eye_dist = cv::norm(lm.left_eye - lm.right_eye);
    if (!(eye_dist > 1e-3f)) {
        return {};
    }

    const float side = 3.2f * eye_dist;
    if (!(side < 1e6f)) {
        return {};
    }
    cv::Point2f center = (lm.right_eye + lm.left_eye + lm.mouth_left +
                          lm.nose_tip + lm.mouth_right) * 0.2f;

    const int x = cvRound(center.x - side * 0.5f);
    const int y = cvRound(center.y - side * 0.5f);
    const int w = cvRound(side);
    const int h = cvRound(side);

    const cv::Rect ideal(x, y, w, h);
    const cv::Rect image_rect(0, 0, img.cols, img.rows);
    const cv::Rect valid = ideal & image_rect;
    if (valid.area() <= 0) {
        return {};
    }

    cv::Mat face_roi;
    cv::Point2f win_tl;
    if (valid == ideal) {
        face_roi = img(valid);
        win_tl = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    } else if (static_cast<long long>(w) * h <= 4LL * static_cast<long long>(img.total())) {
        const int top    = valid.y - y;
        const int bottom = (y + h) - (valid.y + valid.height);
        const int left   = valid.x - x;
        const int right  = (x + w) - (valid.x + valid.width);
        cv::copyMakeBorder(img(valid), face_roi, top, bottom, left, right,
                           border_mode | cv::BORDER_ISOLATED);
        win_tl = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    } else {
        face_roi = img(valid);
        win_tl = cv::Point2f(static_cast<float>(valid.x), static_cast<float>(valid.y));
    }

    lm.right_eye   -= win_tl;
    lm.left_eye    -= win_tl;
    lm.mouth_left  -= win_tl;
    lm.nose_tip    -= win_tl;
    lm.mouth_right -= win_tl;

    return align_face(face_roi, lm, output_size);
}

// ---------------------------------------------------------------------------
// 特征提取
// ---------------------------------------------------------------------------

cv::Mat extract_face_feature(const cv::Mat& img, const Face& det,
                             FeatureExtractor& extractor,
                             const cv::Size& detection_shape) {
    cv::Mat aligned = align_detection(img, det, detection_shape, extractor.inputSize(),
                                      extractor.align_border());
    if (aligned.empty()) {
        return {};
    }
    return extractor.extract(aligned);
}

std::vector<cv::Mat> extract_face_features(const cv::Mat& img,
                                           const std::vector<Face>& dets,
                                           FeatureExtractor& extractor,
                                           const cv::Size& detection_shape) {
    std::vector<cv::Mat> features(dets.size());
    if (dets.empty()) {
        return features;
    }

    const cv::Size out_size = extractor.inputSize();
    std::vector<cv::Mat> aligned;
    std::vector<int> index_map;
    aligned.reserve(dets.size());
    index_map.reserve(dets.size());
    for (int d = 0; d < static_cast<int>(dets.size()); ++d) {
        cv::Mat a = align_detection(img, dets[d], detection_shape, out_size,
                                    extractor.align_border());
        if (!a.empty()) {
            aligned.push_back(std::move(a));
            index_map.push_back(d);
        }
    }
    if (aligned.empty()) {
        return features;
    }

    std::vector<cv::Mat> feats;
    if (extractor.batch_mode()) {
        feats = extractor.extract_batch(aligned);
    } else {
        feats.reserve(aligned.size());
        for (const auto& m : aligned) {
            feats.push_back(extractor.extract(m));
        }
    }

    for (size_t k = 0; k < feats.size() && k < index_map.size(); ++k) {
        features[index_map[k]] = std::move(feats[k]);
    }
    return features;
}

bool features_from_img(const cv::Mat& img, Detector& detector,
                       FeatureExtractor& extractor, std::vector<cv::Mat>& features,
                       const cv::Size& display_shape, int rfactor) {
    // 内部 clone，避免修改调用方传入的原始图像
    cv::Mat work = letterbox(img, display_shape);

    std::vector<Face> dets = detector.detect(work);
    if (dets.empty()) {
        return false;
    }

    features = extract_face_features(work, dets, extractor, detector.shape());
    return true;
}

// ---------------------------------------------------------------------------
// FeatureExtractor
// ---------------------------------------------------------------------------

FeatureExtractor::FeatureExtractor(const std::string& model_path,
                                   cv::Size input_shape,
                                   const DNNConfig& config) :
        model(cv::dnn::readNet(model_path)),
        input_shape(std::move(input_shape)) {
    model.setPreferableBackend(config.backend);
    model.setPreferableTarget(config.target);
    // 实测：CPU 后端 batch 无收益（OpenCV 逐样本跑图 + 内部 blob 重建），
    // 仅 GPU/NPU 等异步后端启用批量前向。
    batch_mode_ = (config.target != cv::dnn::DNN_TARGET_CPU);
}

cv::Mat FeatureExtractor::extract(const cv::Mat& img) const {
    cv::Mat blob_faceAligned, output;
    cv::dnn::blobFromImage(img, blob_faceAligned, 1. / 255.,
                           input_shape, cv::Scalar(), true, false);
    blob_faceAligned = (blob_faceAligned - 0.5) / 0.5;
    model.setInput(blob_faceAligned);
    model.forward().copyTo(output);
    return output;
}

std::vector<cv::Mat> FeatureExtractor::extract_batch(const std::vector<cv::Mat>& imgs) const {
    std::vector<cv::Mat> results;
    if (imgs.empty()) {
        return results;
    }

    cv::Mat blob;
    cv::dnn::blobFromImages(imgs, blob, 1. / 255.,
                            input_shape, cv::Scalar(), true, false);
    blob = (blob - 0.5) / 0.5;

    model.setInput(blob);
    cv::Mat output = model.forward();

    int num = static_cast<int>(imgs.size());
    int dims = output.dims;
    int per_sample = 1;
    for (int d = 1; d < dims; ++d) {
        per_sample *= output.size[d];
    }

    cv::Mat flat = output.reshape(1, num); // num x per_sample
    results.reserve(num);
    for (int i = 0; i < num; ++i) {
        results.push_back(flat.row(i).clone());
    }
    return results;
}

// ---------------------------------------------------------------------------
// 相似度
// ---------------------------------------------------------------------------

double NormSimilarity::similarity(const cv::Mat& feature, const cv::Mat& ref_feature) {
    cv::Mat t;
    cv::normalize(feature, t);
    return 2 - cv::norm(t - ref_feature);
}

bool NormSimilarity::feature_fusion(std::vector<cv::Mat>& features, cv::Mat& feature) {
    if (features.empty()) {
        return false;
    }
    feature = cv::Mat::zeros(features[0].size(), features[0].type());
    for (const auto& f : features) {
        feature += f;
    }
    feature /= static_cast<double>(features.size());
    return true;
}

double CosSimilarity::similarity(const cv::Mat& feature, const cv::Mat& ref_feature) {
    return feature.dot(ref_feature) / (cv::norm(feature));
}

bool CosSimilarity::feature_fusion(std::vector<cv::Mat>& features, cv::Mat& feature) {
    if (features.empty()) {
        return false;
    }
    feature = cv::Mat::zeros(features[0].size(), features[0].type());
    for (const auto& f : features) {
        cv::Mat normalized;
        cv::normalize(f, normalized);
        feature += normalized;
    }
    cv::normalize(feature, feature);
    return true;
}

}  // namespace facerec

