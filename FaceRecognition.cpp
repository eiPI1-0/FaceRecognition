#include "FaceRecognition.h"

#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>



Detector::Detector(const std::string& model_path, cv::Size input_shape, float conf_thresh,
                   float nms_thresh, int keep_top_k,
                   const DNNConfig& config) :
        model(cv::dnn::readNet(model_path)), input_shape(std::move(input_shape)), conf_thresh(conf_thresh),
        nms_thresh(nms_thresh), keep_top_k(keep_top_k), pb(input_shape, input_shape) {
    model.setPreferableBackend(config.backend);
    model.setPreferableTarget(config.target);
}

std::vector<Face>& Detector::detect(cv::Mat& img) {
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, input_shape, cv::Scalar());
    model.setInput(blob);
    model.forward(output_blobs, output_names);
    dets = pb.decode(output_blobs[0], output_blobs[1], output_blobs[2], conf_thresh);
    if (!dets.empty()) {
        nms(dets, nms_thresh);
        if (dets.size() > keep_top_k) {
            dets.erase(dets.begin() + keep_top_k, dets.end());
        }
    }
    detected_nums = dets.size();
    return dets;
}

size_t Detector::size() const {
    return detected_nums;
}

std::vector<Face>& Detector::faces() {
    return dets;
}

cv::Size& Detector::shape() {
    return input_shape;
}

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

// 用 5 点相似变换对齐人脸。
cv::Mat align_face(const cv::Mat& face_roi,
                   const Landmarks_5& lm,
                   const cv::Size& output_size) {
    if (face_roi.empty() || output_size.width <= 0 || output_size.height <= 0) {
        return {};
    }

    // 模板按输出尺寸缩放（模板定义在 112x112 帧）
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

    // YuNet 的 left/right 指人物自身左右，模板槽位按图像方位定义，两者可能左右相反
    //（镜像照片、大角度侧脸时更明显）。这里枚举眼睛/嘴的两两交换，取拟合残差最小的
    // 配对，避免靠命名约定硬猜。
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

// 纯函数：根据原图 + 检测框，对齐单人脸到固定尺寸（默认 112x112），不提取特征。
// det 的 bbox/landmarks 坐标系与 detection_shape 一致（默认 160x120）。
cv::Mat align_detection(const cv::Mat& img, const Face& det,
                        const cv::Size& detection_shape,
                        const cv::Size& output_size) {
    if (img.empty()) {
        return {};
    }

    // 将关键点从检测坐标系 (detection_shape) 映射到原图坐标系
    const float sx = static_cast<float>(img.cols) / detection_shape.width;
    const float sy = static_cast<float>(img.rows) / detection_shape.height;

    Landmarks_5 lm;
    lm.right_eye    = cv::Point2f(det.landmarks.right_eye.x * sx,    det.landmarks.right_eye.y * sy);
    lm.left_eye     = cv::Point2f(det.landmarks.left_eye.x * sx,     det.landmarks.left_eye.y * sy);
    lm.mouth_left   = cv::Point2f(det.landmarks.mouth_left.x * sx,   det.landmarks.mouth_left.y * sy);
    lm.nose_tip     = cv::Point2f(det.landmarks.nose_tip.x * sx,     det.landmarks.nose_tip.y * sy);
    lm.mouth_right  = cv::Point2f(det.landmarks.mouth_right.x * sx,  det.landmarks.mouth_right.y * sy);

    const float eye_dist = cv::norm(lm.left_eye - lm.right_eye);
    if (eye_dist < 1e-3f) {
        return {};
    }

    // 取方形 ROI：112x112 模板窗口对应约 3.2 倍眼距（模板眼距 35.24px），
    // 紧致 bbox 会裁掉旋转/缩放后需要的像素。
    const float side = 3.2f * eye_dist;
    cv::Point2f center = (lm.right_eye + lm.left_eye + lm.mouth_left +
                          lm.nose_tip + lm.mouth_right) * 0.2f;

    cv::Rect face_area(cvRound(center.x - side * 0.5f), cvRound(center.y - side * 0.5f),
                       cvRound(side), cvRound(side));
    // 越界时收缩到图内（相似变换由关键点决定，收缩只影响可见像素，不改变对齐结果）
    face_area &= cv::Rect(0, 0, img.cols, img.rows);
    if (face_area.area() <= 0) {
        return {};
    }

    cv::Mat face_roi = img(face_area);

    // 关键点转成 ROI 内坐标
    const cv::Point2f roi_tl(static_cast<float>(face_area.x), static_cast<float>(face_area.y));
    lm.right_eye   -= roi_tl;
    lm.left_eye    -= roi_tl;
    lm.mouth_left  -= roi_tl;
    lm.nose_tip    -= roi_tl;
    lm.mouth_right -= roi_tl;

    return align_face(face_roi, lm, output_size);
}

// 纯函数：根据原图 + 检测框，提取单人脸对齐后的特征，不修改输入图。
cv::Mat extract_face_feature(const cv::Mat& img, const Face& det,
                             FeatureExtractor& extractor,
                             const cv::Size& detection_shape,
                             const cv::Size& output_size) {
    cv::Mat aligned = align_detection(img, det, detection_shape, output_size);
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

    // 先对齐全部人脸，只保留对齐成功的，并记住其原始下标
    const cv::Size out_size = extractor.inputSize();
    std::vector<cv::Mat> aligned;
    std::vector<int> index_map;
    aligned.reserve(dets.size());
    index_map.reserve(dets.size());
    for (int d = 0; d < static_cast<int>(dets.size()); ++d) {
        cv::Mat a = align_detection(img, dets[d], detection_shape, out_size);
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

    // 按原始下标写回，保证 features[i] 与 dets[i] 对应
    for (size_t k = 0; k < feats.size() && k < index_map.size(); ++k) {
        features[index_map[k]] = std::move(feats[k]);
    }
    return features;
}

bool features_from_img(cv::Mat& img, Detector& detector, FeatureExtractor& extractor,
                       std::vector<cv::Mat>& features, const cv::Size& display_shape,
                       int rfactor) {
    // 内部 clone，避免修改调用方传入的原始图像
    // letterbox：保持长宽比缩放到 display_shape 并补边。
    // 与 main.cpp 的实时预处理完全一致，注册/推理尺度才可比。
    cv::Mat work = letterbox(img, display_shape);

    detector.detect(work);
    if (detector.size() == 0) {
        return false;
    }

    // features[i] 与 detector.faces()[i] 一一对应
    features = extract_face_features(work, detector.faces(), extractor, detector.shape());
    return true;
}

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

cv::Mat FeatureExtractor::extract(const cv::Mat& img) {
    cv::Mat blob_faceAligned, output;
    cv::dnn::blobFromImage(img, blob_faceAligned, 1. / 255.,
                           input_shape, cv::Scalar(), true, false);
    blob_faceAligned = (blob_faceAligned - 0.5) / 0.5;
    model.setInput(blob_faceAligned);
    model.forward().copyTo(output);
    return output;
}

std::vector<cv::Mat> FeatureExtractor::extract_batch(const std::vector<cv::Mat>& imgs) {
    std::vector<cv::Mat> results;
    if (imgs.empty()) {
        return results;
    }

    // 所有输入均为对齐后的 112x112 人脸，尺寸一致，可构建 batch blob
    cv::Mat blob;
    cv::dnn::blobFromImages(imgs, blob, 1. / 255.,
                            input_shape, cv::Scalar(), true, false);
    blob = (blob - 0.5) / 0.5;

    model.setInput(blob);
    cv::Mat output = model.forward();

    // 将 batch 输出拆分为每张人脸一份特征向量
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
