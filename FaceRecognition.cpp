#include "FaceRecognition.h"

#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

const double PI = 3.141592653589793;

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

// 对齐单张人脸：在 face_roi 内根据双眼位置做仿射变换，输出固定尺寸的人脸图。
// 参考模板：双眼水平，位于输出图像上方 40% 处，间距占宽度的 30%。
cv::Mat align_face(const cv::Mat& face_roi,
                   const cv::Point2f& left_eye,
                   const cv::Point2f& right_eye,
                   const cv::Size& output_size) {
    if (face_roi.empty()) {
        return {};
    }

    float dx = right_eye.x - left_eye.x;
    float dy = right_eye.y - left_eye.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist < 1e-3f) {
        return {};
    }

    // 参考眼睛位置（输出图像坐标系）
    cv::Point2f left_ref(output_size.width * 0.35f, output_size.height * 0.40f);
    cv::Point2f right_ref(output_size.width * 0.65f, output_size.height * 0.40f);
    float ref_dist = right_ref.x - left_ref.x;

    double scale = ref_dist / dist;
    double angle = std::atan2(dy, dx) * 180.0 / PI;

    cv::Point2f center = (left_eye + right_eye) * 0.5f;
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, scale);

    // 平移：将人脸双眼中点对齐到参考双眼中点
    cv::Point2f ref_center = (left_ref + right_ref) * 0.5f;
    rot.at<double>(0, 2) += ref_center.x - center.x;
    rot.at<double>(1, 2) += ref_center.y - center.y;

    cv::Mat aligned;
    cv::warpAffine(face_roi, aligned, rot, output_size);
    return aligned;
}

// 纯函数：根据原图 + 检测框，提取单人脸对齐后的特征，不修改输入图。
// det 的 bbox/landmarks 坐标系与 detection_shape 一致（默认 160x120），
// 内部自动映射到原图坐标系后再做裁剪与对齐。
cv::Mat extract_face_feature(const cv::Mat& img, const Face& det,
                             FeatureExtractor& extractor,
                             const cv::Size& detection_shape,
                             const cv::Size& output_size) {
    if (img.empty()) {
        return {};
    }

    // 将检测坐标系 (detection_shape) 映射到原图坐标系
    float sx = static_cast<float>(img.cols) / detection_shape.width;
    float sy = static_cast<float>(img.rows) / detection_shape.height;

    cv::Point2f tl(det.bbox.top_left.x * sx, det.bbox.top_left.y * sy);
    cv::Point2f br(det.bbox.bottom_right.x * sx, det.bbox.bottom_right.y * sy);
    cv::Rect face_area(cv::Point(cvRound(tl.x), cvRound(tl.y)),
                       cv::Point(cvRound(br.x), cvRound(br.y)));

    cv::Mat face_roi;
    try {
        face_roi = img(face_area);
    } catch (const cv::Exception&) {
        return {};
    }

    // 眼睛点映射到原图，再减去 ROI 左上角得到 ROI 内坐标
    cv::Point2f roi_tl(static_cast<float>(face_area.x), static_cast<float>(face_area.y));
    cv::Point2f left_eye(det.landmarks.left_eye.x * sx - roi_tl.x,
                         det.landmarks.left_eye.y * sy - roi_tl.y);
    cv::Point2f right_eye(det.landmarks.right_eye.x * sx - roi_tl.x,
                          det.landmarks.right_eye.y * sy - roi_tl.y);

    cv::Mat aligned = align_face(face_roi, left_eye, right_eye, output_size);
    if (aligned.empty()) {
        return {};
    }
    return extractor.extract(aligned);
}

bool features_from_img(cv::Mat& img, Detector& detector, FeatureExtractor& extractor,
                       std::vector<cv::Mat>& features, const cv::Size& display_shape,
                       int rfactor) {
    // 内部 clone，避免修改调用方传入的原始图像
    cv::Mat work = img.clone();
    cv::resize(work, work, display_shape);

    detector.detect(work);
    if (detector.size() == 0) {
        return false;
    }

    features.clear();
    for (const Face& det : detector.faces()) {
        // det.bbox 处于 input_shape (160x120) 坐标系，work = input_shape * rfactor
        cv::Point2f tl = det.bbox.top_left * rfactor;
        cv::Point2f br = det.bbox.bottom_right * rfactor;
        cv::Rect face_area(cv::Point(cvRound(tl.x), cvRound(tl.y)),
                           cv::Point(cvRound(br.x), cvRound(br.y)));

        cv::Mat face_roi;
        try {
            face_roi = work(face_area);
        } catch (const cv::Exception&) {
            features.emplace_back();
            continue;
        }

        // 眼睛点从 input_shape 坐标系映射到 work，再减去 ROI 左上角得到 ROI 内坐标
        cv::Point2f roi_tl(static_cast<float>(face_area.x), static_cast<float>(face_area.y));
        cv::Point2f left_eye = det.landmarks.left_eye * rfactor - roi_tl;
        cv::Point2f right_eye = det.landmarks.right_eye * rfactor - roi_tl;

        cv::Mat aligned = align_face(face_roi, left_eye, right_eye, cv::Size(112, 112));
        if (aligned.empty()) {
            features.emplace_back();
        } else {
            features.emplace_back(extractor.extract(aligned));
        }
    }

    return true;
}

FeatureExtractor::FeatureExtractor(const std::string& model_path,
                                   cv::Size input_shape,
                                   const DNNConfig& config) :
        model(cv::dnn::readNet(model_path)),
        input_shape(std::move(input_shape)) {
    model.setPreferableBackend(config.backend);
    model.setPreferableTarget(config.target);
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
