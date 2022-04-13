//
// Created by eipi10 on 1/30/21.
//

#include "FaceRecognition.h"

#include <opencv2/core.hpp>
#include <utility>

const double PI = 3.141592653589793;

void read_directory(const std::string &name, std::vector<std::string> &v) {
    DIR *dirp = opendir(name.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.emplace_back(dp->d_name);
    }
    closedir(dirp);
}

Detector::Detector(const std::string &model_path, cv::Size input_shape, float conf_thresh,
                   float nms_thresh, int keep_top_k,
                   cv::dnn::Backend backend, cv::dnn::Target target) :
        model(cv::dnn::readNet(model_path)), input_shape(std::move(input_shape)), conf_thresh(conf_thresh),
        nms_thresh(nms_thresh), keep_top_k(keep_top_k), pb(input_shape, input_shape) {
    model.setPreferableBackend(backend);
    model.setPreferableTarget(target);
}

std::vector<Face> &Detector::detect(cv::Mat &img) {
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

std::vector<Face> &Detector::faces() {
    return dets;
}

cv::Size &Detector::shape() {
    return input_shape;
}

void face_alignment(const cv::Mat &img, cv::Mat &faceImagAligned,
                    const cv::Point &left_eye, const cv::Point &right_eye,
                    const cv::Point &left_eye_ref, const cv::Point &right_eye_ref) {
    float dist_ref = right_eye_ref.x - left_eye_ref.x;
    float dx = right_eye.x - left_eye.x;
    float dy = right_eye.y - left_eye.y;
    float dist = sqrt(dx * dx + dy * dy);

    double scale = dist_ref / dist;

    double angle = atan2(dy, dx) * 180 / PI;

    cv::Point2f center = cv::Point2f(0.5 * (left_eye.x + right_eye.x),
                                     0.5 * (left_eye.y + right_eye.y));
// calculate rotation matrix
    cv::Mat rot = getRotationMatrix2D(center, angle, scale); // translation
    rot.at<double>(0, 2) += img.size().width * 0.5 - center.x;
    rot.at<double>(1, 2) += left_eye_ref.y - center.y;
// apply affine transform
    cv::Mat imgIn = img.clone();
    imgIn.convertTo(imgIn, CV_32FC3, 1. / 255.);
    warpAffine(imgIn, faceImagAligned, rot, img.size());
    faceImagAligned.convertTo(faceImagAligned, CV_8UC3, 255);
}

bool features_from_img(cv::Mat &img, Detector &detector, FeatureExtractor &extractor,
                       std::vector<cv::Mat> &features, const cv::Size &display_shape,
                       int rfactor) {
    detector.detect(img);
    cv::resize(img, img, display_shape);
    if (detector.size() == 0) { return false; }
    cv::Rect face_area;
    cv::Mat face, t;
    features.clear();
    for (Face &i: detector.faces()) {
        face_area = cv::Rect(i.bbox.top_left * rfactor, i.bbox.bottom_right * rfactor);
        try {
            face = cv::Mat(img, face_area);
        } catch (cv::Exception &e) {
            features.emplace_back(cv::Mat());
            continue;
        }
        face_alignment(face, face, i.landmarks.left_eye, i.landmarks.right_eye,
                       i.landmarks.left_eye, i.landmarks.right_eye);
        features.emplace_back(std::move(extractor.extract(face)));
    }

    return true;
}

FeatureExtractor::FeatureExtractor(const std::string &model_path,
                                   cv::Size input_shape,
                                   cv::dnn::Backend backend,
                                   cv::dnn::Target target) :
        model(cv::dnn::readNet(model_path)),
        input_shape(std::move(input_shape)) {
    model.setPreferableBackend(backend);
    model.setPreferableTarget(target);
}

cv::Mat FeatureExtractor::extract(const cv::Mat &img) {
    cv::Mat blob_faceAligned, output;
    cv::dnn::blobFromImage(img, blob_faceAligned, 1. / 255.,
                           input_shape, cv::Scalar(), true, false);
    model.setInput(blob_faceAligned);
    model.forward().copyTo(output);
    return output;
}

double NormSimilarity::similarity(const cv::Mat &feature, const cv::Mat &ref_feature) {
    cv::Mat t;
    cv::normalize(feature, t);
    return 2-cv::norm(t - ref_feature);
}

bool NormSimilarity::feature_fusion(std::vector<cv::Mat> &features, cv::Mat &feature) {
    if (features.empty()) { return false; }
    features[0].copyTo(feature);
    for (auto& i: features) {
        feature = (feature + i) / 2.0;
    }
    return true;
}

double CosSimilarity::similarity(const cv::Mat &feature, const cv::Mat &ref_feature) {
    return feature.dot(ref_feature) / (cv::norm(feature));
}

bool CosSimilarity::feature_fusion(std::vector<cv::Mat> &features, cv::Mat &feature) {
    if (features.empty()) { return false; }
    features[0].copyTo(feature);
    feature.setTo(cv::Scalar::all(0.0));
    for (auto& i: features) {
        cv::Mat t;
        cv::normalize(i, t);
        cv::normalize(feature+t, feature);
    }
    return true;
}
