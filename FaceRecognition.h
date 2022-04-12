//
// Created by eipi10 on 1/30/21.
//

#ifndef FACERECOGNITION_FACERECOGNITION_H
#define FACERECOGNITION_FACERECOGNITION_H

#include <string>
#include <vector>
#include <dirent.h>
#include "priorbox.hpp"
#include "utils.hpp"

#include "opencv2/opencv.hpp"

void read_directory(const std::string& name, std::vector<std::string> &v);

class Detector {
public:
    Detector(const std::string& model_path, cv::Size input_shape,
             float conf_thresh=0.6, float nms_thresh=0.3, int keep_top_k=750,
             cv::dnn::Backend backend=cv::dnn::DNN_BACKEND_OPENCV,
             cv::dnn::Target target=cv::dnn::DNN_TARGET_CPU);
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
    size_t detected_nums;
};

void face_alignment(const cv::Mat& img, cv::Mat& faceImagAligned,
                    const cv::Point& left_eye, const cv::Point& right_eye,
                    const cv::Point& left_eye_ref, const cv::Point& right_eye_ref);

class FeatureExtractor {
public:
    FeatureExtractor(const std::string& model_path, cv::Size input_shape,
                     cv::dnn::Backend backend=cv::dnn::DNN_BACKEND_OPENCV,
                     cv::dnn::Target target=cv::dnn::DNN_TARGET_CPU);
    cv::Mat extract(const cv::Mat& img);
private:
    cv::dnn::Net model;
    cv::Size input_shape;
};

bool features_from_img(cv::Mat& img, Detector& detector, FeatureExtractor& extractor,
                       std::vector<cv::Mat>& features, const cv::Size& display_shape, int rfactor);

bool feauture_fusion(std::vector<cv::Mat> &features, cv::Mat &feature);

class Similarity {
public:
    virtual double similarity(const cv::Mat& feature, const cv::Mat& ref_feature)=0;
};

class NormSimilarity: public Similarity {
public:
    double similarity(const cv::Mat& feature, const cv::Mat& ref_feature) override;
};

class CosSimilarity: public Similarity {
public:
    double similarity(const cv::Mat& feature, const cv::Mat& ref_feature) override;
};

template<typename T>
class FeatureDatabase {
public:
    FeatureDatabase(std::string data_path, Detector& detector,
                    FeatureExtractor& extractor, const cv::Size& display_shape,
                    int rfactor, float threshold=0.8):
            path(std::move(data_path)), threshold(threshold), similar_calculator(new T()) {
        read_directory(path, names);
        std::vector<std::string> photos;
        cv::Mat k;
        std::vector<cv::Mat> features;
        for (auto i = 2; i < names.size(); ++i) {
            cv::glob(path + names[i], photos);
            std::vector<cv::Mat> temp;
            for (const auto& j: photos) {
                k = cv::imread(j);
                if (features_from_img(k, detector, extractor, features,
                                      display_shape, rfactor)) {
                    std::cout << "\t" << detector.size() << " faces loaded from: " << j << std::endl;
                } else {
                    std::cout << "\tNo faces loaded from: " << j << std::endl;
                    continue;
                }
                cv::Mat t;
                features[0].copyTo(t);
                temp.emplace_back(std::move(t));
            }
            cv::Mat fused_feature;
            if (!temp.empty()) {
                temp[0].copyTo(fused_feature);
                feauture_fusion(temp, fused_feature);
            }
            feature_database.emplace_back(std::move(fused_feature));
        }
        data_len = feature_database.size();
    }

    ~FeatureDatabase() { free(similar_calculator); }

    bool select_max_mean_similarity(const cv::Mat& feature, float& similarity,
                                    std::string& name) {
        double total;
        similarity = -1.;
        if (feature.empty()) { return false; }
        for (int i = 0; i < data_len; ++i) {
            const std::vector<cv::Mat> &ref_features = feature_database[i];
            if (ref_features.empty()) { continue; }
            total = 0.;
            for (const cv::Mat &ref_feature : ref_features) {
                total += similar_calculator->similarity(feature, ref_feature);
            }
            total /= double(ref_features.size());
            if (similarity < total) {
                similarity = total;
                name = names[i + 2];
            }
        }
        return similarity >= threshold;
    }

    bool select_max_similarity(const cv::Mat& feature, float& similarity,
                                    std::string& name) {
        double total;
        similarity = -1.;
        if (feature.empty()) { return false; }
        for (int i = 0; i < data_len; ++i) {
            const cv::Mat &ref_feature = feature_database[i];
            if (ref_feature.empty()) { continue; }
            total = similar_calculator->similarity(feature, ref_feature);
            if (similarity < total) {
                similarity = total;
                name = names[i + 2];
            }
        }
        return similarity >= threshold;
    }

    size_t size() const { return data_len; }
    friend void read_directory(const std::string& name, std::vector<std::string> &v);
private:
    std::string path;
    Similarity *similar_calculator;
    std::vector<cv::Mat> feature_database;
    std::vector<std::string> names;
    size_t data_len;
    float threshold;
};

class FaceRecognition {
public:
    FaceRecognition(float recogn_thresh=0.9);
private:
    cv::dnn::Net detector, feature_extractor;
    float conf_thresh, nms_thresh, recogn_thresh;
    int keep_top_k;
};

#endif //FACERECOGNITION_FACERECOGNITION_H
