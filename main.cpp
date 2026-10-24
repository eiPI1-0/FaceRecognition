#include <iostream>
#include "FaceRecognition.h"
#include "face_tracker.hpp"

using namespace cv;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./main <camera_id>" << std::endl;
        return -1;
    }

    float recogn_thresh = 0.2f;
    const int rfactor = 6;
    Size img_shape = {160, 120};
    Size display_shape = img_shape * rfactor;
    Size features_model_shape = {112, 112};
    String detection_model = "./models/yunet_120x160.onnx";
    String recognize_model = "./models/mobilefacenet.onnx";
    String data_path = "./data/";

    Detector detector(detection_model, img_shape);
    FeatureExtractor extractor(recognize_model, features_model_shape);
    FeatureDatabase<CosSimilarity> database(data_path, detector, extractor,
                                            display_shape, rfactor, recogn_thresh);

    // 人脸跟踪器：每 5 帧全图检测一次，中间帧用 IoU 关联 + 匀速预测
    FaceTracker tracker(0.45f, 5, 15, 3);

    cv::VideoCapture cap;
    cv::Mat im;
    cv::TickMeter cvtm;
    std::vector<cv::Mat> features;

    String title = "FaceRecognition@eipi10";

    if (isdigit(argv[1][0])) {
        cap.open(argv[1][0] - '0');
        if (!cap.isOpened()) {
            std::cerr << "Cannot open the camera." << std::endl;
            return 0;
        }
    }

    if (cap.isOpened()) {
        while (true) {
            cap >> im;
            if (im.empty()) {
                break;
            }

            cvtm.start();

            // 跟踪器决定本帧是否跑全图 DNN 检测，返回已关联 ID 的人脸
            tracker.update(im, detector);
            const std::vector<Face>& dets = tracker.faces();

            // 逐人脸提取特征（纯函数，不修改原图）
            features.clear();
            features.reserve(dets.size());
            for (const auto& det : dets) {
                cv::Mat feat = extract_face_feature(im, det, extractor, detector.shape());
                features.push_back(std::move(feat));
            }

            int i = 0;
            float max_similar;
            std::string max_name;
            for (const auto& feature : features) {
                if (!feature.empty() &&
                    database.select_max_similarity(feature, max_similar, max_name)) {
                } else {
                    max_name = std::string("Unknown");
                }
                if (i < static_cast<int>(dets.size())) {
                    cv::putText(im, max_name, dets[i].bbox.top_left * rfactor,
                                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(0, 255, 0), 1);
                }
                ++i;
            }

            cvtm.stop();

            std::string timeLabel = cv::format("Inference time: %.2f ms", cvtm.getTimeMilli());
            cv::putText(im, timeLabel, cv::Point(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            draw(im, dets, rfactor);

            cvtm.reset();
            cv::imshow(title, im);
            if ((cv::waitKey(1) & 0xFF) == 27)
                break;
        }
    }

    return 0;
}
