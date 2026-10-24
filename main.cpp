#include <iostream>
#include <string>
#include "FaceRecognition.h"
#include "face_tracker.hpp"

using namespace cv;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./main <camera_id> [cpu|cuda|openvino] [batch|nobatch]" << std::endl;
        return -1;
    }

    DNNConfig dnn_config = DNNConfig::cpu();
    int batch_mode = 0;  // 0=auto(按后端), 1=强制 batch, -1=强制逐张

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "cpu") {
            dnn_config = DNNConfig::cpu();
        } else if (arg == "cuda") {
            dnn_config = DNNConfig::cuda();
        } else if (arg == "openvino") {
            dnn_config = DNNConfig::openvino();
        } else if (arg == "batch") {
            batch_mode = 1;
        } else if (arg == "nobatch") {
            batch_mode = -1;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }

    float recogn_thresh = 0.2f;
    const int rfactor = 6;
    Size img_shape = {160, 120};
    Size display_shape = img_shape * rfactor;
    Size features_model_shape = {112, 112};
    String detection_model = "./models/yunet_120x160.onnx";
    String recognize_model = "./models/mobilefacenet.onnx";
    String data_path = "./data/";

    Detector detector(detection_model, img_shape, 0.6f, 0.3f, 750, dnn_config);
    FeatureExtractor extractor(recognize_model, features_model_shape, dnn_config);
    if (batch_mode != 0) {
        extractor.set_batch_mode(batch_mode > 0);
    }

    FeatureDatabase<CosSimilarity> database(data_path, detector, extractor,
                                            display_shape, rfactor, recogn_thresh);
    std::cout << "Loaded " << database.size() << " person(s) from " << data_path << std::endl;

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

            // 对齐 + 提特征（纯函数，不修改原图）；features[i] 与 dets[i] 一一对应
            features = extract_face_features(im, dets, extractor, detector.shape());

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
