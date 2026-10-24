#include <iostream>
#include <string>
#include <charconv>
#include "FaceRecognition.h"
#include "face_tracker.hpp"

using namespace cv;

namespace {

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " <source> [cpu|cuda|openvino] [batch|nobatch]\n"
              << "  <source>           camera index (e.g. 0) or an image/video path or a URL\n"
              << "  cpu|cuda|openvino  DNN backend (default: cpu)\n"
              << "  batch|nobatch      force batch embedding (default: auto by backend)\n";
}

}  // namespace

int main(int argc, char *argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return -1;
    }

    const std::string source = argv[1];   // 摄像头索引，或图片/视频/URL 路径

    DNNConfig dnn_config = DNNConfig::cpu();
    int batch_mode = 0;  // 0=auto(按后端), 1=强制 batch, -1=强制逐张
    bool show_help = false;

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
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return -1;
        }
    }
    if (show_help) {
        print_usage(argv[0]);
        return 0;
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

    // 跟踪器参数（实测于本机 CPU，960x720 帧，1 张人脸）：
    //   YuNet 检测 0.93 ms/帧，MobileFaceNet 提特征 4.47 ms/脸 —— 提取才是瓶颈，
    //   跳过检测每帧只省 ~0.75 ms，却让框最大误差从 5.9px 涨到 18px。
    // 故默认每帧检测：跟踪器用于维持 ID、平滑，以及在检测漏掉人脸（遮挡等）时
    // 用匀速预测补帧。硬件较弱时可调大 detect_every 换取性能，代价是框的跟随延迟。
    // min_hits=1 => 首次检测到即出框，避免新人脸要等 N 次确认才显示。
    FaceTracker tracker(0.45f, /*detect_every=*/1, /*max_lost=*/15, /*min_hits=*/1);

    // 纯数字（不限一位）→ 摄像头索引；其它一律交给 VideoCapture，
    // 它接受图片、视频、URL、GStreamer 流水线等。注意单个数字 "0".."9" 若按
    // 设备名传给 VideoCapture 会被 FFMPEG 后端拒绝，所以数字必须走索引分支。
    int camera_index = -1;
    if (std::from_chars(source.data(), source.data() + source.size(), camera_index).ec ==
        std::errc()) {
        camera_index = std::stoi(source);
    }

    cv::VideoCapture cap;
    if (camera_index >= 0) {
        cap.open(camera_index);
    } else {
        cap.open(source);
    }
    if (!cap.isOpened()) {
        std::cerr << "Cannot open source: " << source << std::endl;
        return -1;
    }
    // 文件/图片源：读完就退出，但保留窗口到按键，否则结果一闪而过。
    const bool live_camera = camera_index >= 0;

    cv::Mat im;
    cv::TickMeter cvtm;
    std::vector<cv::Mat> features;

    String title = "FaceRecognition@eipi10";

    while (true) {
        cap >> im;
        if (im.empty()) {
            break;
        }

        // 统一缩放到 display_shape：
        //  (1) draw()/putText() 按 rfactor 把 160x120 检测坐标映射到图像坐标，
        //      该假设要求图像恰好是 input_shape * rfactor；
        //  (2) 注册照片也是按 display_shape 预处理后再检测的，
        //      blobFromImage 不保持长宽比，两者尺度必须一致特征才可比。
        cv::resize(im, im, display_shape);

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
        const int key = cv::waitKey(live_camera ? 1 : 0);   // 文件源：停在最后一帧
        if ((key & 0xFF) == 27)
            break;
    }

    return 0;
}
