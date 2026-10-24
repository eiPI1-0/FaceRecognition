#include <iostream>
#include <string>
#include <charconv>
#include "FaceRecognition.h"
#include "face_tracker.hpp"
#include "config.hpp"

using namespace cv;

namespace {

// 后端名 -> 后端配置。合法性已由 validate_config 保证，这里只做映射。
DNNConfig make_dnn_config(const std::string& backend) {
    if (backend == "cuda") return DNNConfig::cuda();
    if (backend == "openvino") return DNNConfig::openvino();
    return DNNConfig::cpu();
}

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

    // 全部可调项的唯一来源是 PipelineConfig；key 名单见 config.cpp 的 knob 表。
    PipelineConfig cfg;
    bool show_help = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "cpu") {
            cfg.backend = "cpu";
        } else if (arg == "cuda") {
            cfg.backend = "cuda";
        } else if (arg == "openvino") {
            cfg.backend = "openvino";
        } else if (arg == "batch") {
            cfg.batch_mode = 1;
        } else if (arg == "nobatch") {
            cfg.batch_mode = -1;
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

    std::string cfg_err;
    std::vector<std::string> cfg_warnings;
    if (!validate_config(cfg, cfg_err, cfg_warnings)) {
        std::cerr << "Invalid config: " << cfg_err << std::endl;
        return -1;
    }
    for (const auto& warning : cfg_warnings) {
        std::cerr << "config warning: " << warning << std::endl;
    }

    const DNNConfig dnn_config = make_dnn_config(cfg.backend);
    const Size img_shape = cfg.detector_input();
    const Size display_shape = cfg.display_shape();
    const int rfactor = cfg.rfactor;

    Detector detector(cfg.detector_model, img_shape, cfg.conf_thresh, cfg.nms_thresh,
                      cfg.keep_top_k, dnn_config);
    FeatureExtractor extractor(cfg.embedder_model, cfg.embedder_input(), dnn_config);
    if (cfg.batch_mode != 0) {
        extractor.set_batch_mode(cfg.batch_mode > 0);
    }

    FeatureDatabase<CosSimilarity> database(cfg.data_dir, detector, extractor,
                                            display_shape, rfactor, cfg.recogn_thresh);
    std::cout << "Loaded " << database.size() << " person(s) from " << cfg.data_dir << std::endl;

    // 跟踪器默认值来自实测（本机 CPU，960x720，1 张人脸）：YuNet 检测 0.93 ms/帧，
    // MobileFaceNet 提特征 4.47 ms/脸 —— 提取才是瓶颈，跳过检测每帧只省 ~0.75 ms，
    // 却让框最大误差从 5.9px 涨到 18px，故默认每帧检测。详见 knob 表里的注释。
    FaceTracker tracker(cfg.tracker_iou, cfg.detect_every, cfg.max_lost, cfg.min_hits);

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

        // 统一 letterbox 到 display_shape（保持长宽比 + 居中补黑边）：
        //  (1) draw()/putText() 按 rfactor 把 160x120 检测坐标映射到图像坐标，
        //      该假设要求图像恰好是 input_shape * rfactor —— 补边后正好满足，
        //      且补边是居中的，坐标无需额外偏移；
        //  (2) 注册照片走同样的预处理，两者尺度一致特征才可比；
        //  (3) 直接拉伸会把竖构图人脸横向拉扁，检测分数明显下降。
        im = letterbox(im, display_shape);

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
