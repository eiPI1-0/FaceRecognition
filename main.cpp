#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <charconv>
#include "FaceRecognition.h"
#include "face_tracker.hpp"
#include "config.hpp"

using namespace cv;

namespace {

// 跨帧的识别状态。标签挂在**轨迹 ID** 上而不是框上：框每帧都在动、还可能被外推，
// ID 才是"同一个人"的载体。
struct Recognizer {
    std::map<int, std::pair<std::string, long>> label_of;   // 轨迹 ID -> (名字, 出证据的帧号)
    std::vector<cv::Mat> features;                          // 复用，省掉每帧重新分配
};

struct FrameOut {
    std::vector<Face> faces;             // 与 tracks 等长（含纯预测轨迹），画框用
    std::vector<char> measured;          // 1 = 本帧有检测支撑，0 = 外推（draw 会画成细线/另色）
    std::vector<std::string> labels;     // 与 faces 等长
};

// 识别策略：只有"本帧有检测支撑"的轨迹才去对齐+提特征；纯预测的沿用缓存里上一次
// 的真证据（超过 max_lost 帧就作废）。
//
// 为什么不拿外推的几何去识别：预测的关键点 = 上一帧实测值 + 速度×帧数，它喂进嵌入
// 模型会产出一个"看起来完全正常"的垃圾特征 —— 没有任何异常信号，表现就是幽灵框上
// 冒出随机名字。顺带省下预测帧每脸 4.47 ms。
// 反过来，"这次不像任何人"也是真证据：必须把旧标签清掉（写空名），否则人脸走开了
// 名字还赖在框上。
FrameOut recognize_frame(Recognizer& rec,
                         const std::vector<TrackView>& tracks,
                         long frame_no,
                         int max_lost,
                         FeatureDatabase<CosSimilarity>& database,
                         const cv::Mat& im,
                         FeatureExtractor& extractor,
                         const cv::Size& det_shape) {
    FrameOut out;
    out.faces.reserve(tracks.size());
    out.measured.reserve(tracks.size());
    out.labels.reserve(tracks.size());

    std::vector<Face> to_embed;              // 只含有检测支撑的轨迹
    std::vector<size_t> embed_of;            // to_embed[k] -> tracks 下标
    for (const auto& t : tracks) {
        out.faces.push_back(t.face);
        out.measured.push_back(t.measured ? 1 : 0);
        if (t.measured) {
            to_embed.push_back(t.face);
            embed_of.push_back(size_t(&t - tracks.data()));
        }
    }

    // features[k] 与 to_embed[k] 一一对应；对齐失败的人脸对应空 Mat
    rec.features = extract_face_features(im, to_embed, extractor, det_shape);
    for (size_t k = 0; k < embed_of.size(); ++k) {
        std::string name;
        if (!rec.features[k].empty()) {
            float sim = -1.0f;
            std::string hit;
            if (database.select_max_similarity(rec.features[k], sim, hit)) {
                name = hit;
            }
        }
        rec.label_of[tracks[embed_of[k]].id] = {name, frame_no};
    }

    for (auto it = rec.label_of.begin(); it != rec.label_of.end();) {
        if (frame_no - it->second.second > max_lost) {
            it = rec.label_of.erase(it);
        } else {
            ++it;
        }
    }

    for (const auto& t : tracks) {
        const auto it = rec.label_of.find(t.id);
        out.labels.push_back((it != rec.label_of.end() && !it->second.first.empty())
                                 ? it->second.first
                                 : std::string("Unknown"));
    }
    return out;
}

// 后端名 -> 后端配置。合法性已由 validate_config 保证，这里只做映射。
DNNConfig make_dnn_config(const std::string& backend) {
    if (backend == "cuda") return DNNConfig::cuda();
    if (backend == "openvino") return DNNConfig::openvino();
    return DNNConfig::cpu();
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " <source> [cpu|cuda|openvino] [batch|nobatch] [--config <path>]\n"
              << "                   [--set key=value] [--dump-config]\n"
              << "  <source>           camera index (e.g. 0) or an image/video path or a URL\n"
              << "  cpu|cuda|openvino  DNN backend (default: cpu)\n"
              << "  batch|nobatch      force batch embedding (default: auto by backend)\n"
              << "  --config <path>    load a config file (flat `key: value` lines)\n"
              << "  --set key=value    override one knob; repeatable\n"
              << "  --dump-config      print the effective config and exit\n"
              << "Precedence: built-in defaults < --config < CLI options.\n"
              << "Run `--dump-config` for the full list of keys and what they do.\n";
}

// 建流水线 + 跑循环。异常（模型读盘失败、后端不可用）由 main 的 catch 统一兜住。
int run_pipeline(const PipelineConfig& cfg, const std::string& source) {
    const DNNConfig dnn_config = make_dnn_config(cfg.backend);
    const Size img_shape = cfg.detector_input();
    const Size display_shape = cfg.display_shape();
    const int rfactor = cfg.rfactor;

    Detector detector(cfg.detector_model, img_shape, cfg.conf_thresh, cfg.nms_thresh,
                      cfg.keep_top_k, dnn_config,
                      {cfg.out_loc, cfg.out_conf, cfg.out_iou});
    FeatureExtractor extractor(cfg.embedder_model, cfg.embedder_input(), dnn_config);
    if (cfg.batch_mode != 0) {
        extractor.set_batch_mode(cfg.batch_mode > 0);
    }

    // align.border -> OpenCV 边界填充类型。挂在 extractor 上（不是全局变量），并且
    // 必须在构造 FeatureDatabase 之前设：注册就发生在那个构造函数里，注册与推理
    // 用了不同的填充，gallery 与查询特征就不可比（同一预处理是硬约束）。
    int border_mode = cv::BORDER_CONSTANT;
    if (cfg.align_border == "reflect") border_mode = cv::BORDER_REFLECT;
    else if (cfg.align_border == "replicate") border_mode = cv::BORDER_REPLICATE;
    extractor.set_align_border(border_mode);

    FeatureDatabase<CosSimilarity> database(cfg.data_dir, detector, extractor,
                                            display_shape, rfactor, cfg.recogn_thresh);
    std::cout << "Loaded " << database.size() << " person(s) from " << cfg.data_dir << std::endl;

    // 跟踪器默认值来自实测（本机 CPU，960x720，1 张人脸）：YuNet 检测 0.93 ms/帧，
    // MobileFaceNet 提特征 4.47 ms/脸 —— 提取才是瓶颈，跳过检测每帧只省 ~0.75 ms，
    // 却让框最大误差从 5.9px 涨到 18px，故默认每帧检测。关联与生命周期参数见 knob 注释。
    TrackerParams tparams;
    tparams.iou_threshold = cfg.tracker_iou;
    tparams.dist_ratio    = cfg.tracker_dist;
    tparams.detect_every  = cfg.detect_every;
    tparams.max_lost      = cfg.max_lost;
    tparams.min_hits      = cfg.min_hits;
    FaceTracker tracker(tparams);

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
    Recognizer recognizer;
    long frame_no = 0;

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

        // 跟踪器决定本帧是否跑全图 DNN 检测；输出是带稳定 ID 的轨迹视图
        ++frame_no;
        tracker.update(im, detector);
        const std::vector<TrackView>& tracks = tracker.tracks();

        const FrameOut fr = recognize_frame(recognizer, tracker.tracks(), frame_no,
                                            tracker.params().max_lost, database,
                                            im, extractor, detector.shape());
        for (size_t i = 0; i < fr.faces.size(); ++i) {
            cv::putText(im, fr.labels[i], fr.faces[i].bbox.top_left * rfactor,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 1);
        }

        cvtm.stop();

        std::string timeLabel = cv::format("Inference time: %.2f ms", cvtm.getTimeMilli());
        cv::putText(im, timeLabel, cv::Point(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        draw(im, fr.faces, rfactor, &fr.measured);

        cvtm.reset();
        cv::imshow(title, im);
        const int key = cv::waitKey(live_camera ? 1 : 0);   // 文件源：停在最后一帧
        if ((key & 0xFF) == 27)
            break;
    }

    return 0;
}

}  // namespace

int main(int argc, char *argv[])
{
    // 全部可调项的唯一来源是 PipelineConfig；key 名单见 config.cpp 的 knob 表。
    // 优先级：内置默认 < 配置文件 < 命令行。故命令行覆盖项先记下来，加载文件之后再应用。
    PipelineConfig cfg;
    std::string source;        // 摄像头索引，或图片/视频/URL 路径
    std::string config_path;
    // CLI 覆盖按出现顺序记录，全部在配置文件之后应用 => 优先级
    // 内置默认 < 配置文件 < 命令行（后出现的覆盖先出现的）。
    // backend/batch 这类便捷 token 也走同一条通道，不另设专用变量。
    std::vector<std::pair<std::string, std::string>> overrides;
    bool show_help = false;
    bool dump_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "cpu" || arg == "cuda" || arg == "openvino") {
            overrides.emplace_back("dnn.backend", arg);
        } else if (arg == "batch") {
            overrides.emplace_back("dnn.batch_mode", "1");
        } else if (arg == "nobatch") {
            overrides.emplace_back("dnn.batch_mode", "-1");
        } else if (arg == "--set" || arg.rfind("--set=", 0) == 0) {
            std::string assignment = (arg == "--set") ? ((i + 1 < argc) ? argv[++i] : std::string())
                                                      : arg.substr(std::string("--set=").size());
            const size_t eq = assignment.find('=');
            if (eq == std::string::npos) {
                std::cerr << "--set expects key=value, got '" << assignment << "'" << std::endl;
                return -1;
            }
            overrides.emplace_back(assignment.substr(0, eq), assignment.substr(eq + 1));
        } else if (arg == "--config") {
            if (i + 1 >= argc) {
                std::cerr << "--config expects a path" << std::endl;
                return -1;
            }
            config_path = argv[++i];
        } else if (arg.rfind("--config=", 0) == 0) {
            config_path = arg.substr(std::string("--config=").size());
        } else if (arg == "--dump-config") {
            dump_only = true;
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return -1;
        } else if (!source.empty()) {
            std::cerr << "More than one source: " << source << " and " << arg << std::endl;
            return -1;
        } else {
            source = arg;
        }
    }

    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }

    std::string cfg_err;
    std::vector<std::string> cfg_warnings;
    if (!config_path.empty() && !load_config(config_path, cfg, cfg_err, cfg_warnings)) {
        std::cerr << "Invalid config: " << cfg_err << std::endl;
        return -1;
    }
    for (const auto& override : overrides) {
        const Knob* knob = nullptr;
        std::vector<std::string> candidates;
        if (!find_knob(override.first, &knob, &candidates)) {
            std::cerr << "Unknown knob: " << override.first;
            if (!candidates.empty()) {
                std::cerr << "; did you mean";
                for (const auto& c : candidates) std::cerr << " " << c;
            } else {
                std::cerr << "; valid keys:";
                for (const auto& k : knobs()) std::cerr << " " << k.key;
            }
            std::cerr << std::endl;
            return -1;
        }
        if (!set_knob(*knob, override.second, cfg, cfg_err)) {
            std::cerr << "Invalid config: " << cfg_err << std::endl;
            return -1;
        }
    }

    // 所有输入都应用完，才解析资产路径：三条相对路径统一挂到 paths.root 上。
    // root 未指定时就是 cwd，所以只调阈值的配置文件不会挪动资产位置（那类意外比
    // "必须从 build/ 运行"更难查）。要锚定就在配置文件里显式写 `paths.root: .`。
    resolve_asset_paths(cfg, ".");

    if (!validate_config(cfg, cfg_err, cfg_warnings)) {
        std::cerr << "Invalid config: " << cfg_err << std::endl;
        return -1;
    }
    for (const auto& warning : cfg_warnings) {
        std::cerr << "config warning: " << warning << std::endl;
    }

    if (dump_only) {
        dump_config(std::cout, cfg);
        return 0;
    }
    if (source.empty()) {
        std::cerr << "No source given." << std::endl;
        print_usage(argv[0]);
        return -1;
    }

    // 模型文件在场且真是权重，才值得去建流水线：否则报错来自 OpenCV 的 importer，
    // 用户看不出是哪个 knob 指错了。
    std::string asset_err;
    if (!check_model_files(cfg, asset_err)) {
        std::cerr << "Invalid config: " << asset_err << std::endl;
        return -1;
    }

    try {
        return run_pipeline(cfg, source);
    } catch (const cv::Exception& e) {
        // 模型文件缺失、后端没编进 OpenCV（本机既无 cuda 也无 openvino 插件）都会抛
        // cv::Exception。配置项让这些变成用户可犯的错，所以要报清原因而非 core dump。
        std::cerr << "Fatal: " << e.what() << std::endl;
        return -1;
    }
}

