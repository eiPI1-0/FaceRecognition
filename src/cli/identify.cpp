#include "cli/cli.hpp"
#include "cli/args.hpp"

#include "app/api.hpp"
#include "app/config.hpp"
#include "core/engine.hpp"
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace cli;   // 实现文件内引入，不污染任何头文件

void identify_usage() {
    std::cout
        << "Usage: facerec identify --image <path> [options]\n"
        << "\nOptions:\n"
        << "  --image <path>            输入图片（必填）\n"
        << "  --top-k <N>               返回前 N 名候选（默认 5）\n"
        << "  --min-score <F>           候选/匹配的最低分（默认 0，用配置阈值决定 matched）\n"
        << "  --include <what>          landmarks | crops | embedding，可重复\n"
        << "  --detector-input <WxH>    请求级检测输入尺寸，如 640x480\n"
        << "  --config <path>           配置文件\n"
        << "  --set key=value           覆盖单个配置项，可重复\n"
        << "  --dump-config             打印生效配置并退出\n"
        << "Output: JSON to stdout; logs to stderr. Exit 0 even when no face is found.\n";
}

int cmd_identify(int argc, char* argv[]) {
    facerec::IdentifyOptions opt;
    ConfigArgs cfg_args;
    std::string image_path;
    bool dump_only = false;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--image" || arg.rfind("--image=", 0) == 0) {
            image_path = (arg == "--image") ? ((i + 1 < argc) ? argv[++i] : std::string())
                                            : arg.substr(8);
        } else if (arg == "--top-k" || arg.rfind("--top-k=", 0) == 0) {
            const std::string v = (arg == "--top-k") ? ((i + 1 < argc) ? argv[++i] : "")
                                                     : arg.substr(8);
            try { opt.top_k = std::stoi(v); } catch (...) {
                std::cerr << "--top-k expects an integer, got '" << v << "'" << std::endl;
                return -1;
            }
        } else if (arg == "--min-score" || arg.rfind("--min-score=", 0) == 0) {
            const std::string v = (arg == "--min-score") ? ((i + 1 < argc) ? argv[++i] : "")
                                                         : arg.substr(12);
            try { opt.min_score = std::stof(v); } catch (...) {
                std::cerr << "--min-score expects a number, got '" << v << "'" << std::endl;
                return -1;
            }
        } else if (arg == "--include" || arg.rfind("--include=", 0) == 0) {
            const std::string v = (arg == "--include") ? ((i + 1 < argc) ? argv[++i] : "")
                                                       : arg.substr(10);
            if (v == "landmarks") opt.include_landmarks = true;
            else if (v == "crops") opt.include_crops = true;
            else if (v == "embedding") opt.include_embedding = true;
            else {
                std::cerr << "--include expects landmarks|crops|embedding, got '" << v << "'"
                          << std::endl;
                return -1;
            }
        } else if (arg == "--detector-input" || arg.rfind("--detector-input=", 0) == 0) {
            const std::string v = (arg == "--detector-input")
                                      ? ((i + 1 < argc) ? argv[++i] : "")
                                      : arg.substr(std::string("--detector-input=").size());
            const size_t x = v.find('x');
            if (x == std::string::npos) {
                std::cerr << "--detector-input expects WxH, got '" << v << "'" << std::endl;
                return -1;
            }
            try {
                opt.detector_w = std::stoi(v.substr(0, x));
                opt.detector_h = std::stoi(v.substr(x + 1));
            } catch (...) {
                std::cerr << "--detector-input expects WxH, got '" << v << "'" << std::endl;
                return -1;
            }
        } else if (arg == "--json") {
            // JSON 是唯一输出格式，保留这个参数是为了让调用方显式表达意图
        } else if (arg == "--config" || arg.rfind("--config=", 0) == 0) {
            if (!parse_config_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "--set" || arg.rfind("--set=", 0) == 0) {
            if (!parse_set_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "--dump-config") {
            dump_only = true;
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg.rfind("-", 0) != 0 && image_path.empty()) {
            image_path = arg;   // 位置参数兼容：facerec identify photo.jpg
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }

    if (show_help) { identify_usage(); return 0; }

    PipelineConfig cfg;
    if (!load_pipeline_config(cfg_args, cfg)) return -1;

    if (dump_only) {
        dump_config(std::cout, cfg);
        return 0;
    }
    {
        std::string size_err;
        if (!facerec::valid_detector_size(opt.detector_w, opt.detector_h, &size_err)) {
            std::cerr << "--detector-input: " << size_err << std::endl;
            return -1;
        }
    }
    if (image_path.empty()) {
        std::cerr << "identify requires --image <path>" << std::endl;
        identify_usage();
        return -1;
    }
    if (!check_assets(cfg)) return -1;

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << facerec::error_to_json("cannot read image: " + image_path, "bad_image")
                  << std::endl;
        return 2;
    }

    try {
        facerec::Pipeline pipeline(cfg);
        facerec::IdentifyResult result = pipeline.identify(img, opt, image_path);
        std::cout << facerec::identify_to_json(result, opt) << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << facerec::error_to_json(e.what(), "pipeline_error") << std::endl;
        return 3;
    }
    return 0;
}
