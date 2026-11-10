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

void compare_usage() {
    std::cout
        << "Usage: facerec compare <img-a> <img-b> [options]\n"
        << "\nOptions:\n"
        << "  --config <path>           配置文件\n"
        << "  --set key=value           覆盖单个配置项，可重复\n"
        << "  --dump-config             打印生效配置并退出\n"
        << "Output: {\"status\":\"ok\",\"cosine\":0.95,...} to stdout.\n"
        << "Exit 0 on success, 2 when a face is missing in either image.\n"
        << "Each image contributes its most confident face (identify's faces[0]), not its\n"
        << "largest one; cosine is 1.0 for the same face, around 0 for unrelated people.\n";
}

int cmd_compare(int argc, char* argv[]) {
    std::string img_a, img_b;
    ConfigArgs cfg_args;
    bool dump_only = false;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" || arg.rfind("--config=", 0) == 0) {
            if (!parse_config_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "--set" || arg.rfind("--set=", 0) == 0) {
            if (!parse_set_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "--dump-config") {
            dump_only = true;
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg.rfind("-", 0) != 0) {
            if (img_a.empty()) img_a = arg;
            else if (img_b.empty()) img_b = arg;
            else {
                std::cerr << "compare: unexpected extra argument: " << arg << std::endl;
                return -1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }

    if (show_help) { compare_usage(); return 0; }

    PipelineConfig cfg;
    if (!load_pipeline_config(cfg_args, cfg)) return -1;

    if (dump_only) {
        dump_config(std::cout, cfg);
        return 0;
    }

    if (img_a.empty() || img_b.empty()) {
        std::cerr << "compare requires two image paths" << std::endl;
        compare_usage();
        return -1;
    }

    if (!check_assets(cfg)) return -1;

    try {
        facerec::Pipeline pipeline(cfg);
        cv::Mat a = cv::imread(img_a, cv::IMREAD_COLOR);
        cv::Mat b = cv::imread(img_b, cv::IMREAD_COLOR);
        if (a.empty()) {
            std::cout << facerec::error_to_json("cannot read image: " + img_a, "bad_image")
                      << std::endl;
            return 2;
        }
        if (b.empty()) {
            std::cout << facerec::error_to_json("cannot read image: " + img_b, "bad_image")
                      << std::endl;
            return 2;
        }

        cv::Mat fa = pipeline.embed_image(a, 0);
        cv::Mat fb = pipeline.embed_image(b, 0);
        if (fa.empty() || fb.empty()) {
            std::cout << facerec::error_to_json(
                "no face detected in one of the images", "no_face") << std::endl;
            return 2;
        }

        const double dot = fa.dot(fb);
        const double na = cv::norm(fa);
        const double nb = cv::norm(fb);
        const double cos = (na > 0 && nb > 0) ? dot / (na * nb) : 0.0;

        std::cout << "{\"status\":\"ok\",\"cosine\":" << facerec::format_float(cos, 6)
                  << ",\"image_a\":\"" << facerec::json_escape(img_a) << "\","
                  << "\"image_b\":\"" << facerec::json_escape(img_b) << "\"}"
                  << std::endl;
        return 0;
    } catch (const cv::Exception& e) {
        std::cout << facerec::error_to_json(e.what(), "pipeline_error") << std::endl;
        return 3;
    }
}
