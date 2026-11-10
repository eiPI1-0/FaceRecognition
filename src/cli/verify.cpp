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

void verify_usage() {
    std::cout
        << "Usage: facerec verify <img> --name <person> [options]\n"
        << "\nOptions:\n"
        << "  --name <person>           要验证的身份（必填）\n"
        << "  --config <path>           配置文件\n"
        << "  --set key=value           覆盖单个配置项，可重复\n"
        << "  --dump-config             打印生效配置并退出\n"
        << "Output (always JSON, whatever the exit code):\n"
        << "  {\"status\":\"ok\",\"verified\":true,\"cosine\":0.95,\"matched\":\"BruceLi\",\n"
        << "   \"person\":\"BruceLi\",\"threshold\":0.20}\n"
        << "Exit: 0 verified | 1 NOT verified (valid answer) | 2 cannot answer (no face /\n"
        << "      unreadable) | 3 pipeline error. Non-zero on \"no\" is deliberate: an auth\n"
        << "      check that always exits 0 makes every script that forgets to parse the\n"
        << "      JSON pass silently.\n"
        << "The face compared is the most confident one (identify's faces[0]), not the largest.\n";
}

int cmd_verify(int argc, char* argv[]) {
    std::string img_path;
    std::string person;
    ConfigArgs cfg_args;
    bool dump_only = false;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--name" || arg.rfind("--name=", 0) == 0) {
            person = (arg == "--name") ? ((i + 1 < argc) ? argv[++i] : std::string())
                                       : arg.substr(7);
        } else if (arg == "--config" || arg.rfind("--config=", 0) == 0) {
            if (!parse_config_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "--set" || arg.rfind("--set=", 0) == 0) {
            if (!parse_set_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "--dump-config") {
            dump_only = true;
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg.rfind("-", 0) != 0) {
            img_path = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }

    if (show_help) { verify_usage(); return 0; }

    PipelineConfig cfg;
    if (!load_pipeline_config(cfg_args, cfg)) return -1;

    if (dump_only) {
        dump_config(std::cout, cfg);
        return 0;
    }

    if (img_path.empty()) {
        std::cerr << "verify requires an image path" << std::endl;
        verify_usage();
        return -1;
    }
    if (person.empty()) {
        std::cerr << "verify requires --name <person>" << std::endl;
        verify_usage();
        return -1;
    }

    if (!check_assets(cfg)) return -1;

    try {
        facerec::Pipeline pipeline(cfg);
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << facerec::error_to_json("cannot read image: " + img_path, "bad_image")
                      << std::endl;
            return 2;
        }

        cv::Mat feat = pipeline.embed_image(img, 0);
        if (feat.empty()) {
            std::cout << facerec::error_to_json("no face detected in the image", "no_face")
                      << std::endl;
            return 2;
        }

        float score = 0.0f;
        std::string name;
        // best_match 在低于阈值时返回 false，但 score/name 仍写最像的那位 ——
        // “最像谁，但不够像”必须能解释，所以 threshold 一起输出。
        const bool above_thresh =
            pipeline.best_match(feat, cfg.recogn_thresh, score, name);

        const bool verified = above_thresh && name == person;
        std::cout << "{\"status\":\"ok\",\"verified\":" << (verified ? "true" : "false")
                  << ",\"cosine\":" << facerec::format_float(score, 6)
                  << ",\"matched\":\"" << facerec::json_escape(name) << "\","
                  << "\"person\":\"" << facerec::json_escape(person) << "\","
                  << "\"threshold\":" << facerec::format_float(cfg.recogn_thresh, 4) << ","
                  << "\"image\":\"" << facerec::json_escape(img_path) << "\"}"
                  << std::endl;
        // “不”是一个合法答案，但退出码不能装作成功：见 verify_usage。
        return verified ? 0 : 1;
    } catch (const cv::Exception& e) {
        std::cout << facerec::error_to_json(e.what(), "pipeline_error") << std::endl;
        return 3;
    }
}
