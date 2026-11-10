#include "cli/cli.hpp"
#include "cli/args.hpp"

#include "app/api.hpp"
#include "app/config.hpp"
#include "serve/mcp.hpp"
#include "serve/server.hpp"
#include <httplib.h>

#include <iostream>
#include <string>
#include <vector>

// .cpp 里用 using-directive（头文件里绝不）：内核在 `facerec::`，见 RFPLAN §6.1。
using namespace facerec;

using namespace cli;   // 实现文件内引入，不污染任何头文件

int cmd_mcp(int argc, char* argv[]) {
    ConfigArgs cfg_args;
    facerec::McpOptions opt;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value_of = [&](const char* name) -> std::string {
            const std::string eq = std::string(name) + "=";
            if (arg.rfind(eq, 0) == 0) return arg.substr(eq.size());
            return (i + 1 < argc) ? argv[++i] : std::string();
        };
        if (arg == "--transport" || arg.rfind("--transport=", 0) == 0) {
            const std::string v = value_of("--transport");
            if (v == "stdio") opt.transport = facerec::McpTransport::Stdio;
            else if (v == "http") opt.transport = facerec::McpTransport::Http;
            else {
                std::cerr << "--transport expects stdio|http, got '" << v << "'" << std::endl;
                return -1;
            }
        } else if (arg == "--host" || arg.rfind("--host=", 0) == 0) {
            opt.host = value_of("--host");
        } else if (arg == "--port" || arg.rfind("--port=", 0) == 0) {
            try { opt.port = std::stoi(value_of("--port")); } catch (...) {
                std::cerr << "--port expects an integer" << std::endl;
                return -1;
            }
        } else if (arg == "--workers" || arg.rfind("--workers=", 0) == 0) {
            try { opt.workers = std::stoi(value_of("--workers")); } catch (...) {
                std::cerr << "--workers expects an integer" << std::endl;
                return -1;
            }
        } else if (arg == "--max-request-mb" || arg.rfind("--max-request-mb=", 0) == 0) {
            try { opt.max_request_mb = std::stoi(value_of("--max-request-mb")); } catch (...) {
                std::cerr << "--max-request-mb expects an integer" << std::endl;
                return -1;
            }
        } else if (int q = parse_queue_arg(arg, i, argc, argv, opt); q != 0) {
            if (q < 0) return -1;
        } else if (arg == "--quiet") {
            opt.quiet = true;
        } else if (arg == "--config" || arg.rfind("--config=", 0) == 0) {
            if (!parse_config_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "--set" || arg.rfind("--set=", 0) == 0) {
            if (!parse_set_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }

    if (show_help) {
        std::cout << "Usage: facerec mcp [--transport stdio|http] [--host H] [--port P]\n"
                  << "                   [--workers N] [--max-request-mb 32] [--quiet]\n"
                  << "                   [--queue-timeout-ms 5000] [--http-threads N]\n"
                  << "                   [--queue-max-pending 0]\n"
                  << "                   [--config <path>] [--set key=value]\n"
                  << "\nTools (read-only): identify_faces, compare_faces, list_people, get_face_crop\n"
                  << "Enrollment is deliberately NOT exposed as a tool.\n"
                  << "\nstdio: newline-delimited JSON-RPC on stdin/stdout (logs on stderr)\n"
                  << "http:  POST /mcp\n";
        return 0;
    }

    PipelineConfig cfg;
    if (!load_pipeline_config(cfg_args, cfg)) return -1;
    if (!check_assets(cfg)) return -1;

    try {
        return facerec::run_mcp(cfg, opt);
    } catch (const cv::Exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return -1;
    }
}
