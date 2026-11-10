#include "cli/cli.hpp"
#include "cli/args.hpp"

#include "app/api.hpp"
#include "app/config.hpp"
#include "serve/auth.hpp"
#include "serve/mcp.hpp"
#include "serve/server.hpp"

#include <iostream>
#include <string>
#include <vector>

// .cpp 里用 using-directive（头文件里绝不）：内核在 `facerec::`，见 RFPLAN §6.1。
using namespace facerec;

using namespace cli;   // 实现文件内引入，不污染任何头文件

int cmd_serve(int argc, char* argv[]) {
    ConfigArgs cfg_args;
    facerec::ServerOptions opt;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value_of = [&](const char* name) -> std::string {
            const std::string eq = std::string(name) + "=";
            if (arg.rfind(eq, 0) == 0) return arg.substr(eq.size());
            return (i + 1 < argc) ? argv[++i] : std::string();
        };
        if (arg == "--host" || arg.rfind("--host=", 0) == 0) {
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
        } else if (arg == "--cv-threads" || arg.rfind("--cv-threads=", 0) == 0) {
            try { opt.cv_threads = std::stoi(value_of("--cv-threads")); } catch (...) {
                std::cerr << "--cv-threads expects an integer" << std::endl;
                return -1;
            }
        } else if (arg == "--cache-entries" || arg.rfind("--cache-entries=", 0) == 0) {
            try { opt.cache_entries = std::stoi(value_of("--cache-entries")); } catch (...) {
                std::cerr << "--cache-entries expects an integer" << std::endl;
                return -1;
            }
        } else if (int q = parse_queue_arg(arg, i, argc, argv, opt); q != 0) {
            if (q < 0) return -1;
        } else if (arg == "--reuseport") {
            opt.reuseport = true;
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
        std::cout << "Usage: facerec serve [--host 127.0.0.1] [--port 8080] [--workers N]\n"
                  << "                     [--cv-threads 1] [--max-request-mb 32] [--cache-entries 256]\n"
                  << "                     [--reuseport] [--quiet] [--queue-timeout-ms 5000]\n"
                  << "                     [--http-threads N] [--queue-max-pending 0]\n"
                  << "                     [--config <path>] [--set key=value]\n"
                  << "  --workers     pipeline instances / concurrent requests (default: CPU count)\n"
                  << "  --cv-threads  OpenCV internal thread pool per worker (default: 1)\n"
                  << "  --reuseport   SO_REUSEPORT: run N processes on one port (kernel load-balances)\n"
                  << "  --queue-timeout-ms  end-to-end budget for WAITING FOR CAPACITY (queue stage\n"
                  << "                  + pool stage together); past it a request gets 503 instead of\n"
                  << "                  being served for a caller that has already given up\n"
                  << "                  (-1 = never shed, 0 = no waiting at all)\n"
                  << "  --http-threads  HTTP worker threads (default = --workers)\n"
                  << "  --queue-max-pending   hard cap on connections queued in front of the workers\n"
                  << "                  (0 = unbounded). Backstop only: hitting it closes the socket\n"
                  << "                  instead of answering 503, so prefer --queue-timeout-ms\n"
                  << "Endpoints: POST /v1/identify /v1/embed /v1/compare, GET /v1/people,\n"
                  << "           GET /healthz, GET /readyz\n";
        return 0;
    }

    PipelineConfig cfg;
    if (!load_pipeline_config(cfg_args, cfg)) return -1;
    if (!check_assets(cfg)) return -1;

    try {
        return facerec::run_server(cfg, opt);
    } catch (const cv::Exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return -1;
    }
}
