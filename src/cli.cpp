#include "cli.hpp"

#include "auth.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "api.hpp"
#include "config.hpp"
#include "engine.hpp"
#include "mcp.hpp"
#include "server.hpp"

namespace {

// data/<person>/ 的目录名同时就是索引里的 person_id，所以它必须是单个安全的路径分量：
// 不能有 '/'、'\'、前导 '-'（会被当成选项），也不能是 '.'/'..'（能把注册写到库外）。
// 空白归一为 '_'，与 gallery 索引自己的规则一致（行式格式里空白是注入）。
std::string sanitize_person_name(const std::string& raw, std::string* reason) {
    std::string out = raw;
    for (char& c : out) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') c = '_';
    }
    if (out.empty() || out == "." || out == ".." || out.front() == '-' ||
        out.find('/') != std::string::npos || out.find('\\') != std::string::npos) {
        if (reason)
            *reason =
                "must be one path component (no '/', '\\', leading '-', and not '.' or '..')";
        return {};
    }
    return out;
}


// ---------------------------------------------------------------------------
// 配置装配（identify / enroll / serve / mcp 共用，避免四个外壳各写一份）
// ---------------------------------------------------------------------------

struct ConfigArgs {
    std::string config_path;
    std::vector<std::pair<std::string, std::string>> overrides;
};

// 依次应用：内置默认 < 配置文件 < CLI 覆盖，然后解析资产路径并校验。
// 失败时把错误写到 stderr 并返回 false（调用方统一 return -1）。
bool load_pipeline_config(const ConfigArgs& args, PipelineConfig& cfg) {
    std::string err;
    std::vector<std::string> warnings;
    if (!args.config_path.empty() &&
        !load_config(args.config_path, cfg, err, warnings)) {
        std::cerr << "Invalid config: " << err << std::endl;
        return false;
    }
    for (const auto& ov : args.overrides) {
        const Knob* knob = nullptr;
        std::vector<std::string> candidates;
        if (!find_knob(ov.first, &knob, &candidates)) {
            std::cerr << "Unknown knob: " << ov.first << "; did you mean";
            if (!candidates.empty()) {
                for (const auto& c : candidates) std::cerr << " " << c;
            } else {
                std::cerr << " nothing close; valid keys:";
                for (const auto& k : knobs()) std::cerr << " " << k.key;
            }
            std::cerr << std::endl;
            return false;
        }
        if (!set_knob(*knob, ov.second, cfg, err)) {
            std::cerr << "Invalid config: " << err << std::endl;
            return false;
        }
    }
    resolve_asset_paths(cfg, ".");
    if (!validate_config(cfg, err, warnings)) {
        std::cerr << "Invalid config: " << err << std::endl;
        return false;
    }
    for (const auto& w : warnings) std::cerr << "config warning: " << w << std::endl;
    return true;
}

// --set key=value / --set=key=value。解析失败时返回 false。
bool parse_set_arg(const std::string& arg, int& i, int argc, char* argv[],
                   ConfigArgs& out) {
    std::string assignment;
    if (arg == "--set") {
        if (i + 1 >= argc) {
            std::cerr << "--set expects key=value" << std::endl;
            return false;
        }
        assignment = argv[++i];
    } else {
        assignment = arg.substr(std::string("--set=").size());
    }
    const size_t eq = assignment.find('=');
    if (eq == std::string::npos) {
        std::cerr << "--set expects key=value, got '" << assignment << "'" << std::endl;
        return false;
    }
    out.overrides.emplace_back(assignment.substr(0, eq), assignment.substr(eq + 1));
    return true;
}

// --config path / --config=path
bool parse_config_arg(const std::string& arg, int& i, int argc, char* argv[],
                      ConfigArgs& out) {
    if (arg == "--config") {
        if (i + 1 >= argc) {
            std::cerr << "--config expects a path" << std::endl;
            return false;
        }
        out.config_path = argv[++i];
    } else {
        out.config_path = arg.substr(std::string("--config=").size());
    }
    return true;
}

// 模型文件不在场时，报错要指名是哪个 knob 指错了（而不是 OpenCV importer 的原话）。
bool check_assets(const PipelineConfig& cfg) {
    std::string err;
    std::string asset_hint;
    if (!check_model_files(cfg, err, &asset_hint)) {
        std::cerr << "Invalid config: " << err << asset_hint << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// identify
// ---------------------------------------------------------------------------

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

}  // namespace

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

// ---------------------------------------------------------------------------
// enroll
// ---------------------------------------------------------------------------

int cmd_enroll(int argc, char* argv[]) {
    std::string name;
    bool dry_run = true;
    std::vector<std::string> photos;
    ConfigArgs cfg_args;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--name" || arg.rfind("--name=", 0) == 0) {
            name = (arg == "--name") ? ((i + 1 < argc) ? argv[++i] : std::string())
                                     : arg.substr(7);
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--yes") {
            dry_run = false;
        } else if (arg == "--config" || arg.rfind("--config=", 0) == 0) {
            if (!parse_config_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "--set" || arg.rfind("--set=", 0) == 0) {
            if (!parse_set_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg.rfind("-", 0) != 0) {
            photos.push_back(arg);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }

    if (show_help) {
        std::cout << "Usage: facerec enroll --name <person> [--dry-run|--yes] <photo> ...\n"
                  << "  --dry-run (default) validate photos, print the plan, write nothing\n"
                  << "  --yes              copy the photos that contain a usable face into\n"
                  << "                     <data_dir>/<person>/ (the next run rebuilds the gallery index)\n";
        return 0;
    }
    if (name.empty()) {
        std::cerr << "enroll requires --name <person>" << std::endl;
        return -1;
    }
    if (photos.empty()) {
        std::cerr << "enroll requires at least one photo" << std::endl;
        return -1;
    }

    PipelineConfig cfg;
    if (!load_pipeline_config(cfg_args, cfg)) return -1;
    if (!check_assets(cfg)) return -1;

    // --yes 曾经只把 dry_run 改成 false 就报 ok：写入路径上静默什么也不做，
    // 是注册接口最坏的错（调用者以为人已入库）。现在它真的复制照片，并且把
    // 没有可用脸的排除在外。
    try {
        facerec::Pipeline pipeline(cfg);
        std::vector<std::pair<std::string, bool>> checked;   // photo -> 是否含可用脸
        int with_face = 0;
        for (const auto& p : photos) {
            cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "enroll: cannot read " << p << std::endl;
                checked.emplace_back(p, false);
                continue;
            }
            facerec::IdentifyOptions opt;   // 只要检测/对齐，不需要候选
            facerec::IdentifyResult r = pipeline.identify(img, opt, p);
            const bool ok = std::any_of(r.faces.begin(), r.faces.end(),
                                        [](const facerec::FaceResult& f) { return f.aligned; });
            if (ok) ++with_face;
            checked.emplace_back(p, ok);
        }

        std::string person = name;
        std::vector<std::string> written, skipped;
        for (const auto& c : checked) {
            if (!c.second) skipped.push_back(c.first);
        }
        if (!dry_run) {
            std::string why;
            person = sanitize_person_name(name, &why);
            if (person.empty()) {
                std::cerr << "enroll: invalid --name '" << name << "': " << why << std::endl;
                return -1;
            }
            if (person != name) {
                std::cerr << "enroll: person name normalized to '" << person << "'" << std::endl;
            }
            const std::filesystem::path dir = std::filesystem::path(cfg.data_dir) / person;
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec) {
                std::cerr << "enroll: cannot create " << dir.string() << ": " << ec.message()
                          << std::endl;
                return -1;
            }
            for (const auto& c : checked) {
                if (!c.second) continue;
                const std::filesystem::path src{c.first};
                std::filesystem::path target = dir / src.filename();
                for (int n = 1; std::filesystem::exists(target, ec); ++n) {
                    target = dir / (src.stem().string() + "_" + std::to_string(n) +
                                    src.extension().string());
                }
                std::filesystem::copy_file(c.first, target, std::filesystem::copy_options::none,
                                           ec);
                if (ec) {
                    std::cerr << "enroll: cannot write " << target.string() << ": "
                              << ec.message() << std::endl;
                    return -1;
                }
                written.push_back(target.string());
            }
            if (written.empty()) {
                std::cerr << "enroll: no photo contains a usable face; nothing written to "
                          << dir.string() << std::endl;
                return -1;
            }
        }

        std::cout << "{\"status\":\"ok\",\"person\":\"" << facerec::json_escape(person)
                  << "\",\"photos_total\":" << photos.size()
                  << ",\"photos_with_face\":" << with_face
                  << ",\"dry_run\":" << (dry_run ? "true" : "false")
                  << ",\"gallery_size\":" << pipeline.gallery_size()
                  << ",\"gallery_dir\":\"" << facerec::json_escape(cfg.data_dir + person) << "\""
                  << ",\"written\":[";
        for (size_t i = 0; i < written.size(); ++i) {
            std::cout << (i ? "," : "") << "\"" << facerec::json_escape(written[i]) << "\"";
        }
        std::cout << "],\"skipped\":[";
        for (size_t i = 0; i < skipped.size(); ++i) {
            std::cout << (i ? "," : "") << "\"" << facerec::json_escape(skipped[i]) << "\"";
        }
        std::cout << "]";
        if (dry_run) {
            // dry-run 时把"真正执行会发生什么"写清楚，而不是只报计划名。
            std::cout << ",\"would_write_to\":\"" << facerec::json_escape(cfg.data_dir + name)
                      << "\"";
        }
        std::cout << ",\"reload\":\"a running server keeps its in-memory gallery; restart to "
                     "load new photos\"}" << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << facerec::error_to_json(e.what(), "pipeline_error") << std::endl;
        return 3;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// token：生成 bearer 凭据
// ---------------------------------------------------------------------------

int cmd_token(int argc, char* argv[]) {
    std::string name;
    std::string tokens_file;
    bool explicit_file = false;
    bool show_help = false;
    ConfigArgs cfg_args;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--name" || arg.rfind("--name=", 0) == 0) {
            name = (arg == "--name") ? ((i + 1 < argc) ? argv[++i] : std::string())
                                     : arg.substr(7);
        } else if (arg == "--tokens-file" || arg.rfind("--tokens-file=", 0) == 0) {
            tokens_file = (arg == "--tokens-file")
                              ? ((i + 1 < argc) ? argv[++i] : std::string())
                              : arg.substr(14);
            explicit_file = true;
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
        std::cout << "Usage: facerec token --name <caller-id> [--tokens-file <path>]\n"
                  << "  Generates a random bearer token and prints it once. With --tokens-file\n"
                  << "  (or auth.tokens_file from the config) the `name:sha256:<hex>` line is\n"
                  << "  appended to that file with mode 0600; the token itself is never stored.\n";
        return 0;
    }
    if (name.empty()) {
        std::cerr << "token requires --name <caller-id>" << std::endl;
        return -1;
    }
    if (name.find_first_of(" \t:") != std::string::npos) {
        std::cerr << "token: --name must not contain whitespace or ':'" << std::endl;
        return -1;
    }

    PipelineConfig cfg;
    if (!load_pipeline_config(cfg_args, cfg)) return -1;
    if (!explicit_file) tokens_file = cfg.auth_tokens_file;

    const std::string token = facerec::generate_token();
    const std::string line = facerec::token_file_line(name, token);

    if (tokens_file.empty()) {
        std::cout << "token: " << token << '\n'
                  << "line:  " << line << '\n'
                  << "\nThis token is shown once. Append the line to your token table, or pass\n"
                  << "--tokens-file to have this command write it (mode 0600)." << std::endl;
        return 0;
    }

    // 追加一行。文件是凭据表：先确保 0600，再写。
    namespace fs = std::filesystem;
    {
        std::error_code ec;
        if (!fs::exists(tokens_file, ec)) {
            std::ofstream create(tokens_file, std::ios::app);
            if (!create) {
                std::cerr << "token: cannot create " << tokens_file << std::endl;
                return -1;
            }
        }
        fs::permissions(tokens_file, fs::perms::owner_read | fs::perms::owner_write,
                        fs::perm_options::replace, ec);
        if (ec) {
            std::cerr << "token: cannot set mode 0600 on " << tokens_file << ": " << ec.message()
                      << std::endl;
            return -1;
        }
    }
    {
        std::ofstream out(tokens_file, std::ios::app);
        if (!out) {
            std::cerr << "token: cannot append to " << tokens_file << std::endl;
            return -1;
        }
        out << line << '\n';
        if (!out) {
            std::cerr << "token: write failed on " << tokens_file << std::endl;
            return -1;
        }
    }
    // 重新载入一次：确认这一行真的能被解析（打错的名字/重复名在这里就暴露，而不是等 401）。
    {
        facerec::TokenTable check;
        std::string err;
        if (!check.load(tokens_file, err)) {
            std::cerr << "token: wrote the line but the table no longer parses: " << err
                      << std::endl;
            return -1;
        }
    }

    std::cout << "token: " << token << '\n'
              << "line:  " << line << '\n'
              << "file:  " << tokens_file << " (mode 0600, " << '\'' << "name:sha256:" << '\''
              << " lines; only the hash is stored)\n"
              << "\nUse it as:  curl -H 'Authorization: Bearer " << token.substr(0, 8)
              << "...' ..." << std::endl;
    return 0;
}

// ---------------------------------------------------------------------------
// serve / mcp
// ---------------------------------------------------------------------------

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
                  << "                     [--quiet]\n"
                  << "                     [--config <path>] [--set key=value]\n"
                  << "  --workers     pipeline instances / concurrent requests (default: CPU count)\n"
                  << "  --cv-threads  OpenCV internal thread pool per worker (default: 1)\n"
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