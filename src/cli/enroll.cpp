#include "cli/cli.hpp"
#include "cli/args.hpp"

#include "app/api.hpp"
#include "app/config.hpp"
#include "app/gallery.hpp"
#include "core/engine.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace cli;   // 实现文件内引入，不污染任何头文件

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
