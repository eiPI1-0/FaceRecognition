#include "cli/args.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "app/config.hpp"

// args.cpp 现在需要 facerec（`PipelineConfig` / `Knob` / `knobs()` 都在里面）。
using namespace facerec;


namespace cli {

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

}  // namespace cli
