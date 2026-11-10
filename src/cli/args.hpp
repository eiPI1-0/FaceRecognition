#ifndef FACEREC_CLI_ARGS_HPP
#define FACEREC_CLI_ARGS_HPP

// 各子命令共用的参数装配：内置默认 < 配置文件 < CLI 覆盖。四个外壳共用同一份，
// 不在每个 cmd_* 里重写一遍（历史上 --http-threads 就分叉过一次）。
// 头文件里不写 using namespace：这里全是限定名（`facerec::PipelineConfig`）。

#include <iostream>   // parse_queue_arg 的模板定义在这里，报错要能打印
#include <string>
#include <utility>
#include <vector>

#include "app/config.hpp"

namespace cli {

struct ConfigArgs {
    std::string config_path;
    std::vector<std::pair<std::string, std::string>> overrides;
};

// data/<person>/ 的目录名同时就是索引里的 person_id，所以它必须是单个安全的路径分量：
// 不能有 '/'、'\'、前导 '-'（会被当成选项），也不能是 '.'/'..'（能把注册写到库外）。
// 空白归一为 '_'，与 gallery 索引自己的规则一致（行式格式里空白是注入）。
std::string sanitize_person_name(const std::string& raw, std::string* reason);

// 依次应用：内置默认 < 配置文件 < CLI 覆盖，然后解析资产路径并校验。
// 失败时把错误写到 stderr 并返回 false（调用方统一 return -1）。
bool load_pipeline_config(const ConfigArgs& args, facerec::PipelineConfig& cfg);

// --set key=value / --set=key=value。解析失败时返回 false。
bool parse_set_arg(const std::string& arg, int& i, int argc, char* argv[],
                   ConfigArgs& out);

// 拥塞控制的三个旗标：serve 与 mcp **共用一份解析**。它们调的是同一个机制（任务队列 +
// pipeline 池），分两处写已经让 --http-threads 分叉过一次（mcp.cpp 里记着：一边设了一边没设，
// 两个外壳的并发语义就不一样了）。模板而不是重载：两个 Options 结构不同但字段同名，
// 谁漏了字段就是编译错误，而不是又一次静默分叉。
// 返回：0 = 不是这里的旗标；1 = 已消费；-1 = 用法错（已打印，调用方直接 return -1）。
template <class Opt>
int parse_queue_arg(const std::string& arg, int& i, int argc, char* argv[], Opt& opt) {
    auto value_of = [&](const char* name) -> std::string {
        const std::string eq = std::string(name) + "=";
        if (arg.rfind(eq, 0) == 0) return arg.substr(eq.size());
        return (i + 1 < argc) ? argv[++i] : std::string();
    };
    auto integer = [&](const char* name, int* out) -> bool {
        try {
            *out = std::stoi(value_of(name));
        } catch (...) {
            std::cerr << name << " expects an integer" << std::endl;
            return false;
        }
        return true;
    };
    if (arg == "--http-threads" || arg.rfind("--http-threads=", 0) == 0) {
        if (!integer("--http-threads", &opt.http_threads)) return -1;
        if (opt.http_threads < 1) {
            std::cerr << "--http-threads must be >= 1 (default = same as --workers)" << std::endl;
            return -1;
        }
        return 1;
    }
    if (arg == "--queue-timeout-ms" || arg.rfind("--queue-timeout-ms=", 0) == 0) {
        if (!integer("--queue-timeout-ms", &opt.queue_timeout_ms)) return -1;
        if (opt.queue_timeout_ms < -1) {
            // 0 的含义与以前一致：**零**等待容忍（当场拿不到就 503），不是"关闭"。
            // 把"关闭"藏在 0 里会静默翻转一个已有取值的含义（那是最坏的改法），
            // 所以用显式的 -1。
            std::cerr << "--queue-timeout-ms must be >= -1 "
                         "(-1 = never shed, 0 = no waiting at all)" << std::endl;
            return -1;
        }
        return 1;
    }
    if (arg == "--queue-max-pending" || arg.rfind("--queue-max-pending=", 0) == 0) {
        if (!integer("--queue-max-pending", &opt.queue_max_pending)) return -1;
        if (opt.queue_max_pending < 0) {
            std::cerr << "--queue-max-pending must be >= 0 (0 = unbounded)" << std::endl;
            return -1;
        }
        return 1;
    }
    return 0;
}

// --config path / --config=path
bool parse_config_arg(const std::string& arg, int& i, int argc, char* argv[],
                      ConfigArgs& out);

// 模型文件不在场时，报错要指名是哪个 knob 指错了（而不是 OpenCV importer 的原话）。
bool check_assets(const facerec::PipelineConfig& cfg);

}  // namespace cli

#endif  // FACEREC_CLI_ARGS_HPP
