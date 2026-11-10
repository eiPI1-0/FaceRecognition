#include <filesystem>
#include <iostream>
#include <string>

#include "cli/cli.hpp"
#include "cli/demo.hpp"

// 顶层 usage。**定义在入口而不是 cli/**：下面的命令名单就是分发表，两者必须同时改，
// 分开写就会有一个说谎（cli/* 里的每个命令文件另有自己的 --help）。
void print_usage(const char* argv0) {
    std::cout << "facerec - face detection + recognition pipeline\n"
              << "Usage: " << argv0 << " <command> [options]\n"
              << "\nCommands:\n"
              << "  identify  --image <path> [options]   单帧无状态人脸识别，JSON 输出到 stdout\n"
              << "  enroll    --name <person> [--dry-run|--yes] <photo> ...  离线建库验证\n"
              << "  serve     [--host H] [--port P] [--workers N]   HTTP 服务端\n"
              << "  demo      <source> [cpu|cuda|openvino] ...       实时窗口演示（旧行为）\n"
              << "  mcp       [--transport stdio|http] [--port P]    MCP 工具（给 LLM 调用）\n"
              << "  token     --name <id> [--tokens-file <path>]     生成 bearer 凭据（只显示一次）\n"
              << "  compare   <img-a> <img-b>              两张图主脸的余弦相似度（1.0=同人）\n"
              << "  verify    <img> --name <person>        这张图是否是某个人（否 -> 退出码非 0）\n"
              << "  evaluate  <dataset-dir>              量出阈值的 FAR/FRR（1:1 与 1:N）后定 recognize.thresh\n"
              << "\nLegacy: '" << argv0 << " 0' is treated as '" << argv0 << " demo 0'.\n"
              << "Run '" << argv0 << " <command> --help' for command usage.\n"
              << "Run '" << argv0 << " identify --image <path> --dump-config' to dump effective config.\n";
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    // 显式子命令入口。子命令拿到的是 `argv+1`，故参数个数为 argc-1。
    if (cmd == "identify") {
        return cmd_identify(argc - 1, argv + 1);
    } else if (cmd == "enroll") {
        return cmd_enroll(argc - 1, argv + 1);
    } else if (cmd == "serve") {
        return cmd_serve(argc - 1, argv + 1);
    } else if (cmd == "mcp") {
        return cmd_mcp(argc - 1, argv + 1);
    } else if (cmd == "token") {
        return cmd_token(argc - 1, argv + 1);
    } else if (cmd == "compare") {
        return cmd_compare(argc - 1, argv + 1);
    } else if (cmd == "verify") {
        return cmd_verify(argc - 1, argv + 1);
    } else if (cmd == "evaluate") {
        return cmd_evaluate(argc - 1, argv + 1);
    } else if (cmd == "demo") {
        return run_demo(argc - 1, argv + 1);
    }

    // 兼容旧形式：`./main 0`、`./main photo.jpg`，以及 `./main --dump-config` 这类
    // 纯选项调用。历史上它们都是"直接跑演示循环"，文档与 AGENTS.md 的验证配方仍在用，
    // 所以不当作错误。选项本身是明确的意图，只在把参数当 source 时才提示一次。
    if (cmd.rfind("-", 0) == 0) {
        return run_demo(argc, argv);
    }
    // 裸词（无 '/'、无 "://"、非序号、当前目录下也不存在）几乎一定是拼错的子命令，
    // 比如 ./main compaer：当 source 用会得到 "Cannot open source: compaer"，
    // 用户看不出是命令名写错了。旧形式 ./main 0 / ./main photo.jpg / ./main data/*.jpeg
    // 都仍然走演示路径。
    const bool looks_like_source =
        !cmd.empty() &&
        (cmd[0] == '/' || cmd.find('/') != std::string::npos ||
         cmd.find("://") != std::string::npos ||
         cmd.find_first_not_of("0123456789") == std::string::npos ||
         std::filesystem::exists(cmd));
    if (!looks_like_source) {
        std::cerr << "Unknown subcommand: " << cmd << std::endl
                  << "  subcommands: identify | enroll | serve | mcp | token | compare | verify | evaluate | demo"
                  << std::endl
                  << "  (a bare word is not a source: paths need '/', a URL needs '://', "
                  << "a camera index is all digits)" << std::endl;
        return -1;
    }
    if (cmd[0] != '-') {
        std::cerr << "note: treating '" << cmd << "' as a demo source; "
                  << "the subcommands are identify|enroll|serve|mcp|token|compare|verify|evaluate|demo"
                  << std::endl;
        return run_demo(argc, argv);
    }

    std::cerr << "Unknown command: " << cmd << std::endl;
    print_usage(argv[0]);
    return -1;
}
