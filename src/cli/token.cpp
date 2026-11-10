#include "cli/cli.hpp"
#include <filesystem>
#include "cli/args.hpp"
#include "serve/auth.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace cli;   // 实现文件内引入，不污染任何头文件

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
