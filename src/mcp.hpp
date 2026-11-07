#ifndef FACERECOGNITION_MCP_HPP
#define FACERECOGNITION_MCP_HPP

#include <string>

#include "config.hpp"

namespace facerec {

// MCP 传输方式。
enum class McpTransport {
    Stdio,   // 换行分隔的 JSON-RPC，桌面 / IDE 客户端
    Http,    // POST /mcp，远程 / 多客户端
};

struct McpOptions {
    McpTransport transport = McpTransport::Stdio;
    std::string host = "127.0.0.1";
    int port = 8081;
    int workers = 0;             // 仅 Http：Pipeline 实例数 / 并发度
    int max_request_mb = 32;
    int max_image_pixels = 40000000;
    int cache_entries = 256;     // 识别结果缓存条目数；0 = 关闭
    bool quiet = false;
};

// 阻塞运行 MCP 服务端，返回进程退出码。
int run_mcp(const PipelineConfig& cfg, const McpOptions& opt);

}  // namespace facerec

#endif  // FACERECOGNITION_MCP_HPP