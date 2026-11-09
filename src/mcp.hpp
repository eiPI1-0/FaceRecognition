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
    // 拿不到 pipeline 时的排队上限（与 serve 同一个旋钮）：超时后 tool 结果是
    // "server busy"，并发一条 pool_busy 事件。
    int queue_timeout_ms = 5000;
    // HTTP 传输的线程数。0 = 等于 workers。之前这里是 httplib 的默认值
    // max(8, cores-1)，与 --workers 无关 —— 于是 pipeline 池才是真正的准入点，
    // 而 serve 却不是，两个外壳的行为分叉了。现在两边同一个规则。
    int http_threads = 0;
    bool quiet = false;
};

// 阻塞运行 MCP 服务端，返回进程退出码。
int run_mcp(const PipelineConfig& cfg, const McpOptions& opt);

}  // namespace facerec

#endif  // FACERECOGNITION_MCP_HPP