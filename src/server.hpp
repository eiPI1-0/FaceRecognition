#ifndef FACERECOGNITION_SERVER_HPP
#define FACERECOGNITION_SERVER_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "config.hpp"

namespace httplib { class Server; }

namespace facerec {

// HTTP 服务端的运行参数（超出 PipelineConfig 的那部分）。
struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    int workers = 0;             // 0 = 按 cv::getNumberOfCPUs() 自动，最少 1
    int max_request_mb = 32;     // 请求体上限（防解压炸弹/误传大文件）
    int max_image_pixels = 40000000;  // 解码后像素上限
    // 识别结果缓存条目数（LRU，按条目数与字节数双限）。0 = 关闭。
    int cache_entries = 256;
    // OpenCV 内部线程池大小。默认 1：本机实测单请求延迟不变（p50 10.6ms 两种设置相同），
    // 但 4/8 worker 下吞吐 +32%/+26%、p99 更低 —— 每个 worker 再开 16 线程只会互相抢核。
    // 单 worker + 大 batch 或 GPU 后端时才值得调大。
    int cv_threads = 1;
    bool quiet = false;          // 抑制每请求访问日志
    // 端口冲突策略。false（默认）：SO_REUSEADDR 开 + SO_REUSEPORT 关 —— 端口被活着的
    // 监听者占用时**启动失败**，而不是与它静默分摊连接（httplib 的默认值正是后者，见
    // server.cpp 里那段注释）。true：开 SO_REUSEPORT，多个进程绑同一端口由内核分流，
    // 用于横向扩展（单进程 8 worker 已贴近 CPU 墙，193 qps）。
    bool reuseport = false;
};

// 装好 SIGINT/SIGTERM/SIGPIPE 后 listen，阻塞到收到停止信号。
// HTTP 与 MCP 两个外壳共用一份：优雅停机和 SIGPIPE 屏蔽不能只有一边有。
// 返回 0 = 正常停机，1 = listen 失败（端口占用等）。
//
// 形参写成 ::httplib::Server 是有原因的：前向声明必须在全局作用域，
// 写成 facerec::httplib 会让本命名空间里所有 httplib:: 都解析到这个只有
// 一个空类的命名空间，后面每个 httplib::Request 都编不过。
int listen_with_signals(::httplib::Server& svr, const std::string& host, int port,
                        bool reuseport);

// 启动 HTTP 服务端，阻塞直到 SIGINT/SIGTERM。
// 返回 0 表示正常停机；非 0 表示启动失败（端口占用、模型加载失败等）。
int run_server(const PipelineConfig& cfg, const ServerOptions& opt);

}  // namespace facerec

#endif  // FACERECOGNITION_SERVER_HPP