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
    // **端到端**的"等容量"上限：从连接进入 HTTP 任务队列那一刻起算，到借到 pipeline 为止。
    // 超了就回 503，而不是无限排队。它决定背压落在这里还是落到客户端，所以是运维旋钮
    // 而不是实现细节。
    //
    // 它同时管两个准入点，且**连续**生效（见 admission.hpp）：
    //   队列段  已等超过它 → pre_routing 直接 503 `queue_expired`，不碰 pipeline；
    //   池  段  预算 = 它 − 队列段已等掉的（不会两段各等 5s）。
    // 早期版本只在池段用它，队列段无上限 —— 最坏情况因此是"队列 ∞ + 池 5s"，p50 30ms 的
    // 服务能让客户端等 60s。现在一个数界定端到端。
    // 取值：N>0 = 端到端上限；**0 = 零容忍**（当场拿不到就 503，含义与以前一致）；
    // -1 = 不限（关闭 shedding，宁排队也不拒）。0 不能代指"关闭"，因为那是静默翻转一个
    // 已有取值。
    // 可观测信号：queue_expired_total / queue_dropped_total / queue_wait_ms 直方图 /
    // `pool_busy` 事件 / http_requests_total{status="503"}。
    int queue_timeout_ms = 5000;
    // 任务队列的**连接**数硬上限（httplib 按连接排队，不按请求，见 timed_queue.hpp）。
    // 0 = 不设上限（默认）。这是兜底而不是主控制：主控制是上面那个 deadline（回正经 503）。
    // 触顶时 httplib 直接关 socket —— 请求还没被解析，无法按 HTTP 应答，客户端看到的是
    // 连接被关（curl: (52)）而不是 503，所以只在"护住 fd/内存"时才调它，并配合
    // queue_dropped_total 观察。0 默认意味着：宁可排队后拿 503，也不要无应答。
    int queue_max_pending = 0;
    // HTTP 层线程数。0 = 等于 workers（默认，也是最保守的正确值：解码与推理都不超订，
    // 内存上限就是 workers 张解码后的图）。此时 handler 永远拿得到 pipeline，
    // PipelinePool 不构成准入点，503 与 pool_busy 都不会出现 —— 排队发生在 httplib 的
    // 任务队列里，服务端看不见。
    // 设为 > workers 时 PipelinePool 才真正成为准入点：等不到就 503 + pool_busy 事件，
    // 排队时长可测（waited_ms）。代价是 http_threads 张并发解码的图（受 max-image-pixels
    // 约束）与 CPU 超订，只有在"解码/排队要和推理重叠"或"pipeline 比核数少（RSS 预算）"
    // 时才值得。
    int http_threads = 0;
    bool quiet = false;          // 抑制每请求访问日志（只影响日志量，不影响指标）
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