#ifndef FACERECOGNITION_SERVER_HPP
#define FACERECOGNITION_SERVER_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "config.hpp"

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
};

// 启动 HTTP 服务端，阻塞直到 SIGINT/SIGTERM。
// 返回 0 表示正常停机；非 0 表示启动失败（端口占用、模型加载失败等）。
int run_server(const PipelineConfig& cfg, const ServerOptions& opt);

}  // namespace facerec

#endif  // FACERECOGNITION_SERVER_HPP