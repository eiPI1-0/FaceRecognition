#ifndef FACERECOGNITION_PIPELINE_POOL_HPP
#define FACERECOGNITION_PIPELINE_POOL_HPP

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include "app/api.hpp"
#include "app/config.hpp"

namespace facerec {

// 固定 N 个 Pipeline 实例 + 有界并发。
//
// 为什么是"池"而不是"共享一个 Pipeline"：cv::dnn::Net 的 setInput/forward 会写实例
// 内部状态，不是线程安全的。每个 worker 独占一个实例，并发度天然等于池大小；请求多于
// 池容量时在这里排队 —— 这就是背压（配合 HTTP 的 task queue / MCP 的调用方）。
// 池满且等到超时，acquire 返回 nullptr，由上层回 503。
class PipelinePool {
public:
    PipelinePool(const PipelineConfig& cfg, int size);

    // 借出一个 Pipeline；超时返回 nullptr。
    Pipeline* acquire(std::chrono::milliseconds timeout);
    void release(Pipeline* p);

    size_t size() const { return pipelines_.size(); }
    // 每个实例各自建过 gallery，大小相同；取第一个报告即可。
    size_t gallery_size() const {
        return pipelines_.empty() ? 0 : pipelines_.front()->gallery_size();
    }
    // 库本身（启动自检用）。所有实例同一份，取第一个。
    const Gallery& gallery() const {
        static const Gallery empty;
        return pipelines_.empty() ? empty : pipelines_.front()->gallery();
    }
    // 缓存指纹：所有实例的配置与 gallery 相同，取第一个即可。
    std::string fingerprint() const {
        return pipelines_.empty() ? std::string() : pipelines_.front()->fingerprint();
    }

private:
    std::vector<std::unique_ptr<Pipeline>> pipelines_;
    std::vector<bool> free_;
    std::mutex mu_;
    std::condition_variable cv_;
};

// 作用域内的借出/归还，异常安全。
class Lease {
public:
    Lease(PipelinePool& pool, std::chrono::milliseconds timeout)
        : pool_(pool), p_(pool.acquire(timeout)) {}
    ~Lease() { if (p_) pool_.release(p_); }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    // 移动是为了能把"借出 + 计时 + 满了发事件"抽成一个函数返回（见 server.cpp 的
    // acquire_logged）；复制一直是禁止的，双 release 会比编译错误更糟。
    Lease(Lease&& o) noexcept : pool_(o.pool_), p_(o.p_) { o.p_ = nullptr; }
    Pipeline* get() const { return p_; }
    explicit operator bool() const { return p_ != nullptr; }

private:
    PipelinePool& pool_;
    Pipeline* p_;
};

}  // namespace facerec

#endif  // FACERECOGNITION_PIPELINE_POOL_HPP