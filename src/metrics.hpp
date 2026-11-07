#ifndef FACERECOGNITION_METRICS_HPP
#define FACERECOGNITION_METRICS_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace facerec {

// 固定桶的直方图。桶边界按本项目的实测尺度挑（毫秒）：检测 ~1 ms、提取 ~5 ms/脸、
// 单请求 ~10 ms、放大到 1 s 兜住异常。
//
// 只存计数，不做在线分位数：Prometheus 端用 histogram_quantile 算，服务端保持无锁、
// 可随时读。
class Histogram {
public:
    // 桶边界由调用方给：耗时是毫秒、相似度是 [0,1] 的 cosine，两者共用一套边界只会
    // 把分数塞进第一个桶，什么也看不出来。
    Histogram(const char* name, const char* help, const double* bounds, size_t n)
        : name_(name), help_(help), bounds_(bounds), n_(n) {}

    void observe(double value) {
        size_t idx = 0;
        while (idx < n_ && value > bounds_[idx]) ++idx;
        counts_[idx].fetch_add(1, std::memory_order_relaxed);
        // 和用微秒整数累加：原子的 double 在 C++17 里不是无锁的，而这里只需要一个和。
        sum_milli_.fetch_add(static_cast<uint64_t>(value * 1000.0),
                             std::memory_order_relaxed);
        total_.fetch_add(1, std::memory_order_relaxed);
    }

    // Prometheus 文本格式（含 _bucket/_sum/_count）。前缀固定 facerec_。
    void write_prometheus(std::string& out, const std::string& labels) const;

    const char* name() const { return name_; }
    const char* help() const { return help_; }

    static constexpr size_t kMaxBuckets = 16;

private:
    const char* name_;
    const char* help_;
    const double* bounds_;
    size_t n_;
    std::array<std::atomic<uint64_t>, kMaxBuckets> counts_{};
    std::atomic<uint64_t> sum_milli_{0};
    std::atomic<uint64_t> total_{0};
};

class Counter {
public:
    explicit Counter(const char* name, const char* help) : name_(name), help_(help) {}
    void inc(uint64_t delta = 1) { value_.fetch_add(delta, std::memory_order_relaxed); }
    uint64_t value() const { return value_.load(std::memory_order_relaxed); }
    void write_prometheus(std::string& out, const std::string& labels) const;
    const char* name() const { return name_; }
    const char* help() const { return help_; }

private:
    const char* name_;
    const char* help_;
    std::atomic<uint64_t> value_{0};
};

// 全部指标。一个进程一份（server / mcp 各持有一个）。
//
// 这些量能提前两周发现"模型换了但 gallery 没换"这类事故：
// unknown 率突变、match score 分布左移、detect/embed 直方图台阶式变化。
// 桶边界（见 metrics.cpp）。
namespace detail {
extern const double kLatencyMsBounds[12];
extern const size_t kLatencyMsCount;
extern const double kCosineBounds[10];
extern const size_t kCosineCount;
}  // namespace detail

class Metrics {
public:
    Metrics();

    // 请求级
    Counter requests_total{"requests_total", "HTTP/MCP requests served"};
    Counter request_errors_total{"request_errors_total",
                                 "Requests answered with an error"};
    Counter faces_total{"faces_detected_total", "Faces detected across requests"};
    Counter matches_known_total{"matches_known_total",
                                "Faces matched to an enrolled person"};
    Counter matches_unknown_total{"matches_unknown_total",
                                  "Faces below the recognition threshold"};
    Counter cache_hits_total{"cache_hits_total", "Identify results served from cache"};
    Counter cache_misses_total{"cache_misses_total", "Identify results computed"};
    Counter cache_evictions_total{"cache_evictions_total",
                                  "Cache entries dropped by the LRU bound"};

    // 阶段耗时（毫秒）
    Histogram decode_ms{"decode_ms", "Image decode time (ms)",
                        detail::kLatencyMsBounds, detail::kLatencyMsCount};
    Histogram detect_ms{"detect_ms", "Face detection time (ms)",
                        detail::kLatencyMsBounds, detail::kLatencyMsCount};
    Histogram embed_ms{"embed_ms", "Alignment + embedding time (ms)",
                       detail::kLatencyMsBounds, detail::kLatencyMsCount};
    Histogram request_ms{"request_ms", "Total time per identify request (ms)",
                         detail::kLatencyMsBounds, detail::kLatencyMsCount};

    // 分数分布（cosine，[0,1]）：能看出"最近匹配变差了"
    Histogram match_score{"match_score", "Best cosine per detected face",
                          detail::kCosineBounds, detail::kCosineCount};

    std::string to_prometheus() const;
};

}  // namespace facerec

#endif  // FACERECOGNITION_METRICS_HPP