#ifndef FACERECOGNITION_RESULT_CACHE_HPP
#define FACERECOGNITION_RESULT_CACHE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "api.hpp"

namespace facerec {

// 识别结果缓存：键 = 图像字节的 sha256 + 影响结果的请求参数 + 进程指纹。
//
// 命中路径**不经过解码**：键是从原始字节算的，所以一次命中省掉 imdecode（5–10 ms）+
// 检测 + 提取（6–15 ms），只花 sha256（约 200 MB/s，一张 50 KB 的 JPEG 约 0.25 ms）。
// LLM 会重试、网关会重放、用户会重复上传同一张图 —— 这条路径很值。
//
// 键里包含**所有影响输出的请求参数**（top_k / min_score / include_*），而不是"缓存核心
// 再按需裁剪"：后者要为 crop/embedding 这类派生字段设计重算路径，代码多一倍、边界情况
// 多一倍，换来的只是"同一张图不同 top_k 也能命中"。重试场景本来就是同一组参数。
class ResultCache {
public:
    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        size_t entries = 0;
        size_t bytes = 0;
    };

    // 进程指纹：换模型 / 重注册后人脸数据会变，旧结果必须整体失效。
    // 由调用方在启动时计算一次（gallery 在进程生命周期内不变）。
    ResultCache(std::string fingerprint, size_t max_entries = 256,
                size_t max_bytes = 128u * 1024 * 1024);

    // 关闭缓存（--cache-entries 0）。之后 get 永远 miss、put 忽略。
    bool enabled() const { return max_entries_ > 0; }

    bool get(const std::string& key, IdentifyResult& out);
    void put(const std::string& key, const IdentifyResult& result);
    Stats stats() const;

    // 组装缓存键。image_sha256 是原始字节的摘要（由调用方算，命中路径上只算这一次）。
    std::string make_key(const std::string& image_sha256, const IdentifyOptions& opt,
                         int detector_w, int detector_h) const;

private:
    void evict_locked();
    static size_t entry_bytes(const IdentifyResult& r);

    std::string fingerprint_;
    size_t max_entries_;
    size_t max_bytes_;

    struct Entry {
        IdentifyResult result;
        size_t bytes = 0;
        uint64_t last_used = 0;
    };
    mutable std::mutex mu_;
    std::unordered_map<std::string, Entry> map_;
    size_t bytes_ = 0;
    uint64_t clock_ = 0;
    mutable std::atomic<uint64_t> hits_{0};
    mutable std::atomic<uint64_t> misses_{0};
};

}  // namespace facerec

#endif  // FACERECOGNITION_RESULT_CACHE_HPP