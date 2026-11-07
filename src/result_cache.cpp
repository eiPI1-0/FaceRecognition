#include "result_cache.hpp"

#include <algorithm>
#include <sstream>

namespace facerec {

ResultCache::ResultCache(std::string fingerprint, size_t max_entries, size_t max_bytes)
    : fingerprint_(std::move(fingerprint)), max_entries_(max_entries), max_bytes_(max_bytes) {}

std::string ResultCache::make_key(const std::string& image_sha256,
                                  const IdentifyOptions& opt, int detector_w,
                                  int detector_h) const {
    std::ostringstream os;
    os << fingerprint_ << '|' << image_sha256 << '|' << detector_w << 'x' << detector_h
       << '|' << opt.top_k << '|' << opt.min_score << '|' << opt.include_crops
       << opt.include_embedding << opt.include_landmarks;
    return os.str();
}

size_t ResultCache::entry_bytes(const IdentifyResult& r) {
    size_t total = sizeof(IdentifyResult);
    total += r.faces.size() * sizeof(FaceResult);
    for (const auto& f : r.faces) {
        total += f.crop_png_b64.size();
        total += f.embedding.size() * sizeof(float);
        total += f.top_k.size() * sizeof(FaceMatch);
        total += 32;   // person_id 的近似开销
    }
    return total;
}

bool ResultCache::get(const std::string& key, IdentifyResult& out) {
    if (!enabled()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = map_.find(key);
    if (it == map_.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    it->second.last_used = ++clock_;
    out = it->second.result;   // 拷贝：调用方拿到独立对象，可在锁外序列化
    hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ResultCache::put(const std::string& key, const IdentifyResult& result) {
    if (!enabled()) return;
    const size_t bytes = entry_bytes(result);
    if (bytes > max_bytes_) return;   // 单条就超预算：不缓存，别把整个缓存挤空

    std::lock_guard<std::mutex> lock(mu_);
    auto it = map_.find(key);
    if (it != map_.end()) {
        bytes_ -= it->second.bytes;
        it->second.result = result;
        it->second.bytes = bytes;
        it->second.last_used = ++clock_;
        bytes_ += bytes;
        return;
    }
    Entry entry;
    entry.result = result;
    entry.bytes = bytes;
    entry.last_used = ++clock_;
    map_.emplace(key, std::move(entry));
    bytes_ += bytes;
    evict_locked();
}

void ResultCache::evict_locked() {
    while ((map_.size() > max_entries_ || bytes_ > max_bytes_) && !map_.empty()) {
        auto oldest = map_.begin();
        for (auto cur = map_.begin(); cur != map_.end(); ++cur) {
            if (cur->second.last_used < oldest->second.last_used) oldest = cur;
        }
        bytes_ -= oldest->second.bytes;
        map_.erase(oldest);
    }
}

ResultCache::Stats ResultCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    Stats s;
    s.hits = hits_.load(std::memory_order_relaxed);
    s.misses = misses_.load(std::memory_order_relaxed);
    s.entries = map_.size();
    s.bytes = bytes_;
    return s;
}

}  // namespace facerec