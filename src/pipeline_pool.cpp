#include "pipeline_pool.hpp"

#include <algorithm>

namespace facerec {

PipelinePool::PipelinePool(const PipelineConfig& cfg, int size) {
    if (size < 1) size = 1;
    pipelines_.reserve(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i) {
        pipelines_.push_back(std::make_unique<Pipeline>(cfg));
    }
    free_.assign(static_cast<size_t>(size), true);
}

Pipeline* PipelinePool::acquire(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    auto take_free = [&]() -> Pipeline* {
        for (size_t i = 0; i < free_.size(); ++i) {
            if (free_[i]) {
                free_[i] = false;
                return pipelines_[i].get();
            }
        }
        return nullptr;
    };
    if (Pipeline* p = take_free()) {
        return p;
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    if (!cv_.wait_until(lock, deadline, [&] {
            return std::find(free_.begin(), free_.end(), true) != free_.end();
        })) {
        return nullptr;
    }
    return take_free();
}

void PipelinePool::release(Pipeline* p) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (size_t i = 0; i < pipelines_.size(); ++i) {
            if (pipelines_[i].get() == p) {
                free_[i] = true;
                break;
            }
        }
    }
    cv_.notify_one();
}

}  // namespace facerec