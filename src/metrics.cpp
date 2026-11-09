#include "metrics.hpp"

#include <sstream>

namespace facerec {

namespace detail {
// 毫秒：1/2/5/10/20/50/100/200/500/1k/5k/30k 覆盖实测的 1–90 ms 区间并兜住异常。
const double kLatencyMsBounds[12] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 30000};
const size_t kLatencyMsCount = 12;
// cosine：0.2 是默认识别阈值，0.6/0.9 是"明显同一人"的分界。
const double kCosineBounds[10] = {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99};
const size_t kCosineCount = 10;
}  // namespace detail

namespace {

std::string label_suffix(const std::string& labels) { return labels; }

}  // namespace

void Histogram::write_prometheus(std::string& out, const std::string& labels) const {
    const std::string ls = label_suffix(labels);
    out += "# HELP facerec_";
    out += name_;
    out += ' ';
    out += help_;
    out += "\n# TYPE facerec_";
    out += name_;
    out += " histogram\n";

    uint64_t cumulative = 0;
    for (size_t i = 0; i < n_; ++i) {
        cumulative += counts_[i].load(std::memory_order_relaxed);
        std::ostringstream line;
        line << "facerec_" << name_ << "_bucket{le=\"" << bounds_[i] << "\"";
        if (!ls.empty()) line << ls;
        line << "} " << cumulative << "\n";
        out += line.str();
    }
    std::ostringstream inf;
    inf << "facerec_" << name_ << "_bucket{le=\"+Inf\"";
    if (!ls.empty()) inf << ls;
    inf << "} " << total_.load(std::memory_order_relaxed) << "\n";
    out += inf.str();

    const double sum_ms =
        static_cast<double>(sum_milli_.load(std::memory_order_relaxed)) / 1000.0;
    std::ostringstream sum;
    sum << "facerec_" << name_ << "_sum" << (ls.empty() ? "" : ls) << " " << sum_ms << "\n";
    out += sum.str();
    std::ostringstream count;
    count << "facerec_" << name_ << "_count" << (ls.empty() ? "" : ls) << " "
          << total_.load(std::memory_order_relaxed) << "\n";
    out += count.str();
}

void Counter::write_prometheus(std::string& out, const std::string& labels) const {
    out += "# HELP facerec_";
    out += name_;
    out += ' ';
    out += help_;
    out += "\n# TYPE facerec_";
    out += name_;
    out += " counter\nfacerec_";
    out += name_;
    out += labels;
    out += ' ';
    out += std::to_string(value());
    out += '\n';
}

void LabeledCounter::write_prometheus(std::string& out) const {
    out += "# HELP facerec_";
    out += name_;
    out += ' ';
    out += help_;
    out += "\n# TYPE facerec_";
    out += name_;
    out += " counter\n";
    std::lock_guard<std::mutex> lk(m_);
    for (const auto& kv : values_) {  // std::map：已按标签字典序
        out += "facerec_";
        out += name_;
        out += '{';
        out += kv.first;
        out += "} ";
        out += std::to_string(kv.second);
        out += '\n';
    }
}

Metrics::Metrics() = default;

std::string Metrics::to_prometheus() const {
    std::string out;
    out.reserve(8192);
    const std::string none;
    http_requests.write_prometheus(out);
    faces_total.write_prometheus(out, none);
    matches_known_total.write_prometheus(out, none);
    matches_unknown_total.write_prometheus(out, none);
    cache_hits_total.write_prometheus(out, none);
    cache_misses_total.write_prometheus(out, none);
    cache_evictions_total.write_prometheus(out, none);
    decode_ms.write_prometheus(out, none);
    detect_ms.write_prometheus(out, none);
    embed_ms.write_prometheus(out, none);
    request_ms.write_prometheus(out, none);
    queue_wait_ms.write_prometheus(out, none);
    match_score.write_prometheus(out, none);
    return out;
}

}  // namespace facerec