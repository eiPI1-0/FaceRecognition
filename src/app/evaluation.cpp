#include "app/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace facerec {

// ---------------------------------------------------------------------------
// ScoreDist
// ---------------------------------------------------------------------------

int ScoreDist::bin_of(double x) {
    // 两端溢出各自归到一个边界 bin：评测里不该因为一个越界值丢样本，也不该
    // 因为浮点误差把 1.0000001 当成新 bin。
    if (!(x > -1.0)) return 0;        // 含 NaN -> bin 0（最低），不会污染高分尾部
    if (x >= 1.0) return kBins;
    const int i = static_cast<int>(std::floor((x + 1.0) * 0.5 * kBins));
    return std::min(i, kBins);
}

void ScoreDist::add(float score) {
    if (h_.empty()) h_.assign(kBins + 1, 0ULL);
    if (!(score == score)) return;    // NaN 直接丢：它进任何阈值表都是静默污染
    ++h_[bin_of(score)];
    ++n_;
    sum_ += score;
}

double ScoreDist::tail(double t) const {
    if (!n_ || h_.empty()) return 0.0;
    const int b = bin_of(t);
    unsigned long long ge = 0;
    for (int i = b; i <= kBins; ++i) ge += h_[i];
    return static_cast<double>(ge) / static_cast<double>(n_);
}

double ScoreDist::quantile(double q) const {
    if (!n_ || h_.empty()) return 2.0;
    const long double want = static_cast<long double>(q) * n_;
    unsigned long long ge = 0;
    for (int i = kBins; i >= 0; --i) {
        // 找到最小的 bin 下标 i 使 S_i = sum_{j>=i} h_j <= q*n；阈值取该 bin 的下边界。
        if (static_cast<long double>(ge + h_[i]) > want) {
            if (i == kBins) return 2.0;   // 最高 bin 本身就超预算：没有可达阈值
            return -1.0 + static_cast<double>(i + 1) * (2.0 / kBins);
        }
        ge += h_[i];
    }
    return -1.0;   // q >= 1：全分布都满足
}

// ---------------------------------------------------------------------------
// 评测
// ---------------------------------------------------------------------------

namespace {

// 与 CosSimilarity::feature_fusion 同一条规则：每张先归一化，求和，再归一化。
// （均值与求和只差一个正系数，归一化后等价 —— 但写成求和是为了和引擎一模一样，
// 免得将来 fusion 改了加权、这里还按均值算。）
std::vector<float> fuse(const std::vector<std::vector<float>>& photos) {
    if (photos.empty() || photos[0].empty()) return {};
    std::vector<float> acc(photos[0].size(), 0.0f);
    for (const auto& f : photos) {
        if (f.size() != acc.size()) continue;    // 维度不齐：跳过，不静默截断
        double n = 0.0;
        for (float v : f) n += static_cast<double>(v) * v;
        n = std::sqrt(n);
        if (!(n > 0.0)) continue;
        for (size_t i = 0; i < acc.size(); ++i) acc[i] += f[i] / static_cast<float>(n);
    }
    double n = 0.0;
    for (float v : acc) n += static_cast<double>(v) * v;
    n = std::sqrt(n);
    if (!(n > 0.0)) return {};
    for (float& v : acc) v /= static_cast<float>(n);
    return acc;
}

// 特征已在外部归一化过（评测外壳按此约定），所以这里只保证一次归一化成立：
// 模型原始输出的范数约 15，不归一化点积就不是相似度。
std::vector<float> unit(std::vector<float> f) {
    double n = 0.0;
    for (float v : f) n += static_cast<double>(v) * v;
    n = std::sqrt(n);
    if (!(n > 0.0)) return f;
    for (float& v : f) v /= static_cast<float>(n);
    return f;
}

double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}

}  // namespace

double one_n_far_from_pair_rate(double pair_rate, size_t comparisons) {
    if (comparisons == 0 || pair_rate <= 0.0) return 0.0;   // 没有对手 -> 不可能误接受
    if (pair_rate >= 1.0) return 1.0;
    return 1.0 - std::pow(1.0 - pair_rate, static_cast<double>(comparisons));
}

double pair_rate_for_one_n_far(double target_far, size_t comparisons) {
    if (comparisons == 0) return 0.0;
    if (target_far <= 0.0 || target_far >= 1.0) return -1.0;   // FAR 0 / FAR 1：无解，不是"阈值很大"
    return 1.0 - std::pow(1.0 - target_far, 1.0 / static_cast<double>(comparisons));
}

double EvalReport::thresh_for_far(double target_far, size_t gallery_n) const {
    const double p = pair_rate_for_one_n_far(target_far, gallery_n);
    if (p < 0.0) return 2.0;
    return impostor11.quantile(p);
}

EvalReport evaluate_dataset(const std::vector<EvalPerson>& people,
                            const std::vector<double>& thresholds) {
    EvalReport rep;

    // 先归一化所有特征，再按人数分组保留索引（跳过没有可用照片的人，否则
    // "别人的最大值"会因为空 gallery 记录而假性偏低）。
    std::vector<const EvalPerson*> ps;
    for (const auto& p : people) {
        if (p.gallery.empty() || p.probe.empty()) continue;
        ps.push_back(&p);
    }
    rep.persons = ps.size();
    for (const auto* p : ps) {
        rep.gallery_photos += p->gallery.size();
        rep.probe_photos += p->probe.size();
    }
    if (ps.empty()) return rep;

    std::vector<std::vector<float>> fused(ps.size());
    std::vector<std::vector<std::vector<float>>> gallery(ps.size()), probe(ps.size());
    for (size_t i = 0; i < ps.size(); ++i) {
        for (const auto& f : ps[i]->gallery) gallery[i].push_back(unit(f));
        for (const auto& f : ps[i]->probe) probe[i].push_back(unit(f));
        fused[i] = fuse(ps[i]->gallery);
    }

    // 1:1：探针 vs 注册侧的每一张照片（同一人 = genuine，不同人 = impostor）。
    // 1:N：探针 vs 融合后的记录；自己的分与"别人的最大值"必须成对记下来。
    for (size_t i = 0; i < ps.size(); ++i) {
        for (const auto& q : probe[i]) {
            double own = fused[i].empty() ? -2.0 : dot(q, fused[i]);
            double best_other = -2.0;
            for (size_t j = 0; j < ps.size(); ++j) {
                if (j == i) continue;
                const double s = dot(q, fused[j]);
                if (s > best_other) best_other = s;
                for (const auto& g : gallery[j]) rep.impostor11.add(static_cast<float>(dot(q, g)));
            }
            for (const auto& g : gallery[i]) rep.genuine11.add(static_cast<float>(dot(q, g)));
            if (own > -2.0) rep.genuine1n.add(static_cast<float>(own));
            if (best_other > -2.0) rep.max1n.add(static_cast<float>(best_other));
            rep.one_n.emplace_back(static_cast<float>(own), static_cast<float>(best_other));
        }
    }

    // EER（1:1 语义）：扫 bin 边界，找 far 与 frr 交叉处，线性插值。
    if (rep.genuine11.count() && rep.impostor11.count()) {
        double prev_far = 0.0, prev_frr = 0.0, prev_t = -1.0;
        for (int b = 0; b <= ScoreDist::kBins; ++b) {
            const double t = -1.0 + static_cast<double>(b) * (2.0 / ScoreDist::kBins);
            const double far = rep.impostor11.tail(t);
            const double frr = 1.0 - rep.genuine11.tail(t);
            if (b > 0 && (frr - prev_far) * (prev_frr - far) <= 0.0) {
                const double d0 = prev_frr - prev_far, d1 = frr - far;
                const double w = (d0 == d1) ? 0.0 : d0 / (d0 - d1);
                rep.eer_thresh = prev_t + w * (t - prev_t);
                rep.eer_rate = 0.5 * ((prev_far + far) + (prev_frr + frr));
                break;
            }
            prev_far = far, prev_frr = frr, prev_t = t;
        }
    }

    std::vector<double> ts = thresholds;
    if (ts.empty()) {
        for (double t = 0.05; t < 1.0; t += 0.05) ts.push_back(t);
    }
    for (double t : ts) {
        EvalRow r;
        r.thresh = t;
        r.far_11 = rep.impostor11.tail(t);
        r.frr_11 = 1.0 - rep.genuine11.tail(t);
        size_t fa = 0, fr = 0;
        for (const auto& p : rep.one_n) {
            if (p.second >= t) ++fa;                                   // 别人被当成本人
            if (p.first < t || p.first <= p.second) ++fr;              // 本人没被返回
        }
        const double n = rep.one_n.empty() ? 1.0 : static_cast<double>(rep.one_n.size());
        r.far_1n = static_cast<double>(fa) / n;
        r.frr_1n = static_cast<double>(fr) / n;
        rep.rows.push_back(r);
    }
    return rep;
}

std::string EvalReport::to_text(size_t gallery_n) const {
    std::ostringstream o;
    o << "identities=" << persons << "  gallery_photos=" << gallery_photos
      << "  probe_photos=" << probe_photos << "  gallery_size(1:N)=" << gallery_n << "\n";
    char buf[128];
    std::snprintf(buf, sizeof buf, "genuine 1:1  mean=%.4f  n=%zu\n",
                  genuine11.mean(), genuine11.count());
    o << buf;
    std::snprintf(buf, sizeof buf, "impostor 1:1 mean=%.4f  n=%zu\n",
                  impostor11.mean(), impostor11.count());
    o << buf;
    std::snprintf(buf, sizeof buf, "genuine 1:N  mean=%.4f  n=%zu\n",
                  genuine1n.mean(), genuine1n.count());
    o << buf;
    std::snprintf(buf, sizeof buf, "max-other 1:N mean=%.4f  n=%zu\n",
                  max1n.mean(), max1n.count());
    o << buf;
    std::snprintf(buf, sizeof buf, "EER (1:1): threshold=%.4f at rate=%.5f\n\n",
                  eer_thresh, eer_rate);
    o << buf;

    o << "thresh   FAR1:1     FRR1:1     FAR1:N     FRR1:N\n";
    for (const auto& r : rows) {
        std::snprintf(buf, sizeof buf, "%5.2f   %-9.6f %-9.6f %-9.6f %-9.6f\n",
                      r.thresh, r.far_11, r.frr_11, r.far_1n, r.frr_1n);
        o << buf;
    }

    o << "\nthreshold for a target FAR, 1:1 (measured pairs) and 1:N (independence model,"
         " n = gallery size):\n";
    // “这个样本量能量到多小的 FAR”必须一并输出。两层限制：
    //   pair 层：一个误接受对 = 1/n_pairs，更小的尾概率分不出。
    //   1:N 层：gallery 规模 n 把 pair 层分辨率放大 ~n 倍（N 个对里只要有 1 个中就
    //   算误接受），所以 n 越大，可证的 FAR 越高。不写这两行，表格会假装支持 FAR=1e-9。
    const double resolution = impostor11.count() ? 1.0 / static_cast<double>(impostor11.count()) : 0.0;
    const double certifiable = one_n_far_from_pair_rate(resolution, gallery_n);
    std::snprintf(buf, sizeof buf,
                  "measurement resolution: one impostor pair = %.3e of the 1:1 tail "
                  "(%zu pairs); the smallest 1:N FAR this sample can certify at n=%zu is %.3e\n",
                  resolution, impostor11.count(), gallery_n, certifiable);
    o << buf;
    // 一个阈值如果没有任何实测对超出它，那它背后的 FAR 是**外推**而不是测量：
    // 宁可写 unreachable，也不要给一个看起来精确到 1e-4 的数。
    auto fmt = [this](double t) {
        if (t > 1.0 || impostor11.tail(t) == 0.0) return std::string("unreachable ");
        std::string s(12, ' ');
        std::snprintf(&s[0], s.size(), "%.4f", t);
        return s;
    };
    for (double far : {1e-2, 1e-3, 1e-4, 1e-5}) {
        o << "  FAR=" << std::scientific << far << std::defaultfloat
          << "  1:1 t=" << fmt(impostor11.quantile(far))
          << "  1:N(n=" << gallery_n << ") t=" << fmt(thresh_for_far(far, gallery_n));
        if (certifiable > 0.0 && far < certifiable) o << "   <- below what this sample can certify";
        o << "\n";
    }
    return o.str();
}

std::string EvalReport::to_csv(size_t gallery_n) const {
    std::ostringstream o;
    o << std::setprecision(8);
    o << "gallery_size,thresh,far_1_1,frr_1_1,far_1_n,frr_1_n\n";
    for (const auto& r : rows) {
        o << gallery_n << ',' << r.thresh << ',' << r.far_11 << ',' << r.frr_11 << ','
          << r.far_1n << ',' << r.frr_1n << '\n';
    }
    return o.str();
}

}  // namespace facerec
