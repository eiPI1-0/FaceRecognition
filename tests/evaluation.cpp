// 阈值评测（evaluation）的回归测试 —— 不需要模型，不需要数据。
//
// 为什么值得一个测试文件：这里锁的四条都是"算错不会报错、只会给出一份看起来正常的
// 阈值表"的东西，而那份表会直接变成线上 recognize.thresh：
//   1. 1:N 的失败是**联合**事件：本人分数够但别人更高，也是"没返回正确身份"。只看
//      边际分布（本人 < t）会把这种失败算成正确。
//   2. gallery 侧的融合规则必须与 CosSimilarity::feature_fusion 一致（每张先归一化、
//      求和、再归一化）。写成"先求均值再归一化"数值等价，但写成"取第一张"就悄悄变了。
//   3. ScoreDist 的 NaN 必须丢弃：一个 NaN 进了阈值表，tail/quantile 全都不可信。
//   4. thresh_for_far 的独立性模型必须与它声称的公式一致（否则"库涨到 20 万人阈值定
//      多少"这一列是编的）。
//   5. one_n_far_from_pair_rate 是启动自检与报告共用的**唯一**算术。它的 n 是对比次数
//      不是库大小（G 还是 G-1 取决于问的是未注册还是已注册探针），算错只会让启动时
//      那句"这个阈值会错接受多少"偏乐观，不会报错。
//      测试里还钉住了它与 LFW 实测行的差距（模型 0.0114 vs 实测 0.0131）——独立性
//      是理想化，同一个注册记录会出现在多个冒充对里。不记这个差，下一个人就会把
//      模型值当成实测值去设阈值。

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "app/evaluation.hpp"

// .cpp 里用 using-directive（头文件里绝不）：内核在 `facerec::`，见 RFPLAN §6.1。
using namespace facerec;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

bool near(double a, double b, double tol = 1e-4) { return std::fabs(a - b) <= tol; }

// 2 维单位向量：角度制。用它造出**已知余弦**的分数，断言就能精确到小数点后几位。
std::vector<float> dir(double deg) {
    const double r = deg * 3.14159265358979312 / 180.0;
    return {static_cast<float>(std::cos(r)), static_cast<float>(std::sin(r))};
}

void check_score_dist() {
    facerec::ScoreDist d;
    check(d.count() == 0 && d.tail(0.0) == 0.0, "empty dist: tail()=0 (not a division by zero)");
    check(d.quantile(0.0) > 1.0, "empty dist: quantile()=+2 (unreachable threshold)");

    for (float v : {0.10f, 0.20f, 0.30f, 0.40f, 0.90f}) d.add(v);
    d.add(std::nanf(""));            // 必须被丢掉
    d.add(-3.0f);                    // 越界：clip 到最低 bin，不新增样本
    d.add(3.0f);                     // 越界：clip 到最高 bin
    check(d.count() == 7, "NaN dropped, out-of-range clamped (count stays 7)");

    // P(>= t)：t=0.30 含 0.30/0.40/0.90 + 越界上端 = 4/7
    check(near(d.tail(0.30), 4.0 / 7.0, 1e-9), "tail(0.30) counts the boundary bin (4/7)");
    check(near(d.tail(0.95), 1.0 / 7.0, 1e-9), "tail(0.95) = only the overflow bin (1/7)");
    // 上端溢出 bin 代表"分数 >= 1"：任何高于 1 的阈值仍然应该算上它，否则 tail 会在
    // t=1 处不连续（一个 1.0 分的对不应该因为阈值稍微提高就消失）。
    check(near(d.tail(1.5), 1.0 / 7.0, 1e-9), "scores >= 1 count for any threshold above 1");

    // 分位数单调
    const double q10 = d.quantile(0.10), q50 = d.quantile(0.50);
    check(q10 > q50, "quantile is monotone (q10 > q50)");
    check(near(d.tail(q50), 0.5, 0.001 + 1.0 / 7.0 * 0 + 1e-9) || d.tail(q50) <= 0.5,
          "tail(quantile(q)) <= q by construction");

    facerec::ScoreDist big;
    for (int i = 0; i < 1000; ++i) big.add(static_cast<float>(i) / 1000.0f);   // 0 .. 0.999
    check(big.quantile(0.0) >= 0.999 && big.quantile(0.0) <= 1.0005,
          "quantile(0) sits just above the largest sample");

    facerec::ScoreDist capped;
    capped.add(1.0f);   // 分数域上端本身有样本：FAR=0 在此不可达
    check(capped.quantile(0.0) > 1.0, "quantile(0) with a sample at the domain edge = unreachable");
}

// 1:N 的联合事件：本人分数够、但别人更高 —— 1:1 表里"正确接受"，1:N 表里必须记失败。
void check_one_n_is_joint() {
    facerec::EvalPerson a, b;
    a.name = "A"; a.gallery = {dir(0.0)};   a.probe = {dir(-10.0)};
    b.name = "B"; b.gallery = {dir(-10.0)}; b.probe = {dir(-10.0)};

    const auto rep = facerec::evaluate_dataset({a, b}, {0.98, 0.99});
    check(rep.persons == 2 && rep.gallery_photos == 2 && rep.probe_photos == 2,
          "2 identities x 1 gallery + 1 probe");
    check(rep.genuine11.count() == 2 && rep.impostor11.count() == 2,
          "1:1 pair counts: 2 genuine, 2 impostor");
    check(rep.one_n.size() == 2, "1:N records = one per probe");

    const facerec::EvalRow* r98 = nullptr;
    for (const auto& r : rep.rows) if (near(r.thresh, 0.98)) r98 = &r;
    check(r98 != nullptr, "requested threshold present in the table");
    if (!r98) return;
    // 两个 genuine 对 (cos10=0.9848, cos0=1.0) 都 >= 0.98 -> 1:1 全对
    check(near(r98->frr_11, 0.0), "1:1 FRR at 0.98 = 0 (both genuine pairs above)");
    // 但 A 的探针在 fusedB 上拿到 1.0 > 自己的 0.9848 -> 身份没被返回
    check(near(r98->frr_1n, 0.5), "1:N FRR at 0.98 = 1/2 (out-ranked, not below-threshold)");
    check(near(r98->far_1n, 1.0), "1:N FAR at 0.98 = 1 (both probes find an impostor above)");

    const facerec::EvalRow* r99 = nullptr;
    for (const auto& r : rep.rows) if (near(r.thresh, 0.99)) r99 = &r;
    if (r99) check(near(r99->far_1n, 0.5), "1:N FAR tightens when the threshold rises (1 -> 1/2)");
}

// 融合规则：两张 ±3° 的照片，融合后应在 0° —— 1:N 的本人分因此是 1.0，
// 而 1:1 的两张各自只有 cos3。两个均值不同就是这个区分。
void check_fusion_rule() {
    facerec::EvalPerson a, b;
    a.name = "A"; a.gallery = {dir(-3.0), dir(3.0)}; a.probe = {dir(0.0)};
    b.name = "B"; b.gallery = {dir(90.0)};           b.probe = {dir(90.0)};

    const auto rep = facerec::evaluate_dataset({a, b}, {});
    const double c3 = std::cos(3.0 * 3.14159265358979312 / 180.0);
    const double c0 = 1.0;                       // B: probe 90 deg vs own gallery 90 deg
    check(near(rep.genuine1n.mean(), 1.0, 1e-5),
          "fused gallery entry sits at the mean direction (own score 1.0)");
    check(near(rep.genuine11.mean(), (2.0 * c3 + c0) / 3.0, 1e-5),
          "1:1 genuine mean = per-photo cosines, not the fused one");
    check(rep.genuine1n.mean() > rep.genuine11.mean(),
          "fusion is not identity: fused mean > per-photo mean");
}

// 独立性模型：FAR(n,t) = 1-(1-p)^n，反推出的阈值必须满足该式（bin 宽内）。
//
// 这里用**高维随机单位向量**而不是平面角度：真实嵌入的分数分布就是这种“宽而连续”
// 的形状；平面角度会在分数顶端聚成一堆（同一个 bin 里分不出高低），导致“库大 10 倍
// ⇒ 阈值更严”在分箱分辨率下变成等号 —— 那是测试自建的假阳性。
void check_independence_model() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto rand_unit = [&](int dim) {
        std::vector<float> v(dim);
        double n2 = 0.0;
        for (auto& x : v) { x = nd(rng); n2 += static_cast<double>(x) * x; }
        const double n = std::sqrt(n2);
        for (auto& x : v) x /= static_cast<float>(n);
        return v;
    };
    std::vector<facerec::EvalPerson> people;
    for (int i = 0; i < 60; ++i) {
        facerec::EvalPerson p;
        p.name = "p" + std::to_string(i);
        for (int k = 0; k < 2; ++k) p.gallery.push_back(rand_unit(8));
        for (int k = 0; k < 2; ++k) p.probe.push_back(rand_unit(8));
        people.push_back(std::move(p));
    }
    const auto rep = facerec::evaluate_dataset(people, {});
    // 120 个探针 × 别人家的 118 张注册照 = 14160 对；自己家的 2 张是 genuine。
    check(rep.persons == 60 && rep.impostor11.count() == 120 * 118,
          "60 identities: impostor pair count exact");
    check(rep.genuine11.count() == 240 && rep.one_n.size() == 120,
          "genuine pair count, one 1:N record per probe");

    const double far = 0.05;
    const double t = rep.thresh_for_far(far, 60);
    const double p_expected = 1.0 - std::pow(1.0 - far, 1.0 / 60.0);
    check(t > -1.0 && t < 2.0, "thresh_for_far returns a usable threshold");
    check(std::fabs(rep.impostor11.tail(t) - p_expected) <= 0.0005 + 1e-9,
          "model inverse agrees with the histogram (within one bin)");
    // 同一个目标 FAR，库大 100 倍 => 阈值必须更严。
    check(rep.thresh_for_far(far, 6000) > rep.thresh_for_far(far, 60),
          "same target FAR, 100x gallery => stricter threshold (the whole point of 1:N)");
    check(rep.thresh_for_far(0.0, 60) > 1.0, "target FAR 0 -> unreachable threshold");

    // 报告不得把“量不出来”当成“量到了”：没有 unreachable / resolution 两行，读者
    // 就会把 FAR=1e-9 那一列当真（和当年把 0.2 当真是一样的错）。14160 个对证不出
    // 6000 人库的 FAR=1e-5，那一行必须是 unreachable。
    const std::string text = rep.to_text(6000);
    check(text.find("unreachable") != std::string::npos,
          "report marks unresolvable thresholds instead of printing a fake number");
    check(text.find("measurement resolution") != std::string::npos,
          "report states the smallest FAR this sample can certify");
}

// 启动自检用的那条算术：单配对率 -> 1:N 率。它是唯一路径（serve/mcp 启动行、/readyz、
// 报告的建议列都走它），而它不在任何模型的保护范围内 —— 算错了没人报错，只会把
// "这个阈值在你的库里会错接受多少"说得比实际乐观。
void check_pair_rate_to_one_n() {
    // 无碰撞：每个配对都没超阈 → 1:N 也是 0（不是"未知"）。
    check(facerec::one_n_far_from_pair_rate(0.0, 1000) == 0.0,
          "pair rate 0 -> 1:N rate 0 (exact)");
    // n 是**对比次数**。n=0 就是没有对手；n=1 就是单对本身。
    check(facerec::one_n_far_from_pair_rate(0.5, 0) == 0.0, "no comparisons -> FAR 0");
    check(near(facerec::one_n_far_from_pair_rate(0.0126, 1), 0.0126),
          "one comparison -> the pair rate itself");
    check(near(facerec::one_n_far_from_pair_rate(0.0126, 2), 1.0 - 0.9874 * 0.9874, 1e-9),
          "two comparisons -> the binomial union, not 2p");
    // 两个常见调用点：未注册脸对 G 人的库用 n=G；已注册探针（evaluate 的 far_1n 实测列）
    // 用 n=G-1。LFW 实测行：p=1.2639e-2、610 人库、实测 far_1n=0.99672。
    // LFW 实跑对照（t=0.20、610 人库）：模型 1-(1-1.2639e-2)^609 = 0.9996，实测 0.9967。
    // 两者都说"这个阈值下几乎必错"，但不相等：同一个注册记录会出现在很多个冒充对里，
    // 所以那些"对"并不独立。这是已知偏差，不是 bug：它只在 p 很小时才紧（t=0.45：
    // 模型 0.0114 vs 实测 0.0131）。
    check(near(facerec::one_n_far_from_pair_rate(0.012638967, 609), 0.9996, 1e-4),
          "LFW t=0.20 model gives 0.9996 (measured 0.9967: same conclusion, not identical)");
    check(near(facerec::one_n_far_from_pair_rate(1.8843037e-05, 609), 0.011410, 1e-5),
          "LFW t=0.45 model gives 0.0114 (measured 0.0131 - the idealization is 13% low)");
    check(facerec::one_n_far_from_pair_rate(0.012638967, 610)
              > facerec::one_n_far_from_pair_rate(0.012638967, 609),
          "an unenrolled face has one more opponent than an enrolled probe (n=G vs G-1)");
    // 小 p 大 N：递推与独立 Monte-Carlo 抽样一致。
    std::mt19937 rng(4242);
    std::bernoulli_distribution hit(0.002);
    const size_t n = 5000;
    int above = 0, trials = 100000;
    for (int s = 0; s < trials; ++s) {
        bool any = false;
        for (size_t k = 0; k + 1 < n && !any; ++k) any = hit(rng);
        if (any) ++above;
    }
    const double mc = static_cast<double>(above) / trials;
    const double model = facerec::one_n_far_from_pair_rate(0.002, n);
    check(std::fabs(mc - model) < 0.004,
          "binomial model matches 1e5 independent draws (MC " + std::to_string(mc) +
          " vs model " + std::to_string(model) + ")");
    // 反向函数是正向的逆：拿 p 算 FAR，再从 FAR 算回 p。
    const double p = 0.0017, far = facerec::one_n_far_from_pair_rate(p, 800);
    check(near(facerec::pair_rate_for_one_n_far(far, 800), p, 1e-9), "inverse round-trips");
    // FAR 目标 0 不是"阈值拉到无穷"，是无解：返回 -1 而不是 0，否则调用方会把
    // "p=0"当成一个可实现的尾概率去查分位数。
    check(facerec::pair_rate_for_one_n_far(0.0, 800) < 0.0, "target FAR 0 -> no solution (-1)");
    check(facerec::pair_rate_for_one_n_far(1.0, 800) < 0.0, "target FAR 1 -> no solution (-1)");
    check(facerec::one_n_far_from_pair_rate(1.0, 5) == 1.0, "pair rate 1 -> 1:N rate 1");
}

}  // namespace

int main() {
    std::printf("ScoreDist\n");
    check_score_dist();
    std::printf("1:N is a joint event\n");
    check_one_n_is_joint();
    std::printf("gallery fusion rule\n");
    check_fusion_rule();
    std::printf("independence model for 1:N\n");
    check_independence_model();
    std::printf("pair rate -> 1:N rate (the startup self-check's arithmetic)\n");
    check_pair_rate_to_one_n();

    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "ok", g_failures);
    return g_failures ? 1 : 0;
}
