#ifndef FACERECOGNITION_EVALUATION_HPP
#define FACERECOGNITION_EVALUATION_HPP

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// 阈值评测：把 recognize.thresh 从"一个看着合理的默认值"变成"量出来的数"。
//
// 这里只有**纯统计**：输入是已经算好的特征（unit-norm），输出是分数分布与阈值表。
// 没有模型、没有 IO、没有网络，所以能在 ctest 里用手造数据精确断言。数据集读取与
// 推理循环在 cli.cpp 的 evaluate 外壳里 —— 它必须走生产同款预处理（letterbox +
// 对齐 + 同一个模型），否则量出来的阈值不适用于线上。
//
// 为什么必须区分两种语义（这是本文件存在的核心理由）：
//   1:1 = /v1/compare：拿两张图的特征比一次。
//   1:N = /v1/identify：拿查询特征对**整个 gallery 取最大值**。
// 同一个阈值下 1:N 的误接受率远高于 1:1 —— 每个探针都要和 N 个人比一次，最大值
// 自然比单次更靠近噪声尾部。默认阈值当年是在 2 人 gallery 上定的，所以它的危险
// 不是"偏松"，而是**不会随 gallery 变大而自动收紧**。
// ---------------------------------------------------------------------------

namespace facerec {

// 分数分布：[-1,1] 上的等宽直方图。一次评测会产出百万级分数对；排序要 O(n log n)
// 加 8 字节/对的内存，而直方图内存固定（4001 个计数器 = 32 KB），阈值分辨率
// 0.0005 —— 远细于模型本身的噪声，也细到能区分"0.42 与 0.45"这种真实差别。
class ScoreDist {
public:
    static constexpr int kBins = 4000;   // bin 宽 = 2/kBins = 0.0005

    void add(float score);

    size_t count() const { return n_; }
    double mean() const { return n_ ? static_cast<double>(sum_) / n_ : 0.0; }

    // P(score >= t)。按 bin **下边界**计入，所以对尾部概率是略微高估（保守方向：
    // 由它反推的阈值只会比真值更严，不会放行更多请求）。空分布返回 0。
    double tail(double t) const;

    // 最小 t 使 P(score >= t) <= q。q=0.001 就是"99.9% 分位数"。精度一个 bin。
    // 注意 q 小于分箱分辨率（1/count）时，返回的是**超过所有实测样本**的边界：
    // 它满足不等式，但那个 FAR 并未被测出。调用方必须用 tail(t)==0 识别这种情形
    // （报告里写成 unreachable），否则会把外推当成测量。空分布返回 +2。
    double quantile(double q) const;

private:
    static int bin_of(double x);
    std::vector<unsigned long long> h_;   // kBins+1 个（两端溢出各归一个 bin）
    size_t n_ = 0;
    long double sum_ = 0;
};

// 一个人的划分：gallery = 注册侧（融合成一条 gallery 记录的照片），
// probe = 留出侧（**从不**参与注册的照片）。特征已由调用方归一化。
struct EvalPerson {
    std::string name;
    std::vector<std::vector<float>> gallery;
    std::vector<std::vector<float>> probe;
};

// 一行阈值扫描结果。
struct EvalRow {
    double thresh;
    double far_11;   // 1:1 误接受率：非同一人的对 >= t
    double frr_11;   // 1:1 误拒绝率：同一人的对 < t
    double far_1n;   // 1:N 误接受率：探针在**别人**里取到的最大值 >= t
    double frr_1n;   // 1:N 误拒绝率：正确身份没被返回（自己分数 < t，或自己不是第一名）
};

struct EvalReport {
    size_t persons = 0;
    size_t gallery_photos = 0;
    size_t probe_photos = 0;

    ScoreDist genuine11;    // 同一人的 (probe, gallery 照片) 对
    ScoreDist impostor11;   // 不同人的 (probe, gallery 照片) 对
    ScoreDist genuine1n;    // probe 与自己融合记录的分
    ScoreDist max1n;        // probe 在别人融合记录里的最大值

    // 每个探针的 (自己的分, 别人的最大值)：1:N 的 far/frr 必须按**联合**事件算，
    // 只看边际分布会把"自己分数够但别人更高"这种错当成正确。
    std::vector<std::pair<float, float>> one_n;

    double eer_thresh = 0.0;   // 等误率点（FAR == FRR），1:1 语义
    double eer_rate = 0.0;

    std::vector<EvalRow> rows;

    // 独立性模型下，1:N（gallery 规模 n）达到 target FAR 所需的阈值：
    //   FAR(n, t) = 1 - (1 - p(t))^n,  p(t) = 单对误接受率
    // 用于回答"我的库从 200 人涨到 20 万人时阈值该定多少"。实测 far_1n 会同时给出，
    // 两者对不上就说明存在独立性之外的因素（例如 LFW 里同名不同人/同一人不同名）。
    double thresh_for_far(double target_far, size_t gallery_n) const;

    // 人读表格（含模型阈值建议列）。gallery_n 用于表头与 1:N 实测列的语义说明。
    std::string to_text(size_t gallery_n) const;
    // 机器读（运维拿去选点）。
    std::string to_csv(size_t gallery_n) const;
};

// 由**单对比**误接受率 p 推"n 次独立对比里至少一次误接受"的概率：1-(1-p)^n。
// 不是预测而是算术：单对误接受率 1.26% 时，向 610 人取最大值是 1-0.9874^609 = 99.7%。
//
// n 是**对比次数**，不是库大小 —— 这两个约定在不同调用点相差 1，必须各自明说（否则
// 注释里写 n、代码里传 n-1，就是这里以前的情形）：
//   - 未注册的脸进了 G 人的库：它对全部 G 条记录取最大值 -> n = G（启动自检的问题）；
//   - 已注册探针的 1:N（evaluate 的 far_1n 实测列）：自己那条不算对手 -> n = G-1。
double one_n_far_from_pair_rate(double pair_rate, size_t comparisons);

// 反向：要"n 次对比里至少一次误接受"的概率不超过 target，单对比尾概率必须 <= 此值。
// target=0 返回 -1：没有任何有限阈值能做到 FAR 精确为 0（那不是"阈值很大"，是无解）。
double pair_rate_for_one_n_far(double target_far, size_t comparisons);

// 主入口：特征 -> 报告。gallery 侧融合 = 均值再归一化（与 CosSimilarity::feature_fusion
// 同一条规则），点积即相似度 —— 与生产一致：identify 走 CosSimilarity（库侧已
// unit-norm），compare 走显式双归一化 cosine。
// thresholds 为空时用内置扫描点（0.05 起、步长 0.05）。
EvalReport evaluate_dataset(const std::vector<EvalPerson>& people,
                            const std::vector<double>& thresholds = {});

}  // namespace facerec

#endif  // FACERECOGNITION_EVALUATION_HPP
