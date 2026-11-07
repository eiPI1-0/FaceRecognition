// PriorBox::decode 的两道门的回归测试（PLAN.md D6 的第一条）。
//
// 为什么这两条断言值得一个测试文件：它们是**唯一**挡住"模型输出里的垃圾 anchor 变成幽灵脸"
// 的东西，而两处都很容易被后人"简化"成更直觉、但语义错误的写法：
//   - 分数门写成 `score < ignore_score`（NaN 与两个比较都是 false → 把 NaN 放行）；
//   - 几何门被当成"整洁"删掉（分数被 clamp 到 [0,1]，垃圾 conf 能伪造 0.99 分）。
// 把分数门改回 `<` 时，第 1 个用例必须变红。做不到这一点的测试没有价值。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "priorbox.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

// detector input 160x120、输出同尺寸时的先验数：
//   stride 8  : 20x15 格 x 3 anchor = 900
//   stride 16 : 10x7  格 x 2        = 140
//   stride 32 : 5x3   格 x 2        = 30
//   stride 64 : 2x1   格 x 3        = 6
constexpr int kAnchors = 1076;

struct Blobs {
    cv::Mat loc, conf, iou;
    Blobs()
        : loc(1, kAnchors * 14, CV_32F, cv::Scalar(0.f)),
          conf(1, kAnchors * 2, CV_32F, cv::Scalar(0.f)),
          iou(1, kAnchors, CV_32F, cv::Scalar(0.f)) {}
};

void set_anchor(Blobs& b, int i, float cls, float iou_score, float dx, float dy, float dw,
                float dh) {
    b.conf.at<float>(0, i * 2 + 1) = cls;
    b.iou.at<float>(0, i) = iou_score;
    float* l = b.loc.ptr<float>(0) + i * 14;
    l[0] = dx;
    l[1] = dy;
    l[2] = dw;
    l[3] = dh;
}

bool finite_box(const Face& f) {
    const BndBox_xyxy& b = f.bbox;
    return std::isfinite(b.top_left.x) && std::isfinite(b.top_left.y) &&
           std::isfinite(b.bottom_right.x) && std::isfinite(b.bottom_right.y) &&
           b.bottom_right.x > b.top_left.x && b.bottom_right.y > b.top_left.y;
}

}  // namespace

int main() {
    PriorBox pb(cv::Size(160, 120), cv::Size(160, 120));

    // ---- 1. NaN 分数一个都不许通过 -------------------------------------------
    // 这就是 fused 导出的实际故障形态：尾部 anchor 的 conf 是未初始化内存，
    // sqrt(cls*iou) 变成 NaN。旧写法 `score < ignore_score` 对 NaN 为 false → 放行。
    {
        Blobs b;
        for (int i = 0; i < kAnchors; ++i) {
            set_anchor(b, i, std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.f, 0.f, 0.f, 0.f);
        }
        const auto dets = pb.decode(b.loc, b.conf, b.iou);
        check(dets.empty(), "all-NaN scores -> 0 candidates (score gate rejects NaN)");
    }

    // ---- 2. 分数有限但几何退化：必须被几何门拦下 ------------------------------
    // dw/dh 很小 -> w = prior.w * exp(dw * 0.1) 塌成亚像素框。分数是伪造的 0.99。
    {
        Blobs b;
        for (int i = 0; i < kAnchors; ++i) {
            set_anchor(b, i, 1.0f, 1.0f, 0.f, 0.f, -80.f, -80.f);
        }
        const auto dets = pb.decode(b.loc, b.conf, b.iou);
        check(dets.empty(), "plausible score + collapsed box -> 0 candidates (geometry gate)");
    }

    // ---- 3. loc 里出现 NaN/Inf：同样不许通过 --------------------------------
    {
        Blobs b;
        for (int i = 0; i < kAnchors; ++i) {
            set_anchor(b, i, 1.0f, 1.0f, std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f, 0.f);
        }
        const auto dets = pb.decode(b.loc, b.conf, b.iou);
        check(dets.empty(), "NaN loc with high score -> 0 candidates (finite-loc gate)");
    }

    // ---- 4. 阳性对照：一个真实候选必须原样出来（门不能把好数据也拦掉） ---------
    {
        Blobs b;
        // 唯一一个超阈值的 anchor，box 落在画面中央附近。
        set_anchor(b, 0, 1.0f, 1.0f, 0.f, 0.f, 0.f, 0.f);
        const auto dets = pb.decode(b.loc, b.conf, b.iou, /*ignore_score=*/0.3f);
        const bool one = dets.size() == 1;
        check(one, "single confident anchor -> exactly 1 candidate");
        if (one) {
            check(finite_box(dets[0]), "the candidate's box is finite and non-degenerate");
            check(dets[0].score > 0.99f, "score is the usual sqrt(cls*iou) ~ 1.0");
            const BndBox_xyxy& bb = dets[0].bbox;
            const float ix = std::min(bb.bottom_right.x, 160.f) - std::max(bb.top_left.x, 0.f);
            const float iy = std::min(bb.bottom_right.y, 120.f) - std::max(bb.top_left.y, 0.f);
            check(ix > 0.f && iy > 0.f,
                  "the candidate overlaps the detector input (edge-clipped anchors stay valid)");
        }
    }

    // ---- 5. 阈值语义：低于 ignore_score 的分数被丢弃，且边界是闭区间 -----------
    {
        Blobs b;
        set_anchor(b, 0, 0.09f, 1.0f, 0.f, 0.f, 0.f, 0.f);   // sqrt(0.09) = 0.3 == 阈值
        const auto at = pb.decode(b.loc, b.conf, b.iou, 0.3f);
        check(at.size() == 1, "score exactly at ignore_score is kept (>= is inclusive)");

        Blobs c;
        set_anchor(c, 0, 0.0899f, 1.0f, 0.f, 0.f, 0.f, 0.f);
        const auto below = pb.decode(c.loc, c.conf, c.iou, 0.3f);
        check(below.empty(), "score just below ignore_score is dropped");
    }

    // ---- 6. 头/网格契约破坏必须报错而不是静默返回空 ---------------------------
    // 返回空会被上层读成"这张图没有脸"，与真漏检不可区分 —— 那是更坏的失败模式。
    {
        Blobs b;
        cv::Mat short_iou(1, kAnchors - 1, CV_32F, cv::Scalar(0.f));
        bool threw = false;
        try {
            (void)pb.decode(b.loc, b.conf, short_iou);
        } catch (const cv::Exception&) {
            threw = true;
        }
        check(threw, "anchor-count mismatch throws instead of returning an empty list");
    }

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}