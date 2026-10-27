#ifndef FACE_TRACKER_HPP
#define FACE_TRACKER_HPP

#include <vector>
#include "face_def.hpp"
#include "opencv2/opencv.hpp"

class Detector; // 前置声明，避免循环依赖

// 跟踪器交给应用的一帧视图：几何 + 跨帧稳定 ID + 本帧是否有检测支撑。
//
// measured=false 表示这一帧没有匹配到任何检测，框和关键点都是按速度外推的。
// 这个区分是有意暴露的：外推出来的关键点是上一帧的值加上位移，喂给嵌入模型
// 只会产出看似正常的垃圾特征（表现为幽灵框上冒出随机名字），所以应用据此
// 决定"要不要为它跑识别"，识别结果则按 id 沿用上一次的真证据。
// 跟踪器参数（全部是实测调出来的，理由写在 knob 注释里）。放在类外是因为类内
// 默认实参 Params{} 在类定义结束前用不到这些默认成员初始化（编译期硬约束）。
struct TrackerParams {
    float iou_threshold = 0.30f;  // 关联门一：框 IoU 下限
    float dist_ratio    = 0.60f;  // 关联门二：允许的中心位移 / 框平均尺寸
    int detect_every      = 1;    // 每几帧跑一次全图检测
    int max_lost          = 5;    // 已确认轨迹允许纯外推的最长帧数
    int min_hits          = 2;    // 命中几次检测后才输出（>1 会晚一帧上屏）
};

struct TrackView {
    Face face;              // 坐标系同 Detector 的 input_shape（默认 160x120）
    int id = -1;            // 跨帧稳定 ID（识别标签按它缓存）
    bool measured = false;  // 本帧有检测支撑（true）还是纯预测（false）
};

// 人脸跟踪器：恒定速度预测 + 全局最优度量的关联 + 轨迹生命周期管理。
//
// 设计要点（都有实测支撑，见 .agents/AGENTS.md）：
//  1. 关联用"框 IoU 或 归一化中心位移"两条门，且候选对全局按分数降序贪心接受，
//     不是按检测顺序贪婪认领 —— 贴边时同一个脸的框几何会突变（整脸框 vs 可见部分框，
//     相邻帧 IoU 只有 0.37），单靠 IoU 阈值会把一个脸拆成多条并行轨迹，画面上就是
//     "突然冒出好几个框"。
//  2. 轨迹分两阶段：命中 min_hits 次检测才上屏（单帧误检不会画出来、也不会被识别），
//     确认后允许 max_lost 帧纯外推（默认 5，不是过去那种 15 帧的残影）。
//  3. 同一目标被拆成多条轨迹时，末尾做一次合并（按框 IoU / 归一化中心距离），
//     留下证据更强的那条 —— 关联性再好也挡不住偶发拆分，去重是最后一道闸。
//  4. 预测只做刚体平移（框与关键点一起挪），绝不单独夹取框边角，否则贴边时框被压扁
//     → 下一轮关联更难 → 又拆一条轨迹；完全出画的轨迹由 prune() 清掉。
class FaceTracker {
public:
    using Params = TrackerParams;   // 兼容写法：FaceTracker::Params

    explicit FaceTracker(const TrackerParams& params = TrackerParams{});

    // 每帧调用。内部按 detect_every 决定是否调用 detector.detect()。
    void update(cv::Mat& img, Detector& detector);

    const std::vector<TrackView>& tracks() const { return views_; }
    size_t size() const { return views_.size(); }
    const Params& params() const { return p_; }

private:
    struct Track {
        int id = -1;
        cv::Rect2f box;                 // input_shape 坐标系
        Landmarks_5 lm{};
        float score = 0.f;              // 上次匹配到的检测分数
        cv::Point2f velocity{};         // 每帧位移（不是两次检测的总位移）
        int hits = 0;                   // 匹配到的检测次数
        int age = 0;                    // 距上次匹配到检测的帧数（0 = 本帧有检测）
    };

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);
    static cv::Point2f center(const cv::Rect2f& r) { return (r.tl() + r.br()) * 0.5f; }

    void advance(Track& t);
    void associate(const std::vector<Face>& dets);
    void dedup();
    void prune(const cv::Size& bounds);
    void build_views();

    std::vector<Track> tracks_;
    std::vector<TrackView> views_;
    Params p_;
    int next_id_ = 0;
    int frame_count_ = 0;
};

#endif // FACE_TRACKER_HPP
