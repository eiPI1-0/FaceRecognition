#ifndef DETECTOR_HEAD_HPP
#define DETECTOR_HEAD_HPP

#include <memory>
#include <utility>
#include <vector>

#include "face_def.hpp"
#include "priorbox.hpp"

#include "opencv2/core.hpp"

// YuNet 家族的不同导出把检测结果放在不同的张量布局里，解码方式因此不同。
// 头类型从模型自己的输出层名推断（见 make_detector_head），不需要用户配置；
// 认不出来时抛异常并列出图的输出名，而不是静默返回 0 张脸。
class DetectorHead {
public:
    virtual ~DetectorHead() = default;

    // 需要向 net.forward() 请求的输出张量名（顺序与 decode() 的入参一致）。
    virtual const std::vector<cv::String>& outputNames() const = 0;

    // 解码成 Face 列表；坐标一律处于 detector input_shape 坐标系（与 PriorBox 路径一致）。
    // 布局与先验网格不符时抛 cv::Exception —— 越界读比报错难查得多。
    virtual std::vector<Face> decode(const std::vector<cv::Mat>& outs, float score_thresh) const = 0;

    // 人可读的头名，用于日志与错误信息。
    virtual const char* name() const = 0;
};

// layer_names: net.getLayerNames()。
// fused_names: 配置里的 detector.out_* 三个名字，只用于融合头。
// 已知两种布局：
//   fused     —— 单个 loc(14 float/anchor) + conf(2/anchor) + iou(1/anchor)，
//                4 层 stride 8/16/32/64，先验由 PriorBox 生成（120x160 那份导出）
//   per-level —— cls_{s}/obj_{s}/bbox_{s}/kps_{s}，每格 1 个 anchor，stride 写在名字里
//                （YuNet-G 风格，opencv_zoo face_detection_yunet_2023mar.onnx）
std::unique_ptr<DetectorHead> make_detector_head(const std::vector<cv::String>& layer_names,
                                                 cv::Size input_shape,
                                                 const std::vector<cv::String>& fused_names);

#endif
