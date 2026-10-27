#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <limits>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

#include <opencv2/core.hpp>

// 流水线的全部可调项，main.cpp 的唯一来源。
// 库层（FaceRecognition.h）绝不包含本头：FeatureDatabase<T> 是 header-only 模板，
// 把 app 层概念漏进库头 = 每次调参全量重编。
struct PipelineConfig {
    // 资产路径。三条相对路径都相对 asset_root 解析（见 resolve_asset_paths）；
    // asset_root 为空表示"未指定"，由调用方给默认值（无配置文件时就是 "."）。
    std::string asset_root;
    std::string detector_model = "./models/yunet_120x160.onnx";
    std::string embedder_model = "./models/mobilefacenet.onnx";
    std::string data_dir       = "./data/";

    // 模型契约：改错不会崩，只会输出垃圾。只应在换模型文件时一同修改。
    int detector_input_w = 160, detector_input_h = 120;   // YuNet + PriorBox::decode 的先验框布局
    int embedder_input_w = 112, embedder_input_h = 112;   // ArcFace 对齐模板 / MobileFaceNet 输入

    // 检测
    float conf_thresh = 0.6f;
    float nms_thresh  = 0.3f;
    int   keep_top_k  = 750;
    // 检测头三个输出张量的名字（本仓库的 YuNet 导出：loc / conf / iou）
    std::string out_loc = "loc", out_conf = "conf", out_iou = "iou";

    // 识别与显示
    float recogn_thresh = 0.2f;
    int   rfactor       = 6;      // display_shape = detector_input * rfactor（派生，不可单独设）

    // 跟踪（关联与生命周期，见 face_tracker.hpp 的设计说明）
    // 关联走两条门取分：框 IoU，或"中心位移 < dist_ratio×框平均尺寸"。
    // 只靠 IoU 会在贴边时把一个脸拆成多条并行轨迹：同一个脸的框几何会在"整脸框"和
    // "只剩可见部分的框"之间跳，实测相邻帧 IoU 只有 0.37，低于旧的 0.45 单一阈值
    // → 一个脸两条轨迹同时画（画面表现就是"突然冒出好几个框"）。
    float tracker_iou  = 0.30f;
    float tracker_dist = 0.60f;   // 距离门系数：0.6×框尺寸 ≈ 同目标一步的合理位移
    int   detect_every = 1;       // 实测：检测 0.93ms vs 提特征 4.47ms/脸，调大省不了多少
    int   max_lost     = 5;       // 旧默认 15：一次误检/一次拆分能挂 16 帧残影，太长
    int   min_hits     = 2;       // 命中 2 次检测才上屏：单帧误检从此不上屏也不被识别

    // 人脸贴边时，对齐用的 side×side 窗口会超出图像边界，超出的部分怎么填：
    // constant（黑边，默认）| reflect（镜像）| replicate（边缘复制）。
    // 实测（2 身份 × 13 增强图 × 4 条边 × 裁掉窗口 25/50/75% = 312 例，指标 = 与未裁剪
    // 参考特征的 cosine）：constant 0.507 / reflect 0.509 / replicate 0.489 —— replicate
    // 明显更差、reflect 在噪声范围内，所以默认保持旧的黑边行为。
    std::string align_border = "constant";
    // 试过"贴边人脸再做一次水平镜像的 TTA 取高分"：实测 312 例贴边裁剪里平均 cosine 只
    // +0.006、312 例只多救回 1 例，代价是每张受影响的脸多一次 forward。已删除，别按直觉
    // 加回来 —— 镜像不找回丢掉的像素，只把缺口挪到窗口另一侧（模型对镜像本就近乎不变，
    // self vs self-flip cosine 0.97/0.98）。贴边人脸的旋钮只有 align.border。

    // 后端
    std::string backend = "cpu";  // cpu | cuda | openvino
    int batch_mode = 0;           // 0=按后端自动, 1=强制 batch, -1=强制逐张

    cv::Size detector_input() const { return cv::Size(detector_input_w, detector_input_h); }
    cv::Size embedder_input() const { return cv::Size(embedder_input_w, embedder_input_h); }
    cv::Size display_shape()  const { return detector_input() * rfactor; }
};

// 单一真源：每个可调项在这里描述一次（点分 key + 成员指针 + 取值区间）。
// 配置文件解析、--set 覆盖、--dump-config 全部由这张表生成，不许出现绕过表的分支。
// 成员指针类型必须取别名：`int PipelineConfig::*` 直接写在模板实参里 g++ 会报
// "two or more data types in declaration of 'type name'"。
using IntMember   = int PipelineConfig::*;
using FloatMember = float PipelineConfig::*;
using StrMember   = std::string PipelineConfig::*;

struct Knob {
    const char* key;
    std::variant<IntMember, FloatMember, StrMember> member;
    double lo = -std::numeric_limits<double>::infinity();   // string 类型忽略区间
    double hi =  std::numeric_limits<double>::infinity();
    const char* group = "";      // dump 时的分组标题
    const char* comment = "";    // dump 时写在条目上方的说明（生成的配置自解释）
};

const std::vector<Knob>& knobs();

// 按 key 查表；未找到返回 false，并在 candidates 里给出相近的合法 key（拼错时提示用）。
bool find_knob(const std::string& key, const Knob** knob, std::vector<std::string>* candidates = nullptr);

// 从字符串设值（--set 用），带类型与区间校验；失败时 err 带 key 名与实际值。
bool set_knob(const Knob& knob, const std::string& value, PipelineConfig& cfg, std::string& err);

// 读 flat 点分 key 的配置文件（cv::FileStorage，零新依赖）。未知 key -> 警告并给候选；
// 类型不符或越界 -> 硬失败，err 指名 key。文件不存在也失败（显式给的路径必须存在）。
bool load_config(const std::string& path, PipelineConfig& cfg,
                 std::string& err, std::vector<std::string>& warnings);

// 把三条相对资产路径挂到 asset_root 上（绝对路径原样保留）。在所有输入
// （默认值 / 配置文件 / --set）都应用完之后调用一次，避免"半解析"的歧义。
void resolve_asset_paths(PipelineConfig& cfg, const std::string& default_root);

// 区间校验 + 后端合法性 + 模型契约警告。返回 false 时 err 指名 key。
bool validate_config(const PipelineConfig& cfg, std::string& err, std::vector<std::string>& warnings);

// 模型文件能不能真被加载：不存在 / 0 字节 / 其实是个 Git LFS 指针。
// 与 validate_config 分开，因为 --dump-config 不需要模型在场；而这些检查只在真要跑
// 流水线时才该拦。报的错带 knob 名，用户才知道改哪一行。
bool check_model_files(const PipelineConfig& cfg, std::string& err);

// 把当前配置写成 flat 点分 key 的 YAML 风格文本（可读、可 grep、与 --set 同构）。
void dump_config(std::ostream& os, const PipelineConfig& cfg);

#endif  // CONFIG_HPP
