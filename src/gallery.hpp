#ifndef FACERECOGNITION_GALLERY_HPP
#define FACERECOGNITION_GALLERY_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "config.hpp"
#include "engine.hpp"

namespace facerec {

// person_id / provenance 的归一（空白 -> '_'）：索引是行式格式，空白是注入；
// 建库、写索引、比较目录名都必须用同一个规则，否则同一个人会有两个名字。
std::string normalize_entry_id(const std::string& s);

// 一条注册记录：一个人一份**已归一化**的融合特征 + 来源信息。
//
// 关键点：融合特征一旦落盘就必须是 unit-norm。CosSimilarity::similarity 只除以查询向量的
// 范数，假设库侧已归一化（否则返回值变成 ‖ref‖·cos ≈ 15 而不是相似度）；把这条契约放在
// 写入侧，读取方就不可能拿到未归一化的向量。
struct GalleryEntry {
    std::string person_id;
    int dim = 0;
    std::vector<float> fused;    // unit-norm
    size_t n_photos = 0;
    std::string provenance;      // 建库时的照片文件名，逗号分隔（便于追查）
};

// 可持久化的 face gallery。
//
// 为什么要有它：原来的 FeatureDatabase 在构造函数里边读图边跑 detect+embed，于是
// (1) 每个 worker 各算一遍（16 worker = 16 次完整注册），(2) 启动慢，(3) 无法热更新，
// (4) 结果缓存拿不到"gallery 变了吗"这个信息。现在注册一次、落盘、所有 worker 读同一个
// 文件，并用内容摘要作为缓存指纹的一部分。
//
// 失效判定用 **源文件戳**（每张照片的路径/大小/mtime + 两个模型文件的路径/大小/mtime），
// 而不是时间戳猜测：加一张照片、换一张照片、换模型，戳都会变。stamp 只 stat，不解码，
// 所以每次启动都校验得起。
class Gallery {
public:
    struct SourceStamp;   // 见 gallery.cpp

    // 载入索引；索引不存在、损坏、或源文件戳不匹配时**重建**并按 need_save 写回。
    // 返回值永不为空（链路失败时是一个空 gallery + ok=false）。
    static Gallery load_or_build(const PipelineConfig& cfg, Detector& detector,
                                 FeatureExtractor& extractor, const cv::Size& display_shape,
                                 int rfactor, bool* rebuilt = nullptr,
                                 std::string* err = nullptr);

    bool save(const std::string& path, std::string* err = nullptr) const;
    static Gallery load(const std::string& path, std::string* err = nullptr);

    // 是否真的注册了人；空 gallery 时匹配永远返回 false。
    bool empty() const { return entries_.empty(); }
    size_t size() const { return entries_.size(); }
    const std::vector<GalleryEntry>& entries() const { return entries_; }
    int dim() const { return dim_; }
    // 内容摘要：gallery 内容或模型变了，指纹就变。直接进结果缓存的键。
    const std::string& fingerprint() const { return fingerprint_; }
    const std::string& embedder_sha256() const { return embedder_sha256_; }

    // 一次 GEMM 排序全部人员。query 先归一化，与被归一化的库侧取点积 = cosine。
    // 返回 [(person_index, score)] 按分数降序，最多 k 条（k<=0 表示全部）。
    std::vector<std::pair<size_t, float>> rank(const cv::Mat& feature, int k) const;

    // 最高分。返回 true 表示 >= threshold；否则 false（name 仍写最高分那位，便于调用方
    // 解释"最像谁但不够像"）。
    bool best_match(const cv::Mat& feature, float threshold, float& score,
                    std::string& name) const;

private:
    std::string fingerprint_;
    std::string embedder_sha256_;
    int dim_ = 0;
    std::vector<GalleryEntry> entries_;
};

}  // namespace facerec

#endif  // FACERECOGNITION_GALLERY_HPP