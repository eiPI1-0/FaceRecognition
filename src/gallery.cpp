#include "gallery.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

#include "encoding.hpp"
#include "sha256.hpp"

namespace facerec {

namespace {

constexpr const char* kMagic = "facerec-gallery";
constexpr const char* kFormatVersion = "1";

struct FileStamp {
    std::string path;
    long long size = 0;
    long long mtime = 0;
};

bool stat_file(const std::string& path, FileStamp& out) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return false;
    out.path = path;
    out.size = static_cast<long long>(st.st_size);
    out.mtime = static_cast<long long>(st.st_mtime);
    return true;
}

// 源文件戳：两个模型 + 每张注册照片的 (path, size, mtime)，排序后取摘要。
// 只 stat、不解码，所以每次启动校验都很便宜。
std::string source_stamp(const PipelineConfig& cfg) {
    std::ostringstream os;
    FileStamp fs;
    for (const std::string* path : {&cfg.detector_model, &cfg.embedder_model}) {
        if (stat_file(*path, fs)) {
            os << fs.path << ':' << fs.size << ':' << fs.mtime << ';';
        } else {
            os << *path << ":missing;";
        }
    }
    // 注册照片：data/<person>/*
    std::vector<std::string> persons = list_person_dirs(cfg.data_dir);
    std::vector<std::string> files;
    for (const auto& person : persons) {
        std::vector<cv::String> photos;
        cv::glob(cfg.data_dir + person + "/*", photos);
        for (const auto& p : photos) {
            files.push_back(person + "/" + std::string(p.substr(p.find_last_of('/') + 1)));
            if (stat_file(p, fs)) {
                os << fs.path << ':' << fs.size << ':' << fs.mtime << ';';
            }
        }
    }
    std::sort(files.begin(), files.end());
    for (const auto& f : files) os << f << ';';
    return sha256_hex(os.str());
}

void normalize_unit(std::vector<float>& v) {
    double norm = 0.0;
    for (float x : v) norm += static_cast<double>(x) * x;
    norm = std::sqrt(norm);
    if (norm <= 0.0) return;
    for (float& x : v) x = static_cast<float>(x / norm);
}

std::string floats_to_b64(const std::vector<float>& v) {
    std::string raw;
    raw.resize(v.size() * sizeof(float));
    std::memcpy(raw.data(), v.data(), raw.size());
    return base64_encode(raw);
}

bool b64_to_floats(const std::string& b64, size_t expected, std::vector<float>& out) {
    const std::string raw = base64_decode(b64);
    if (raw.size() != expected * sizeof(float)) return false;
    out.resize(expected);
    std::memcpy(out.data(), raw.data(), raw.size());
    return true;
}

}  // namespace

// 归一规则只有一处：建库时用它，写索引时再用一次（幂等），目录名比较也用它。
// 声明在 gallery.hpp —— MCP/CLI 需要把目录名与 entry.person_id 对齐。
std::string normalize_entry_id(const std::string& s) {
    // person_id 里不许出现空白/换行：索引是行式格式，那是格式注入。
    std::string out = s;
    for (char& c : out) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') c = '_';
    }
    return out;
}

// ---------------------------------------------------------------------------

std::vector<std::pair<size_t, float>> Gallery::rank(const cv::Mat& feature, int k) const {
    std::vector<std::pair<size_t, float>> out;
    if (feature.empty() || entries_.empty()) return out;
    if (static_cast<int>(feature.total()) != dim_) return out;   // 维度不匹配 = 不同模型

    // 查询向量归一化：点积即 cosine（库侧已 unit-norm）。
    cv::Mat query;
    feature.convertTo(query, CV_32F);
    query = query.reshape(1, 1);
    const double n = cv::norm(query);
    if (n <= 0.0) return out;
    query /= n;

    // 一次 GEMM：query(1×dim) * M(dim×N)。N 是人数，dim 是 512 —— 一万人约 5 MFLOP，
    // 比逐人虚函数调用快一个数量级，也不需要每位一个 cv::Mat。
    cv::Mat matrix(static_cast<int>(entries_.size()), dim_, CV_32F);
    for (size_t i = 0; i < entries_.size(); ++i) {
        std::copy(entries_[i].fused.begin(), entries_[i].fused.end(),
                  matrix.ptr<float>(static_cast<int>(i)));
    }
    cv::Mat scores;   // 1 x N
    cv::gemm(query, matrix, 1.0, cv::noArray(), 0.0, scores, cv::GEMM_2_T);

    out.reserve(entries_.size());
    const float* s = scores.ptr<float>(0);
    for (size_t i = 0; i < entries_.size(); ++i) {
        out.emplace_back(i, s[i]);
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    if (k > 0 && static_cast<int>(out.size()) > k) out.resize(static_cast<size_t>(k));
    return out;
}

bool Gallery::best_match(const cv::Mat& feature, float threshold, float& score,
                         std::string& name) const {
    score = -1.0f;
    name.clear();
    const auto ranked = rank(feature, 1);
    if (ranked.empty()) return false;
    score = ranked.front().second;
    name = entries_[ranked.front().first].person_id;
    return score >= threshold;
}

// ---------------------------------------------------------------------------
// 建库
// ---------------------------------------------------------------------------

Gallery Gallery::load_or_build(const PipelineConfig& cfg, Detector& detector,
                               FeatureExtractor& extractor, const cv::Size& display_shape,
                               int rfactor, bool* rebuilt, std::string* err) {
    if (rebuilt) *rebuilt = false;
    const std::string stamp = source_stamp(cfg);
    const std::string index_path = cfg.gallery_index;

    if (!index_path.empty()) {
        std::string load_err;
        Gallery loaded = load(index_path, &load_err);
        if (!load_err.empty()) {
            // 索引损坏不算错误：重建即可，但要让人看见原因。
            if (err) *err = load_err;
        } else if (loaded.fingerprint_ == stamp) {
            return loaded;   // 命中：不解码任何照片
        }
    }

    if (rebuilt) *rebuilt = true;

    Gallery g;
    g.fingerprint_ = stamp;
    {
        std::string raw;
        std::ifstream in(cfg.embedder_model, std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        raw = ss.str();
        if (!raw.empty()) g.embedder_sha256_ = sha256_hex(raw);
    }

    // 注册：每人把该目录下所有能用的脸融合成一份 unit-norm 特征（与原 FeatureDatabase
    // 完全相同的融合规则 —— 注册与查询必须同一预处理，这是本项目的硬约束）。
    CosSimilarity similarity;
    for (const auto& person : list_person_dirs(cfg.data_dir)) {
        std::vector<cv::String> photos;
        cv::glob(cfg.data_dir + person + "/*", photos);
        std::vector<cv::Mat> person_features;
        std::ostringstream provenance;
        bool first = true;
        for (const auto& photo : photos) {
            cv::Mat img = cv::imread(photo, cv::IMREAD_COLOR);
            if (img.empty()) continue;
            std::vector<cv::Mat> feats;
            if (!features_from_img(img, detector, extractor, feats, display_shape, rfactor) ||
                feats.empty() || feats[0].empty()) {
                continue;
            }
            person_features.push_back(feats[0].clone());
            if (!first) provenance << ',';
            first = false;
            const std::string p(photo);
            provenance << p.substr(p.find_last_of('/') + 1);
        }
        if (person_features.empty()) continue;

        cv::Mat fused;
        if (!similarity.feature_fusion(person_features, fused)) continue;
        GalleryEntry entry;
        // id 与 provenance 在建立时就归一：索引是行式格式，写入侧会再归一一次；
        // 若这里不归一，目录 `John Doe` 在冷启动时是 "John Doe"、索引重载后是 "John_Doe"，
        // 同一个人在同一服务里就有了两个名字。
        entry.person_id = normalize_entry_id(person);
        entry.dim = static_cast<int>(fused.total());
        entry.fused.assign(fused.ptr<float>(), fused.ptr<float>() + fused.total());
        normalize_unit(entry.fused);   // 写入侧强制 unit-norm（读取侧的硬契约）
        entry.n_photos = person_features.size();
        entry.provenance = normalize_entry_id(provenance.str());
        if (g.dim_ == 0) {
            g.dim_ = entry.dim;
        } else if (entry.dim != g.dim_) {
            continue;   // 不同模型的维度混在一个库里 = 数据损坏，跳过而不是污染
        }
        g.entries_.push_back(std::move(entry));
    }
    std::sort(g.entries_.begin(), g.entries_.end(),
              [](const GalleryEntry& a, const GalleryEntry& b) {
                  return a.person_id < b.person_id;
              });

    if (!index_path.empty()) {
        std::string save_err;
        if (!g.save(index_path, &save_err) && err) *err = save_err;
    }
    return g;
}

// ---------------------------------------------------------------------------
// 持久化：行式文本（可 grep、可 diff、不引 JSON 库）
// ---------------------------------------------------------------------------

bool Gallery::save(const std::string& path, std::string* err) const {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        if (err) *err = "cannot write gallery index: " + path;
        return false;
    }
    out << kMagic << ' ' << kFormatVersion << '\n';
    out << "fingerprint " << fingerprint_ << '\n';
    out << "embedder_sha256 " << (embedder_sha256_.empty() ? "-" : embedder_sha256_) << '\n';
    out << "dim " << dim_ << '\n';
    for (const auto& e : entries_) {
        out << "entry " << normalize_entry_id(e.person_id) << ' ' << e.n_photos << ' '
            << e.dim << ' ' << (e.provenance.empty() ? "-" : normalize_entry_id(e.provenance))
            << ' ' << floats_to_b64(e.fused) << '\n';
    }
    out.flush();
    if (!out) {
        if (err) *err = "failed while writing gallery index: " + path;
        return false;
    }
    return true;
}

Gallery Gallery::load(const std::string& path, std::string* err) {
    Gallery g;
    std::ifstream in(path);
    if (!in) {
        if (err) *err = "gallery index not found: " + path;
        return g;
    }
    std::string line;
    if (!std::getline(in, line)) {
        if (err) *err = "empty gallery index: " + path;
        return g;
    }
    {
        std::istringstream head(line);
        std::string magic, version;
        head >> magic >> version;
        if (magic != kMagic || version != kFormatVersion) {
            if (err) *err = "unsupported gallery index format: " + path;
            return g;
        }
    }
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string kind;
        ss >> kind;
        if (kind == "fingerprint") {
            ss >> g.fingerprint_;
        } else if (kind == "embedder_sha256") {
            ss >> g.embedder_sha256_;
            if (g.embedder_sha256_ == "-") g.embedder_sha256_.clear();
        } else if (kind == "dim") {
            ss >> g.dim_;
        } else if (kind == "entry") {
            GalleryEntry e;
            std::string provenance, b64;
            ss >> e.person_id >> e.n_photos >> e.dim >> provenance >> b64;
            if (e.person_id.empty() || e.dim <= 0 || b64.empty()) {
                if (err) *err = "malformed gallery entry in " + path;
                return Gallery{};
            }
            if (provenance != "-") e.provenance = provenance;
            if (!b64_to_floats(b64, static_cast<size_t>(e.dim), e.fused)) {
                if (err) *err = "corrupt feature blob in " + path;
                return Gallery{};
            }
            if (g.dim_ == 0) g.dim_ = e.dim;
            if (e.dim != g.dim_) {
                if (err) *err = "mixed feature dims in " + path;
                return Gallery{};
            }
            g.entries_.push_back(std::move(e));
        }
        // 未知行忽略：向前兼容（未来加字段不会让旧二进制读不动）
    }
    return g;
}

}  // namespace facerec