// Augmentation A/B harness (PLAN.md D7).
//
// 给定一个检测器（模型路径 + 输入尺寸），对 data/ 里每个身份的注册照片做：
//   1. 检测人脸，5-点相似变换对齐到 112×112（ArcFace 参考模板）
//   2. 生成 12 个扭曲变体（±25° 旋转、0.85–1.15 缩放、微平移、亮度 ±40）
//   3. MobileFaceNet 提取每张变体的 512-d 特征
//   4. 统计同身份内 cosine 均值 vs 跨身份 cosine 均值，margin = 内 - 跨
//
// 用 true cosine（a·b/(‖a‖‖b‖)），不依赖 CosSimilarity::similarity 的
// "stored 侧已 unit-norm" 假设。mt19937 seed 12345 保证可复现。
//
// 用法:
//   augmentation_ab <det_model> <det_w> <det_h> [data_dir]
//
// 示例:
//   augmentation_ab models/yunet_120x160.onnx 160 120   # fused 默认
//   augmentation_ab models/face_detection_yunet_2023mar.onnx 320 320  # per-level
//
// 预期输出（fused 基线，来自 AGENTS.md）：within ≈0.845, between ≈−0.029, margin ≈0.874

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/engine.hpp"
#include "core/face_def.hpp"

// .cpp 里用 using-directive（头文件里绝不）：内核在 `facerec::`，见 RFPLAN §6.1。
using namespace facerec;

namespace fs = std::filesystem;

namespace {

std::mt19937 rng(12345);

double true_cosine(const cv::Mat& a, const cv::Mat& b) {
    double na = cv::norm(a), nb = cv::norm(b);
    if (na < 1e-9 || nb < 1e-9) return 0.0;
    return a.dot(b) / (na * nb);
}

// 给对齐后的 112×112 人脸图生成一组 warp 变体（含原图）。
std::vector<cv::Mat> warp_variants(const cv::Mat& aligned, int n = 12) {
    std::vector<cv::Mat> out;
    out.push_back(aligned.clone());  // 索引 0：原图

    const int h = aligned.rows, w = aligned.cols;
    const cv::Point2f center(w * 0.5f, h * 0.5f);

    for (int i = 0; i < n; ++i) {
        // 旋转 ±25°，缩放 0.85–1.15，平移 ±8 px，亮度 ±40
        double angle = (rng() % 5000) / 10000.0 * 50.0 - 25.0;
        double scale = 0.85 + (rng() % 3000) / 10000.0 * 0.30;
        float tx = (rng() % 1700) / 10000.0 * 16.0f - 8.0f;
        float ty = (rng() % 1700) / 10000.0 * 16.0f - 8.0f;
        double brightness = (rng() % 8001) / 10000.0 * 80.0 - 40.0;

        cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);
        M.at<double>(0, 2) += tx;
        M.at<double>(1, 2) += ty;

        cv::Mat warped;
        cv::warpAffine(aligned, warped, M, {w, h}, cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        if (std::abs(brightness) > 0.5) {
            warped.convertTo(warped, -1, 1.0, brightness);
        }
        out.push_back(std::move(warped));
    }
    return out;
}

// 给定检测器配置，对 data_dir 下每个身份注册，返回 {within_mean, between_mean, margin}。
struct ABResult {
    double within_mean = 0.0;
    double between_mean = 0.0;
    double margin = 0.0;
    int n_within = 0, n_between = 0;
};

ABResult run_ab(const std::string& det_model, int det_w, int det_h,
                const std::string& data_dir) {
    DNNConfig cfg = DNNConfig::cpu();
    Detector detector(det_model, {det_w, det_h}, 0.6f, 0.3f, 750, cfg);
    FeatureExtractor extractor("models/mobilefacenet.onnx", {112, 112}, cfg);

    // 每人：所有变体的特征向量
    std::vector<std::vector<cv::Mat>> per_person_feats;
    std::vector<std::string> person_names;

    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (!entry.is_directory()) continue;
        const std::string person = entry.path().filename().string();
        std::vector<cv::Mat> person_feats;

        for (const auto& photo : fs::directory_iterator(entry.path())) {
            if (!photo.is_regular_file()) continue;
            cv::Mat img = cv::imread(photo.path().string(), cv::IMREAD_COLOR);
            if (img.empty()) continue;

            std::vector<Face> dets = detector.detect(img);
            if (dets.empty()) continue;

            // 取第一张检测到的人脸对齐
            cv::Mat aligned = align_detection(img, dets[0], {det_w, det_h}, {112, 112},
                                              cv::BORDER_CONSTANT);
            if (aligned.empty()) continue;

            auto variants = warp_variants(aligned, 12);
            for (const auto& v : variants) {
                cv::Mat feat = extractor.extract(v);
                if (!feat.empty()) person_feats.push_back(std::move(feat));
            }
        }

        if (!person_feats.empty()) {
            per_person_feats.push_back(std::move(person_feats));
            person_names.push_back(person);
            std::printf("  %s: %d variant embeddings\n", person.c_str(),
                        (int)per_person_feats.back().size());
        }
    }

    // 同身份内两两 cosine
    double within_sum = 0.0;
    int within_n = 0;
    for (const auto& feats : per_person_feats) {
        for (size_t i = 0; i < feats.size(); ++i) {
            for (size_t j = i + 1; j < feats.size(); ++j) {
                within_sum += true_cosine(feats[i], feats[j]);
                ++within_n;
            }
        }
    }

    // 跨身份两两 cosine
    double between_sum = 0.0;
    int between_n = 0;
    for (size_t i = 0; i < per_person_feats.size(); ++i) {
        for (size_t j = i + 1; j < per_person_feats.size(); ++j) {
            for (const auto& a : per_person_feats[i]) {
                for (const auto& b : per_person_feats[j]) {
                    between_sum += true_cosine(a, b);
                    ++between_n;
                }
            }
        }
    }

    ABResult r;
    r.within_mean = within_n ? within_sum / within_n : 0.0;
    r.between_mean = between_n ? between_sum / between_n : 0.0;
    r.margin = r.within_mean - r.between_mean;
    r.n_within = within_n;
    r.n_between = between_n;
    return r;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "Usage: %s <det_model> <det_w> <det_h> [data_dir]\n"
                     "  Example: %s models/yunet_120x160.onnx 160 120\n"
                     "           %s models/face_detection_yunet_2023mar.onnx 320 320\n",
                     argv[0], argv[0], argv[0]);
        return 1;
    }

    std::string det_model = argv[1];
    int det_w = std::atoi(argv[2]);
    int det_h = std::atoi(argv[3]);
    std::string data_dir = (argc > 4) ? argv[4] : "data/";

    std::printf("=== Augmentation A/B ===\n");
    std::printf("Detector: %s (%dx%d)\n", det_model.c_str(), det_w, det_h);
    std::printf("Data dir: %s\n", data_dir.c_str());
    std::printf("Embedder: models/mobilefacenet.onnx (112x112, 512-d)\n");
    std::printf("Warp seed: 12345, 12 warps + original per identity\n\n");

    ABResult r = run_ab(det_model, det_w, det_h, data_dir);

    std::printf("\n=== Result ===\n");
    std::printf("within-identity mean cosine : %+.6f (%d pairs)\n", r.within_mean,
                r.n_within);
    std::printf("between-identity mean cosine: %+.6f (%d pairs)\n", r.between_mean,
                r.n_between);
    std::printf("margin                      : %+.6f\n", r.margin);
    std::printf("\nFused baseline (AGENTS.md): within +0.845, between −0.029, margin +0.874\n");

    return 0;
}
