#include "cli/cli.hpp"
#include "cli/args.hpp"

#include "app/api.hpp"
#include "app/config.hpp"
#include "app/evaluation.hpp"
#include "serve/pipeline_pool.hpp"
#include "app/gallery.hpp"
#include "core/engine.hpp"
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <thread>
#include <random>
#include <charconv>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// .cpp 里用 using-directive（头文件里绝不）：内核在 `facerec::`，见 RFPLAN §6.1。
using namespace facerec;

using namespace cli;   // 实现文件内引入，不污染任何头文件

void print_evaluate_usage() {
    std::cout <<
        "Usage: facerec evaluate <dataset-dir> [options]\n"
        "  <dataset-dir> must contain one subdirectory per identity, each holding\n"
        "  that person's photos (same layout as ./data/, but with held-out photos).\n"
        "\n"
        "  Per identity the sorted photo list is split: first --gallery-photos become\n"
        "  the enrollment side, the next --probe-photos are held out and NEVER enrolled.\n"
        "  Scores are produced by the production path (letterbox + YuNet + landmark\n"
        "  align + the configured embedder), so the thresholds apply to the deployed\n"
        "  pipeline, not to a reimplementation.\n"
        "\n"
        "  --gallery-photos N  photos enrolled per identity (default 2)\n"
        "  --probe-photos N    held-out photos per identity (default 2)\n"
        "  --limit N           evaluate at most N identities (0 = all, default 0)\n"
        "  --seed N            seed for the --limit sample (default 12345)\n"
        "  --workers N         pipelines in parallel (default 4)\n"
        "  --csv <path>        also write the threshold table as CSV\n"
        "  --config/--set      normal config plumbing (models, detector input, ...)\n"
        "\n"
        "Reports FAR/FRR for BOTH matcher semantics: 1:1 (= /v1/compare, one comparison)\n"
        "and 1:N (= /v1/identify, max over the whole gallery). A threshold tuned for one\n"
        "is wrong for the other, and 1:N gets worse as the gallery grows.\n";
}

namespace {

// 图片后缀判定只写一次：照片挑选与作业表必须用同一个规则，否则"这一层过滤掉了
// .jpeg，那一层又收进来"会让注册侧/留出侧的划分依赖于书写顺序。
bool is_image_path(const std::string& p) {
    std::string low = p;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);
    for (const char* e : {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".pbm"}) {
        const size_t n = std::strlen(e);
        if (low.size() > n && low.compare(low.size() - n, n, e) == 0) return true;
    }
    return false;
}

}  // namespace

int cmd_evaluate(int argc, char* argv[]) {
    int gallery_photos = 2;
    int probe_photos = 2;
    int limit = 0;
    int workers = 4;
    unsigned seed = 12345;
    std::string dataset, csv_path;
    ConfigArgs cfg_args;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take_int = [&](const std::string& prefix, int* out) -> bool {
            // arg == prefix（值在下一个 argv）时 substr(prefix.size()+1) 会越界抛
            // out_of_range：先判长度。
            std::string v = (arg.size() > prefix.size() + 1) ? arg.substr(prefix.size() + 1)
                                                             : std::string();
            if (v.empty()) {
                if (i + 1 >= argc) { std::cerr << prefix << " needs a value" << std::endl; return false; }
                v = argv[++i];
            }
            const char* b = v.c_str();
            const char* e = b + v.size();
            const auto res = std::from_chars(b, e, *out);
            if (res.ec != std::errc() || res.ptr != e) {
                std::cerr << prefix << " needs an integer, got: " << v << std::endl;
                return false;
            }
            return true;
        };
        bool is_prefixed = false;
        for (const std::string p : {"--gallery-photos", "--probe-photos", "--limit", "--workers"}) {
            if (arg == p || arg.rfind(p + "=", 0) == 0) {
                is_prefixed = true;
                int* tgt = (p == "--gallery-photos") ? &gallery_photos
                         : (p == "--probe-photos") ? &probe_photos
                         : (p == "--limit") ? &limit : &workers;
                if (!take_int(p, tgt)) return -1;
            }
        }
        if (is_prefixed) continue;
        if (arg == "--seed" || arg.rfind("--seed=", 0) == 0) {
            int s = 0;
            if (!take_int("--seed", &s)) return -1;
            seed = static_cast<unsigned>(s < 0 ? 0 : s);
        } else if (arg == "--csv" || arg.rfind("--csv=", 0) == 0) {
            csv_path = (arg == "--csv") ? ((i + 1 < argc) ? argv[++i] : std::string())
                                        : arg.substr(6);
        } else if (arg == "--config" || arg.rfind("--config=", 0) == 0) {
            if (!parse_config_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "--set" || arg.rfind("--set=", 0) == 0) {
            if (!parse_set_arg(arg, i, argc, argv, cfg_args)) return -1;
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg.rfind("-", 0) != 0) {
            dataset = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return -1;
        }
    }

    if (show_help) { print_evaluate_usage(); return 0; }
    if (dataset.empty()) {
        std::cerr << "evaluate needs <dataset-dir> (see --help)" << std::endl;
        return -1;
    }
    if (gallery_photos < 1 || probe_photos < 1 || workers < 1) {
        std::cerr << "--gallery-photos, --probe-photos and --workers must be >= 1" << std::endl;
        return -1;
    }

    PipelineConfig cfg;
    if (!load_pipeline_config(cfg_args, cfg)) return -1;
    if (!check_assets(cfg)) return -1;

    // 评测的 gallery 是自己算的（注册侧/留出侧的划分正是被测对象），所以必须断开
    // Pipeline 自带的 gallery：否则 (1) 每个 worker 都去注册一遍数据集，(2) 更糟的是
    // 可能把 build/gallery.idx 覆写成空库 —— 那是线上缓存文件，评测进程不该碰它。
    cfg.data_dir.clear();
    cfg.gallery_index.clear();

    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_directory(dataset, ec)) {
        std::cerr << "not a directory: " << dataset << std::endl;
        return -1;
    }
    if (dataset.back() != '/') dataset += '/';

    // 身份选择先于照片选择：--limit 抽样必须在**所有人**里做，否则只从字母表前段
    // 取人会把样本偏置（LFW 前段是同一批名人）。
    std::vector<std::string> ids = list_person_dirs(dataset);
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](const std::string& id) {
        std::vector<cv::String> found;
        cv::glob(dataset + id + "/*", found, false);   // glob 返回 void，只能拿输出参数
        int photos = 0;
        for (const auto& p : found)
            if (is_image_path(std::string(p))) ++photos;
        return photos < gallery_photos + probe_photos;
    }), ids.end());
    std::sort(ids.begin(), ids.end());

    const size_t total_ids = ids.size();
    if (limit > 0 && static_cast<size_t>(limit) < total_ids) {
        std::mt19937 rng(seed);
        std::shuffle(ids.begin(), ids.end(), rng);
        ids.resize(static_cast<size_t>(limit));
        std::sort(ids.begin(), ids.end());   // 抽样后再排序，照片切分仍按名字确定
    }
    if (ids.empty()) {
        std::cerr << "no identity in " << dataset << " has " << gallery_photos
                  << "+" << probe_photos << " readable photos" << std::endl;
        return 1;
    }

    // 作业表：(identity index, photo path)。imread 的成本差异大，所以用共享原子游标
    // 而不是静态切片；每个线程独占一个 Pipeline（cv::dnn::Net 不是线程安全的）。
    struct Job { size_t who; std::string path; };
    std::vector<Job> jobs;
    for (size_t k = 0; k < ids.size(); ++k) {
        std::vector<cv::String> found;
        cv::glob(dataset + ids[k] + "/*", found, false);
        for (const auto& p : found) {
            std::string s = std::string(p);
            if (is_image_path(s)) jobs.push_back({k, s});
        }
    }
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
        if (a.who != b.who) return a.who < b.who;
        return a.path < b.path;
    });

    std::vector<std::vector<float>> feats(jobs.size());
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0}, usable{0};
    const auto t0 = std::chrono::steady_clock::now();
    std::cerr << "embedding " << jobs.size() << " photos from " << ids.size()
              << " identities (" << workers << " pipelines)..." << std::endl;

    {
        facerec::PipelinePool pool(cfg, workers);
        const int nthreads = static_cast<int>(pool.size());
        std::vector<std::thread> threads;
        threads.reserve(nthreads);
        for (int w = 0; w < nthreads; ++w) {
            threads.emplace_back([&] {
                // 一个 worker 全程只借一次：借两次会在两次之间让出 pipeline，白白多一次
                // 排队，而且这里的超时语义应该是“永不放弃”。
                facerec::Lease lease(pool, std::chrono::milliseconds(24 * 3600 * 1000));
                if (!lease) return;
                size_t j;
                while ((j = next.fetch_add(1)) < jobs.size()) {
                    cv::Mat img = cv::imread(jobs[j].path, cv::IMREAD_COLOR);
                    if (!img.empty()) {
                        // face_index 0 = 置信度最高的脸：与 /v1/compare 和 compare_faces
                        // 完全同一个取脸规则，换规则就等于换评测。
                        cv::Mat f = lease.get()->embed_image(img, 0);
                        if (!f.empty() && !f.isContinuous()) f = f.clone();
                        if (!f.empty()) {
                            feats[j].assign(f.begin<float>(), f.end<float>());
                            usable.fetch_add(1);
                        }
                    }
                    const size_t d = done.fetch_add(1) + 1;
                    if (d % 500 == 0) {
                        const double s = std::chrono::duration<double>(
                                             std::chrono::steady_clock::now() - t0).count();
                        std::cerr << "  " << d << "/" << jobs.size() << "  "
                                  << static_cast<long>(s) << "s" << std::endl;
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    // 按身份切分：只用**有可用脸**的照片，注册侧与留出侧不相交。
    std::vector<facerec::EvalPerson> people(ids.size());
    for (size_t k = 0; k < ids.size(); ++k) people[k].name = ids[k];
    for (size_t j = 0; j < jobs.size(); ++j) {
        if (feats[j].empty()) continue;
        auto& p = people[jobs[j].who];
        if (static_cast<int>(p.gallery.size()) < gallery_photos) p.gallery.push_back(feats[j]);
        else if (static_cast<int>(p.probe.size()) < probe_photos) p.probe.push_back(feats[j]);
    }
    size_t dropped = 0;
    for (auto& p : people) {
        if (static_cast<int>(p.gallery.size()) < gallery_photos || p.probe.empty()) {
            p.gallery.clear();
            p.probe.clear();
            ++dropped;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    std::cerr << "embedded in " << std::chrono::duration<double>(t1 - t0).count()
              << "s; usable faces " << usable.load() << "/" << jobs.size()
              << ", identities dropped " << dropped << std::endl;

    // 扫描点里带上当前默认阈值：运维要看得见“线上这个值落在表的哪一行”。
    // float 0.45 转 double 是 0.449999988，与扫描点的 0.45 不是同一个数：不吸附就会在表
    // 里并排出现两行同义阈值（CSV 里一行印成 0.44999999），读表的人会以为那是两个不同的
    // 测量。吸附只在本就落在网格上时发生；0.375 这种默认值仍原样进表。
    std::vector<double> ts;
    for (int i = 1; i <= 19; ++i) ts.push_back(i * 0.05);   // 累加会漂，乘法不漂
    const double t_grid = std::round(static_cast<double>(cfg.recogn_thresh) * 20.0) / 20.0;
    ts.push_back(std::fabs(t_grid - cfg.recogn_thresh) < 1e-4 ? t_grid
                                                              : static_cast<double>(cfg.recogn_thresh));
    std::sort(ts.begin(), ts.end());
    ts.erase(std::unique(ts.begin(), ts.end()), ts.end());

    const facerec::EvalReport rep = facerec::evaluate_dataset(people, ts);
    std::cout << rep.to_text(rep.persons);
    std::cout << "\ncurrent default recognize.thresh = "
              << facerec::format_float(cfg.recogn_thresh, 4) << "\n";
    for (const auto& r : rep.rows) {
        if (std::abs(r.thresh - cfg.recogn_thresh) < 1e-9) {
            std::printf("  at that threshold: FAR 1:1=%.6f  FRR 1:1=%.6f  "
                        "FAR 1:N=%.6f  FRR 1:N=%.6f\n",
                        r.far_11, r.frr_11, r.far_1n, r.frr_1n);
        }
    }
    if (total_ids > ids.size()) {
        std::cout << "note: sampled " << ids.size() << " of " << total_ids
                  << " eligible identities (seed " << seed << ")\n";
    }
    if (!csv_path.empty()) {
        std::ofstream out(csv_path);
        if (!out) { std::cerr << "cannot write csv: " << csv_path << std::endl; return 1; }
        out << rep.to_csv(rep.persons);
        std::cerr << "csv: " << csv_path << std::endl;
    }
    return 0;
}
