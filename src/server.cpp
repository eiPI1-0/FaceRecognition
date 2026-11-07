#include "server.hpp"

#include "auth.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sys/socket.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "api.hpp"
#include "encoding.hpp"
#include "json_util.hpp"
#include "pipeline_pool.hpp"
#include "metrics.hpp"
#include "result_cache.hpp"
#include "sha256.hpp"

// httplib 对 `Content-Type: application/x-www-form-urlencoded` 的请求体有一道**独立于**
// set_payload_max_length 的 8192 字节硬上限（httplib.h:6160），超了直接 413。图像请求体
// 经常是几 MB：客户端只要没显式写 image/*（curl --data-binary 的默认 Content-Type 就是
// form-urlencoded），就会收到一个与图片大小毫无关系的 413。真正的上限是运行时设置的
// set_payload_max_length，所以这里把编译期那道抬到不可能触发。
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH (64 * 1024 * 1024)
#include "httplib.h"

namespace facerec {

namespace {

// 请求体 -> 图像字节。两种输入形态：
//   1) Content-Type: image/*       请求体就是图片字节（最省带宽，推荐）
//   2) Content-Type: application/json  {"image_b64": "..."}
bool image_bytes_from_request(const httplib::Request& req, std::string& bytes,
                              std::string& err) {
    const std::string ct = req.get_header_value("Content-Type");
    if (ct.find("application/json") != std::string::npos) {
        const std::string b64 = json::string(req.body, "image_b64");
        if (b64.empty()) {
            err = "missing field: image_b64";
            return false;
        }
        bytes = base64_decode(b64);
    } else {
        bytes = req.body;
    }
    if (bytes.empty()) {
        err = "empty image body";
        return false;
    }
    return true;
}

cv::Mat decode_image(const std::string& bytes, int max_pixels, std::string& err,
                     double& ms_decode) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uchar> buf(bytes.begin(), bytes.end());
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    ms_decode = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
    if (img.empty()) {
        err = "cannot decode image (unsupported or corrupt)";
        return {};
    }
    if (max_pixels > 0 && img.total() > static_cast<size_t>(max_pixels)) {
        err = "image too large: " + std::to_string(img.total()) +
              " pixels (limit " + std::to_string(max_pixels) + ")";
        return {};
    }
    return img;
}

// 一次识别请求的指标：脸数、已知/未知、分数分布、各阶段耗时、端到端耗时。
void observe_result(Metrics& m, const IdentifyResult& r, double decode_ms,
                    std::chrono::steady_clock::duration elapsed) {
    m.faces_total.inc(r.faces.size());
    for (const auto& f : r.faces) {
        if (f.matched) {
            m.matches_known_total.inc();
        } else {
            m.matches_unknown_total.inc();
        }
        // 分数是 cosine；负值（完全不像）落进第一个桶，趋势仍然可读。
        if (!f.top_k.empty()) m.match_score.observe(f.top_k.front().score);
    }
    if (decode_ms > 0) m.decode_ms.observe(decode_ms);
    m.detect_ms.observe(r.ms_detect);
    m.embed_ms.observe(r.ms_embed);
    m.request_ms.observe(std::chrono::duration<double, std::milli>(elapsed).count());
}

void set_json(httplib::Response& res, int status, const std::string& body) {
    res.status = status;
    res.set_content(body, "application/json; charset=utf-8");
}

IdentifyOptions options_from_json(const std::string& body) {
    IdentifyOptions opt;
    double num = 0;
    if (json::number(body, "top_k", num)) opt.top_k = static_cast<int>(num);
    if (json::number(body, "min_score", num)) opt.min_score = static_cast<float>(num);
    bool flag = false;
    if (json::boolean(body, "include_crops", flag)) opt.include_crops = flag;
    if (json::boolean(body, "include_embedding", flag)) opt.include_embedding = flag;
    if (json::boolean(body, "include_landmarks", flag)) opt.include_landmarks = flag;
    for (const auto& inc : json::string_array(body, "include")) {
        if (inc == "crops") opt.include_crops = true;
        else if (inc == "embedding") opt.include_embedding = true;
        else if (inc == "landmarks") opt.include_landmarks = true;
    }
    if (opt.top_k <= 0) opt.top_k = 1;
    if (opt.top_k > 100) opt.top_k = 100;
    // 请求级检测尺寸：{"detector_input":[w,h]} 或 {"detector_input":{"w":..,"h":..}}
    const std::string sub = json::object(body, "detector_input");
    if (!sub.empty()) {
        int w = 0, h = 0;
        if (json::integer(sub, "w", w) && json::integer(sub, "h", h)) {
            opt.detector_w = w;
            opt.detector_h = h;
        }
    } else {
        const auto arr = json::string_array(body, "detector_input");
        (void)arr;
    }
    return opt;
}

// SIGINT/SIGTERM -> 停止接受新连接并退出 listen 循环。
// 信号handler 里只能做 async-signal-safe 的事（不能分配、不能加锁、不能写 iostream），
// 所以只能提前把一个裸指针交出去，handler 只做 svr->stop() 这一件事。
httplib::Server* g_server = nullptr;

void on_signal(int) {
    if (g_server) g_server->stop();
}

}  // namespace

int listen_with_signals(::httplib::Server& svr, const std::string& host, int port,
                        bool reuseport) {
    // 端口冲突的策略必须显式，不能交给第三方库的默认值。实测（本机 Linux）：httplib 的
    // default_socket_options（httplib.h:1867）在 Linux 上开的是 SO_REUSEPORT 而**不是**
    // SO_REUSEADDR，于是第二个进程绑同一个端口不会失败，而是与老进程**静默地各接一半连接**。
    // 对服务端这是最坏的一种"能用"：改代码重启后老进程还在，一半流量打到旧二进制，
    // 现象是"时灵时不灵"，日志里什么都没有。
    //   默认：SO_REUSEADDR 开（重启不受 TIME_WAIT 影响）+ SO_REUSEPORT 关
    //         -> 端口被活着的监听者占用时立刻 EADDRINUSE，启动失败并说明原因。
    //   --reuseport：显式打开，用于多进程同端口横向扩展（见 deploy/facerec@.service）。
    svr.set_socket_options([reuseport](socket_t sock) {
        const int yes = 1, on = reuseport ? 1 : 0;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&yes), sizeof(yes));
        setsockopt(sock, SOL_SOCKET, SO_REUSEPORT,
                   reinterpret_cast<const char*>(&on), sizeof(on));
    });
    // Nagle 必须关。httplib 的 CPPHTTPLIB_TCP_NODELAY 默认是 **false**（httplib.h:90），
    // 于是 accept 出来的每个连接都开着 Nagle；而响应是**两段**写出的（header 一次 send、
    // body 一次 send），第二段是个小包，撞上客户端的延迟 ACK 就要等 ~40ms 才发出去。
    // 实测（本机，裸 socket vs http.client 打 /healthz，服务端不做任何推理）：
    //   裸 socket 单连接   p50 = 0.10 ms
    //   http.client keep-alive  p50 = 41.0 ms，strace 里第二段 body 被扣住 41.5 ms
    // 这 41ms 是**每个响应**都付的固定成本，跟模型、并发、缓存都无关；之前记录的所有
    // 延迟数字里都含着它。关掉后同一条链路与裸 socket 持平。
    svr.set_tcp_nodelay(true);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    // httplib 的写路径没有用 MSG_NOSIGNAL，向已关闭的连接写数据会送 SIGPIPE，
    // 而它的默认动作是终止进程。这里不是“防御性编程”：不屏蔽的话，一个客户端
    // 提前断开就能带走整个服务。
    std::signal(SIGPIPE, SIG_IGN);
    g_server = &svr;
    const bool ok = svr.listen(host, port);
    g_server = nullptr;
    return ok ? 0 : 1;
}

int run_server(const PipelineConfig& cfg, const ServerOptions& opt) {
    // ---- 鉴权与审计 ----
    // 安全是绑定约束而不是开关：没有 token 表就只允许绑 loopback。这样"忘了配鉴权又对外监听"
    // 这个最坏组合在启动阶段就不可能成立（PLAN.md D3）。
    TokenTable auth;
    if (!cfg.auth_tokens_file.empty()) {
        std::string err;
        if (!auth.load(cfg.auth_tokens_file, err)) {
            std::cerr << "Fatal: " << err << std::endl;
            return -1;
        }
    }
    if (auth.empty() && !is_loopback_host(opt.host)) {
        std::cerr << "Refusing to listen on " << opt.host << " without authentication.\n"
                  << "  Either bind loopback (--host 127.0.0.1) and put a reverse proxy in "
                     "front, or set auth.tokens_file.\n"
                  << "  Create a token table:  ./main token --name admin --tokens-file "
                     "./tokens.txt" << std::endl;
        return -1;
    }
    AuditSink audit;
    {
        std::string err;
        if (!audit.configure(cfg.log_audit, err)) {
            std::cerr << "Fatal: log.audit: " << err << std::endl;
            return -1;
        }
    }

    // 并发度 = 核数（默认）。CPU 上超过核数只会增加排队与上下文切换；
    // OpenCV 内部线程池在 P4 会单独压成 1，这里先只控制 worker 数。
    int workers = opt.workers > 0 ? opt.workers
                                  : static_cast<int>(cv::getNumberOfCPUs());
    if (workers < 1) workers = 1;

    // 并发模型是"多实例并行"，不是"一个 Net 内并行"：把 OpenCV 的内部线程池压到 1，
    // 让每个 worker 独占一个核（`--cv-threads` 可改）。实测（Release，cache off，
    // 4×W 并发客户端，本机 16 核，单脸图像）：
    //   workers=4  ≈135 qps / p50 74ms  / max 117ms
    //   workers=8  ≈193 qps / p50 100ms / max 150ms
    //   workers=16 ≈187 qps / p50 185ms / max 307ms   (64 客户端，已超订)
    // 再往上加 worker 只增排队（每 pipeline ≈7MB RSS + 一核）；要再快就多进程。
    // 注：早先记的 165/191/175 是在 httplib 默认 listen backlog=5 下测的 —— 当时并发
    // 恰好没超队列容量，所以 qps 接近，但那个 1.0s 的连接尾巴一直藏着。
    cv::setNumThreads(opt.cv_threads > 0 ? opt.cv_threads : 1);

    std::cout << "Building " << workers << " pipeline instance(s)..." << std::endl;
    PipelinePool pool(cfg, workers);
    std::cout << "Ready: " << pool.size() << " worker(s), gallery "
              << pool.gallery_size() << " person(s) from " << cfg.data_dir << std::endl;

    // 指纹在启动时算一次：gallery 在进程生命周期内不变，缓存必须跟着它失效。
    ResultCache cache(pool.fingerprint(), static_cast<size_t>(opt.cache_entries));
    Metrics metrics;
    std::cout << "Result cache: "
              << (cache.enabled() ? std::to_string(opt.cache_entries) + " entries" : "disabled")
              << std::endl;

    httplib::Server svr;
    svr.new_task_queue = [workers] { return new httplib::ThreadPool(workers); };
    svr.set_payload_max_length(static_cast<size_t>(opt.max_request_mb) * 1024 * 1024);
    svr.set_read_timeout(10, 0);
    svr.set_write_timeout(30, 0);
    svr.set_keep_alive_max_count(100);

    // 鉴权 + 结构化审计（PLAN.md D1/D3）：探针免鉴权，其余全要 bearer；
    // 401 永远记审计，成功行受 --quiet 影响。
    install_access_control(svr, auth, opt.quiet, audit, /*is_mcp=*/false, &metrics);

    // ---- 指标 ----
    svr.Get("/metrics", [&metrics, &cache](const httplib::Request&, httplib::Response& res) {
        // 缓存命中率来自 cache 自己的原子计数，比在这里重复计数少一处偏差。
        const ResultCache::Stats st = cache.stats();
        std::string body = metrics.to_prometheus();
        body += "# HELP facerec_cache_entries Resident cache entries\n";
        body += "# TYPE facerec_cache_entries gauge\nfacerec_cache_entries " +
                std::to_string(st.entries) + "\n";
        body += "# HELP facerec_cache_bytes Resident cache bytes\n";
        body += "# TYPE facerec_cache_bytes gauge\nfacerec_cache_bytes " +
                std::to_string(st.bytes) + "\n";
        res.set_content(body, "text/plain; version=0.0.4; charset=utf-8");
    });

    // ---- 探针 ----
    svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        set_json(res, 200, "{\"status\":\"ok\"}");
    });

    svr.Get("/readyz", [&cfg, &cache](const httplib::Request&, httplib::Response& res) {
        const ResultCache::Stats st = cache.stats();
        std::ostringstream os;
        os << "{\"status\":\"ready\",\"model\":{\"detector\":\""
           << json_escape(cfg.detector_model) << "\",\"embedder\":\""
           << json_escape(cfg.embedder_model) << "\"},\"detector_input\":["
           << cfg.detector_input_w << "," << cfg.detector_input_h << "]"
           << ",\"cache\":{\"hits\":" << st.hits << ",\"misses\":" << st.misses
           << ",\"entries\":" << st.entries << ",\"bytes\":" << st.bytes << "}}";
        set_json(res, 200, os.str());
    });

    svr.Get("/v1/people", [&pool](const httplib::Request&, httplib::Response& res) {
        Lease lease(pool, std::chrono::milliseconds(2000));
        if (!lease) {
            set_json(res, 503, error_to_json("server busy", "unavailable"));
            return;
        }
        set_json(res, 200, people_to_json(*lease.get()));
    });

    // ---- 识别 ----
    svr.Post("/v1/identify", [&pool, &opt, &cache, &metrics](const httplib::Request& req,
                                                             httplib::Response& res) {
        const auto t_request = std::chrono::steady_clock::now();
        // 请求数与错误数不在这里计：auth.cpp 的访问日志钩子按 {path,status} 统一计，
        // 否则 400 bad_request / 503 server busy 这类早退分支必然漏计。
        std::string bytes, err;
        if (!image_bytes_from_request(req, bytes, err)) {
            set_json(res, 400, error_to_json(err, "bad_request"));
            return;
        }
        IdentifyOptions iopt = options_from_json(req.body);
        if (iopt.detector_w == 0 && iopt.detector_h == 0) {
            iopt.detector_w = 0;   // 用配置默认值
        }
        if (!valid_detector_size(iopt.detector_w, iopt.detector_h, &err)) {
            set_json(res, 400, error_to_json(err, "bad_request"));
            return;
        }

        // 缓存键算在**解码之前**：命中路径连 imdecode 都省掉。
        const std::string key =
            cache.make_key(sha256_hex(bytes), iopt, iopt.detector_w, iopt.detector_h);
        IdentifyResult cached;
        if (cache.get(key, cached)) {
            metrics.cache_hits_total.inc();
            observe_result(metrics, cached, 0.0,
                           std::chrono::steady_clock::now() - t_request);
            res.set_header("X-Cache", "HIT");
            set_json(res, 200, identify_to_json(cached, iopt));
            return;
        }

        double ms_decode = 0;
        cv::Mat img = decode_image(bytes, opt.max_image_pixels, err, ms_decode);
        if (img.empty()) {
            set_json(res, 400, error_to_json(err, "bad_image"));
            return;
        }

        Lease lease(pool, std::chrono::milliseconds(5000));
        if (!lease) {
            set_json(res, 503, error_to_json("server busy", "unavailable"));
            return;
        }
        try {
            IdentifyResult result = lease.get()->identify(img, iopt, "request");
            result.ms_decode = ms_decode;
            cache.put(key, result);
            metrics.cache_misses_total.inc();
            observe_result(metrics, result, ms_decode,
                           std::chrono::steady_clock::now() - t_request);
            res.set_header("X-Cache", "MISS");
            set_json(res, 200, identify_to_json(result, iopt));
        } catch (const cv::Exception& e) {
            set_json(res, 500, error_to_json(e.what(), "pipeline_error"));
        }
    });

    // ---- 提特征 ----
    svr.Post("/v1/embed", [&pool, &opt](const httplib::Request& req,
                                        httplib::Response& res) {
        std::string bytes, err;
        if (!image_bytes_from_request(req, bytes, err)) {
            set_json(res, 400, error_to_json(err, "bad_request"));
            return;
        }
        double dummy = 0;
        cv::Mat img = decode_image(bytes, opt.max_image_pixels, err, dummy);
        if (img.empty()) {
            set_json(res, 400, error_to_json(err, "bad_image"));
            return;
        }
        int face_index = 0;
        double num = 0;
        if (json::number(req.body, "face_index", num)) face_index = static_cast<int>(num);

        Lease lease(pool, std::chrono::milliseconds(5000));
        if (!lease) {
            set_json(res, 503, error_to_json("server busy", "unavailable"));
            return;
        }
        try {
            cv::Mat feat = lease.get()->embed_image(img, face_index);
            if (feat.empty()) {
                set_json(res, 422, error_to_json("no face at index", "no_face"));
                return;
            }
            std::ostringstream os;
            os << "{\"status\":\"ok\",\"dim\":" << feat.total() << ",\"embedding\":[";
            const float* p = feat.ptr<float>();
            const int dims = static_cast<int>(feat.total());
            for (int i = 0; i < dims; ++i) {
                if (i) os << ",";
                os << format_float(p[i], 6);
            }
            os << "]}";
            set_json(res, 200, os.str());
        } catch (const cv::Exception& e) {
            set_json(res, 500, error_to_json(e.what(), "pipeline_error"));
        }
    });

    // ---- 比较两个特征 ----
    svr.Post("/v1/compare", [](const httplib::Request& req, httplib::Response& res) {
        // 只接受两个已归一化的特征数组：{"a":[512 floats],"b":[...]}
        auto parse_vec = [&req](const std::string& key, std::vector<float>& out) {
            const std::string pat = "\"" + key + "\"";
            size_t pos = req.body.find(pat);
            if (pos == std::string::npos) return false;
            pos = req.body.find('[', pos);
            if (pos == std::string::npos) return false;
            const size_t end = req.body.find(']', pos);
            if (end == std::string::npos) return false;
            std::stringstream ss(req.body.substr(pos + 1, end - pos - 1));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                try { out.push_back(std::stof(tok)); } catch (...) { return false; }
            }
            return !out.empty();
        };
        std::vector<float> a, b;
        if (!parse_vec("a", a) || !parse_vec("b", b)) {
            set_json(res, 400, error_to_json("expected {\"a\":[...],\"b\":[...]}", "bad_request"));
            return;
        }
        if (a.size() != b.size()) {
            set_json(res, 400, error_to_json("dimension mismatch", "bad_request"));
            return;
        }
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += static_cast<double>(a[i]) * b[i];
            na += static_cast<double>(a[i]) * a[i];
            nb += static_cast<double>(b[i]) * b[i];
        }
        const double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
        set_json(res, 200, "{\"status\":\"ok\",\"cosine\":" + format_float(cos, 6) + "}");
    });

    // ---- 启动 ----
    std::cout << "Listening on http://" << opt.host << ":" << opt.port
              << "  (workers=" << workers << (opt.reuseport ? ", SO_REUSEPORT" : "") << ")"
              << std::endl;
    {
        std::ostringstream os;
        os << "\"host\":\"" << json_escape(opt.host) << "\",\"port\":"
           << opt.port << ",\"workers\":" << workers << ",\"persons\":" << pool.gallery_size()
           << ",\"gallery_fingerprint\":\"" << json_escape(pool.fingerprint())
           << "\",\"detector\":\"" << json_escape(cfg.detector_model)
           << "\",\"embedder\":\"" << json_escape(cfg.embedder_model)
           << "\",\"auth\":\"" << (auth.empty() ? "off" : std::to_string(auth.size()) + " token(s)")
           << "\"";
        audit_event(audit, "startup", os.str());
    }

    if (listen_with_signals(svr, opt.host, opt.port, opt.reuseport) != 0) {
        std::cerr << "Failed to listen on " << opt.host << ":" << opt.port
                  << (opt.reuseport ? " (SO_REUSEPORT)" : "") << std::endl;
        return 1;
    }
    std::cout << "Stopped." << std::endl;
    return 0;
}

}  // namespace facerec