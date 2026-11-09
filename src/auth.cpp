#include "auth.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <random>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <atomic>
#include <sstream>

#include "admission.hpp"  // 本请求已为"等容量"等了多久（队列段与池段共用）
#include "api.hpp"      // json_escape（api 层已经是所有外壳的共同依赖）
#include "metrics.hpp"    // 请求计数：唯一计数点在本文件的访问日志钩子
#include "sha256.hpp"

namespace facerec {

thread_local std::string tl_caller;
thread_local std::chrono::steady_clock::time_point tl_request_start;

const std::string& current_caller() { return tl_caller; }

namespace {

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool is_hex64(const std::string& s) {
    if (s.size() != 64) return false;
    for (char c : s) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!ok) return false;
    }
    return true;
}

// 常量时间比较：不因为第一个不同字节就提前返回，避免用比对耗时逐字节猜 token。
// volatile 防止编译器把它优化成 memcmp。
bool ct_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    volatile unsigned char diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = static_cast<unsigned char>(diff | (static_cast<unsigned char>(a[i] ^ b[i])));
    }
    return diff == 0;
}

// `Authorization: Bearer <token>`，大小写不敏感地匹配 scheme。
std::string bearer(const httplib::Request& req) {
    const std::string auth = req.get_header_value("Authorization");
    if (auth.size() < 7) return {};
    std::string scheme = auth.substr(0, 7);
    std::transform(scheme.begin(), scheme.end(), scheme.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    if (scheme != "bearer ") return {};
    return trim(auth.substr(7));
}

bool is_probe_path(const std::string& path) { return path == "/healthz" || path == "/readyz"; }

// 不 shed 的路径：探针 + 指标。两者都不借 pipeline， shed 它们不会减轻任何负载；
// 而饱和时杀掉探针会把一次过载变成一次重启（k8s 看到的是 pod 挂，不是服 503），
// 杀掉 /metrics 则恰好抹掉诊断过载所需的数据。
bool sheddable_path(const std::string& path) {
    return !is_probe_path(path) && path != "/metrics";
}

// 把一次请求折成稳定基数的标签对。**不能**把 req.path 原样当标签：任何客户端循环打
// /random-N 都能给 map 加一条永久记录，把 /metrics 变成内存与磁盘的双重炸弹。
// 白名单是 serve 与 mcp 全部公开路由的并集；其余（含扫描器探路、404）一律归 other。
std::string metric_labels(const std::string& path, int status) {
    static const char* kKnown[] = {"/v1/identify", "/v1/embed", "/v1/compare", "/v1/people",
                                   "/mcp",         "/healthz", "/readyz",    "/metrics"};
    const char* slot = "other";
    for (const char* p : kKnown) {
        if (path == p) {
            slot = p;
            break;
        }
    }
    std::string out = "path=\"";
    out += slot;
    out += "\",status=\"";
    out += std::to_string(status);
    out += '"';
    return out;
}

}  // namespace

bool is_loopback_host(const std::string& host) {
    return host == "127.0.0.1" || host == "localhost" || host == "::1" || host == "[::1]";
}

bool TokenTable::load(const std::string& path, std::string& err) {
    entries_.clear();
    std::ifstream in(path);
    if (!in) {
        err = "cannot read token file: " + path;
        return false;
    }
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        const size_t a = t.find(':');
        const size_t b = (a == std::string::npos) ? std::string::npos : t.find(':', a + 1);
        const std::string where = path + ":" + std::to_string(line_no);
        if (a == std::string::npos || b == std::string::npos) {
            err = where + ": expected `name:sha256:<64 hex>`, got `" + t.substr(0, 60) + "`";
            return false;
        }
        Entry e;
        e.name = trim(t.substr(0, a));
        const std::string algo = trim(t.substr(a + 1, b - a - 1));
        e.digest_hex = trim(t.substr(b + 1));
        if (e.name.empty() || e.name.find_first_of(" \t") != std::string::npos) {
            err = where + ": token name must be non-empty and contain no whitespace";
            return false;
        }
        if (algo != "sha256") {
            err = where + ": unsupported algorithm `" + algo + "` (only sha256 is accepted)";
            return false;
        }
        if (!is_hex64(e.digest_hex)) {
            err = where + ": digest must be 64 lowercase hex characters";
            return false;
        }
        for (const Entry& prev : entries_) {
            if (prev.name == e.name) {
                err = where + ": duplicate token name `" + e.name + "`";
                return false;
            }
        }
        entries_.push_back(std::move(e));
    }
    if (entries_.empty()) {
        err = "token file is empty: " + path;
        return false;
    }
    return true;
}

bool TokenTable::authenticate(const std::string& token, std::string& caller) const {
    if (token.empty()) return false;
    const std::string digest = sha256_hex(token);
    // 扫完全表：命中位置不是秘密，但没有理由让它变成一条可观察的分支。
    const Entry* hit = nullptr;
    for (const Entry& e : entries_) {
        if (ct_equal(digest, e.digest_hex)) hit = &e;
    }
    if (!hit) return false;
    caller = hit->name;
    return true;
}

// 本次请求是否被 shed：只用来让下面的 logger 跳过审计行（见那里的理由）。
// 指标计数不受它影响 —— 计数在 logger 的第一行，早于任何提前返回。
thread_local bool tl_shed = false;

void install_access_control(httplib::Server& svr, const TokenTable& tokens, bool quiet,
                            const Log& log, bool is_mcp, Metrics* metrics,
                            long long queue_deadline_ms) {
    const char* svc = is_mcp ? "mcp" : "http";

    svr.set_pre_routing_handler([&tokens, svc, queue_deadline_ms, &log, metrics](
                                    const httplib::Request& req,
                                    httplib::Response& res) {
        tl_caller.clear();
        tl_shed = false;
        tl_request_start = std::chrono::steady_clock::now();
        // 先把连接被 worker 取走时的排队时长结算给本请求（keep-alive 的第 2+ 条拿到 0，
        // 它从未在 accept 队列里排过队，见 admission.hpp）。
        admission::begin_request();

        // ---- 边缘 shedding：在任何每请求开销之前 ----
        // 已经等过 deadline 的请求，再花 15ms 推理大概率是在为一个已经离开的客户端烧 CPU，
        // 而那些 CPU 正是让别人不再排队所需的东西 —— 不 shed 就是死亡螺旋。
        // 放在鉴权之前：shed 的决定必须不依赖任何请求内容，也不能自己变成新的成本来源。
        if (queue_deadline_ms > 0 && sheddable_path(req.path) &&
            admission::request_wait_ms() > static_cast<double>(queue_deadline_ms)) {
            tl_shed = true;
            if (metrics) metrics->queue_expired_total.inc();
            res.status = 503;
            // 不带 Retry-After 的 503 会让客户端立即重试，把刚 shed 掉的负载重新推进来。
            res.set_header("Retry-After", "1");
            res.set_content("{\"status\":\"error\",\"error\":\"server busy: admission "
                            "deadline exceeded before serving\",\"code\":\"queue_expired\"}",
                            "application/json");
            // 饱和这个**事实**要出声，但只出一次：每请求一行就是 P1 那次事故本身
            //（日志把服务压垮）。真相在 counter 与直方图里，不在日志行数里。
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                log.event(LogLevel::Warn, "queue_shed_start",
                          std::string("\"svc\":\"") + svc + "\",\"deadline_ms\":" +
                              std::to_string(queue_deadline_ms) +
                              ",\"note\":\"subsequent sheds are counted in "
                              "facerec_queue_expired_total, not logged\"");
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        if (tokens.empty() || is_probe_path(req.path)) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        std::string caller;
        if (!tokens.authenticate(bearer(req), caller)) {
            // 安全事件不受 quiet 影响：拒绝行在下面的 logger 里必出（status >= 400）。
            res.status = 401;
            res.set_header("WWW-Authenticate", "Bearer realm=\"facerec\"");
            res.set_content("{\"status\":\"error\",\"error\":\"unauthorized\"}",
                            "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        tl_caller = caller;
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.set_logger([quiet, &log, svc, metrics](const httplib::Request& req,
                                               const httplib::Response& res) {
        // 计数在**所有提前返回之前**：quiet 与"探针成功不记日志"只影响日志量，
        // 不该让流量从指标里消失（否则 quiet 生产部署的 requests 恒为 0）。
        if (metrics) {
            metrics->http_requests.inc(metric_labels(req.path, res.status));
        }
        // shed 不写审计行。容量事件与安全事件不同：401 必出是因为它是安全事件，
        // 而一淘并发下的 503 每请求一行会把服务自己压垮（P1 的根因就是日志）。
        // 可观测性由 counter + 直方图 + 上面那条一次性 warn 承担，不是靠日志行数。
        if (tl_shed) return;
        // 探针成功不记（k8s 每 10s 打一次会把日志淹没），探针失败要记。
        if (is_probe_path(req.path) && res.status < 400) return;
        if (quiet && res.status < 400) return;
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                     tl_request_start)
                .count();
        // 只拼字段：时间戳与 "audit" 键由 Log 统一加（同一份流里两类行的 ts 格式必须一致）。
        std::ostringstream os;
        os << "\"svc\":\"" << svc << "\",\"method\":\"" << json_escape(req.method)
           << "\",\"path\":\"" << json_escape(req.path) << "\",\"status\":" << res.status
           << ",\"ms\":" << static_cast<long long>(ms) << ",\"caller\":\""
           << json_escape(current_caller()) << "\",\"bytes\":" << req.body.size();
        if (res.status == 401) {
            // 拒绝永远是安全事件：即使 --quiet 也记，且带上对端与"是否出示了 token"。
            os << ",\"auth\":\"denied\",\"peer\":\""
               << json_escape(req.remote_addr.empty() ? std::string("?") : req.remote_addr)
               << "\",\"presented\":"
               << (req.get_header_value("Authorization").empty() ? "false" : "true");
        }
        log.audit("request", os.str());
    });
}

// 生成一个新 token：32 字节 CSPRNG → 64 hex（十六进制没有 URL/头部转义问题，
// 长度固定也就没有"看起来像"的分支）。
std::string generate_token() {
    std::random_device rd;
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 32; ++i) {
        const unsigned int b = rd() & 0xffu;
        out.push_back(kHex[b >> 4]);
        out.push_back(kHex[b & 0x0f]);
    }
    return out;
}

std::string token_file_line(const std::string& name, const std::string& token) {
    return name + ":sha256:" + sha256_hex(token);
}

}  // namespace facerec