#include "mcp.hpp"

#include "auth.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <csignal>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "api.hpp"
#include "encoding.hpp"
#include "httplib.h"
#include "json_util.hpp"
#include "pipeline_pool.hpp"
#include "result_cache.hpp"
#include "server.hpp"
#include "sha256.hpp"

namespace facerec {

namespace {

constexpr const char* kProtocolVersion = "2025-06-18";
constexpr const char* kServerName = "facerec";
constexpr const char* kServerVersion = "1.0.0";

// ---------------------------------------------------------------------------
// JSON-RPC 信封
// ---------------------------------------------------------------------------

std::string rpc_result(const std::string& id, const std::string& result_json) {
    if (id.empty()) return {};   // notification：不回应
    return "{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"result\":" + result_json + "}";
}

std::string rpc_error(const std::string& id, int code, const std::string& message) {
    if (id.empty()) return {};
    return "{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"error\":{\"code\":" +
           std::to_string(code) + ",\"message\":\"" + json_escape(message) + "\"}}";
}

// id 可能是数字或字符串，原样抄回去即可（不做解析，MCP 允许两者）。
std::string extract_id(const std::string& msg) {
    const size_t pos = msg.find("\"id\"");
    if (pos == std::string::npos) return {};
    size_t cur = msg.find(':', pos + 4);
    if (cur == std::string::npos) return {};
    cur = msg.find_first_not_of(" \t\r\n", cur + 1);
    if (cur == std::string::npos) return {};
    if (msg[cur] == '"') {
        const size_t close = msg.find('"', cur + 1);
        if (close == std::string::npos) return {};
        return msg.substr(cur, close - cur + 1);
    }
    size_t end = cur;
    while (end < msg.size() && (std::isdigit(static_cast<unsigned char>(msg[end])) ||
                                msg[end] == '-' || msg[end] == '+' || msg[end] == '.' ||
                                msg[end] == 'e' || msg[end] == 'E')) {
        ++end;
    }
    if (end == cur) return {};
    return msg.substr(cur, end - cur);
}

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

const char* kIdentifySchema = R"({
  "type":"object",
  "properties":{
    "image_b64":{"type":"string","description":"Base64 of the image bytes (JPEG/PNG)."},
    "top_k":{"type":"integer","default":5,"minimum":1,"maximum":100,
             "description":"How many ranked candidates per face."},
    "min_score":{"type":"number","default":0,
                 "description":"Minimum cosine for a candidate to be listed. 0 = use the server's recognize threshold."},
    "include_landmarks":{"type":"boolean","default":false,
             "description":"Include the 5 landmarks in original-image pixels."}
  },
  "required":["image_b64"]
})";

const char* kCompareSchema = R"({
  "type":"object",
  "properties":{
    "image_a_b64":{"type":"string","description":"Base64 of the first image."},
    "image_b_b64":{"type":"string","description":"Base64 of the second image."}
  },
  "required":["image_a_b64","image_b_b64"]
})";

const char* kCropSchema = R"({
  "type":"object",
  "properties":{
    "image_b64":{"type":"string","description":"Base64 of the image bytes."},
    "face_index":{"type":"integer","default":0,
                  "description":"Which detected face (0-based, same order as identify_faces)."}
  },
  "required":["image_b64"]
})";

const char* kListSchema = R"({"type":"object","properties":{}})";

struct ToolDef {
    const char* name;
    const char* description;
    const char* schema;
};

const ToolDef kTools[] = {
    {"identify_faces",
     "Detect faces in one image and match them against the enrolled gallery. Returns, per "
     "face, the bounding box in original-image pixels and a match verdict. IMPORTANT: "
     "'matched' is null when the best cosine is below the server threshold, which means "
     "'unknown person' — do NOT infer an identity from the ranked 'top_k' candidates, they "
     "are scores only. An empty 'faces' array means no face was detected, which is not an "
     "error. Read-only; never enrolls or stores anything.",
     kIdentifySchema},
    {"compare_faces",
     "Compare the largest face in two images by cosine similarity of their embeddings "
     "(1.0 = same identity, ~0 = unrelated). Returns the cosine and both matched persons "
     "when they are known. Read-only.",
     kCompareSchema},
    {"list_people",
     "List the person ids that are enrolled in the gallery. Returns identifiers only — "
     "never embeddings, photos or biometric data. Read-only.",
     kListSchema},
    {"get_face_crop",
     "Return the aligned 112x112 face crop (PNG) for one detected face. Useful to show a "
     "user which face was matched. Read-only.",
     kCropSchema},
};

std::string text_content(const std::string& s) {
    return "{\"type\":\"text\",\"text\":\"" + json_escape(s) + "\"}";
}

std::string image_content(const std::string& b64, const char* mime) {
    return "{\"type\":\"image\",\"data\":\"" + b64 + "\",\"mimeType\":\"" + mime + "\"}";
}

std::string tool_ok(const std::vector<std::string>& parts) {
    std::string out = "{\"content\":[";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += ",";
        out += parts[i];
    }
    out += "],\"isError\":false}";
    return out;
}

std::string tool_err(const std::string& message) {
    return "{\"content\":[" + text_content(message) + "],\"isError\":true}";
}

// 工具参数里的图片：base64 -> 字节 -> Mat。失败时给出人可读的原因。
cv::Mat image_from_arg(const std::string& args, const std::string& key, int max_pixels,
                       std::string& err) {
    const std::string b64 = json::string(args, key);
    if (b64.empty()) {
        err = "missing required field: " + key;
        return {};
    }
    const std::string bytes = base64_decode(b64);
    if (bytes.empty()) {
        err = "image_b64 is not valid base64";
        return {};
    }
    std::vector<uchar> buf(bytes.begin(), bytes.end());
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) {
        err = "cannot decode image (unsupported or corrupt)";
        return {};
    }
    if (max_pixels > 0 && img.total() > static_cast<size_t>(max_pixels)) {
        err = "image too large: " + std::to_string(img.total()) + " pixels";
        return {};
    }
    return img;
}

std::string handle_identify(Pipeline& pipeline, const std::string& args, int max_pixels,
                            ResultCache* cache, const std::string& image_sha) {
    IdentifyOptions opt;
    json::integer(args, "top_k", opt.top_k);
    if (opt.top_k <= 0) opt.top_k = 5;
    if (opt.top_k > 100) opt.top_k = 100;
    double min_score = 0;
    if (json::number(args, "min_score", min_score)) opt.min_score = static_cast<float>(min_score);
    json::boolean(args, "include_landmarks", opt.include_landmarks);

    // 缓存键用 base64 串的摘要，所以命中路径连 base64 解码和 imdecode 都省掉。
    const std::string key =
        cache ? cache->make_key(image_sha, opt, opt.detector_w, opt.detector_h) : std::string();
    IdentifyResult r;
    if (!(cache && cache->get(key, r))) {
        std::string err;
        cv::Mat img = image_from_arg(args, "image_b64", max_pixels, err);
        if (img.empty()) return tool_err(err);
        r = pipeline.identify(img, opt, "mcp");
        if (cache) cache->put(key, r);
    }

    // 给人读的摘要 + 给程序用的 JSON 一起返回：只给 JSON 时模型容易自己编解释，
    // 只给文字时下游没法解析。
    std::ostringstream summary;
    summary << "Detected " << r.faces.size() << " face(s) in a "
            << r.image_w << "x" << r.image_h << " image.";
    for (const auto& f : r.faces) {
        summary << "\n- face " << f.index << " bbox=["
                << format_float(f.bbox[0], 0) << "," << format_float(f.bbox[1], 0) << ","
                << format_float(f.bbox[2], 0) << "," << format_float(f.bbox[3], 0) << "]";
        if (f.matched) {
            summary << " matched '" << f.matched->person_id << "' (cosine "
                    << format_float(f.matched->score, 3) << ")";
        } else if (!f.aligned) {
            summary << " could not be aligned (unknown)";
        } else {
            summary << " unknown person (no candidate above threshold)";
        }
        if (!f.top_k.empty()) {
            summary << "; best candidate score "
                    << format_float(f.top_k.front().score, 3);
        }
    }
    summary << "\nA null match means 'unknown' — do not infer identity from candidate "
               "scores.";

    return tool_ok({text_content(summary.str()), text_content(identify_to_json(r, opt))});
}

std::string handle_compare(Pipeline& pipeline, const std::string& args, int max_pixels) {
    std::string err;
    cv::Mat a = image_from_arg(args, "image_a_b64", max_pixels, err);
    if (a.empty()) return tool_err(err);
    cv::Mat b = image_from_arg(args, "image_b_b64", max_pixels, err);
    if (b.empty()) return tool_err(err);

    cv::Mat fa = pipeline.embed_image(a, 0);
    cv::Mat fb = pipeline.embed_image(b, 0);
    if (fa.empty() || fb.empty()) {
        return tool_err("no face detected in one of the images");
    }
    const double dot = fa.dot(fb);
    const double na = cv::norm(fa);
    const double nb = cv::norm(fb);
    const double cos = (na > 0 && nb > 0) ? dot / (na * nb) : 0.0;

    std::ostringstream summary;
    summary << "Cosine similarity between the largest face in each image: "
            << format_float(cos, 4)
            << " (1.0 = same identity, around 0 = unrelated).";
    std::ostringstream body;
    body << "{\"status\":\"ok\",\"cosine\":" << format_float(cos, 6) << "}";
    return tool_ok({text_content(summary.str()), text_content(body.str())});
}

std::string handle_list_people(Pipeline& pipeline, const std::string&, int) {
    // 摘要必须与下面的 JSON 同一个源：用目录列表会把空目录/无可用脸的目录说成"已注册"，
    // 而那种人永远不可能匹配 —— LLM 拿到就会去推断一个不存在的身份。
    const auto& entries = pipeline.gallery().entries();
    std::ostringstream summary;
    summary << entries.size() << " person(s) enrolled:";
    for (const auto& e : entries) summary << " " << e.person_id;
    if (entries.empty()) summary << " (none)";

    // 目录存在但库里没有：注册失败（空目录，或照片里没有可用的脸）——值得说出来。
    std::vector<std::string> empty_dirs;
    for (const auto& d : list_person_dirs(pipeline.config().data_dir)) {
        const std::string want = normalize_entry_id(d);
        const bool present = std::any_of(entries.begin(), entries.end(),
                                         [&](const GalleryEntry& e) {
                                             return e.person_id == want;
                                         });
        if (!present) empty_dirs.push_back(d);
    }
    if (!empty_dirs.empty()) {
        summary << "  [enrolled but no usable photos:";
        for (const auto& d : empty_dirs) summary << " " << d;
        summary << "]";
    }
    return tool_ok({text_content(summary.str()), text_content(people_to_json(pipeline))});
}

std::string handle_crop(Pipeline& pipeline, const std::string& args, int max_pixels) {
    std::string err;
    cv::Mat img = image_from_arg(args, "image_b64", max_pixels, err);
    if (img.empty()) return tool_err(err);

    int face_index = 0;
    json::integer(args, "face_index", face_index);
    if (face_index < 0) face_index = 0;

    IdentifyOptions opt;
    opt.include_crops = true;
    IdentifyResult r = pipeline.identify(img, opt, "mcp");
    if (face_index >= static_cast<int>(r.faces.size())) {
        return tool_err("face_index " + std::to_string(face_index) + " out of range: image has " +
                        std::to_string(r.faces.size()) + " face(s)");
    }
    const FaceResult& f = r.faces[static_cast<size_t>(face_index)];
    if (f.crop_png_b64.empty()) {
        return tool_err("could not align the requested face (no usable landmarks)");
    }
    std::ostringstream summary;
    summary << "Aligned 112x112 crop of face " << face_index
            << " (the same crop the embedder saw).";
    return tool_ok({text_content(summary.str()), image_content(f.crop_png_b64, "image/png")});
}

std::string dispatch_tool(Pipeline& pipeline, const std::string& name,
                          const std::string& args, int max_pixels, ResultCache* cache,
                          const std::string& image_sha) {
    if (name == "identify_faces") {
        return handle_identify(pipeline, args, max_pixels, cache, image_sha);
    }
    if (name == "compare_faces") return handle_compare(pipeline, args, max_pixels);
    if (name == "list_people") return handle_list_people(pipeline, args, max_pixels);
    if (name == "get_face_crop") return handle_crop(pipeline, args, max_pixels);
    return {};   // 未知工具由调用方按协议错误处理
}

// schema 用 raw string 书写（可读），但 stdio 传输是**换行分帧**的：一条消息里出现裸
// 换行就会把 JSON 截断成两条。JSON 允许对象内部的空白，所以发送前把换行换成空格。
std::string one_line(const char* s) {
    std::string out;
    out.reserve(std::strlen(s) + 1);
    for (const char* p = s; *p; ++p) {
        out += (*p == '\n' || *p == '\r') ? ' ' : *p;
    }
    return out;
}

std::string tools_list_json() {
    std::ostringstream os;
    os << "{\"tools\":[";
    bool first = true;
    for (const auto& t : kTools) {
        if (!first) os << ",";
        first = false;
        os << "{\"name\":\"" << t.name << "\",\"description\":\""
           << json_escape(t.description) << "\",\"inputSchema\":" << one_line(t.schema)
           << "}";
    }
    os << "]}";
    return os.str();
}

// 单条 JSON-RPC 消息 -> 响应（notification 返回空串）。
// 这是唯一的协议实现：stdio 与 HTTP 两种传输都走这里，行为不可能分叉。
std::string handle_message(PipelinePool& pool, const std::string& msg, int max_pixels,
                           ResultCache* cache) {
    const std::string id = extract_id(msg);
    const std::string method = json::string(msg, "method");
    if (method.empty()) {
        return rpc_error(id, -32600, "invalid request: missing method");
    }

    if (method == "initialize") {
        const std::string client_version = json::string(msg, "protocolVersion");
        std::ostringstream os;
        os << "{\"protocolVersion\":\""
           << json_escape(client_version.empty() ? kProtocolVersion : client_version)
           << "\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"" << kServerName
           << "\",\"version\":\"" << kServerVersion << "\"},"
           << "\"instructions\":\"Face detection + recognition. identify_faces reports "
              "\\\"matched\\\": null for unknown people — never infer an identity from "
              "candidate scores. Enrollment is deliberately not exposed as a tool.\"}";
        return rpc_result(id, os.str());
    }

    if (method == "notifications/initialized" || method == "notifications/cancelled" ||
        method == "notifications/roots/list_changed") {
        return {};   // notification：无响应
    }

    if (method == "ping") {
        return rpc_result(id, "{}");
    }

    if (method == "tools/list") {
        return rpc_result(id, tools_list_json());
    }

    if (method == "tools/call") {
        const std::string params = json::object(msg, "params");
        const std::string name = json::string(params, "name");
        if (name.empty()) {
            return rpc_error(id, -32602, "invalid params: missing tool name");
        }
        const std::string args = json::object(params, "arguments");

        Lease lease(pool, std::chrono::milliseconds(10000));
        if (!lease) {
            return rpc_result(id, tool_err("server busy, try again"));
        }
        // 缓存键用参数里 base64 串的摘要：同一张图 -> 同一串 -> 同一键，
        // 且不必先解码就能查缓存。
        const std::string image_sha = sha256_hex(json::string(args, "image_b64"));
        std::string result = dispatch_tool(*lease.get(), name, args, max_pixels, cache, image_sha);
        if (result.empty()) {
            return rpc_error(id, -32602, "unknown tool: " + name);
        }
        return rpc_result(id, result);
    }

    return rpc_error(id, -32601, "method not found: " + method);
}

// ---------------------------------------------------------------------------
// 传输
// ---------------------------------------------------------------------------

// stdio：每行一条 JSON-RPC 消息。stdout 是协议通道，**任何日志都必须走 stderr**。
int serve_stdio(PipelinePool& pool, const McpOptions& opt, ResultCache& cache) {
    std::cerr << "facerec mcp: stdio transport, " << pool.size() << " worker(s), gallery "
              << pool.gallery_size() << " person(s)" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::string response = handle_message(pool, line, opt.max_image_pixels, &cache);
        if (!response.empty()) {
            std::cout << response << '\n' << std::flush;
        }
    }
    std::cerr << "facerec mcp: stdin closed, exiting" << std::endl;
    return 0;
}

struct Access {
    TokenTable tokens;
    AuditSink audit;
};

int serve_http(PipelinePool& pool, const McpOptions& opt, ResultCache& cache,
               const Access& access) {
    httplib::Server svr;
    svr.set_payload_max_length(static_cast<size_t>(opt.max_request_mb) * 1024 * 1024);

    // 与 serve 同一套：/healthz 免鉴权，其余要 bearer；审计行不含 person_id（PLAN.md D1/D3）。
    install_access_control(svr, access.tokens, opt.quiet, access.audit, /*is_mcp=*/true);

    svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json; charset=utf-8");
    });

    // streamable HTTP 传输：客户端 POST 一条 JSON-RPC 消息，服务端回 JSON。
    // 通知（无 id）按规范回 202 Accepted 且不携带响应体。
    svr.Post("/mcp", [&pool, &opt, &cache](const httplib::Request& req,
                                            httplib::Response& res) {
        const std::string response =
            handle_message(pool, req.body, opt.max_image_pixels, &cache);
        if (response.empty()) {
            res.status = 202;
            return;
        }
        res.status = 200;
        res.set_content(response, "application/json; charset=utf-8");
    });

    // 部分客户端会先 GET 建立 SSE 流。本实现只支持请求-响应，明确回 405 而不是挂住。
    svr.Get("/mcp", [](const httplib::Request&, httplib::Response& res) {
        res.status = 405;
        res.set_content(
            "{\"error\":\"this server implements JSON-RPC request/response only; POST /mcp\"}",
            "application/json; charset=utf-8");
    });

    std::cerr << "facerec mcp: HTTP transport on http://" << opt.host << ":" << opt.port
              << "/mcp, " << pool.size() << " worker(s), gallery " << pool.gallery_size()
              << " person(s)" << std::endl;
    // 与 HTTP 外壳同一份停机/SIGPIPE 处理：两个入口的行为必须一致，否则修一个忘一个。
    if (listen_with_signals(svr, opt.host, opt.port, false) != 0) {
        std::cerr << "Failed to listen on " << opt.host << ":" << opt.port << std::endl;
        return 1;
    }
    return 0;
}

}  // namespace

int run_mcp(const PipelineConfig& cfg, const McpOptions& opt) {
    int workers = opt.workers > 0 ? opt.workers : 1;
    if (opt.transport == McpTransport::Http && opt.workers <= 0) {
        workers = static_cast<int>(cv::getNumberOfCPUs());
    }
    if (workers < 1) workers = 1;

    // 鉴权与审计：规则与 serve 完全一致（同一个 install_access_control）。
    Access access;
    if (!cfg.auth_tokens_file.empty()) {
        std::string err;
        if (!access.tokens.load(cfg.auth_tokens_file, err)) {
            std::cerr << "Fatal: " << err << std::endl;
            return -1;
        }
    }
    if (opt.transport == McpTransport::Http && access.tokens.empty() &&
        !is_loopback_host(opt.host)) {
        std::cerr << "Refusing to listen on " << opt.host << " without authentication.\n"
                  << "  Either bind loopback (--host 127.0.0.1), or set auth.tokens_file.\n"
                  << "  Create a token table:  ./main token --name admin --tokens-file "
                     "./tokens.txt" << std::endl;
        return -1;
    }
    {
        std::string err;
        if (!access.audit.configure(cfg.log_audit, err)) {
            std::cerr << "Fatal: log.audit: " << err << std::endl;
            return -1;
        }
    }

    // 与 serve 同一结论：并发靠多实例，不靠 Net 内部线程池。
    cv::setNumThreads(1);
    PipelinePool pool(cfg, workers);
    ResultCache cache(pool.fingerprint(), static_cast<size_t>(opt.cache_entries));
    {
        std::ostringstream os;
        os << "\"op\":\"startup\",\"transport\":\""
           << (opt.transport == McpTransport::Http ? "http" : "stdio") << "\",\"persons\":"
           << pool.gallery_size() << ",\"gallery_fingerprint\":\""
           << json_escape(pool.fingerprint()) << "\",\"auth\":\""
           << (access.tokens.empty() ? "off"
                                     : std::to_string(access.tokens.size()) + " token(s)")
           << "\"";
        audit_event(access.audit, "startup", os.str());
    }
    return opt.transport == McpTransport::Http ? serve_http(pool, opt, cache, access)
                                               : serve_stdio(pool, opt, cache);
}

}  // namespace facerec