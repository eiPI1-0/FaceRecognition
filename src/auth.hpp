#ifndef FACERECOGNITION_AUTH_HPP
#define FACERECOGNITION_AUTH_HPP

// 进程内 bearer 鉴权 + 结构化审计日志（PLAN.md D1/D3）。
//
// 为什么在进程内做：TLS 和边缘身份交给反向代理，但"服务自己不知道谁在调"会让审计无从归因，
// 也让 `--host 0.0.0.0` 成为一条静默的裸奔路径。这里的规则是：**安全是绑定约束，不是开关**
// —— 非 loopback 监听且没有 token 表时，启动直接失败。
//
// token 表只存 sha256，磁盘上没有任何可用凭据；比对是常量时间的。

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <httplib.h>

namespace facerec {

class TokenTable {
public:
    // 行式文件：`name:sha256:<64 hex>`，`#` 注释与空行忽略。任何坏行都失败并给出 file:line
    // （静默跳过坏行 = 一个打错的 token 永远不生效，而现象是 401）。
    bool load(const std::string& path, std::string& err);
    bool empty() const { return entries_.empty(); }
    size_t size() const { return entries_.size(); }

    // 命中则写出 caller 名并返回 true。比较与表的大小无关地扫过所有条目？
    // 不：条目数本身不是秘密（它写在配置文件里），泄漏"第几条命中"没有意义；
    // 需要恒定的是**同一个 token 的逐字节比较**。
    bool authenticate(const std::string& token, std::string& caller) const;

private:
    struct Entry {
        std::string name;
        std::string digest_hex;   // 64 个小写 hex
    };
    std::vector<Entry> entries_;
};

// 当前请求的调用方（线程局部）。
// httplib 一个连接一个线程、同一线程同时只处理一个请求，所以请求处理期间这个值不会被别的请求
// 覆盖；pre_routing 写入，handler 读出来放进审计行。
extern thread_local std::string tl_caller;
const std::string& current_caller();

bool is_loopback_host(const std::string& host);

// 审计/访问日志：一行一个 JSON 对象。查询行不含 person_id / 名字 / embedding / 图像字节，
// 见 PLAN.md D1 的理由。
struct AuditSink {
    enum Kind { Off, Stderr, File };
    Kind kind = Stderr;
    std::string path;

    // spec: "off" | "stderr" | 其它=文件路径（追加写）。文件打不开算配置错误。
    bool configure(const std::string& spec, std::string& err);
    void emit(const std::string& json_object) const;
};

// 把鉴权与访问日志装到任何一个 httplib::Server 上（serve 与 mcp --transport http 共用）。
//   - 除 /healthz、/readyz 外全部需要 `Authorization: Bearer <token>`（token 表为空则放行，
//     此时启动阶段已保证只绑 loopback）；
//   - 401 带 WWW-Authenticate，且**永远**记审计，不受 quiet 影响（安全事件不能被静音）；
//   - 成功行的 body 细节由 handler 自己补（它知道 faces/matched/cache），这里只记
//     method/path/status/ms/caller/bytes。
void install_access_control(httplib::Server& svr, const TokenTable& tokens, bool quiet,
                            const AuditSink& audit, bool is_mcp);

// 非请求事件（startup / enroll / gallery 重建）：fields 是已序列化好的 JSON 片段。
void audit_event(const AuditSink& sink, const char* kind, const std::string& fields);

// 生成一个新 token（32 字节 CSPRNG 的十六进制）与它在该文件里应有的那一行。
// 分两步给调用方：token 只显示一次，文件里只存 sha256。
std::string generate_token();
std::string token_file_line(const std::string& name, const std::string& token);

}  // namespace facerec

#endif  // FACERECOGNITION_AUTH_HPP