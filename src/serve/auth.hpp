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

// 只用到 httplib::Server 的**名字**（参数与返回值都是引用/指针），所以前向声明而不是
// include：这样 `./main token`（只调 TokenTable）不必编译整个 HTTP 库。
// 注意必须在**全局作用域**写 `namespace httplib { class Server; }`——写进 `facerec` 里
// 会造出一个假的 `facerec::httplib::Server`，声明能用、链接期才找不到定义。
// 需要完整类型的是 auth.cpp（它定义 install_access_control）与 server/mcp 那两个外壳。
namespace httplib { class Server; }

#include "common/log.hpp"

namespace facerec {

class Metrics;

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

// 把鉴权与访问日志装到任何一个 httplib::Server 上（serve 与 mcp --transport http 共用）。
//   - 除 /healthz、/readyz 外全部需要 `Authorization: Bearer <token>`（token 表为空则放行，
//     此时启动阶段已保证只绑 loopback）；
//   - 401 带 WWW-Authenticate，且**永远**记审计，不受 quiet 影响（安全事件不能被静音）；
//   - 成功行的 body 细节由 handler 自己补（它知道 faces/matched/cache），这里只记
//     method/path/status/ms/caller/bytes。
// metrics 可为空（mcp server 不持有 Metrics）：非空时每个请求计一次
// `http_requests_total{path,status}`。计数点放在这里而不是各 handler，是为了让
// "加路由忘了计数" 与 "某个分支漏了错误码" 结构上不可能发生。
// 审计行经 log.audit() 写出（不受 log.level 影响；查询行不含 person_id / 名字 /
// embedding / 图像字节，见 PLAN.md D1 的理由）。
// queue_deadline_ms = 端到端等容量上限（与 handler 借 pipeline 用的是同一个值，见
// ServerOptions::queue_timeout_ms）。>0 时，已在任务队列里等过它的连接在进入 handler 前
// 就被 shed：回 503 + Retry-After，不碰 pipeline。探针与 /metrics **不 shed** ——
// 它们不花 pipeline，而在饱和时杀掉探针会把一次过载变成一次重启。
void install_access_control(httplib::Server& svr, const TokenTable& tokens, bool quiet,
                            const Log& log, bool is_mcp, Metrics* metrics,
                            long long queue_deadline_ms = 0);

// 生成一个新 token（32 字节 CSPRNG 的十六进制）与它在该文件里应有的那一行。
// 分两步给调用方：token 只显示一次，文件里只存 sha256。
std::string generate_token();
std::string token_file_line(const std::string& name, const std::string& token);

}  // namespace facerec

#endif  // FACERECOGNITION_AUTH_HPP