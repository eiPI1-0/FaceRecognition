// 准入控制（admission）的回归测试。
//
// 为什么值得一个测试文件：这里锁的三条都是**改成"看起来更直觉"就坏**的东西：
//   1. 池段预算 = 端到端上限 − 队列段已等掉的。写成 `min(deadline, 剩余)` 或对负数不
//      取 floor，端到端等待就会变成两段各一次（改之前的真实行为是"队列 ∞ + 池 5s"）。
//   2. 等待记账的 keep-alive 语义：httplib 按**连接**排队，一个任务里连续服务多条请求。
//      第 2 条请求没有在 accept 队列排过队，必须记 0。写成"每次都拿连接的时间"会把
//      服务时间当排队时间，于是把正在被服务的连接误 shed。
//      （本文件第 2 组断言来自一次真实的近失：三个函数里各写 `static thread_local` 会得到
//       三个不同变量，写入者与读取者根本不是同一份状态 —— 编译通过，行为是错的。）
//   3. Metrics::to_prometheus() 是手写列表，漏一行不会报错：指标存在但 /metrics 里看不见。
//      这条断言把"忘了加一行"从静默失效变成测试红。

#include <cstdio>
#include <string>

#include "admission.hpp"
#include "metrics.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

void check_pool_budget() {
    using facerec::admission::pool_budget_ms;
    check(pool_budget_ms(300, 48) == 252, "deadline 300, queued 48 -> pool budget 252");
    check(pool_budget_ms(100, 100) == 0, "deadline 100, queued 100 -> pool budget 0");
    // 负预算会传给 cv::wait_until（now()+负值 = 立即超时），语义上是巧合而不是设计。
    check(pool_budget_ms(100, 250) == 0, "queued past deadline -> floored at 0, never negative");
    // 0 的含义与 --queue-timeout-ms 的历史一致：**零容忍**，不是"关闭"。
    check(pool_budget_ms(0, 0) == 0, "deadline 0 means zero tolerance, not 'off'");
    check(pool_budget_ms(-1, 99999) == facerec::admission::kUnlimitedWaitMs,
          "deadline -1 = 'off' explicitly, regardless of queued time");
    // 24h 而不是 chrono::max()：now()+max() 会溢出 steady_clock。
    check(facerec::admission::kUnlimitedWaitMs < (1LL << 40),
          "unlimited sentinel is far in the future but cannot overflow now()+d");
}

void check_wait_accounting() {
    using namespace facerec::admission;
    // 连接被 worker 取走时排了 120ms -> 该连接的第一条请求拿到 120。
    note_pickup_wait(120.0);
    begin_request();
    check(request_wait_ms() == 120.0, "first request on a connection inherits its queue wait");
    // keep-alive 的第 2 条：上一条答完立刻读出来，从未在 accept 队列排队。
    begin_request();
    check(request_wait_ms() == 0.0, "second request on same connection waits 0 (not service time)");
    begin_request();
    check(request_wait_ms() == 0.0, "and stays 0 for the rest of the connection");
    // stdio 传输根本没有队列：没有 note_pickup_wait 时读到 0，不是上一次的残值。
    begin_request();
    check(request_wait_ms() == 0.0, "task without a queue (stdio) reports 0, not a stale value");
    // 写入者与读取者必须是同一份状态（跨 inline 函数的单一实例）。
    note_pickup_wait(7.0);
    begin_request();
    check(request_wait_ms() == 7.0, "writer and reader share one thread_local instance");
}

void check_metrics_are_emitted() {
    const std::string out = facerec::Metrics().to_prometheus();
    // 新指标必须真的出现在 /metrics 里：只加成员不加 write_prometheus 行是静默失效。
    for (const char* needle : {"facerec_queue_wait_ms_count",
                               "facerec_queue_expired_total",
                               "facerec_queue_dropped_total"}) {
        check(out.find(needle) != std::string::npos,
              std::string("/metrics exposes ") + needle);
    }
    // 队列直方图按连接计数 —— HELP 里必须说清，否则运维会拿它当 QPS 读。
    check(out.find("per connection") != std::string::npos,
          "queue_wait HELP states it is per connection, not per request");
}

}  // namespace

int main() {
    std::printf("pool budget (end-to-end deadline, continuous across both admission points):\n");
    check_pool_budget();
    std::printf("\nqueue-wait accounting (httplib queues connections, not requests):\n");
    check_wait_accounting();
    std::printf("\nmetrics surface:\n");
    check_metrics_are_emitted();
    std::printf("%s\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
