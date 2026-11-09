#ifndef FACERECOGNITION_ADMISSION_HPP
#define FACERECOGNITION_ADMISSION_HPP

namespace facerec {
namespace admission {

// "等容量"的账本：一个请求从进入 HTTP 任务队列到借到 pipeline，等多久。
//
// 为什么要有这个文件：同一个资源等待发生在**两个**地方 ——
//   1) HTTP 任务队列（accept 线程 enqueue，worker 线程 pop）；
//   2) PipelinePool（handler 里 borrow，上限 pool.busy_timeout_ms）。
// 两处各自设上限是错的：最坏情况会变成"队列 ∞ + 池 5s"，客户端等了 60s 才被服务，
// 而 p50 是 30ms。这里把等待做成**一个可传递的量**，让 deadline 在两段之间连续：
// 队列段已经等掉的，从池段的预算里扣掉。
//
// 线程模型：值存在 worker 线程的 thread_local 里。写入者是 TimedTaskQueue（pop 出任务
// 的那一刻），读取者是 pre_routing 钩子与 acquire_logged —— 都是同一个 worker 线程、
// 同一个任务的动态嵌套内，所以不需要锁。
//
// keep-alive 语义（关键，容易想错）：httplib 的任务队列按**连接**排队，不是按请求 ——
// 一个任务里 `process_and_close_socket` 会连续服务最多 keep_alive_max_count 个请求。
// 所以"连接被 worker 取走"这个等待只对**该连接的第一条请求**成立；第二条请求是上一条
// 答完后立刻读出来的，它从未在 accept 队列里排过队，等待记 0。begin_request() 的
// "取一次就清空"正是实现这个语义，而不是随手写的清理。
// 单一实例：inline 函数里的 static 在所有 TU 里是同一个对象（头文件里放 thread_local
// 变量的唯一安全写法 —— 在三个函数里各写一遍 static thread_local 会得到三个变量）。
// -1 = 本连接没有待结算的排队等待。
inline double& pending_pickup() {
    static thread_local double v = -1.0;
    return v;
}
inline double& this_request_wait() {
    static thread_local double v = 0.0;
    return v;
}

inline void note_pickup_wait(double ms) { pending_pickup() = ms; }

// 每条请求开头调用（pre_routing）。把连接被取走时的排队等待结算给这条请求；
// 后续请求拿到 0。
inline void begin_request() {
    this_request_wait() = pending_pickup() < 0.0 ? 0.0 : pending_pickup();
    pending_pickup() = -1.0;
}

// 本请求到目前为止为"等容量"付出的时间（毫秒）。
inline double request_wait_ms() { return this_request_wait(); }

// "不限"的实现：不是给 PipelinePool 加一个"无限等待"语义（那需要它自己的
// 取消/溢出规则），而是一个事实上不可能到达的等待上限。steady_clock 上 now()+24h 不会
// 溢出（now()+ms::max() 会），所以这个值是安全的。
// 用 -1 触发，见 cli.cpp：0 仍是"零容忍"，不能被重定义成"关闭"。
constexpr long long kUnlimitedWaitMs = 24LL * 60 * 60 * 1000;

// 把"端到端上限"换算成池段还能等的毫秒数。deadline < 0 = 不限；deadline == 0 = 零容忍。
inline long long pool_budget_ms(long long deadline, double already_queued_ms) {
    if (deadline < 0) return kUnlimitedWaitMs;
    if (deadline == 0) return 0;
    const long long left = deadline - static_cast<long long>(already_queued_ms);
    return left > 0 ? left : 0;
}

}  // namespace admission
}  // namespace facerec

#endif  // FACERECOGNITION_ADMISSION_HPP
