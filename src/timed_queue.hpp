#ifndef FACERECOGNITION_TIMED_QUEUE_HPP
#define FACERECOGNITION_TIMED_QUEUE_HPP

#include <chrono>
#include <functional>

#include <httplib.h>

#include "admission.hpp"

namespace facerec {

// 包装 httplib::ThreadPool：测量队列等待、给队列加硬上限、并把"连接被 worker 取走时的
// 排队时长"交给 admission（见 admission.hpp），让 deadline 能跨两段连续生效。
//
// cpp-httplib 的 listen_internal 在 **accept 线程**上把每个**连接**包成
// std::function<void()> 推入 task queue（new_task_queue 建的那个），worker 线程 pop 出来
// 执行 `process_and_close_socket(sock)`，那个函数会在同一个任务里连续服务最多
// keep_alive_max_count 条请求。所以：
//   - 队列的粒度是**连接**，不是请求。直方图的 count 是连接数；max_pending 限的也是
//     待处理连接数。（实测：12 条请求走 1 条 keep-alive 连接 → count +1；
//     12 条请求走 12 条连接 → count +12。）
//   - enqueue() 发生在请求字节被读取**之前**，所以这里是唯一真正的"边缘"：超过上限就
//     回 false，httplib 直接 shutdown+close（httplib.h:6388），请求根本不占 worker。
//     代价是客户端看到的是连接被关（curl: (52) Empty reply），不是 503 —— 因为此刻
//     请求还没解析，无法按 HTTP 应答。要 503 就得晚一点（pre_routing 的 deadline shed）。
//
// 两者分工：deadline shed 是主控制（正常过载，回正经 503 + Retry-After）；
// max_pending 是兜底（连 deadline 都来不及结算时，护住 fd/内存），默认关。
//
// wait_cb / drop_cb 在 worker / accept 线程上被调，必须线程安全。空 function = 不记录。
class TimedTaskQueue : public httplib::TaskQueue {
public:
    using WaitCb = std::function<void(double wait_ms)>;
    using DropCb = std::function<void()>;

    // max_pending = 0 表示队列不设上限（只靠 deadline shed）。
    TimedTaskQueue(size_t n, size_t max_pending = 0, WaitCb wait_cb = nullptr,
                   DropCb drop_cb = nullptr)
        : pool_(n, max_pending), wait_cb_(std::move(wait_cb)),
          drop_cb_(std::move(drop_cb)) {}

    bool enqueue(std::function<void()> fn) override {
        const auto t_enqueue = std::chrono::steady_clock::now();
        auto wrapped = [fn = std::move(fn), t_enqueue, cb = wait_cb_]() {
            const auto t_start = std::chrono::steady_clock::now();
            const double wait_ms =
                std::chrono::duration<double, std::milli>(t_start - t_enqueue).count();
            // 先交给 admission：本连接的后续请求与池段预算要用。
            admission::note_pickup_wait(wait_ms);
            if (cb) cb(wait_ms);
            fn();
        };
        if (!pool_.enqueue(std::move(wrapped))) {
            // httplib 会关掉这个 socket。它自己**不计数、不记日志** —— 和当年
            // listen backlog=5 一样，这类拒绝在服务端是完全隐形的，所以必须在这里计数。
            if (drop_cb_) drop_cb_();
            return false;
        }
        return true;
    }

    void shutdown() override { pool_.shutdown(); }
    void on_idle() override { pool_.on_idle(); }

private:
    httplib::ThreadPool pool_;
    WaitCb wait_cb_;
    DropCb drop_cb_;
};

}  // namespace facerec

#endif  // FACERECOGNITION_TIMED_QUEUE_HPP
