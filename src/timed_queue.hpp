#ifndef FACERECOGNITION_TIMED_QUEUE_HPP
#define FACERECOGNITION_TIMED_QUEUE_HPP

#include <chrono>
#include <functional>

#include <httplib.h>

namespace facerec {

// 包装 httplib::ThreadPool，在任务入队与真正执行之间测量等待时长。
//
// cpp-httplib 的 listen_internal 把每个请求包成 std::function<void()> 推入
// task queue（new_task_queue 创建的那个），worker 线程 pop 出来执行。
//   - enqueue() 时记录 t_enqueue（主线程/accept 线程调用）
//   - wrapped fn 真正执行时记录 t_start（worker 线程调用）
//   - 差值 t_start - t_enqueue 即任务在 queue 里堆积的时长
//
// 这覆盖了并发高时任务堆积等 worker 的情况。不覆盖 accept 后、enqueue 前
// 读取请求头/解析的时间（在 httplib 内部，不可见）。和 request_ms（只覆盖
// handler 内部）互补：两者加起来才是客户端感知的完整延迟。
//
// wait_cb 被 worker 线程调用，必须线程安全。传 nullptr（或空 function）表示不记录。
class TimedTaskQueue : public httplib::TaskQueue {
public:
    using WaitCb = std::function<void(double wait_ms)>;

    TimedTaskQueue(size_t n, WaitCb wait_cb = nullptr)
        : pool_(n), wait_cb_(std::move(wait_cb)) {}

    bool enqueue(std::function<void()> fn) override {
        const auto t_enqueue = std::chrono::steady_clock::now();
        auto wrapped = [fn = std::move(fn), t_enqueue,
                        cb = wait_cb_]() {
            const auto t_start = std::chrono::steady_clock::now();
            const double wait_ms =
                std::chrono::duration<double, std::milli>(t_start - t_enqueue).count();
            if (cb) cb(wait_ms);
            fn();
        };
        return pool_.enqueue(std::move(wrapped));
    }

    void shutdown() override { pool_.shutdown(); }
    void on_idle() override { pool_.on_idle(); }

private:
    httplib::ThreadPool pool_;
    WaitCb wait_cb_;
};

}  // namespace facerec

#endif  // FACERECOGNITION_TIMED_QUEUE_HPP
