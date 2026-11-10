#ifndef FACERECOGNITION_LOG_HPP
#define FACERECOGNITION_LOG_HPP

// 进程内唯一的结构化日志出口（JSON lines，一行一个对象）。
//
// 为什么审计与运行日志共用一个出口：
//   - 同一个目的地、同一把锁 → 两类行严格按发生顺序交错，排障时"503 是在索引重建之前
//     还是之后"不需要跨两个文件对时间戳；
//   - 只有一处写文件的代码 → 不会出现"审计落盘了、运行日志丢在 stderr"这种半截状态。
//
// 但两者的**保证不同**，所以是两个方法，不是一个方法加参数：
//   - audit(...)：安全记录。永远写（唯一能关掉它的是 sink=off，那是显式配置动作）。
//   - event(...)：诊断记录。受 log.level 过滤，生产默认 warn。
// 401 永远走 audit，即使 --quiet —— 安全和静音是两个正交的开关。

#include <mutex>
#include <string>

namespace facerec {

// 枚举顺序即严重度顺序（Off 最低），`event()` 用 `<=` 比较，所以 parse 之后不必再排序。
enum class LogLevel { Off = 0, Error = 1, Warn = 2, Info = 3, Debug = 4 };

// "off|error|warn|info|debug"，大小写不敏感。未知值返回 false（调用方给 file:line 报错）。
bool parse_log_level(const std::string& s, LogLevel& out);
const char* log_level_name(LogLevel l);

// ISO-8601 UTC，毫秒精度，始终以 Z 结尾。放在这里而不是各模块各写一份。
std::string iso8601_now();

class Log {
public:
    // 目的地。Sink 前缀是为了不和 LogLevel::Off 撞名（两个 Off 语义不同：一个是
    // "不写任何日志"，一个是"不写诊断日志"）。
    enum SinkKind { SinkOff, SinkStderr, SinkFile };

    // sink_spec: "off" | "stderr" | 其它=文件路径（追加写）。打开一次确认可写：
    // 启动期失败好过运行时每行都静默丢掉。
    bool configure(const std::string& sink_spec, std::string& err);

    // 级别单独设：它来自另一个 knob，且必须能在 sink 之后设置（--set 的顺序无关）。
    bool set_level(const std::string& level_spec, std::string& err);

    // 安全记录，不受 level 影响。
    void audit(const char* kind, const std::string& fields) const;

    // 诊断记录，受 level 过滤。热点路径先问 enabled() 再拼字符串。
    void event(LogLevel level, const char* name, const std::string& fields) const;

    // fields 的约定：对象**成员列表**（`"a":1,"b":2`），外层大括号由本类加。
    // 带括号的写法会被剔掉——这一处约定第一轮就违反过，产出的是一行无法解析的 JSON；
    // 拆成多余两个字符比静默接受垃圾更容易被发现。

    bool enabled(LogLevel level) const {
        return level_ != LogLevel::Off && static_cast<int>(level) <= static_cast<int>(level_);
    }

    LogLevel level() const { return level_; }
    SinkKind kind() const { return kind_; }
    const std::string& path() const { return path_; }

private:
    void write_line(const std::string& line) const;

    SinkKind kind_ = SinkStderr;
    std::string path_;
    LogLevel level_ = LogLevel::Warn;
    // mutable：日志是"读起来像 const、写起来有副作用"的典型；调用方持有 const Log&
    // 才能保证没人中途改级别。
    mutable std::mutex m_;
};

}  // namespace facerec

#endif  // FACERECOGNITION_LOG_HPP
