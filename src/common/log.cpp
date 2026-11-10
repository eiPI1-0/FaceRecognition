#include "common/log.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

namespace facerec {

namespace {

// 小写化后再比：配置文件里写 `WARN` 不该变成启动失败。
std::string lower(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

}  // namespace

bool parse_log_level(const std::string& s, LogLevel& out) {
    const std::string v = lower(s);
    if (v == "off") {
        out = LogLevel::Off;
    } else if (v == "error") {
        out = LogLevel::Error;
    } else if (v == "warn") {
        out = LogLevel::Warn;   // 不收 "warning"：注释只列一个拼写，多一个同义词就是多一个猜谜
    } else if (v == "info") {
        out = LogLevel::Info;
    } else if (v == "debug") {
        out = LogLevel::Debug;
    } else {
        return false;
    }
    return true;
}

const char* log_level_name(LogLevel l) {
    switch (l) {
        case LogLevel::Off: return "off";
        case LogLevel::Error: return "error";
        case LogLevel::Warn: return "warn";
        case LogLevel::Info: return "info";
        case LogLevel::Debug: return "debug";
    }
    return "warn";
}

std::string iso8601_now() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto secs = time_point_cast<seconds>(now);
    const auto ms = duration_cast<milliseconds>(now - secs).count();
    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900,
                  tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<int>(ms));
    return buf;
}

bool Log::configure(const std::string& spec, std::string& err) {
    if (spec == "off") {
        kind_ = SinkOff;
        path_.clear();
        return true;
    }
    if (spec.empty() || spec == "stderr") {
        kind_ = SinkStderr;
        path_.clear();
        return true;
    }
    {
        std::ofstream probe(spec, std::ios::app);
        if (!probe) {
            err = "cannot open log for append: " + spec;
            return false;
        }
    }
    kind_ = SinkFile;
    path_ = spec;
    return true;
}

bool Log::set_level(const std::string& level_spec, std::string& err) {
    LogLevel parsed = LogLevel::Warn;
    if (!parse_log_level(level_spec, parsed)) {
        err = "log.level must be one of off|error|warn|info|debug, got `" + level_spec + "`";
        return false;
    }
    level_ = parsed;
    return true;
}

void Log::write_line(const std::string& line) const {
    if (kind_ == SinkOff) return;
    // 整行在同一把锁里写完：stderr 是共享的，两个 worker 各写半行会产出无法解析的行。
    std::lock_guard<std::mutex> lk(m_);
    if (kind_ == SinkStderr) {
        std::cerr << line << '\n';
        return;
    }
    std::ofstream out(path_, std::ios::app);
    if (out) out << line << '\n';
}

namespace {

// 见 log.hpp：fields 是成员列表。调用方很惯性地会带上外层括号，所以这里剔掉；
// 不剔的话多出的那对 {} 会把整行变成非法 JSON，而这正是最难查的那一类错。
std::string strip_braces(const std::string& fields) {
    if (fields.size() < 2 || fields.front() != '{' || fields.back() != '}') return fields;
    return fields.substr(1, fields.size() - 2);
}

}  // namespace

void Log::audit(const char* kind, const std::string& fields) const {
    if (kind_ == SinkOff) return;
    const std::string f = strip_braces(fields);
    std::ostringstream os;
    os << "{\"audit\":\"" << kind << "\",\"ts\":\"" << iso8601_now() << '"';
    if (!f.empty()) os << ',' << f;
    os << '}';
    write_line(os.str());
}

void Log::event(LogLevel level, const char* name, const std::string& fields) const {
    if (!enabled(level)) return;
    const std::string f = strip_braces(fields);
    std::ostringstream os;
    os << "{\"level\":\"" << log_level_name(level) << "\",\"event\":\"" << name << "\",\"ts\":\""
       << iso8601_now() << '"';
    if (!f.empty()) os << ',' << f;
    os << '}';
    write_line(os.str());
}

}  // namespace facerec
