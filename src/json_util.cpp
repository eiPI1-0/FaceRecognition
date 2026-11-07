#include "json_util.hpp"

namespace facerec::json {

namespace {

// 定位 `"key"` 之后冒号后面的第一个非空白字符的位置。
size_t value_start(const std::string& body, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return std::string::npos;
    pos = body.find(':', pos + pat.size());
    if (pos == std::string::npos) return std::string::npos;
    return body.find_first_not_of(" \t\r\n", pos + 1);
}

std::string unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out += s[i];
            continue;
        }
        const char e = s[++i];
        switch (e) {
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            default: out += e;
        }
    }
    return out;
}

}  // namespace

std::string string(const std::string& body, const std::string& key,
                   const std::string& fallback) {
    size_t pos = value_start(body, key);
    if (pos == std::string::npos || pos >= body.size() || body[pos] != '"') {
        return fallback;
    }
    ++pos;
    std::string raw;
    while (pos < body.size() && body[pos] != '"') {
        if (body[pos] == '\\' && pos + 1 < body.size()) {
            raw += body[pos];
            raw += body[pos + 1];
            pos += 2;
        } else {
            raw += body[pos++];
        }
    }
    if (pos >= body.size()) return fallback;   // 未闭合
    return unescape(raw);
}

bool number(const std::string& body, const std::string& key, double& out) {
    const size_t pos = value_start(body, key);
    if (pos == std::string::npos || pos >= body.size()) return false;
    try {
        size_t used = 0;
        const double v = std::stod(body.substr(pos), &used);
        if (used == 0) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool integer(const std::string& body, const std::string& key, int& out) {
    double v = 0;
    if (!number(body, key, v)) return false;
    out = static_cast<int>(v);
    return true;
}

bool boolean(const std::string& body, const std::string& key, bool& out) {
    const size_t pos = value_start(body, key);
    if (pos == std::string::npos) return false;
    if (body.compare(pos, 4, "true") == 0) { out = true; return true; }
    if (body.compare(pos, 5, "false") == 0) { out = false; return true; }
    return false;
}

std::vector<std::string> string_array(const std::string& body, const std::string& key) {
    std::vector<std::string> out;
    const size_t start = value_start(body, key);
    if (start == std::string::npos || start >= body.size() || body[start] != '[') {
        return out;
    }
    const size_t end = body.find(']', start);
    if (end == std::string::npos) return out;
    size_t cur = start + 1;
    while (cur < end) {
        cur = body.find('"', cur);
        if (cur == std::string::npos || cur >= end) break;
        const size_t close = body.find('"', cur + 1);
        if (close == std::string::npos || close > end) break;
        out.push_back(unescape(body.substr(cur + 1, close - cur - 1)));
        cur = close + 1;
    }
    return out;
}

std::string object(const std::string& body, const std::string& key) {
    const size_t start = value_start(body, key);
    if (start == std::string::npos || start >= body.size() || body[start] != '{') {
        return {};
    }
    int depth = 0;
    bool in_string = false;
    for (size_t i = start; i < body.size(); ++i) {
        const char c = body[i];
        if (in_string) {
            if (c == '\\') { ++i; continue; }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') { in_string = true; continue; }
        if (c == '{') { ++depth; continue; }
        if (c == '}') {
            if (--depth == 0) {
                return body.substr(start + 1, i - start - 1);
            }
        }
    }
    return {};   // 括号不配平
}

}  // namespace facerec::json