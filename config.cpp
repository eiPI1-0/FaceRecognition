#include "config.hpp"


#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace {

// 极小编辑距离，用于 "recog_thresh" 这类拼错的提示
int edit_distance(const std::string& a, const std::string& b) {
    std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = static_cast<int>(j);
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = static_cast<int>(i);
        for (size_t j = 1; j <= b.size(); ++j) {
            const int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
        }
        prev = cur;
    }
    return prev[b.size()];
}

// 去掉分隔符，让 `recogn_thresh` 与 `recognize.thresh` 在拼写纠错时等价
std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return std::string();
    const size_t e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

std::string normalize_key(const std::string& key) {
    std::string out;
    out.reserve(key.size());
    for (char c : key) {
        if (c != '.' && c != '_') out.push_back(c);
    }
    return out;
}

std::string last_segment(const std::string& key) {
    const size_t dot = key.rfind('.');
    return dot == std::string::npos ? key : key.substr(dot + 1);
}

// 读 knob 当前值（数值类型统一成 double；string 返回 false）
bool knob_as_double(const Knob& k, const PipelineConfig& cfg, double* out) {
    if (const auto* p = std::get_if<IntMember>(&k.member)) {
        *out = static_cast<double>(cfg.*(*p));
        return true;
    }
    if (const auto* p = std::get_if<FloatMember>(&k.member)) {
        *out = static_cast<double>(cfg.*(*p));
        return true;
    }
    return false;
}

std::string knob_as_string(const Knob& k, const PipelineConfig& cfg) {
    if (const auto* p = std::get_if<StrMember>(&k.member)) {
        return cfg.*(*p);
    }
    return std::string();
}

std::string format_range(const Knob& k) {
    const bool has_lo = std::isfinite(k.lo);
    const bool has_hi = std::isfinite(k.hi);
    if (has_lo && has_hi) {
        return std::string("[") + std::to_string(k.lo) + ", " + std::to_string(k.hi) + "]";
    }
    if (has_lo) return "> " + std::to_string(k.lo);
    if (has_hi) return "< " + std::to_string(k.hi);
    return "any";
}

}  // namespace

const std::vector<Knob>& knobs() {
    // key 命名 = 配置文件里的 key = --set 的 key = dump 的 key，三者同一字符串。
    // 表顺序 = dump 顺序；group 变化处打印一行分节标题，comment 写在该 key 上方。
    static const std::vector<Knob> table = {
        {"paths.detector_model", &PipelineConfig::detector_model,
         -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
         "paths", "relative paths resolve against the config file's directory"},
        {"paths.embedder_model", &PipelineConfig::embedder_model,
         -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
         "paths", ""},
        {"paths.data_dir", &PipelineConfig::data_dir,
         -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
         "paths", "one subfolder per enrolled person"},

        {"detector.input_w", &PipelineConfig::detector_input_w, 32, 1024,
         "model contract", "MODEL CONTRACT: only change together with the model file."},
        {"detector.input_h", &PipelineConfig::detector_input_h, 32, 1024,
         "model contract", "PriorBox::decode assumes the YuNet 160x120 prior-box layout"},
        {"embedder.input_w", &PipelineConfig::embedder_input_w, 32, 512,
         "model contract", "align_face() targets the same 112x112 ArcFace template"},
        {"embedder.input_h", &PipelineConfig::embedder_input_h, 32, 512,
         "model contract", ""},

        {"detector.conf_thresh", &PipelineConfig::conf_thresh, 0.0, 1.0,
         "detection", "face is kept above this detector confidence"},
        {"detector.nms_thresh", &PipelineConfig::nms_thresh, 0.0, 1.0,
         "detection", ""},
        {"detector.keep_top_k", &PipelineConfig::keep_top_k, 1, 10000,
         "detection", ""},

        {"recognize.thresh", &PipelineConfig::recogn_thresh, 0.0, 1.0,
         "recognition", "cosine above which a match is reported; after the alignment fix "
                        "same-identity sits near 1.0 and cross-identity near 0"},
        {"display.rfactor", &PipelineConfig::rfactor, 1, 16,
         "display", "display frame = detector input x rfactor; letterbox canvas, and the "
                    "scale draw()/putText() map detection coords by"},

        {"tracker.iou", &PipelineConfig::tracker_iou, 0.0, 1.0,
         "tracking", "IoU above which a detection is the same track"},
        {"tracker.detect_every", &PipelineConfig::detect_every, 1, 30,
         "tracking", "measured: detection 0.93 ms vs embedding 4.47 ms/face, so raising this "
                     "saves little (max box error 5.9px at 1, ~18px at 5)"},
        {"tracker.max_lost", &PipelineConfig::max_lost, 0, 1000,
         "tracking", "frames a track survives without a matching detection"},
        {"tracker.min_hits", &PipelineConfig::min_hits, 1, 100,
         "tracking", ""},

        {"dnn.backend", &PipelineConfig::backend,
         -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
         "backend", "cpu | cuda | openvino (cuda/openvino need an OpenCV build with them)"},
        {"dnn.batch_mode", &PipelineConfig::batch_mode, -1, 1,
         "backend", "0 = auto by backend (batch is a regression on the OpenCV CPU backend), "
                    "1 = force batch, -1 = force one-at-a-time"},
    };
    return table;
}

bool find_knob(const std::string& key, const Knob** knob, std::vector<std::string>* candidates) {
    for (const auto& k : knobs()) {
        if (key == k.key) {
            if (knob) *knob = &k;
            return true;
        }
    }
    if (knob) *knob = nullptr;
    if (!candidates) return false;
    // 相近 key 提示：去掉分隔符后比编辑距离（`recogn_thresh` → `recognize.thresh` 只差 3），
    // 或末段相同（拼错末段的常见情形）。
    const std::string want = normalize_key(key);
    for (const auto& k : knobs()) {
        const bool near = edit_distance(want, normalize_key(k.key)) <= 3;
        // 末段相同也能命中：用户漏掉组前缀时（`thresh` → `recognize.thresh`）
        const bool same_tail = last_segment(key) == last_segment(k.key);
        if (near || same_tail) candidates->push_back(k.key);
    }
    return false;
}

bool set_knob(const Knob& knob, const std::string& value, PipelineConfig& cfg, std::string& err) {
    if (auto* p = std::get_if<StrMember>(&knob.member)) {
        if (value.empty()) {
            err = std::string(knob.key) + ": empty value";
            return false;
        }
        cfg.*(*p) = value;
        return true;
    }

    // 严格解析：整串必须是数字，拒绝 "0.5x" 这类尾巴
    const char* begin = value.c_str();
    const char* end = begin + value.size();

    if (auto* p = std::get_if<IntMember>(&knob.member)) {
        int parsed = 0;
        const auto res = std::from_chars(begin, end, parsed);
        if (res.ec != std::errc() || res.ptr != end) {
            err = std::string(knob.key) + ": '" + value + "' is not an integer (expected " +
                  format_range(knob) + ")";
            return false;
        }
        if (parsed < knob.lo || parsed > knob.hi) {
            err = std::string(knob.key) + ": " + value + " is out of range " +
                  format_range(knob);
            return false;
        }
        cfg.*(*p) = parsed;
        return true;
    }

    if (auto* p = std::get_if<FloatMember>(&knob.member)) {
        char* stop = nullptr;
        const double parsed = std::strtod(begin, &stop);
        if (stop != end || stop == begin) {
            err = std::string(knob.key) + ": '" + value + "' is not a number (expected " +
                  format_range(knob) + ")";
            return false;
        }
        if (parsed < knob.lo || parsed > knob.hi) {
            err = std::string(knob.key) + ": " + value + " is out of range " +
                  format_range(knob);
            return false;
        }
        cfg.*(*p) = static_cast<float>(parsed);
        return true;
    }

    err = std::string(knob.key) + ": unsupported knob type";
    return false;
}

bool load_config(const std::string& path, PipelineConfig& cfg,
                 std::string& err, std::vector<std::string>& warnings) {
    // 手写 flat 解析器，而不是 cv::FileStorage：后者要求文件以 `%YAML:1.0` 签名开头
    // （只按内容嗅探、不看 .yml 扩展名），否则抛异常 —— 对一个人手编辑的配置文件是
    // 多余的怪癖。且这样 parse + 区间校验与 --set 完全共用一份实现。
    std::ifstream in(path);
    if (!in) {
        err = "cannot open config file: " + path;
        return false;
    }

    int line_no = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++line_no;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;

        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            err = path + ":" + std::to_string(line_no) + ": expected 'key: value', got '" + line + "'";
            return false;
        }
        const std::string key = trim(line.substr(0, colon));
        std::string value = trim(line.substr(colon + 1));
        if (value.size() >= 2 && value.front() == value.back() &&
            (value.front() == '"' || value.front() == '\'')) {
            value = value.substr(1, value.size() - 2);   // 允许 YAML 式引号
        }

        const Knob* knob = nullptr;
        std::vector<std::string> candidates;
        if (!find_knob(key, &knob, &candidates)) {
            std::string msg = path + ":" + std::to_string(line_no) + ": unknown key '" + key + "'";
            if (!candidates.empty()) {
                msg += "; did you mean";
                for (const auto& c : candidates) msg += " '" + c + "'";
            }
            warnings.push_back(msg + " (ignored)");
            continue;
        }
        if (!set_knob(*knob, value, cfg, err)) {
            err = path + ":" + std::to_string(line_no) + ": " + err;
            return false;
        }
    }
    return true;
}

bool validate_config(const PipelineConfig& cfg, std::string& err,
                     std::vector<std::string>& warnings) {
    for (const auto& k : knobs()) {
        double value = 0.0;
        if (!knob_as_double(k, cfg, &value)) continue;
        if (value < k.lo || value > k.hi) {
            err = std::string(k.key) + ": " + std::to_string(value) + " is out of range " +
                  format_range(k);
            return false;
        }
    }

    if (cfg.backend != "cpu" && cfg.backend != "cuda" && cfg.backend != "openvino") {
        err = "dnn.backend: '" + cfg.backend + "' must be one of cpu, cuda, openvino";
        return false;
    }

    // display_shape = detector_input * rfactor 是 draw()/putText() 坐标映射的前提，
    // 只校验派生量的合理性。
    const cv::Size disp = cfg.display_shape();
    if (disp.width < 160 || disp.height < 120) {
        err = "display.rfactor: " + std::to_string(cfg.rfactor) +
              " gives a display frame smaller than the detector input";
        return false;
    }

    // 模型契约警告：用默认模型文件名却给了别的输入尺寸，几乎一定是接错。
    if (cfg.detector_model == "./models/yunet_120x160.onnx" &&
        (cfg.detector_input_w != 160 || cfg.detector_input_h != 120)) {
        warnings.push_back("detector.input_w/input_h (" + std::to_string(cfg.detector_input_w) +
                           "x" + std::to_string(cfg.detector_input_h) +
                           ") do not match the default YuNet 120x160 model; "
                           "PriorBox::decode assumes that layout");
    }
    if (cfg.embedder_model == "./models/mobilefacenet.onnx" &&
        (cfg.embedder_input_w != 112 || cfg.embedder_input_h != 112)) {
        warnings.push_back("embedder.input_w/input_h do not match the default MobileFaceNet "
                           "112x112 model; align_face() targets the same 112x112 template");
    }
    if (cfg.detect_every > 1) {
        warnings.push_back("tracker.detect_every > 1 trades ~0.75 ms/frame for box lag "
                           "(measured: max box error 5.9px at 1, ~18px at 5)");
    }
    return true;
}

void dump_config(std::ostream& os, const PipelineConfig& cfg) {
    // flat 点分 key：与配置文件、--set 完全同构，可 grep、可 diff。
    // 注释由表里的 group/comment 生成 —— 示例配置永远和代码里的默认值一致。
    os << "# FaceRecognition configuration -- generated by `main --dump-config`\n"
       << "# Load with `--config <path>`, override with `--set key=value`.\n"
       << "# Unknown keys are rejected with the nearest valid ones.\n";

    const char* group = "";
    for (const auto& k : knobs()) {
        if (std::strcmp(group, k.group) != 0) {
            group = k.group;
            os << "\n# --- " << group << " ---\n";
        }
        if (*k.comment) {
            for (const char* line = k.comment; line;) {
                os << "# ";
                const char* nl = std::strchr(line, '\n');
                if (nl) { os.write(line, nl - line) << "\n"; line = nl + 1; }
                else { os << line << "\n"; line = nullptr; }
            }
        }
        os << k.key << ": ";
        if (const auto* p = std::get_if<StrMember>(&k.member)) {
            os << cfg.*(*p) << "\n";
            continue;
        }
        double v = 0.0;
        knob_as_double(k, cfg, &v);
        if (std::get_if<IntMember>(&k.member)) {
            os << static_cast<long>(std::lround(v)) << "\n";
        } else {
            os.precision(2);
            os << std::showpoint << std::fixed << v << "\n";
        }
    }
}
