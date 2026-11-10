#include "common/encoding.hpp"

#include <array>
#include <cstdint>

namespace facerec {

namespace {

constexpr const char* kAlphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const std::array<int8_t, 256>& decode_table() {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int i = 0; i < 64; ++i) {
            t[static_cast<unsigned char>(kAlphabet[i])] = static_cast<int8_t>(i);
        }
        return t;
    }();
    return table;
}

}  // namespace

std::string base64_encode(const std::vector<unsigned char>& buf) {
    std::string out;
    out.reserve(buf.size() * 4 / 3 + 4);
    size_t k = 0;
    for (; k + 2 < buf.size(); k += 3) {
        out += kAlphabet[(buf[k] >> 2) & 0x3F];
        out += kAlphabet[((buf[k] << 4) | (buf[k + 1] >> 4)) & 0x3F];
        out += kAlphabet[((buf[k + 1] << 2) | (buf[k + 2] >> 6)) & 0x3F];
        out += kAlphabet[buf[k + 2] & 0x3F];
    }
    const size_t rem = buf.size() - k;
    if (rem == 1) {
        out += kAlphabet[(buf[k] >> 2) & 0x3F];
        out += kAlphabet[(buf[k] << 4) & 0x3F];
        out += "==";
    } else if (rem == 2) {
        out += kAlphabet[(buf[k] >> 2) & 0x3F];
        out += kAlphabet[((buf[k] << 4) | (buf[k + 1] >> 4)) & 0x3F];
        out += kAlphabet[(buf[k + 1] << 2) & 0x3F];
        out += '=';
    }
    return out;
}

std::string base64_encode(const std::string& raw) {
    return base64_encode(std::vector<unsigned char>(raw.begin(), raw.end()));
}

std::string base64_decode(const std::string& in) {
    const auto& table = decode_table();
    std::string out;
    out.reserve(in.size() * 3 / 4 + 3);
    int accum = 0;
    int bits = 0;
    for (unsigned char c : in) {
        const int8_t v = table[c];
        if (v < 0) continue;   // '=' / 空白 / 非法字符
        accum = (accum << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((accum >> bits) & 0xFF));
        }
    }
    return out;
}

}  // namespace facerec