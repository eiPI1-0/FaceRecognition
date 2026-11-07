#ifndef FACERECOGNITION_SHA256_HPP
#define FACERECOGNITION_SHA256_HPP

#include <string>

// 自包含的 SHA-256（FIPS 180-4）。用来给"同一张图 + 同一组参数"做缓存键。
//
// 为什么不链 OpenSSL：本仓库刻意只依赖 OpenCV + pthread（HTTPS 由反向代理终结），
// 而这里需要的只是一个 32 字节摘要，不是一个密码学库。scripts/get_models.sh 里
// 校验模型用的也是 sha256，所以这不是引入新概念。
namespace facerec {

// 返回 64 个小写十六进制字符。
std::string sha256_hex(const std::string& data);

}  // namespace facerec

#endif  // FACERECOGNITION_SHA256_HPP