#ifndef FACERECOGNITION_ENCODING_HPP
#define FACERECOGNITION_ENCODING_HPP

#include <string>
#include <vector>

// base64（RFC 4648 标准字母表，带 '=' 填充）。HTTP 的 image_b64 字段、MCP 的 image
// content 都要它，所以从各外壳里抽出来共用一份。
namespace facerec {

std::string base64_encode(const std::vector<unsigned char>& buf);
std::string base64_encode(const std::string& raw);
// 忽略 '='、空白与任何非字母表字符（宽容解码：调用方把坏输入交给 imdecode 报错）。
std::string base64_decode(const std::string& in);

}  // namespace facerec

#endif  // FACERECOGNITION_ENCODING_HPP