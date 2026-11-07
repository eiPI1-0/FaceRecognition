#ifndef FACERECOGNITION_JSON_UTIL_HPP
#define FACERECOGNITION_JSON_UTIL_HPP

#include <string>
#include <vector>

// 极小的 JSON 取值器：只处理本仓库自己产生的扁平结构（HTTP 请求体、MCP 工具参数）。
//
// 这**不是**通用解析器：不支持任意嵌套、数字的完整语法、代理对。目标是为"读几个已知
// 字段"这件事消灭一个依赖，而不是重新实现 JSON。所有函数在解析失败时返回默认值，
// 从不抛异常 —— 调用方据此回 400，而不是崩在半途。
//
// 需要读嵌套字段时先用 object() 把子对象切成字符串，再在子串上取值。
namespace facerec::json {

// "key":"value"（带 \" \\ \n \t \r \/ 反转义）
std::string string(const std::string& body, const std::string& key,
                   const std::string& fallback = {});

// "key":123 / "key":1.5
bool number(const std::string& body, const std::string& key, double& out);
bool integer(const std::string& body, const std::string& key, int& out);

// "key":true / "key":false
bool boolean(const std::string& body, const std::string& key, bool& out);

// "key":["a","b"]（元素必须是不含转义的字符串）
std::vector<std::string> string_array(const std::string& body, const std::string& key);

// "key":{...} —— 返回花括号之间的内容（按深度匹配，跳过字符串里的花括号）。
// 找不到或括号不配平时返回空串。
std::string object(const std::string& body, const std::string& key);

}  // namespace facerec::json

#endif  // FACERECOGNITION_JSON_UTIL_HPP