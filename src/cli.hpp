#ifndef FACERECOGNITION_CLI_HPP
#define FACERECOGNITION_CLI_HPP

#include <string>
#include <vector>

// 子命令入口（P1: identify / enroll；serve/mcp 留 P2/P3 打桩）。
int cmd_identify(int argc, char* argv[]);
int cmd_enroll(int argc, char* argv[]);
int cmd_serve(int argc, char* argv[]);
int cmd_mcp(int argc, char* argv[]);
// 生成 bearer token 并（可选）追加到 token 表：`./main token --name ci`。
int cmd_token(int argc, char* argv[]);

// compare <img-a> <img-b>：两张图最大人脸的余弦相似度（1.0 = 同一人）。
int cmd_compare(int argc, char* argv[]);
// verify <img> --name <person>：图中最优人脸是否属于指定身份。
int cmd_verify(int argc, char* argv[]);

#endif // FACERECOGNITION_CLI_HPP
