#ifndef FACEREC_CLI_DEMO_HPP
#define FACEREC_CLI_DEMO_HPP

// 演示用实时循环（摄像头/图片/视频 + 窗口 + 画框）。保留原样，供 `facerec demo`
// 与旧命令行形式 `./main 0` 使用；生产路径是 identify/enroll/serve/mcp。
// 它是这个仓库里唯一碰 highgui 的地方。
int run_demo(int argc, char* argv[]);

#endif  // FACEREC_CLI_DEMO_HPP
