cmake_minimum_required(VERSION 3.10)

project(FaceRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 没有 build type 时 OpenCV 找到的库仍可用，但性能数字没有意义：默认给 Release，
# 需要调试符号时显式 -DCMAKE_BUILD_TYPE=Debug（-O0 -g）。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main
    main.cpp
    config.cpp
    priorbox.cpp
    detector_head.cpp
    utils.cpp
    src/engine.cpp
    src/auth.cpp
    src/api.cpp
    src/cli.cpp
    src/server.cpp
    src/mcp.cpp
    src/encoding.cpp
    src/json_util.cpp
    src/pipeline_pool.cpp
    src/result_cache.cpp
    src/sha256.cpp
    src/gallery.cpp
    src/metrics.cpp
    face_tracker.cpp
)
target_include_directories(main PRIVATE src ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/third_party)

# cpp-httplib 的 listen backlog 默认是 **5**（httplib.h:117，且 listen() 没有 backlog 参数，
# 只能靠这个宏在 include 前定下来）。实测：并发连接数 >5 时 accept 队列溢出，内核直接丢 SYN，
# 客户端在 1 秒后重传 —— 压测里表现为"少数请求 connect 耗时 1014ms、总耗时 ~1.04s"，
# 而服务端只花了 16ms。1024 被 net.core.somaxconn 截断（本机 4096），够用且不撒谎。
target_compile_definitions(main PRIVATE CPPHTTPLIB_LISTEN_BACKLOG=1024)
# httplib 对 `Content-Type: application/x-www-form-urlencoded` 的请求体内建 8KB 上限
# （httplib.h:6160，超了直接 413），而 `curl --data-binary @photo.jpg` 默认发的就是这个
# Content-Type —— 一张 48KB 的照片就会莫名其妙 413。更糟的是这个上限在不同 TU 布局下行为
# 不一致：同一个请求、同一个 httplib.h，2 个 TU 包含它时放行，加入第三个包含它的 TU 后 413
# （实测，见 PLAN.md）。我们的 API 吃原始字节，语义不该随客户端填的 Content-Type 变化，
# 所以把上限显式钉到"远高于 payload 上限"，让 payload 上限成为唯一的体积闸门。
target_compile_definitions(main PRIVATE CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=268435456)
# cpp-httplib 是单头 header-only 库（third_party/httplib.h，v0.15.3，sha256
# a3347656c71c81c3fe510fdb1ff229d7db52eb7df04d3342d340eba0b6ad50ea）。
# 需要 pthread；不引入 OpenSSL，HTTPS 由反向代理终结。
find_package(Threads REQUIRED)
target_link_libraries(main ${OpenCV_LIBS} Threads::Threads)

# ---- 回归测试（PLAN.md D6）----
# 只放"纯函数"层的东西：不需要模型、不需要摄像头、不需要网络，`ctest` 在任何机器上都能跑。
# 服务端契约（鉴权/413/优雅停机）需要权重文件，等有了再单独加，并用 SKIP_RETURN_CODE 跳过缺模型。
option(FACEREC_BUILD_TESTS "Build the regression tests (ctest)" ON)
if(FACEREC_BUILD_TESTS)
    enable_testing()
    add_executable(decode_gates tests/decode_gates.cpp priorbox.cpp)
    target_include_directories(decode_gates PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(decode_gates ${OpenCV_LIBS})
    add_test(NAME decode_gates COMMAND decode_gates)

    # tracker 的六条脚本化序列：纯跟踪入口驱动，不加载模型（见 face_tracker.hpp）。
    add_executable(tracker_scenarios tests/tracker_scenarios.cpp face_tracker.cpp)
    target_include_directories(tracker_scenarios PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(tracker_scenarios ${OpenCV_LIBS})
    add_test(NAME tracker_scenarios COMMAND tracker_scenarios)
endif()
