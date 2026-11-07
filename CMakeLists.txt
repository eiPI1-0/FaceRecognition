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
# cpp-httplib 是单头 header-only 库（third_party/httplib.h，v0.15.3，sha256
# a3347656c71c81c3fe510fdb1ff229d7db52eb7df04d3342d340eba0b6ad50ea）。
# 需要 pthread；不引入 OpenSSL，HTTPS 由反向代理终结。
find_package(Threads REQUIRED)
target_link_libraries(main ${OpenCV_LIBS} Threads::Threads)
