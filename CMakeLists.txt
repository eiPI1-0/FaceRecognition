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
find_package(Threads REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 单一源码根：src/。项目头一律 include 成 "<层>/<文件>"（如 "core/engine.hpp"），
# 于是依赖方向写在 include 行里；层序与检查见 scripts/check_layers.py。
set(FACEREC_ROOT ${CMAKE_SOURCE_DIR})

# ---- 层 common：叶子，不依赖本项目任何东西 ----
add_library(facerec_common STATIC
    src/common/log.cpp
    src/common/json_util.cpp
    src/common/encoding.cpp
    src/common/sha256.cpp
    src/common/metrics.cpp
)
target_include_directories(facerec_common PUBLIC ${FACEREC_ROOT}/src)
target_link_libraries(facerec_common PUBLIC Threads::Threads)

# ---- 层 core：识别内核，只用 OpenCV（不认识 config / JSON / 网络）----
add_library(facerec_core STATIC
    src/core/priorbox.cpp
    src/core/detector_head.cpp
    src/core/utils.cpp
    src/core/engine.cpp
    src/core/face_tracker.cpp
)
target_include_directories(facerec_core PUBLIC ${FACEREC_ROOT}/src)
target_link_libraries(facerec_core PUBLIC facerec_common ${OpenCV_LIBS})

# ---- 层 app：配置 + 库 + 阈值 + 三个外壳共享的结果格式 ----
add_library(facerec_app STATIC
    src/app/config.cpp
    src/app/evaluation.cpp
    src/app/gallery.cpp
    src/app/api.cpp
)
target_include_directories(facerec_app PUBLIC ${FACEREC_ROOT}/src)
target_link_libraries(facerec_app PUBLIC facerec_core)

# ---- cpp-httplib 的两个宏必须对**每个**包含它的 TU 一致 ----
# listen backlog 默认是 **5**（httplib.h:117，且 listen() 没有 backlog 参数，只能靠宏）。
# 实测：并发连接数 >5 时 accept 队列溢出，内核直接丢 SYN，客户端在 1 秒后重传 —— 压测里
# 表现为"少数请求 connect 耗时 1014ms、总耗时 ~1.04s"，而服务端只花了 16ms。1024 被
# net.core.somaxconn 截断（本机 4096），够用且不撒谎。
# 另一个宏：httplib 对 `Content-Type: application/x-www-form-urlencoded` 的请求体内建
# 8KB 上限（httplib.h:6160，超了直接 413），而 `curl --data-binary @photo.jpg` 默认发的就是
# 这个 Content-Type。这个上限在不同 TU 布局下行为不一致：同一个请求、同一个 httplib.h，
# 2 个 TU 包含它时放行，加入第三个 TU 后 413（实测，见 PLAN.md）。我们的 API 吃原始字节，
# 语义不该随客户端填的 Content-Type 变化，所以把上限钉到远高于 payload 上限。
# 按 target 手抄这两个宏 = "新 target 忘了抄就静默换个上限"，所以做成 INTERFACE target：
# 谁包含 httplib，谁就必须链它。（CMake 3.10 没有 add_compile_definitions，3.12 才有。）
add_library(facerec_httplib INTERFACE)
target_include_directories(facerec_httplib INTERFACE ${FACEREC_ROOT}/third_party)
target_compile_definitions(facerec_httplib INTERFACE
    CPPHTTPLIB_LISTEN_BACKLOG=1024
    CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=268435456
)

# ---- 层 serve：传输与容量（HTTP / MCP / 鉴权 / 池 / 缓存）----
add_library(facerec_serve STATIC
    src/serve/auth.cpp
    src/serve/pipeline_pool.cpp
    src/serve/result_cache.cpp
    src/serve/server.cpp
    src/serve/mcp.cpp
)
target_include_directories(facerec_serve PUBLIC ${FACEREC_ROOT}/src)
target_link_libraries(facerec_serve PUBLIC facerec_app facerec_common facerec_httplib)

# ---- 层 cli + 入口 ----
add_executable(main
    src/main.cpp
    src/cli/cli.cpp
)
target_link_libraries(main PRIVATE facerec_serve)

# ---- 回归测试（PLAN.md D6）----
# 只放"纯函数"层的东西：不需要模型、不需要摄像头、不需要网络，`ctest` 在任何机器上都能跑。
# 测试 link target，不再抄源文件清单——清单只有 CMakeLists.txt 一处，漏了就是链接错误。
option(FACEREC_BUILD_TESTS "Build the regression tests (ctest)" ON)
if(FACEREC_BUILD_TESTS)
    enable_testing()

    add_executable(decode_gates tests/decode_gates.cpp)
    target_link_libraries(decode_gates PRIVATE facerec_core)
    add_test(NAME decode_gates COMMAND decode_gates)

    # tracker 的六条脚本化序列：纯跟踪入口驱动，不加载模型（见 face_tracker.hpp）。
    add_executable(tracker_scenarios tests/tracker_scenarios.cpp)
    target_link_libraries(tracker_scenarios PRIVATE facerec_core)
    add_test(NAME tracker_scenarios COMMAND tracker_scenarios)

    # 准入控制：端到端"等容量"预算在两个准入点之间的连续性、以及 httplib 按连接而不是
    # 按请求排队带来的记账语义。不需要模型与网络（metrics.cpp 是纯计数器）。
    add_executable(admission tests/admission.cpp)
    target_link_libraries(admission PRIVATE facerec_common)
    add_test(NAME admission COMMAND admission)

    # 阈值评测的统计层：用已知角度的手造特征断言 FAR/FRR。不需要模型与数据 ——
    # 错的不是分数（那由模型决定），是分数到阈值的映射（那由这份代码决定）。
    add_executable(evaluation tests/evaluation.cpp)
    target_link_libraries(evaluation PRIVATE facerec_app)
    add_test(NAME evaluation COMMAND evaluation)

    # 布局守卫：include 只能向下（见 RFPLAN.md §3.1）。缺 python3 时跳过并留一条 WARNING，
    # 而不是让 ctest 变绿得不明不白。
    find_program(FACEREC_PYTHON3 NAMES python3)
    if(FACEREC_PYTHON3)
        add_test(NAME layers
                 COMMAND ${FACEREC_PYTHON3} ${FACEREC_ROOT}/scripts/check_layers.py ${FACEREC_ROOT})
    else()
        message(WARNING "python3 not found: the `layers` test is disabled")
    endif()

    # augmentation A/B：同一套 warp 种子下，对比不同检测器的 within/between margin。
    add_executable(augmentation_ab tests/augmentation_ab.cpp)
    target_link_libraries(augmentation_ab PRIVATE facerec_core)
    # 不加入 ctest：需要 models/ 和 data/ 资产，且运行时间取决于身份数。
endif()
