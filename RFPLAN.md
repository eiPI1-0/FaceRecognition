# RFPLAN — 源码布局重构（工作文档）

> **这份文档的用途**：上下文被压缩、或换一个会话接手时，**从这里接着做**。
> 诊断、取舍、文件映射、CMake 草案、验证配方、进度台账都在这里；不要重新论证方案。
>
> **状态**：**本计划已完成并定版 `v1.3.0`**（P1 `acc6903` → P3.5 `0f0959a`，逐条见 §10 台账）。
> 下一步：无。P4+ 的四条候选在 §12，**未列入本计划**，每条都带"为什么现在不做"——要动就另开小计划。
> **基线**：git `7d7d27f`（v1.2.0 之后）。重构后实测：`ctest` 5/5（0.71 s，无需权重）、
> clean build **22.74 s**（基线 32.86 s）、改一个子命令重编 1.4-1.9 s（基线 11.55 s）。
>
> **铁律**：每个 Phase 结束必须满足
> 1) `make && ctest` 全绿（P1 起是 5 个测试，含 `layers`）；
> 2) 第 7 节的端到端配方跑过一遍（identify / dump-config 往返 / serve / JSON 日志）；
> 3) 绿了才 commit，才进下一个 Phase。
> 发现方案有问题：先在 §10 台账里记一行、停下来，**不要静默偏离**。
>
> **本文件本身**：未加进 `.gitignore`，也未 commit。各 Phase commit 时用**显式路径**
> `git add`（如 `git add src CMakeLists.txt tests scripts README.md CHANGELOG.md`），
> 不要 `git add -A`，否则会把本文件带进历史 —— 除非明确决定入库。

---

## 0. 目标（一句话）

把所有编译单元收进**一个源码根** `src/`，用**目录表达层**、用**构建图表达依赖**、
用**ctest 断言规则**；根目录只剩构建文件、公开文档、配置示例和四个目录。

**状态：已达成**（P1 `acc6903` / P2 `a6e3ce7` / P3.1 `9536f3f` / P3.2 `c6338e9` / P3.4 `a021182` / P3.5 `0f0959a` / docs `b3e78ca`）。
P4+ 的候选在 §12，**未列入本计划**——要做就先开新的小计划，别顺手做大动作。

---

## 1. 为什么动（诊断，带实测）

### 1.1 六条具体问题

| # | 问题 | 证据（写就本文件时） |
|---|---|---|
| 1 | 两个源码根 + 三个 include 根（`src/`、仓库根、`third_party`） | 同类文件两种写法：`src/*.cpp` 写 `#include "api.hpp"`；`main.cpp:12` 写 `#include "src/cli.hpp"`（全仓唯一一例） |
| 2 | 分层只活在约定里，构建里没有 | `main` 一个 target 平铺 21 个源文件；`augmentation_ab` 手抄 `src/engine.cpp priorbox.cpp detector_head.cpp utils.cpp face_tracker.cpp`；`tests/admission` 手抄 `src/metrics.cpp`。搬/改名要改 4 处清单 |
| 3 | 命名空间是半套 | `facerec::` 只覆盖 `api` 往上；识别内核 `Detector/FeatureExtractor/PriorBox/Face/FaceTracker/align_*` 在全局；`FaceRecognition.h` 是只为不惊动 `main.cpp` 的转发头（唯一用户就是 `main.cpp:8`） |
| 4 | 编译耦合没有边界 | `src/auth.hpp:20` 直接 `#include <httplib.h>`（310 KB）→ `src/cli.cpp` 单 TU **11.55 s**、`src/server.cpp` **10.47 s**；而 `src/engine.cpp` 1.83 s、`src/metrics.cpp` 0.50 s（见 §1.2） |
| 5 | 文件粒度失衡 | `src/cli.cpp` 1266 行 = 8 个子命令 + 12 个共享辅助函数；`main.cpp` 442 行 = 分发 + 采集循环 + 识别状态机 + usage |
| 6 | 文档与物理路径强耦合 | README/PLAN/AGENTS 共约 145 处 `.cpp/.hpp` 路径提及；搬文件必须同步，否则文档立刻开始骗人 |

顺带两个已存在的"宏不一致"隐患：`CPPHTTPLIB_LISTEN_BACKLOG` 与
`CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH` 现在只挂在 `main` target 上。
任何新 target 包含 httplib 时忘了复制，就重现 AGENTS 里"2 个 TU 放行、3 个 TU 413"
那类 bug —— 本质是不同 TU 看到不同宏。这是结构问题，注释治不了。

### 1.2 基线数字（P0 记录，改完用来对比）

| 指标 | 基线 |
|---|---|
| `ctest` | 4/4 passed，0.64–0.81 s |
| 全量 clean build（Release，`-j16`） | **32.86 s** |
| `src/cli.cpp` 单 TU（`-O3`） | **11.55 s** |
| `src/server.cpp` 单 TU | 10.47 s |
| `src/config.cpp` 单 TU | 2.12 s |
| `src/engine.cpp` 单 TU | 1.83 s |
| `src/metrics.cpp` 单 TU | 0.50 s |
| 触及 `src/engine.hpp` + `config.hpp` + `face_tracker.hpp` | 重编 17 个 TU |
| 单 TU 计时配方 | `cmake -S . -B /tmp/frbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build /tmp/frbuild -j16`，再用 python 从 `compile_commands.json` 取命令行、去掉 `-o/-c`、加 `-c -o /tmp/x.o` 计时（见 §7.6） |
| 端到端基线 | `./main identify --image data/BruceLi/*.jpeg` → `"matched":{"person_id":"BruceLi","score":1.0000}`；`./main 1` → `Loaded 2 person(s)` + `Cannot open source: 1`，退出码 255（-1） |

`build/` 里还有一个**过期对象** `CMakeFiles/main.dir/FaceRecognition.cpp.o`（源文件早删了，
CMake 没重新配置过）—— 这也是 P1 必须 clean reconfigure 的证据。

---

## 2. 目标结构

```
FaceRecognition/
├── CMakeLists.txt
├── README.md  CHANGELOG.md  PLAN.md  LICENSE
├── config.example.yml
├── src/
│   ├── main.cpp                     # 只做 argv → 子命令分发
│   ├── common/                      # 叶子层：不 include 任何本项目头
│   │   ├── log.{hpp,cpp}  json_util.{hpp,cpp}  encoding.{hpp,cpp}
│   │   └── sha256.{hpp,cpp}  metrics.{hpp,cpp}
│   ├── core/                        # 识别内核：只用 OpenCV，不认识配置/JSON/网络
│   │   ├── face_def.hpp  priorbox.{hpp,cpp}  detector_head.{hpp,cpp}
│   │   ├── utils.{hpp,cpp}  engine.{hpp,cpp}  face_tracker.{hpp,cpp}
│   ├── app/                         # 配置 + 库 + 阈值 + 三外壳共享的结果格式
│   │   ├── config.{hpp,cpp}  evaluation.{hpp,cpp}
│   │   └── gallery.{hpp,cpp}  api.{hpp,cpp}
│   ├── serve/                       # 传输与容量：HTTP / MCP / 鉴权 / 池 / 缓存
│   │   ├── admission.hpp  timed_queue.hpp  auth.{hpp,cpp}
│   │   ├── pipeline_pool.{hpp,cpp}  result_cache.{hpp,cpp}
│   │   └── server.{hpp,cpp}  mcp.{hpp,cpp}
│   └── cli/                         # 外壳：一个子命令一个文件（P2 后）
│       ├── cli.hpp  args.{hpp,cpp}  demo.{hpp,cpp}
│       └── identify.cpp enroll.cpp token.cpp serve.cpp mcp.cpp
│           evaluate.cpp compare.cpp verify.cpp
├── tests/{decode_gates,tracker_scenarios,admission,evaluation,augmentation_ab}.cpp
├── scripts/{get_models.sh,stress.py,check_layers.py}
├── third_party/httplib.h
└── deploy/
```

---

## 3. 不变量与取舍

### 3.1 三条规则（"优雅"在这里，不在目录名）

1. **依赖只有一个方向，而且写在 include 行里。**
   `common(0) → core(1) → app(2) → serve(3) → cli(4) → src/main.cpp(5)`；
   只能向下 include，**同层可以互引**；项目头一律写成 `"<层>/<文件>"`。
   于是 `#include "core/engine.hpp"` 这一行本身就说明了依赖方向，违规在 diff 里一眼可见。
   （现状其实已接近：`api.hpp`→`config/engine/gallery`、`pipeline_pool.hpp`→`api`、
   `server.cpp`→全部，都在合法方向。规则只是把它说出来并守住。）

2. **构建图 = 结构图。**
   四个静态库 `facerec_common / facerec_core / facerec_app / facerec_serve` + `main`；
   测试 **link target**，不再抄源文件清单。忘了把新 `.cpp` 加进 target → 链接期就炸，
   而不是"编过了但没跑"。

3. **规则由 ctest 断言。**
   第 5 个测试 `layers`：`scripts/check_layers.py` 扫 `src/**`，断言
   (a) 被 include 的层号 ≤ 自己的层号；(b) 项目 include 必须带层前缀；
   (c) 每个 `src/**/*.cpp` 都出现在 `CMakeLists.txt` 里；
   (d) `httplib.h` 用尖括号（third_party 不进规则）。跑一次 ~30 ms。

### 3.2 刻意不做的五件事（以及原因）

| 不做 | 原因 |
|---|---|
| `include/facerec/` 公共头树 | 没有外部消费者（单一可执行文件 + 树内测试）。公共/私有分离 = 多一层目录 + 头文件同步仪式。真要做 SDK 时再拆，那时 `core/` 已是干净边界 |
| 一模块一目录 | 全仓 ~6.4k 行、15 个模块，2 文件一目录是过度切分。目录回答"谁能依赖谁"；某模块长到 1000+ 行再给它开子目录，不预先切 |
| 重写 CHANGELOG / PLAN 的历史路径 | 那是当时的路径，改了就成伪造历史。只在最新一节加"布局调整"条目 |
| 统一 header guard 命名 | 纯 churn（`UTILS_HPP` vs `FACERECOGNITION_LOG_HPP` vs `#pragma once`）。碰到文件时顺手改可以，不为它单独扫全仓 |
| 把 OpenCV 头从引号改尖括号 | 与本次目标无关；`layers` 脚本把 `opencv2/...` 当第三方跳过。以后想统一再说 |

---

## 4. P1 — 纯搬运（无行为变化，一个 commit）

### 4.1 文件去向（`git mv`）

```sh
mkdir -p src/common src/core src/app src/serve src/cli

# 根目录的库文件 → core / app
git mv priorbox.hpp priorbox.cpp detector_head.hpp detector_head.cpp \
       utils.hpp utils.cpp face_tracker.hpp face_tracker.cpp face_def.hpp src/core/
git mv config.hpp config.cpp src/app/
git mv main.cpp src/main.cpp

# src/ 下按层重排
git mv src/engine.hpp src/engine.cpp src/core/
git mv src/log.hpp src/log.cpp src/json_util.hpp src/json_util.cpp \
       src/encoding.hpp src/encoding.cpp src/sha256.hpp src/sha256.cpp \
       src/metrics.hpp src/metrics.cpp src/common/
git mv src/api.hpp src/api.cpp src/gallery.hpp src/gallery.cpp \
       src/evaluation.hpp src/evaluation.cpp src/app/
git mv src/admission.hpp src/timed_queue.hpp src/auth.hpp src/auth.cpp \
       src/pipeline_pool.hpp src/pipeline_pool.cpp src/result_cache.hpp src/result_cache.cpp \
       src/server.hpp src/server.cpp src/mcp.hpp src/mcp.cpp src/serve/
git mv src/cli.hpp src/cli.cpp src/cli/

# 转发头：唯一用户是 main.cpp，直接删（4.3）
git rm FaceRecognition.h
```

完整映射表（旧 → 新）：

| 旧 | 新 |
|---|---|
| `main.cpp` | `src/main.cpp` |
| `config.hpp` / `config.cpp` | `src/app/config.hpp` / `.cpp` |
| `priorbox.hpp` / `.cpp` | `src/core/priorbox.hpp` / `.cpp` |
| `detector_head.hpp` / `.cpp` | `src/core/detector_head.hpp` / `.cpp` |
| `utils.hpp` / `.cpp` | `src/core/utils.hpp` / `.cpp` |
| `face_tracker.hpp` / `.cpp` | `src/core/face_tracker.hpp` / `.cpp` |
| `face_def.hpp` | `src/core/face_def.hpp` |
| `FaceRecognition.h` | 删除 |
| `src/engine.hpp` / `.cpp` | `src/core/engine.hpp` / `.cpp` |
| `src/log.hpp` / `.cpp` | `src/common/log.hpp` / `.cpp` |
| `src/json_util.hpp` / `.cpp` | `src/common/json_util.hpp` / `.cpp` |
| `src/encoding.hpp` / `.cpp` | `src/common/encoding.hpp` / `.cpp` |
| `src/sha256.hpp` / `.cpp` | `src/common/sha256.hpp` / `.cpp` |
| `src/metrics.hpp` / `.cpp` | `src/common/metrics.hpp` / `.cpp` |
| `src/api.hpp` / `.cpp` | `src/app/api.hpp` / `.cpp` |
| `src/gallery.hpp` / `.cpp` | `src/app/gallery.hpp` / `.cpp` |
| `src/evaluation.hpp` / `.cpp` | `src/app/evaluation.hpp` / `.cpp` |
| `src/admission.hpp` | `src/serve/admission.hpp` |
| `src/timed_queue.hpp` | `src/serve/timed_queue.hpp` |
| `src/auth.hpp` / `.cpp` | `src/serve/auth.hpp` / `.cpp` |
| `src/pipeline_pool.hpp` / `.cpp` | `src/serve/pipeline_pool.hpp` / `.cpp` |
| `src/result_cache.hpp` / `.cpp` | `src/serve/result_cache.hpp` / `.cpp` |
| `src/server.hpp` / `.cpp` | `src/serve/server.hpp` / `.cpp` |
| `src/mcp.hpp` / `.cpp` | `src/serve/mcp.hpp` / `.cpp` |
| `src/cli.hpp` / `.cpp` | `src/cli/cli.hpp` / `.cpp`（P2 再拆） |

### 4.2 include 改写

```sh
cd /home/eipi10/MyProjects/FaceRecognition
files=$(git ls-files '*.cpp' '*.hpp' '*.h' | grep -v third_party)
sed -i -E \
  -e 's|#include "(log|json_util|encoding|sha256|metrics)\.hpp"|#include "common/\1.hpp"|' \
  -e 's|#include "(face_def|priorbox|detector_head|utils|engine|face_tracker)\.hpp"|#include "core/\1.hpp"|' \
  -e 's|#include "(config|evaluation|gallery|api)\.hpp"|#include "app/\1.hpp"|' \
  -e 's|#include "(admission|timed_queue|auth|pipeline_pool|result_cache|server|mcp)\.hpp"|#include "serve/\1.hpp"|' \
  -e 's|#include "cli\.hpp"|#include "cli/cli.hpp"|' \
  -e 's|#include "src/cli\.hpp"|#include "cli/cli.hpp"|' \
  $files
```

跑完必须做的三件手工事 + 一次自检：

1. `src/main.cpp`：`#include "FaceRecognition.h"` → `#include "core/engine.hpp"`（§4.3）。
2. 两处 `httplib.h` 改尖括号（`layers` 脚本会检查）：
   `grep -rn '#include "httplib.h"' src` → `src/serve/server.cpp`、`src/serve/mcp.cpp`，
   改成 `#include <httplib.h>`。
3. `git diff` 通读一遍：确认没有注释里的 `#include "..."` 被 sed 误改。
4. 自检：
   - `grep -rn '#include "[a-z_]*\.hpp"' src tests` → 必须为空（没有未限定的项目头）；
   - `grep -rn 'src/cli\.hpp' src tests` → 必须为空。

保留原样、不动：`#include "opencv2/opencv.hpp"`（引号形式，脚本按第三方跳过）、
`third_party/httplib.h` 的位置。

### 4.3 `FaceRecognition.h` 的处置

内容只有一行有效代码：`#include "engine.hpp"`，用户只有 `main.cpp:8`。
**直接删掉**，`src/main.cpp` 改 include `"core/engine.hpp"`。README 的 Source layout 表里
删掉对应行，AGENTS 里"转发头"的描述改成"已删除"。

### 4.4 `CMakeLists.txt` 目标草案（P1 后、P2 前的完整文件）

> 注意：CMake 3.10 没有 `add_compile_definitions`（3.12 才有），所以 httplib 的宏用
> **INTERFACE target** 表达；这样连 `cmake_minimum_required` 都不用动（README 写的是 ≥3.10）。

```cmake
cmake_minimum_required(VERSION 3.10)

project(FaceRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 没有 build type 时 OpenCV 找到的库仍可用，但性能数字没有意义：默认给 Release，
# 需要调试符号时显式 -DCMAKE_BUILD_TYPE=Debug（-O0 -g）。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 单一源码根：src/。include 一律写成 "<层>/<文件>"，路径本身就表达依赖方向
# （层序与检查见 scripts/check_layers.py）。
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

# ---- 层 app：配置 + 库 + 阈值 + 三外壳共享的结果格式 ----
add_library(facerec_app STATIC
    src/app/config.cpp
    src/app/evaluation.cpp
    src/app/gallery.cpp
    src/app/api.cpp
)
target_include_directories(facerec_app PUBLIC ${FACEREC_ROOT}/src)
target_link_libraries(facerec_app PUBLIC facerec_core)

# ---- cpp-httplib：宏必须对**每个**包含它的 TU 一致 ----
# listen backlog 默认 5（httplib.h:117，listen() 没有 backlog 参数，只能靠宏），实测并发
# >5 时 accept 队列溢出、内核丢 SYN，客户端看到 ~1.04 s 的连接耗时；1024 被
# net.core.somaxconn 截断（本机 4096），够用且不撒谎。
# 另一个宏是把 form-urlencoded 请求体上限钉到远高于 payload 上限：httplib 内建 8 KB 上限，
# 而 curl --data-binary 默认发的就是这个 Content-Type；实测这个上限在不同 TU 布局下行为
# 不一致（2 个 TU 包含 httplib 时放行，加第三个 TU 后 413），所以只能用"全局一致"来防。
# 按 target 抄写会让"新 target 忘了抄"变成运行期静默行为，因此做成 INTERFACE target：
# 谁包含 httplib，谁就必须链它。
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

# ---- 层 cli + 入口（P2 之后把 cli.cpp 换成每个子命令一个文件）----
add_executable(main
    src/main.cpp
    src/cli/cli.cpp
)
target_link_libraries(main PRIVATE facerec_serve)

# ---- 回归测试（PLAN.md D6）----
# 只放"纯函数"层的东西：不需要模型、摄像头、网络，`ctest` 在任何机器上都能跑。
# 测试 link target，不再抄源文件清单：清单只有 CMake 一处。
option(FACEREC_BUILD_TESTS "Build the regression tests (ctest)" ON)
if(FACEREC_BUILD_TESTS)
    enable_testing()

    add_executable(decode_gates tests/decode_gates.cpp)
    target_link_libraries(decode_gates PRIVATE facerec_core)
    add_test(NAME decode_gates COMMAND decode_gates)

    add_executable(tracker_scenarios tests/tracker_scenarios.cpp)
    target_link_libraries(tracker_scenarios PRIVATE facerec_core)
    add_test(NAME tracker_scenarios COMMAND tracker_scenarios)

    add_executable(admission tests/admission.cpp)
    target_link_libraries(admission PRIVATE facerec_common)
    add_test(NAME admission COMMAND admission)

    add_executable(evaluation tests/evaluation.cpp)
    target_link_libraries(evaluation PRIVATE facerec_app)
    add_test(NAME evaluation COMMAND evaluation)

    # 布局守卫：include 方向 + "每个 .cpp 都在 CMake 里"（§4.6）。
    find_program(FACEREC_PYTHON3 NAMES python3)
    if(FACEREC_PYTHON3)
        add_test(NAME layers
                 COMMAND ${FACEREC_PYTHON3} ${FACEREC_ROOT}/scripts/check_layers.py ${FACEREC_ROOT})
    else()
        message(WARNING "python3 not found: the `layers` test is disabled")
    endif()

    # augmentation A/B：需要 models/ 与 data/，不进 ctest。
    add_executable(augmentation_ab tests/augmentation_ab.cpp)
    target_link_libraries(augmentation_ab PRIVATE facerec_core)
endif()
```

要点：
- 旧的 `target_include_directories(main PRIVATE src ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/third_party)`
  与 `target_compile_definitions(main PRIVATE ...)` **整段删除**，语义移到 INTERFACE target；
  两段解释性注释（backlog / form-urlencoded）要**保留**，跟着宏走。
- `facerec_core` 链上 `facerec_common` 是为了让"core 允许用 common"这条规则在构建里成立
  （当前没用到，静态库不会拉进无用对象）。
- 测试不需要再写 `target_include_directories`：include 根从 `facerec_*` 的 PUBLIC 继承。

### 4.5 测试 target 映射（改完对一遍）

| 测试 | 原来 | 现在 |
|---|---|---|
| `decode_gates` | `tests/decode_gates.cpp priorbox.cpp` | `tests/decode_gates.cpp` + link `facerec_core` |
| `tracker_scenarios` | `tests/tracker_scenarios.cpp face_tracker.cpp` | `tests/tracker_scenarios.cpp` + link `facerec_core` |
| `admission` | `tests/admission.cpp src/metrics.cpp` | `tests/admission.cpp` + link `facerec_common` |
| `evaluation` | `tests/evaluation.cpp src/evaluation.cpp` | `tests/evaluation.cpp` + link `facerec_app` |
| `augmentation_ab` | 手抄 5 个源文件 | `tests/augmentation_ab.cpp` + link `facerec_core` |
| `layers`（新） | — | `python3 scripts/check_layers.py <repo>` |

`tests/*.cpp` 的 include 由 §4.2 的 sed 一并改写（`#include "priorbox.hpp"` →
`#include "core/priorbox.hpp"` 等）。

### 4.6 `scripts/check_layers.py`（可直接用）

```python
#!/usr/bin/env python3
"""源码布局守卫：include 只能指向同层或下层，且每个 .cpp 都必须在 CMakeLists.txt 里。

层序：common(0) < core(1) < app(2) < serve(3) < cli(4) < src/main.cpp(5)
规则：
  1. 项目 include 必须写成 "<层>/<文件>"（如 "core/engine.hpp"）；未限定即失败。
  2. 被 include 的层号不得大于包含者（同层允许）。
  3. 每个 src/**/*.cpp 的路径必须原样出现在 CMakeLists.txt 里（防"编过了但没跑"）。
  4. third_party 头用尖括号（<httplib.h>），不参与上面的规则。
退出码 0 = 通过；1 = 有违规（逐行打印 文件:行: 原因）。
"""
import pathlib
import re
import sys

LAYERS = {"common": 0, "core": 1, "app": 2, "serve": 3, "cli": 4}
TOP = 5                       # src/main.cpp
QUOTED = re.compile(r'#include\s+"([^"]+)"')


def layer_of(rel):
    return TOP if len(rel.parts) == 1 else LAYERS.get(rel.parts[0], -1)


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    src = root / "src"
    if not src.is_dir():
        print("check_layers: no src/ under %s" % root, file=sys.stderr)
        return 1
    cmake_text = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    problems, sources = [], []
    for path in sorted(src.rglob("*")):
        if path.suffix not in (".cpp", ".hpp", ".h"):
            continue
        rel = pathlib.PurePosixPath(path.relative_to(src).as_posix())
        mine = layer_of(rel)
        if mine < 0:
            problems.append("%s: unknown layer directory %r" % (path, rel.parts[0]))
            continue
        if path.suffix == ".cpp":
            sources.append(rel)
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = QUOTED.search(line)
            if not m:
                continue
            inc = m.group(1)
            if inc == "httplib.h":
                problems.append("%s:%d: use <httplib.h> (angle form) for third_party" % (path, no))
                continue
            p = pathlib.PurePosixPath(inc)
            if len(p.parts) > 1 and p.parts[0] not in LAYERS:
                continue                              # 第三方头（opencv2/...）
            if len(p.parts) == 1:
                problems.append(
                    '%s:%d: unqualified project include "%s" — write "<layer>/<file>"'
                    % (path, no, inc))
                continue
            if not (src / p).exists():
                problems.append('%s:%d: include "%s" does not exist under src/' % (path, no, inc))
                continue
            theirs = LAYERS[p.parts[0]]
            if theirs > mine:
                problems.append(
                    '%s:%d: layer %s includes up into %s: "%s"'
                    % (path, no, rel.parts[0] if len(rel.parts) > 1 else "main", p.parts[0], inc))
    for rel in sources:
        if "src/" + rel.as_posix() not in cmake_text:
            problems.append("src/%s: not listed in CMakeLists.txt (would compile but never link)"
                            % rel.as_posix())
    for p in problems:
        print(p)
    if problems:
        print("check_layers: %d problem(s)" % len(problems), file=sys.stderr)
        return 1
    print("check_layers: %d sources, ok" % len(sources))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

写完后立刻跑：`python3 scripts/check_layers.py .`，期望只有一行
`check_layers: 21 sources, ok`（P1 后 src 下 .cpp：common 5 + core 5 + app 4 + serve 5 +
cli 1 + main 1 = 21；可用 `git ls-files 'src/**/*.cpp' | wc -l` 复核，真实数字填进 §10）。

> 这个脚本（写本计划时）已经在一棵 `/tmp` 假树上验证过：正确布局退出 0；
> 「core 向上 include app」「未限定 include」「引号形式的 httplib.h」「.cpp 漏出
> CMakeLists」四种违规都能被抓到并打印 `文件:行:`。所以 P1.4 通过 = 规则真的成立，
> 不是脚本写歪了。当前（改造前）对它跑会报 71 条问题，那是预期的。

### 4.7 文档同步（P1 必须做，否则文档开始撒谎）

| 文件 | 改什么 |
|---|---|
| `README.md` | §"Source layout"（约 829–852 行的表格）整段换成"层 + 目录"表；同节下散落的当前路径（约 295 行 `src/auth.cpp`、870 行 `utils.cpp`、909 行 `config.cpp`、923/929 行的 recipe 路径）改成新路径。**不**改历史叙述 |
| `.agents/AGENTS.md` | File map 表逐行改路径；删除 `FaceRecognition.h` 行并把 `main.cpp` 行改成 `src/main.cpp`；Gotchas 里 `main.cpp:recognize_frame`、`src/server.cpp`、`src/auth.cpp`、`src/admission.hpp`、`src/timed_queue.hpp`、`src/evaluation.*` 等当前路径全部改；"Current repo state" 段落里 `FaceRecognition.cpp` 已删除的说法保持事实。此文件 gitignore，不随 commit（但必须改对） |
| `PLAN.md` | §7「代码骨架」的树换成 §2 的树；在文件末尾附近加一小节「布局调整（RFPLAN P1–P3）」记录决策与三阶段。历史 P 小节里的旧路径**不改** |
| `CHANGELOG.md` | 在 `## Unreleased` 下新增 `### Changed`：一条源码根/目录层次、一条 CLI 按子命令拆分、一条内核命名空间。不重写 v1.2.0 段落 |

自检：`grep -rn 'src/engine\.cpp\|`main\.cpp`\|priorbox\.cpp' README.md .agents/AGENTS.md`，
剩下的每一条都必须属于"历史记录"，不能属于"当前布局"描述。

### 4.8 P1 验证

见 §7（clean reconfigure 是必须的：源文件路径变了，CMake 缓存里的清单不会自己更新）。
期望：`ctest` **5/5** 通过（新增 `layers`）。

### 4.9 P1 commit

```sh
git add src CMakeLists.txt tests scripts README.md CHANGELOG.md PLAN.md
git commit -m "layout: one source root, four layers, and a build graph that says so"
```
（`.agents/AGENTS.md` 与 `RFPLAN.md` 不入库；如果你想让计划入库，单独一个 commit。）

---

## 5. P2 — 拆 `cli` / `main`（无行为变化，一个 commit）

### 5.1 `src/cli/cli.cpp` 现状解剖（1266 行）

| 行 | 内容 | 去向 |
|---|---|---|
| 1–30 | include 块 | 按需分给各文件（编译驱动，不预先抄） |
| 32–198 | 匿名 namespace 里的共享辅助 | `cli/args.{hpp,cpp}`，提到 `namespace cli` |
| 37–51 | `sanitize_person_name`（调用点只有 enroll:416） | `cli::sanitize_person_name` |
| 57–62 | `struct ConfigArgs` | `cli/args.hpp` |
| 64–99 | `load_pipeline_config`（8 个调用点） | `cli::load_pipeline_config` |
| 101–125 | `parse_set_arg`（8 个调用点） | `cli::parse_set_arg` |
| 127–173 | `parse_queue_arg`（**模板**，`Opt` = `ServerOptions`/`McpOptions`；调用点 643/732） | 定义留在 `cli/args.hpp`（模板必须对调用方可见），或对两种类型显式实例化放 `.cpp`；**不要只把它挪进 `.cpp`** |
| 175–188 | `parse_config_arg` | `cli::parse_config_arg` |
| 190–198 | `check_assets` | `cli::check_assets` |
| 204–217 | `identify_usage` | `cli/identify.cpp` |
| 221–331 | `cmd_identify` | `cli/identify.cpp` |
| 337–483 | `cmd_enroll` | `cli/enroll.cpp` |
| 489–598 | `cmd_token` | `cli/token.cpp` |
| 604–693 | `cmd_serve` | `cli/serve.cpp` |
| 695–771 | `cmd_mcp` | `cli/mcp.cpp` |
| 777–788 | `compare_usage` | `cli/compare.cpp` |
| 794–817 | `print_evaluate_usage` | `cli/evaluate.cpp` |
| 819–833 | 第二个匿名 namespace，`is_image_path`（933/960） | `cli/evaluate.cpp`（匿名 namespace） |
| 835–1069 | `cmd_evaluate` | `cli/evaluate.cpp` |
| 1071–1155 | `cmd_compare` | `cli/compare.cpp` |
| 1161–1177 | `verify_usage` | `cli/verify.cpp` |
| 1179–1266 | `cmd_verify` | `cli/verify.cpp` |

`cmd_*` 的签名与全局名字**不变**（`cli/cli.hpp` 已有声明），所以 `src/main.cpp` 的分发表
一行都不用改。

### 5.2 新增文件与契约

```
src/cli/args.hpp    namespace cli { struct ConfigArgs; bool load_pipeline_config(...);
                    bool parse_set_arg(...); template<class Opt> int parse_queue_arg(...);
                    bool parse_config_arg(...); bool check_assets(...);
                    std::string sanitize_person_name(...); }
src/cli/args.cpp    上面的非模板定义
src/cli/identify.cpp enroll.cpp token.cpp serve.cpp mcp.cpp evaluate.cpp compare.cpp verify.cpp
src/cli/demo.{hpp,cpp}   run_demo() 声明/定义
```

规则：
- **头文件里不许 `using namespace ...`**；实现文件里可以 `using facerec::Foo;` 单名引入。
- 每个命令文件只 include 自己用到的头（编译错误会告诉你少了什么；不要预先堆一堆）。
- `args.hpp` 需要 `#include "app/config.hpp"`（`ConfigArgs` 里是 `PipelineConfig` 相关字段）
  与 `<string>` / `<vector>`；模板 `parse_queue_arg` 的定义放这里。

### 5.3 `src/main.cpp` 剩余内容（442 行 → ~90 行）

| 行 | 内容 | 去向 |
|---|---|---|
| 1–15 | include / `using namespace cv;` | `main.cpp` 只留 C++ 标准头 + `cli/cli.hpp` + `cli/demo.hpp`；`using namespace cv;` 移到 `demo.cpp` |
| 16–241 | 匿名 namespace：`Recognizer`(20–23)、`FrameOut`(25–29)、`recognize_frame`(39–93)、`make_dnn_config`(96–100)、`print_usage`(102–119)、`run_pipeline`(121–239) | 前四个 + `run_pipeline` → `cli/demo.cpp`；`print_usage` 留在 `main.cpp` |
| 245–371 | `run_demo` | `cli/demo.cpp`（声明进 `cli/demo.hpp`） |
| 374–442 | `main` 分发 | 原地不动（唯一改动：include 行） |

`demo.cpp` 需要：`core/engine.hpp`、`core/utils.hpp`、`core/face_tracker.hpp`、`app/config.hpp`、
`app/gallery.hpp`、`app/api.hpp`（若用到 `Pipeline`/结果类型）、`<opencv2/highgui.hpp>`、
`<opencv2/imgproc.hpp>`、`<filesystem>`、`<map>`、`<vector>`。以编译为准。

### 5.4 P2 验证 + commit

§7 全套（`./main 0` 等价路径、`demo`、identify、serve 必须仍然正常）；`ctest` 5/5；
`layers` 自动覆盖新文件（CMake 里要逐个列出 10 个 cli 源文件，漏一个测试就失败）。

```sh
git add src CMakeLists.txt README.md CHANGELOG.md PLAN.md
git commit -m "cli: one file per subcommand, and a demo that is not the entry point"
```

---

## 6. P3 — 语义收尾（两个独立小 commit）

### 6.1 `core` 进 `namespace facerec`

动机：现在是"半个库有名、半个没有"。内核（`Detector`/`FeatureExtractor`/`Similarity`/
`Face`/`BndBox_*`/`PriorBox`/`FaceTracker`/`nms`/`letterbox`/`draw`/`align_*`/`extract_*`/
`features_from_img`/`list_person_dirs`）全在全局命名空间，而 `api` 往上全在 `facerec::`。

做法：
- `src/core/{face_def,priorbox,detector_head,utils,engine,face_tracker}.*` 整体包进
  `namespace facerec { ... }`（`face_def.hpp` 是 `#pragma once`，位置保持不变）。
- 消费方按编译错误逐个限定：`src/app/api.cpp`、`src/app/gallery.cpp`、`src/cli/*.cpp`、
  `src/serve/*.cpp`（`api.hpp` 里已经 `facerec::` 的不用动）。
- 测试/实现文件允许局部 `using namespace facerec;`（**只有 `.cpp`**）：
  `tests/augmentation_ab.cpp`、`tests/decode_gates.cpp`、`tests/tracker_scenarios.cpp`、
  `src/cli/demo.cpp`。
- `Face` 与 OpenCV 无同名冲突；`Detector`/`Similarity` 同理。不要加 `facerec_legacy` alias。

验证：`make && ctest` 5/5；§7.4 的 identify 输出与 P1 逐字符一致（分数 1.0000 / 0.0226）。
commit：`git commit -m "namespace: the recognition core joins facerec"`。

### 6.2 `auth.hpp` 停止 include `<httplib.h>`

动机：`src/cli/cli.cpp` 的 11.55 s 大部分是"为了一个声明编译整个 HTTP 库"。
`server.hpp` 已经有正确写法（全局作用域的 `namespace httplib { class Server; }`，
并且注释解释了为什么不能写进 `facerec`）。

做法：
- `src/serve/auth.hpp`：删 `#include <httplib.h>`，加
  ```cpp
  namespace httplib { class Server; }   // 必须在全局作用域（见 server.hpp 的注释）
  ```
- `src/serve/auth.cpp`：补 `#include <httplib.h>`（它定义 `install_access_control`，
  需要完整类型；现在它是靠 `auth.hpp` 传递才编过的）。
- 重新跑 `layers`：`auth.hpp` 不再包含第三方头，规则不变。

### 6.3 效果测量（写进 commit message 或 §10）

用 §7.6 的 recipe 重新测这些 TU：

| TU | 期望 |
|---|---|
| `src/cli/token.cpp`、`identify.cpp`、`compare.cpp`、`verify.cpp`、`evaluate.cpp` | 从 ~11 s 档降到 OpenCV 头成本档（~2–4 s；与 `src/config.cpp` 2.12 s 同量级） |
| `src/cli/serve.cpp`、`src/cli/mcp.cpp` | 仍偏高（它们合法地构造 `httplib::Server`） |
| `src/serve/server.cpp` | 不变（10.47 s 是 httplib + OpenCV 的真实成本） |

commit：`git commit -m "auth: forward-declare httplib::Server so the CLI stops compiling it"`。

---

## 7. 通用验证配方（复制粘贴；每个 Phase 都跑）

### 7.1 重新配置 + 构建 + 测试

```sh
cd /home/eipi10/MyProjects/FaceRecognition/build
rm -rf CMakeCache.txt CMakeFiles Makefile     # 源文件路径变了，缓存清单不会自己更新
cmake .. && make -j"$(nproc)"
ctest                                          # 期望 5/5 passed（P1 起）
```

### 7.2 配置往返（不需要摄像头）

```sh
cd build
./main --dump-config | head -5
./main --dump-config > /tmp/rt.yml && ./main 1 --config /tmp/rt.yml
# 期望两次都出现 "Loaded 2 person(s) from ./data/"，随后 "Cannot open source: 1"
# 期望退出码 255（-1）
```

### 7.3 端到端识别（无 GUI）

```sh
cd build
./main identify --image data/BruceLi/*.jpeg
# 期望：status ok；matched.person_id = "BruceLi"；score = 1.0000；IvanYang 0.0226
```

### 7.4 serve 冒烟 + JSON 日志必须可解析

```sh
cd build
(timeout 8 ./main serve --port 18099 --workers 1 2>/tmp/serve.err 1>/tmp/serve.out &)
sleep 2.5
for u in healthz readyz metrics; do
  curl -s -o /tmp/r.out -w "/$u -> %{http_code}\n" "http://127.0.0.1:18099/$u"
done
sleep 5
python3 - <<'PY'
import json, sys
bad = 0
for i, line in enumerate(open('/tmp/serve.err'), 1):
    line = line.strip()
    if not line:
        continue
    if not line.startswith('{'):
        print('non-JSON line (allowed only for human output):', line[:100]); continue
    try:
        json.loads(line)
    except Exception as e:
        bad += 1; print('BAD', i, e, line[:160])
print('bad JSON lines:', bad)
sys.exit(1 if bad else 0)
PY
```

期望：三个端点 200（`/healthz` `{"status":"ok"}`，`/readyz` 含 `gallery_fingerprint` 与
`far_1n_*`，`/metrics` 含 `facerec_http_requests_total`）；`bad JSON lines: 0`；
stdout 里是人工 banner（允许非 JSON）；stderr 首行是 `{"audit":"startup",...}`。

### 7.5 MCP 冒烟（可选，stdio 一条 initialize + tools/list）

```sh
cd build
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | ./main mcp --transport stdio
# 期望：两行 JSON，第二行 result.tools 是 4 个只读工具
```

### 7.6 单 TU 编译计时（§1.2 / §6.3 用）

```sh
cmake -S /home/eipi10/MyProjects/FaceRecognition -B /tmp/frbuild \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build /tmp/frbuild -j"$(nproc)"
python3 - <<'PY'
import json, os, shlex, subprocess, sys, time
want = sys.argv[1:] or ['src/cli/identify.cpp', 'src/core/engine.cpp']
cc = json.load(open('/tmp/frbuild/compile_commands.json'))
for entry in cc:
    f = entry['file']
    if not any(f.endswith('/' + w) for w in want):
        continue
    parts = shlex.split(entry['command']); out = []; i = 0
    while i < len(parts):
        if parts[i] == '-o': i += 2; continue
        if parts[i] == '-c': i += 1; continue
        out.append(parts[i]); i += 1
    out += ['-c', '-o', '/tmp/tu_probe.o']
    t = time.time(); r = subprocess.run(out, capture_output=True); dt = time.time() - t
    print('%-28s rc=%d %6.2f s' % (f.split('FaceRecognition/')[-1], r.returncode, dt))
PY
```

### 7.7 GUI 路径（只在需要时；**绝不**用用户的 `DISPLAY=:0`）

```sh
command -v xvfb-run >/dev/null && xvfb-run -a ./main data/BruceLi/*.jpeg
# 或者：Xvfb :99 & DISPLAY=:99 ./main data/BruceLi/*.jpeg
# 期望：窗口显示一帧、按任意键退出；stdout 打印 Loaded 2 person(s)
```

### 7.8 收尾检查

```sh
git status --short          # build/ 被 ignore；只剩预期改动
git diff --stat
grep -rn 'src/[a-z_]*\.cpp' README.md .agents/AGENTS.md | head   # 只剩历史章节
```

---

## 8. 陷阱清单（会咬人的地方）

1. **clean reconfigure**：源文件路径变了，`build/CMakeFiles` 里的旧对象/清单不会自己更新
   （证据：至今还躺着 `FaceRecognition.cpp.o`）。不删缓存就 `cmake ..` 可能"成功"但链旧对象。
2. **`parse_queue_arg` 是模板**：函数体必须对 `ServerOptions` / `McpOptions` 两种实例化可见；
   只挪进 `args.cpp` 会链接失败或编译失败。定义放 `args.hpp`。
3. **`usage` 函数的位置**：`identify_usage` 在匿名 namespace 里（204–217），
   `compare_usage`/`print_evaluate_usage`/`verify_usage` 各自在命令前面，
   `is_image_path` 在**第二个**匿名 namespace（819–833）。按 §5.1 的行号归位，别漏。
4. **`#include "httplib.h"` 两处必须是尖括号**（`src/serve/server.cpp`、`src/serve/mcp.cpp`），
   否则 `layers` 测试失败。
5. **`main.cpp` 的 `using namespace cv;`**：P2 移到 `demo.cpp`；P3 之后 `facerec::` 与 `cv::`
   没有同名冲突（已核对 `Face`/`Detector`/`Similarity`/`draw`/`nms`）。
6. **`FaceRecognition.h` 只有一个用户**（已核对 `grep -rn FaceRecognition.h`），删它安全；
   README 那一行要同步删。
7. **静态库的宏传播**：`facerec_httplib` 必须被 `facerec_serve` PUBLIC 链上，`main` 经由
   `facerec_serve` 拿到；不要退回 `target_compile_definitions(main ...)`（那正是"新 target
   忘抄"的 bug 类别）。CMake 3.10 没有 `add_compile_definitions`（3.12 才引入），别用。
8. **`admission` 测试**：它只 include `serve/admission.hpp` + `common/metrics.hpp`，
   链 `facerec_common` 即可；不要为了它把 serve 库拖进来。
9. **运行时路径不变**：`main.cpp` 里 `./models/...`、`./data/` 是 **cwd 相对**
   （`paths.root` 默认 cwd），搬源码不影响运行；仍然从 `build/` 跑。
10. **文档三类**：现状（README 布局表、AGENTS File map）必须改；历史（CHANGELOG 旧版本、
    PLAN 旧 P 小节）不改；锚点（`config.example.yml` 由 `--dump-config` 生成，永不手改）。
11. **不要 `git add -A`**：本文件与 `.agents/` 未入库。
12. **P3 的命名空间改动面最大**：单独 commit；如果只想要布局收益，P1+P2 已经完整，P3 可回退。

---

## 9. 回退

每个 Phase 一个 commit，回退就是 `git reset --hard <phase 之前的 commit>`：

```sh
git log --oneline -5
# P1 之前是 7d7d27f；P1/P2/P3 各自可单独 revert
git revert --no-commit <P3 的 commit>     # 只退命名空间
```

`build/` 被 gitignore，不受影响；回退后按 §7.1 重新配置即可。

---

## 10. 进度台账（**每完成一步就更新这里**，压缩后靠它恢复）

| 步骤 | 内容 | 状态 | commit |
|---|---|---|---|
| P0 | 记录基线（§1.2） | ✅ | —（基线 `7d7d27f`） |
| P1.1 | `git mv` 全部文件（§4.1） | ✅ 43 个文件，根目录只剩文档 | `acc6903` |
| P1.2 | include 改写 + httplib 尖括号 + 删 `FaceRecognition.h`（§4.2/4.3） | ✅ sed 批量 + 守卫核对；未限定 include 归零 | `acc6903` |
| P1.3 | CMake 重写为 4 库 + INTERFACE httplib + 测试 link（§4.4/4.5） | ✅ clean reconfigure 一次通过编译 | `acc6903` |
| P1.4 | `scripts/check_layers.py` + ctest `layers`（§4.6） | ✅ 21 sources ok；三类负例已实测会失败 | `acc6903` |
| P1.5 | 文档同步（README/AGENTS/PLAN/CHANGELOG，§4.7） | ✅ README 分层表 + AGENTS 新增 Layout 节 + PLAN §7/§14 + CHANGELOG 一条 | `acc6903` |
| P1.6 | 验证 §7 → commit | ✅ ctest 5/5 0.73 s；clean build 30.10 s（基线 32.86 s）；identify 1.000/0.0226；blank→`faces:[]` rc0；`./main 1` rc255；dump==example；serve 200×3 + 日志行全 parse；mcp stdio 4 工具 | `acc6903` |
| P2.1 | 抽 `cli/args.{hpp,cpp}`（§5.2） | ✅ `ConfigArgs` + 6 个 helper；`parse_queue_arg` 是模板，**定义必须留在 .hpp** | `a6e3ce7` |
| P2.2 | 8 个子命令一文件 + usage/`is_image_path` 归位（§5.1） | ✅ 八个命令文件 + `args.cpp` + `demo.cpp`；每个命令保留自己的 `*_usage()`，`is_image_path` 归 identify | `a6e3ce7` |
| P2.3 | 同上 | ✅ 全部搬过去（main.cpp 442 → **97 行**）。**偏离 §5.3**：`print_usage` 留在入口，理由写进 main.cpp 与 cli.hpp——命令名单就是入口的 `if/else`，拆到 `cli/` 必有一个说谎。`sanitize_person_name` 从 compare 块挪到 `args.cpp`（它是 enroll 的输入校验，不是比较逻辑） | `a6e3ce7` |
| P2.4 | 验证 §7 → commit | ✅ ctest 5/5 0.73 s；八个 `--help` 的 diff 只有 include 块；identify 1.000 / blank `faces:[]`；enroll dry-run；token；compare+verify rc0；demo rc255 + Xvfb 下窗口停住（rc124）；dump==example；serve 三探针 200 + 日志行全 parse；mcp stdio 4 工具；clean build 29.83 s | `a6e3ce7` |
| P3.1 | `core` 进 `namespace facerec` + 消费方限定（§6.1） | ✅ 11 个 core 文件全包进去；消费方只加 6 个 `using namespace facerec;`（`cli/*.cpp`+`tests/*.cpp`）；`args.cpp` 当时不加（无 core 依赖）——**P3.4 之后要加**，见 P3.4 行；内容不缩进，与 api.cpp 一致 | `9536f3f` |
| P3.2 | `auth.hpp` 前向声明 / `auth.cpp` 自己 include（§6.2） | ✅ 全局作用域 `namespace httplib { class Server; }`（写进 `facerec` 会编过、链接才炸）；`<httplib.h>` 现在只在 6 个文件里 | `c6338e9` |
| P3.3 | 验证 + 编译耗时对比（§6.3/§7.6） → commit | ✅ `token.cpp` **7.98 → 0.92 s**；`cli/serve` 8.68 / `cli/mcp` 8.64 / `serve/server` 9.90 不变（真的构造 Server）；clean build 29.83 → 29.02 s；ctest 5/5；鉴权路径重测全对（401/401/200 + `caller=p1` + 非 loopback rc=255） | `c6338e9` |
| P3.4 | `src/app/config.{hpp,cpp}` 进 `facerec`（库里最后一个全局公开头） | ✅ `cli/args.hpp` 必须写 `facerec::PipelineConfig`（头文件不写 using-directive）；`cli/args.cpp` 反过来**要**加 using——与 P3.1 那行「args.cpp 不加」相反，已在那行就地更正 | `a021182` |
| P3.5 | 删掉 `cli/{serve,mcp}.cpp` 里**vestigial** 的 `#include <httplib.h>`（两个文件都不用一个 `httplib::` 名字） | ✅ `cli/serve.cpp` 8.68 → **1.40 s**、`cli/mcp.cpp` 8.64 → **1.38 s**；clean build 29.02 → **22.74 s**；`<httplib.h>` 只剩 `serve/{auth.cpp,timed_queue.hpp,server.cpp,mcp.cpp}` 四处 | `0f0959a` |

**恢复协议**（新会话/压缩后第一件事）：

```sh
cd /home/eipi10/MyProjects/FaceRecognition
git log --oneline -5 && git status --short
test -d src/common && echo "P1 已落地（目录已建）"
test -f src/cli/identify.cpp && echo "P2 已落地"
grep -q 'namespace facerec' src/core/engine.hpp && echo "P3.1 已落地"
grep -q 'include <httplib.h>' src/serve/auth.hpp && echo "P3.2 未做"
```

然后从台账里**第一个 ⬜** 继续；动手前先跑一次 §7.1 确认当前状态是绿的
（半截状态会编译失败，这本身也是定位信息）。

**P3 的实测与一条更正（2026-08-22）**
- **我上一条（P2）里的结论是错的，这里更正**：我写「重 TU 贵在 codegen 不在 header 解析，
  所以 P3.2 不是性能修」——依据是 `#include <httplib.h>` 单独 `-fsyntax-only` 只要 1.33 s。
  P3.2 落地后实测：`token.cpp` **7.98 → 0.92 s**（它只调 `TokenTable`，从不碰 HTTP）。
  也就是说 httplib 那 7 s 全在**语义分析 + inline 模板的 codegen**，而 `-fsyntax-only` 恰好看不见这一段。
  ⇒ 规则：**不要用 `-fsyntax-only` 的数字外推编译耗时**；要测就测 `-O3 -c` 全量。
- §6.3 当初的预期（token/identify/compare/verify 降到 2-4 s 档）**方向对、幅度保守**：token 实际 0.92 s。
- 删传递 include 的代价立刻现形：`token.cpp` 原来靠 `<httplib.h>` 传递 `<fstream>`，必须自己 include。
  这类"隐藏传递依赖"是编译器会抓的，不是静默的——但会一次性炸出好几个，留好耐心。
- 命名空间包法：不缩进（与 `api.cpp` 一致），否则 diff 里 1000 行纯空白会把真正的改动埋掉。
- 鉴权必须重测（auth.hpp 是它的头）：无 token 401 + `WWW-Authenticate: Bearer realm="facerec"`；
  错 token 401；对 token 200 且审计 `"caller":"p1"`；`/healthz` 免鉴权 200；
  `--host 0.0.0.0` 且无 token → **rc=255** 拒启动。注意 `token` 的提示行里 token 是**截断显示**
  （`Bearer 77223031...`），脚本要取 `token:` 那一行，取提示行会得到 8 个字符的假 token。

**P2 的实测与发现（2026-08-22）**
- **勘误**：P2 的 commit message 与 CHANGELOG/PLAN/AGENTS 一度把 `src/cli/cli.cpp` 写成 635 行。
  实际是 **1266 行**（`git show a6e3ce7^:src/cli/cli.cpp | wc -l`），三处文档已改；
  已提交的 P2 message 不重写（不是最后一笔，rebase 改历史不值得为一个大数字做）。
- cli 层单 TU：11.55 → **8.67 s**（serve.cpp）；改一个命令的重编：**1.4-1.9 s**。
- cli 层总编译 CPU：11.55 → **37.2 s**（10 个 TU 各自解析 header）；clean build：32.86 → **29.83 s**（-j16）。
- **重 TU 贵在 codegen，不在 header 解析**：token.cpp front-end 1.68 s / -O0 3.59 s / -O3 7.92 s；
  `#include <httplib.h>` 单独 `-fsyntax-only` 只要 **1.33 s**。⇒ §6.3（P3.2 让 `auth.hpp` 不 include
  httplib）按**层正确性**做，不要按性能做；原先写成性能修是猜的。
- `token.cpp` 也必须 include `cli/args.hpp`（它用 `ConfigArgs` / `parse_config_arg` / `parse_set_arg`）：
  拆 cli 时最容易漏的一个 include。
- `evaluate.cpp` 的标准库 include 有几个原本被别的命令顺手带进来，拆出来必须显式补：
  `<atomic> <charconv> <fstream> <random> <thread>` + `serve/pipeline_pool.hpp`。

**执行中发现的新情况记在这里**（一行一条，带日期）：

- **2026-11-10 P1**：旧文档里有两处**关于测试的假话**：AGENTS 写 ctest 四个 target 叫
  `config`/`tracker_harness`（实际是 `decode_gates`/`tracker_scenarios`/`admission`/`evaluation`），
  README 写 "There is no test harness"。已按事实改，并补上 `layers`。
- **2026-11-10 P1**：我一度把 PLAN.md 的历史路径也一起 sed 了（121 行），**违反 §3.2 自己写的决定**
  （历史路径改了就是伪造历史）。已回退到 HEAD 重来：只重写 §7（标题就是"现状"，必须同步）+ 新增
  §14，其余小节的旧路径保持原样并在 §7 开头声明"故意不同步"。
- **2026-11-10 P1**：数字差点说谎——口头说的"full build 7.2 s"其实是增量编译；真正的 clean build 是
  **30.10 s**（基线 32.86 s）。写进文档的编译数字必须标 clean / incremental 与 `-j`。
- **2026-11-10 P1**：P1 的意外收益：旧 CMake 里 `priorbox.cpp` / `face_tracker.cpp` / `metrics.cpp` /
  `evaluation.cpp` 各列 **3 遍**（main + 测试 + augmentation_ab），core 被重复编译。现在一次 clean
  build = **26 个 .o，每个源文件恰好一遍**（`find build/CMakeFiles -name '*.o' | sed 's|.*/||' | uniq -d` 为空）。
- **2026-11-10 P1**：`git add` 漏了 `tests/`（include 也改了）→ 那个 commit 单独 checkout 编不过。
  `git commit --amend` 补上。**教训：commit 前 `git status --short | grep -v '^??'` 必须为空**。
- **2026-11-10 P1**：`src/serve/server.hpp` 里**没有**为单测存在的反向 include（全仓没有任何
  `#include "…​.cpp"`）——我早先在口头计划里写过这条，属凭空捏造，已从 PLAN §14 删除。
  P3.2（`auth.hpp` 不再 include `<httplib.h>`）才是真实存在的那条边。

---

## 11. 附：当前 include 全表（改造前，供核对）

> 这张表是 sed 的输入清单，也是"改完是否漏了"的对照。

```
config.cpp:1        "config.hpp"              → app/config.hpp
config.cpp:14       "log.hpp"                 → common/log.hpp
detector_head.cpp:1 "detector_head.hpp"       → core/detector_head.hpp
detector_head.hpp:8 "face_def.hpp"            → core/face_def.hpp
detector_head.hpp:9 "priorbox.hpp"            → core/priorbox.hpp
face_def.hpp:6      "opencv2/opencv.hpp"      （不动）
face_tracker.cpp:1  "face_tracker.hpp"        → core/face_tracker.hpp
face_tracker.hpp:5  "face_def.hpp"            → core/face_def.hpp
main.cpp:8          "FaceRecognition.h"       → core/engine.hpp（删转发头）
main.cpp:9          "face_tracker.hpp"        → core/face_tracker.hpp
main.cpp:10         "config.hpp"              → app/config.hpp
main.cpp:11         "gallery.hpp"             → app/gallery.hpp
main.cpp:12         "src/cli.hpp"             → cli/cli.hpp
priorbox.cpp:1      "priorbox.hpp"            → core/priorbox.hpp
priorbox.hpp:8      "face_def.hpp"            → core/face_def.hpp
utils.cpp:1         "utils.hpp"               → core/utils.hpp
utils.hpp:7         "face_def.hpp"            → core/face_def.hpp
utils.hpp:8         "opencv2/opencv.hpp"      （不动）
src/api.cpp:1       "api.hpp"                 → app/api.hpp
src/api.cpp:12      "evaluation.hpp"          → app/evaluation.hpp
src/api.hpp:13      "config.hpp"              → app/config.hpp
src/api.hpp:14      "engine.hpp"              → core/engine.hpp
src/api.hpp:15      "gallery.hpp"             → app/gallery.hpp
src/auth.cpp:1      "auth.hpp"                → serve/auth.hpp
src/auth.cpp:15     "admission.hpp"           → serve/admission.hpp
src/auth.cpp:16     "api.hpp"                 → app/api.hpp
src/auth.cpp:17     "metrics.hpp"             → common/metrics.hpp
src/auth.cpp:18     "sha256.hpp"              → common/sha256.hpp
src/auth.hpp:20     "log.hpp"                 → common/log.hpp
src/cli.cpp:1       "cli.hpp"                 → cli/cli.hpp
src/cli.cpp:3       "auth.hpp"                → serve/auth.hpp
src/cli.cpp:4       "sha256.hpp"              → common/sha256.hpp
src/cli.cpp:17      "api.hpp"                 → app/api.hpp
src/cli.cpp:18      "config.hpp"              → app/config.hpp
src/cli.cpp:19      "engine.hpp"              → core/engine.hpp
src/cli.cpp:20      "evaluation.hpp"          → app/evaluation.hpp
src/cli.cpp:21      "mcp.hpp"                 → serve/mcp.hpp
src/cli.cpp:22      "pipeline_pool.hpp"       → serve/pipeline_pool.hpp
src/cli.cpp:23      "server.hpp"              → serve/server.hpp
src/encoding.cpp:1  "encoding.hpp"            → common/encoding.hpp
src/engine.cpp:1    "engine.hpp"              → core/engine.hpp
src/engine.hpp:9    "detector_head.hpp"       → core/detector_head.hpp
src/engine.hpp:10   "face_def.hpp"            → core/face_def.hpp
src/engine.hpp:11   "utils.hpp"               → core/utils.hpp
src/evaluation.cpp:1 "evaluation.hpp"         → app/evaluation.hpp
src/gallery.cpp:1   "gallery.hpp"             → app/gallery.hpp
src/gallery.cpp:19  "encoding.hpp"            → common/encoding.hpp
src/gallery.cpp:20  "evaluation.hpp"          → app/evaluation.hpp
src/gallery.cpp:21  "sha256.hpp"              → common/sha256.hpp
src/gallery.hpp:12  "config.hpp"              → app/config.hpp
src/gallery.hpp:13  "engine.hpp"              → core/engine.hpp
src/json_util.cpp:1 "json_util.hpp"           → common/json_util.hpp
src/log.cpp:1       "log.hpp"                 → common/log.hpp
src/mcp.cpp:1       "mcp.hpp"                 → serve/mcp.hpp
src/mcp.cpp:3       "auth.hpp"                → serve/auth.hpp
src/mcp.cpp:4       "log.hpp"                 → common/log.hpp
src/mcp.cpp:17      "admission.hpp"           → serve/admission.hpp
src/mcp.cpp:18      "api.hpp"                 → app/api.hpp
src/mcp.cpp:19      "encoding.hpp"            → common/encoding.hpp
src/mcp.cpp:20      "httplib.h"               → <httplib.h>（尖括号）
src/mcp.cpp:21      "json_util.hpp"           → common/json_util.hpp
src/mcp.cpp:22      "pipeline_pool.hpp"       → serve/pipeline_pool.hpp
src/mcp.cpp:23      "result_cache.hpp"        → serve/result_cache.hpp
src/mcp.cpp:24      "server.hpp"              → serve/server.hpp
src/mcp.cpp:25      "sha256.hpp"              → common/sha256.hpp
src/mcp.cpp:26      "timed_queue.hpp"         → serve/timed_queue.hpp
src/mcp.hpp:6       "config.hpp"              → app/config.hpp
src/metrics.cpp:1   "metrics.hpp"             → common/metrics.hpp
src/pipeline_pool.cpp:1 "pipeline_pool.hpp"   → serve/pipeline_pool.hpp
src/pipeline_pool.hpp:10 "api.hpp"            → app/api.hpp
src/pipeline_pool.hpp:11 "config.hpp"         → app/config.hpp
src/result_cache.cpp:1 "result_cache.hpp"     → serve/result_cache.hpp
src/result_cache.hpp:11 "api.hpp"             → app/api.hpp
src/server.cpp:1    "server.hpp"              → serve/server.hpp
src/server.cpp:3    "auth.hpp"                → serve/auth.hpp
src/server.cpp:21   "admission.hpp"           → serve/admission.hpp
src/server.cpp:22   "api.hpp"                 → app/api.hpp
src/server.cpp:23   "encoding.hpp"            → common/encoding.hpp
src/server.cpp:24   "json_util.hpp"           → common/json_util.hpp
src/server.cpp:25   "log.hpp"                 → common/log.hpp
src/server.cpp:26   "pipeline_pool.hpp"       → serve/pipeline_pool.hpp
src/server.cpp:27   "metrics.hpp"             → common/metrics.hpp
src/server.cpp:28   "result_cache.hpp"        → serve/result_cache.hpp
src/server.cpp:29   "sha256.hpp"              → common/sha256.hpp
src/server.cpp:30   "timed_queue.hpp"         → serve/timed_queue.hpp
src/server.cpp:40   "httplib.h"               → <httplib.h>（尖括号）
src/server.hpp:8    "config.hpp"              → app/config.hpp
src/sha256.cpp:1    "sha256.hpp"              → common/sha256.hpp
src/timed_queue.hpp:9 "admission.hpp"         → serve/admission.hpp
tests/admission.cpp:17 "admission.hpp"        → serve/admission.hpp
tests/admission.cpp:18 "metrics.hpp"          → common/metrics.hpp
tests/augmentation_ab.cpp:34 "engine.hpp"     → core/engine.hpp
tests/augmentation_ab.cpp:35 "face_def.hpp"   → core/face_def.hpp
tests/decode_gates.cpp:18 "priorbox.hpp"      → core/priorbox.hpp
tests/evaluation.cpp:25 "evaluation.hpp"      → app/evaluation.hpp
tests/tracker_scenarios.cpp:21 "face_def.hpp" → core/face_def.hpp
tests/tracker_scenarios.cpp:22 "face_tracker.hpp" → core/face_tracker.hpp
```

---

## 12. P4+ 候选（**未列入本计划**；要动就另开小计划，一次一条）

P1–P3 已把 §0 的目标做完。下面四条是过程中量到的、但**当前不值得做**的事，写下来免得下次重新推一遍。

| 候选 | 实测依据 | 成本 | 现在的判断 |
|---|---|---|---|
| `PipelineConfig` 下沉到 `core/config/` | `src/core/*` 里 **0 处** include 配置——core 至今不需要它 | `knobs()` 有 30+ 个成员指针，改 include 面覆盖所有层 | **不做**。等 core 真的需要读配置再说；现在下沉只是把 `app` 的名字搬走 |
| 单独的 `facerec_eval` 库 | `evaluation.cpp` 只被 `cli/evaluate.cpp`（3.06 s）与 `tests/evaluation` 用 | CMake 多一个 target、多一处清单 | **不做**。拆它不减任何 TU 的编译时间，只多一个概念 |
| httplib 收进单一 `serve/transport` TU | P3.5 之后 httplib 只在 `serve/{auth.cpp,timed_queue.hpp,server.cpp,mcp.cpp}`；`serve/server.cpp` 9.90 s、`serve/mcp.cpp` 9.71 s 是它的真实成本（它们真的装配 route） | 把 route 装配（>1000 行）改成不透明 handle，接口变抽象 | **暂缓**。剩下唯一便宜的下一步是让 `serve/timed_queue.hpp` 不 include httplib（它 `: public httplib::TaskQueue` 并且按值持有 `httplib::ThreadPool`，所以做不到前向声明——要拆成 pimpl）。做之前先确认有人在乎这 20 s |
| `src/main.cpp` 搬进 `src/cli/` | 无收益 | `layers` 脚本的 TOP 规则与 CMake 都要改 | **不做**。TOP 层=入口是有意表达，不是一个待收的散文件 |
