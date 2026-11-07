# PLAN.md — FaceRecognition 生产化重构计划

> 目的：把现有 demo（采集+推理+渲染耦合在一个 `while` 循环里）重构为可服务化、高并发、可被 LLM 调用的面识别平台。本文件为执行蓝图，每步都必须能独立 `make` 并运行验证，不推翻已验证的步骤。

---

## 1. 现状与问题

现有代码（`main.cpp:run_pipeline`）把以下三者耦合在一个阻塞线程里：

```
cap >> im → letterbox(960×720) → tracker.update → recognize_frame
→ putText → draw → imshow → waitKey
```

- **采集（ingest）** 与 **推理（engine）** 与 **渲染（sink）** 不可分离，无法并发多路流。
- `Detector` / `FeatureExtractor` 持有可变内部状态（`output_blobs`、`dets`、`batch_mode_`），不可重入，多线程直接竞争。
- `FeatureDatabase` 在构造函数里边读图边跑 detect+embed，gallery 与模型实例绑死，无法热更新，启动慢。
- `recognize_frame` 的 `label_of` 是跨帧状态，属于流式语义，不该出现在请求路径。
- `rfactor=6` + `letterbox` 固定 960×720，给 demo 画框合适，对 API 是白扔分辨率（远/小人脸消失）。
- 只有 CLI（单摄像头），没有服务端、没有结构化输出、无法被 LLM 作为工具调用。

---

## 2. 目标架构：三层分离，四个外壳

```
┌─────────────────────────────────────────┐
│           应用层（四个外壳，同一个核心）   │
│  facerec identify   facerec serve     │
│  facerec enroll     facerec mcp        │
└──────────┬──────────────┬──────────────┘
           │              │
┌──────────▼──────────┐  ┌─▼─────────────────┐
│   libfacerec（内核）   │  │   sink 层（可插拔）   │
│  ┌───────────────┐  │  │  窗口 / MJPEG / WSS │
│  │ Engine（无状态） │  │  │  SSE / HTTP 事件流 │
│  │ - detect/embed│  │  │  └──────────────────┘
│  │ - match（gallery 只读）│  │
│  └───────────────┘  │  │   ingest 层（流式）   │
│  Gallery 索引（只读）│  │  │ 摄像头/RTSP/上传帧 │
│  Session 状态（tracker/label_of）│ │ 每路流一线程 │
└─────────────────────┘  └───────────────────┘
```

### 2.1 三层职责

| 层 | 职责 | 状态 | 并发单位 | 生命周期 |
|---|---|---|---|---|
| **ingest** | 摄像头/RTSP/图片/URL → 解码 → 帧+时间戳 | 每路流独立 | 每路 1 线程 | 流启→流停 |
| **engine** | detect → align → embed → match（纯函数式，无跨请求状态） | 无状态 | 每次调用 | 请求级 |
| **sink** | 画框窗口 / MJPEG / WebSocket / JSONL / 告警 | 每路流独立 | 每路 1 线程 | 流启→流停 |

- **`imshow`/`waitKey`/`draw`/`rfactor`/`letterbox(960×720)` 只属于 sink 层。** Engine 对外契约用**原图像素坐标**，缩放系数由调用方/前端决定。
- **Tracker（`FaceTracker`）只属于流式会话（session），不属于 HTTP 请求路径。** 请求之间没有帧序，`max_lost`/速度估计/`label_of` 缓存全部失效，强行塞进 HTTP 会引入跨请求泄漏。

### 2.2 四个外壳（同一个二进制，四个子命令）

```
facerec identify   # 无状态识别，JSON 输出（CLI / 脚本 / LLM）
facerec serve      # HTTP/gRPC 服务端（/v1/identify, /v1/embed, /v1/compare, /v1/people）
facerec enroll     # 离线建库（photos → gallery 索引），不暴露给 LLM
facerec mcp        # MCP 工具（stdio / streamable HTTP），直接链接 lib，不 shell out
```

---

## 3. 核心设计原则（优雅 + 合理 + 高效）

1. **库不依赖应用**：`FaceRecognition.h` / 库层绝不包含 `config.hpp`；`config.hpp` 是 app 层概念。`FeatureDatabase<T>` 是 header-only 模板，配置改动不应触发全库重编。
2. **无状态 engine**：`Engine::detect/embed/match` 标 `const`，不持有跨请求可变状态。多实例 = 多线程/多进程安全。
3. **Gallery 与模型解耦**：enroll 是离线管理操作，产出只读索引（person_id + unit-norm embedding + embedder sha256 + dim）。服务端启动时只读索引，不在构造函数里跑 DNN。
4. **按核并发，不超订**：CPU-bound（detect 0.93 ms + embed 4.47 ms/脸）。线程数 ≈ 核数，OpenCV 内部 `setNumThreads(1)`，避免嵌套并行。
5. **版本先行**：响应和索引都携带 `embedder_sha256` + `gallery_version`，版本不匹配拒绝匹配而非静默低分。
6. **“没脸”不是错，低于阈值也不是错**：`200 + faces:[]` / `matched:null`，区分“没人”“有人但未知”“服务异常”。
7. **安全优先**：enroll/delete 不进 MCP tool 列表；不记录图像字节/embedding/人名映射；请求大小与像素上限；有界队列 + 背压（429/503）。

---

## 4. 接口契约

### 4.1 无状态识别（平面 A，占 95% 流量）

```
POST /v1/identify
  入参：image（文件 / base64 / URL），top_k?, min_score?,
       include:[crops,embedding,landmarks],
       detector_input?:{w,h}   // 请求级，量化成几档：160×120 / 320×240 / 640×480
  出参：200 {faces:[{bbox:[x,y,w,h] /*原图像素*/, det_score, landmarks?,
                    matched:{person_id,name,score}|null,
                    top_k:[{person_id,score}...], crop_png_b64?}],
         model:{embedder_sha256, gallery_version},
         timing_ms:{decode, detect, align, embed, match}}
```

- `detector_input` 请求级可调：合照/远景用 640 检测换召回，实时流仍用 160×120。模型是 shape-dynamic 的（AGENTS 记录过多种尺寸都能解码），只需按 size 缓存 `PriorBox`。
- `min_score`/`top_k` 是请求参数，阈值调优无需重启。
- 缓存键 = `sha256(image_bytes) + gallery_version + model_hash + params`。

### 4.2 其他端点

```
POST /v1/embed   {image or crops} → {vectors:[float32[512]]}
POST /v1/compare {a, b} → {cosine}
GET  /v1/people  → {items:[{person_id, name, enrolled_at, n_photos}]}   // 不含 embedding/照片
GET  /healthz    → 进程活着
GET  /readyz     → model_hash / gallery_version / n_people / ready
```

### 4.3 流式会话（平面 B，tracker 住在这里）

> **已作废（见 D2）**：会话语义不进 `serve`。若需要服务端拉流，做独立进程 `facerec ingest`，
> 内部调平面 A，并复用 `FaceTracker` 作为库；`serve` 永远保持无状态。以下为原始设计，仅存档。


```
POST   /v1/sessions        {source: rtsp://... 或 push} → {session_id}
GET    /v1/sessions/{id}/events   (SSE / WebSocket)
DELETE /v1/sessions/{id}
```

会话有界 LRU + TTL，空闲自动回收；`TrackView.measured`、`label_of`、`max_lost` 全部封在会话内部。

### 4.4 MCP tools（给 LLM 调用）

| tool | 输入 | 输出要点 | 备注 |
|---|---|---|---|
| `identify_faces` | `image_b64` 或 resource URI | faces[] + match/unknown + top_k + det_score | 只读 |
| `compare_faces` | 两张图 / 两个 embedding | cosine + 判定 | 只读 |
| `list_people` | — | person_id + name（**不含 embedding**） | 只读 |
| `get_face_crop` | image + index | 112×112 对齐图（image content） | 只读 |

- **`enroll_person` / `delete_person` 不进 MCP tool。** 生物特征入库是不可逆写入，不交给可能被 prompt injection 驱动的调用方；只走 CLI/管理 API。
- 返回结构化 JSON + 可读摘要一起；`unknown` 时不得让模型猜名字（`matched:null, top_k:[{name, score}]`，description 明确低于阈值不推断身份）。
- 传输：stdio（桌面/IDE）+ streamable HTTP（远程/多人，OAuth/bearer）。图片用 112×112 对齐图，不传原图。

### 4.5 CLI 契约（给脚本与 LLM）

```sh
facerec identify --image a.jpg --json            # stdout 只有 JSON，日志走 stderr
facerec identify --image a.jpg --json --top-k 3 --include crops
facerec enroll --name 张三 --dry-run photos/*.jpg
```

规则：永远非交互；`--json` 是默认；业务性“没找到脸”也是退出码 0（`faces: []`）；服务/参数错误非 0，stderr 也是结构化 JSON；不把图片内容打进日志；enroll/delete 必须 `--dry-run` 或 `--yes`。

---

## 5. 并发与性能模型

### 5.1 CPU（默认，本机 16 核）

```
推荐：多进程 + SO_REUSEPORT，每核一个单线程 worker 进程
  - 每个进程独占一份 Detector + FeatureExtractor（模型 340 KB + 7.9 MB，16 份无所谓）
  - cv::setNumThreads(1) 禁止 OpenCV 内部 parallel_for 再开线程
  - 崩溃隔离：一个 Net 断言不会带走全服务
  - NUMA/亲和性天然正确
前置：nginx/envoy 或直接 SO_REUSEPORT 分发
备选：单进程 N worker 线程池 + setNumThreads(1)（更简单，但崩溃不隔离）
```

- **CPU 上不要 micro-batching。** AGENTS 已测：OpenCV CPU 后端 batch 是 1.31× 变慢（N=8 时 −22%），因为逐样本执行图 + blob 重建。CPU 的并行粒度是“多个独立 Net 实例”，不是“一个 Net 多个样本”。
- 单脸请求 ≈ 5.4 ms 推理 + 5–10 ms JPEG 解码（1080p）。单核上限约 100 req/s 量级，16 核约 1.5k req/s（随脸数分布下降）。超过 worker 数的请求排队，有界队列满了返回 429/503。

### 5.2 GPU / NPU（可选后端）

- 单进程、请求队列 + 动态 batching（2–5 ms 窗口），所有 `forward` 串行化。
- `FeatureExtractor` 已有 `batch_mode_`，GPU/NPU 自动开启（`target != DNN_TARGET_CPU`）。

---

## 6. Gallery 与索引设计

从“构造函数的副作用”变成“可部署的数据”：

```
gallery 索引（sqlite 或单文件 / 目录里的 .bin）
  person_id, display_name, photo_id, embedder_sha256,
  dim, unit_norm float32[512], created_at, provenance
```

- **enroll 是离线操作** (`facerec enroll --name 张三 photos/*.jpg`)，不是请求路径。
- 匹配 = 一次 GEMM（`N×512` 矩阵乘查询向量），而非 per-person 虚函数循环。1 万人 ≈ 5 GFLOP，`cv::gemm` 几十 µs；10 万级再考虑 HNSW/FAISS。
- 保留 `feature_fusion`（每人一份融合特征），但额外存照片级向量，方便阈值标定与重算。
- `CosSimilarity::similarity` 假设库侧已 unit-norm，写入侧必须强制。

---

## 7. 代码骨架（现状，P0–P2 已落地）

```
src/
  engine.{hpp,cpp}   ✅ 库层：Detector/FeatureExtractor/Similarity/align*/extract*/FeatureDatabase<T>
                        detect(const Mat&) const 按值返回；无 GUI、无 config.hpp 依赖
  api.{hpp,cpp}      ✅ 应用层：Pipeline（detector+extractor+gallery）+ IdentifyResult + JSON
                        CLI / HTTP / MCP 三个外壳共用，输出不可能不一致
  cli.{hpp,cpp}      ✅ identify / enroll / serve / mcp 子命令
  server.{hpp,cpp}   ✅ HTTP：cpp-httplib + PipelinePool + 优雅停机 + 背压
  mcp.{hpp,cpp}      ✅ JSON-RPC 2.0 + stdio/HTTP 两种传输 + 4 个只读工具
  json_util.{hpp,cpp} ✅ 容错 JSON 取值（HTTP 与 MCP 共用，杜绝解析分叉）
  encoding.{hpp,cpp}  ✅ base64 编解码（HTTP image_b64 / MCP image content 共用）
  pipeline_pool.{hpp,cpp} ✅ PipelinePool + Lease（serve 与 mcp-http 共用）
  result_cache.{hpp,cpp}  ✅ sha256 键 LRU 结果缓存（命中路径不解码）
  sha256.{hpp,cpp}        ✅ 自包含 FIPS 180-4
  gallery.{hpp,cpp}  ⬜ 只读索引持久化（当前 gallery 仍在 FeatureDatabase 构造里现算）
  session.{hpp,cpp}  ⬜ 流式会话：tracker + label_of + max_lost（serve 的平面 B）
third_party/
  httplib.h          ✅ vendor 的单头 HTTP 库 v0.15.3（sha256 记在 P2 小节）
main.cpp             ✅ 子命令分发 + run_demo()（原 UI 循环原样保留）
```

已完成的关键接口改造：`Detector::detect(const cv::Mat&) const -> std::vector<Face>`（按值返回，
不再持有 `dets`/`output_blobs` 作为跨调用状态）；`FeatureDatabase::select_max_similarity` 标
`const` 并有 `select_top_k(feature, k)`；`features_from_img` 收 `const cv::Mat&`。

---

## 8. 实施路径（每步独立可验证）

### P0：拆库（最优先，不改 `./main 0` 行为）— 已完成

- [x] 新建 `src/engine.hpp` / `src/engine.cpp`：从 `FaceRecognition.h/.cpp` 抽出 `Detector`/`FeatureExtractor`/`Similarity`/`FeatureDatabase<T>`/`align*`/`extract*`，`detect` 改为 `const` 按值返回（不再维护内部 `dets`/`output_blobs` 缓存），`features_from_img` 输入改为 `const cv::Mat&`。
- [x] `FeatureDatabase<T>` 保持 header-only 模板，留在 `src/engine.hpp`，不依赖 `config.hpp`。
- [x] `FaceRecognition.h` 改为转发头 `#include "engine.hpp"`，`main.cpp` 零侵入。
- [x] `CMakeLists.txt`：把 `FaceRecognition.cpp` 换成 `src/engine.cpp`，加 `src` + 根目录 include 路径。
- [x] `face_tracker.cpp`：`#include "FaceRecognition.h"` → `#include "engine.hpp"`。
- [x] `main.cpp` / `run_pipeline` 逻辑不动，`./main 0`/`./main 1` 行为与当前一致（已验证：Loaded 2 person(s)、Cannot open source: 1、退出码非零）。
- [x] `gallery.{hpp,cpp}`：只读索引读写，离线 build，不在构造里跑 DNN（P0 先保留 `FeatureDatabase` 在头文件，P1+ 再拆出持久化索引）。

### P1：CLI（可被脚本/LLM 调用）— 已完成

- [x] `facerec identify --image --json`：stdout 纯 JSON，`faces:[]` 也是 exit 0。
- [x] `facerec enroll --name --dry-run|--yes`：离线建库验证，输出索引摘要（持久化索引写入留 P1+/P2）。
- [x] 验收：`data/BruceLi/*.jpeg` 自匹配 1.000；`data/IvanYang/*.jpg` 自匹配 1.000，交叉 0.0226；`jq`/`python3 -m json.tool` 能吃输出；无人脸图片返回 `faces:[]` exit 0；`--include landmarks|crops|embedding`、`--top-k`、`--min-score` 均正常；`--set` 未知 key 给最近名称建议。
- [x] `--image` 缺失报错并 exit 非零；`./main` 无子命令打印总览用法。

### P2：Serve（HTTP 服务端）— 已完成

- [x] HTTP 库：**cpp-httplib v0.15.3**，单头 vendor 到 `third_party/httplib.h`
      （309,946 B，sha256 `a3347656c71c81c3fe510fdb1ff229d7db52eb7df04d3342d340eba0b6ad50ea`），
      header-only、零构建依赖、只要 pthread。HTTPS 由反向代理终结，不链 OpenSSL。
- [x] `src/api.{hpp,cpp}`：**应用级 `Pipeline` 类**（detector+extractor+gallery）+ `IdentifyResult`
      + JSON 序列化。CLI / HTTP / 未来 MCP 共用同一份实现，三个外壳不可能输出不一致。
- [x] 端点：`POST /v1/identify`、`POST /v1/embed`、`POST /v1/compare`、`GET /v1/people`、
      `GET /healthz`、`GET /readyz`。
- [x] `src/server.{hpp,cpp}`：`PipelinePool`（N 个 Pipeline 实例 + 借用/归还 + 条件变量），
      并发度 = worker 数 = 背压；借不到返回 `503 unavailable`。
- [x] `cv::setNumThreads(1)`（`--cv-threads`，默认 1）；请求体/像素上限；优雅停机（SIGINT/SIGTERM）。
- [x] 向后兼容：`./main 0` 仍走 demo（打印一行 note），`./main demo 0` 显式形式。

**实测（本机 16 核，640×607 单脸图，16 客户端并发，每档 128 请求）：**

| workers | `--cv-threads 1`（默认） |
|---|---|
| 1 | ≈52 qps |
| 2 | ≈96 qps |
| 4 | ≈160 qps，p50 ≈24 ms |
| 8 | ≈220 qps，p50 ≈33 ms |
| 16（64 客户端，超订） | ≈220 qps，不再涨 |

- 单请求延迟（无排队）：p50 **~20–25 ms**（包括 TCP 往返与 JSON 解码），纯推理 ~5.4 ms。
- 旧文档里的 165/191/175 qps 是 cpp-httplib `CPPHTTPLIB_LISTEN_BACKLOG=5` + Nagle（默认 false）下
  测的：污染的是**每个响应**的尾巴，不是并发能力。开 `TCP_NODELAY` 后 p99 从 ~1 s 掉到 ~150 ms，
  p50 从 ~63 ms 掉到 ~25 ms。
- 结论：**8 worker + cv-threads 1** 是这台机器的拐点；16 worker 不再涨（压测端线程 + GIL
  开始主导，qps 的增益被客户端自己吃掉了）。真实客户端应在此基础上重测，并用
  `scripts/stress.py --healthz` 先量出压测器的自身天花板。
### P4-1：请求级 `detector_input`（已完成）

- [x] `Pipeline` 内部按检测尺寸惰性创建 `Detector`（`std::map<(w,h), entry>`，LRU 上限
      8 个尺寸 —— 尺寸来自请求，不设上限就是一条无界内存增长路径）。
- [x] **画布（letterbox 目标）固定为配置值**，不随请求变化。这条边界是正确性的关键：
      对齐裁剪取自画布，所以 gallery 与任何请求的对齐来源分辨率一致，换检测尺寸不会
      把特征推到另一个域。请求级尺寸只改变"送进检测器的像素数"。
- [x] 校验：必须能被 8 整除、在 [32,1024]；否则 400（而不是让 `PriorBox` 在请求中途抛 500）。
- [x] CLI `--detector-input` 也走请求级参数（不再改 config，从而不改画布）。

**实测（同一张图，gallery 仍按 160×120 建的）：**

| detector_input | detect | 与 gallery 的 cosine |
|---|---|---|
| 160×120 | 1.6 ms | 1.0000 |
| 320×240 | 11.4 ms | 0.9635 |
| 640×480 | 35.0 ms | 0.9768 |
| 960×720 | 87.5 ms | 0.9396 |

- 召回换成本是**平方级**的：像素 ×16，检测耗时 ×22。
- cosine 下降到 0.94–0.98 不是"变差"，而是**关键点更准**：gallery 那一侧是 160×120
  下采样的粗关键点，两侧几何不再同源。身份判定仍然正确（阈值 0.2 远未触及），但如果
  某次部署常用大尺寸，就应该同时把 `detector.input_w/h` 调大，让 gallery 与查询同源 ——
  请求级尺寸是"提高召回"的旋钮，不是"免费提精度"的旋钮。**不要把默认值改掉**：
  README/AGENTS 里 0.845/0.874 那组基线都是在 160×120 下测的。

### P4-2：sha256 结果缓存（已完成）

- [x] `src/sha256.{hpp,cpp}`：自包含 FIPS 180-4 实现，三组标准测试向量与 `sha256sum` 逐一比对通过。
      不链 OpenSSL（HTTPS 由反向代理终结，这里只要一个摘要）。
- [x] `src/result_cache.{hpp,cpp}`：LRU，双限（条目数 + 字节数，默认 256 / 128 MB），
      命中统计进 `/readyz`。
- [x] 键 = `sha256(图像字节) + 指纹(模型路径 + gallery 人数 + data_dir mtime + 画布) +
      检测尺寸 + top_k + min_score + include_*`。
- [x] **键算在解码之前**：命中路径省掉 `imdecode` + 检测 + 提取，只花 sha256
      （约 200 MB/s，50 KB 的 JPEG ≈ 0.25 ms）。HTTP 回 `X-Cache: HIT|MISS`。
- [x] 键里包含所有影响输出的参数，而不是"缓存核心再裁剪"：后者要为 crop/embedding
      设计重算路径，代码与边界情况都翻倍，换来的只是"同一张图换 top_k 也命中"。

**实测（8 worker，16 客户端，同一张图重复请求）：**

| | qps | p50 | p99 |
|---|---|---|---|
| 缓存关闭 | 205 | 72.8 ms | 116.5 ms |
| 缓存开启 | **1347** | **4.4 ms** | 67.9 ms |

6.6× 吞吐、16× p50。交替提交两张不同图三轮，结果始终各归各位（没有串图）。
注：这是"重复同一张图"的上界；真实流量的命中率取决于重试/重放比例。

**仍未做（P4 剩余）：** gallery 持久化索引（`src/gallery.cpp`）、流式会话平面 B
（`src/session.cpp`）、指标直方图、鉴权/审计、多进程 `SO_REUSEPORT`。

**踩过并修掉的坑（别重新发现）：**

- `httplib.h:6160` 对 `Content-Type: application/x-www-form-urlencoded` 的请求体有一道 **8192
  字节**硬上限，与 `set_payload_max_length` 无关，超了直接 **413**。`curl --data-binary` 的
  默认 Content-Type 正是 form-urlencoded，所以"传一张 900 KB 的图 → 413"看起来像限流 bug。
  修法：`server.cpp` 在 include 之前把 `CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH`
  抬到 64 MB，真正上限交给运行时的 `--max-request-mb`。注意它只影响 form-urlencoded，
  `image/*` 走的另一条路径没有这个限制（这就是为什么 JPEG 请求一直是 200）。
- `cd dir && cmd &` 会把 `cd` 一起后台化，前台留在原目录 —— 压测脚本因此读不到 `data/`。
- 用 `pgrep -f "main serve ..."` 杀进程会匹配到执行该命令的 shell 自身，实测把自己杀了；
  用启动时记录的 `$!`。
- `curl --data-binary @path/*.jpg` 里的 `@` 会挡住 glob 展开（`@` 与通配符在同一 word），
  先赋给变量再 `@"$VAR"`。

### P1.5：`demo` 子命令与兼容（已完成）

- `./main demo 0` / `./main 0`：原实时窗口循环，代码未变（`main.cpp:run_demo`）。
- 非命令的首个参数一律落到 demo，并打印一行 note；`--dump-config` 等旧形式继续可用。

### P3：MCP（LLM 工具）— 已完成

- [x] `src/mcp.{hpp,cpp}`：JSON-RPC 2.0 核心 + 两种传输，**直接链接 lib**（不 shell out，
      避免每次 tool 调用冷启动 `readNet` 几百 ms）。
- [x] 传输：`stdio`（换行分帧，stdout 是协议通道、日志走 stderr）与 `http`（`POST /mcp`）。
      `GET /mcp` 明确回 405 而不是挂住（本实现是请求-响应，不开 SSE 流）。
- [x] 四个只读 tool：`identify_faces` / `compare_faces` / `list_people` / `get_face_crop`，
      每个都带 JSON Schema。**`enroll_person` 刻意不注册**：实测 `tools/call` 到它返回
      `-32602 unknown tool`，而不是"工具存在但内部拒绝"。
- [x] 协议方法：`initialize`（回显客户端 protocolVersion）、`notifications/*`（静默）、
      `ping`、`tools/list`、`tools/call`；未知方法 `-32601`，缺参数 `-32602`。
- [x] 工具结果是"文本摘要 + JSON"双内容；错误走 `isError:true`（工具级），不是协议错误。
- [x] 验收（实测）：stdio 6 条消息 → 6 条响应（通知不回应）；`tools/list` 四个工具 schema
      正确；`list_people` 3 人；`identify_faces` 自匹配 1.000 + 候选；`compare_faces`
      同图 1.0000 / 跨人 0.0226；`get_face_crop` 返回 image content、越界 `isError`；
      HTTP 传输 initialize / tools/list / tools/call / 202 / 405 / -32601 全部正确。

**踩过并修掉的坑：**

- **stdio 是换行分帧的**：JSON Schema 用 raw string 书写（可读）时含真实换行，直接拼进
  响应会把一条消息截成多条 —— 表现为"6 条请求收到 33 行响应"、客户端 JSON 解析在行内
  中途报错。修法：发射前把 schema 里的 `\n`/`\r` 换成空格（JSON 允许对象内空白）。
  用 HTTP 传输时不会暴露这个 bug，所以**必须用 stdio 测**。

### P3.5：共用模块抽取（已完成）

`src/json_util.{hpp,cpp}`（容错 JSON 取值）、`src/encoding.{hpp,cpp}`（base64）、
`src/pipeline_pool.{hpp,cpp}`（`PipelinePool` + `Lease`）。server 与 mcp 两种传输之前各写
一份，现统一 —— 三处副本的字典序、base64 填充、池化语义不可能再分叉。

### P4：硬化（生产可用）— 部分完成，见 P5

- [ ] 多进程 + SO_REUSEPORT（CPU 生产默认）；GPU 路径动态 batching。
- [x] 指标：按阶段直方图（decode/detect/embed/request）、match_score 分布、unknown 率、cache 命中率（`GET /metrics`，Prometheus 文本格式）。队列深度未埋点。
- [ ] 鉴权（bearer/OAuth）、审计日志（不记图像字节/embedding/人名映射）。输入大小与像素上限已实现（`--max-request-mb`、`max_image_pixels`）。
- [x] 优雅停机：SIGINT/SIGTERM → `svr.stop()`（停止 accept，在途 handler 跑完）。
- [x] gallery 持久化索引（`paths.gallery_index`，带模型/gallery 指纹，失效自动重建）。
- [x] 并发正确性回归：同一张图 24 路并发 → 1 种结果（见 P5）。

---

### P5：并发正确性与运维硬化（本轮，已完成）

**症状**：`serve` 下同一张图有时返回 1 个脸、有时 2 个，且响应里出现 `nan` 分数；压测里
少数请求总耗时稳定 ~1.04 s。两条独立根因，都已定位并修掉。

#### 5.1 fused YuNet 导出最后一级 anchor 不可信（模型/OpenCV 侧）

证据（全部可复现，Debug 构建 + 临时插桩，插桩已删除）：

- NaN 出现在 `forward()` 返回之后、`decode()` 之前；把 align/embed 全关掉仍然出现。
- 与并发**无关**：全进程 forward 加互斥锁串行化 → 仍 12/48 个 forward 的 conf blob 含 NaN；
  `cv::setNumThreads(1)` → 24/48；`net.enableFusion(false)` → 仍复现。**不是竞争**。
- **按 Net 实例确定**：同一进程内 4 条 pipeline 处理同一张图，conf blob 最后 12 个 float 分别是
  `3f78e11c…`（看着像正常 softmax）/ `3f800000 0 …`（1.0/0.0）/ `fffeff82 …`（NaN）；
  同一实例每次调用完全一致，换一张图就换一组值。
- 受影响范围**恰好**是最后一级：prior 网格 300×3 + 70×2 + 15×2 + **2×1×3** = 1076 = 模型声明的
  `[1076,14]/[1076,2]/[1076,1]`，坏的是尾部 6 个 anchor（stride 64）。
- `face_detection_yunet_2023mar.onnx`（per-level 头，320×320）**0/44** 个 forward 受影响。
- 结论：这是这个 fused 导出 + OpenCV 4.13 执行的产物（最后一级的输入缓冲依赖 Net 实例状态），
  不是本仓库的并发 bug。追查过程中一度误判为 tracker/align/并发，别再重复。

修复（`priorbox.cpp:decode`，两道门，都是语义上本来就该成立的断言）：

1. 分数门用 `!(score >= ignore_score)` 而不是 `score < ignore_score`：NaN 与两个比较都是 false，
   旧写法把 NaN 分数**放行**（这正是幽灵框的来源）。
2. 几何门：解码后的框必须是"这张图里的一个框"（w,h ≥ 2 px、≤ 8×画面、与画面交集 ≥ 1% 面积）。
   分数被 clamp 到 [0,1]，垃圾 conf 也能伪造 0.99 分，但垃圾 loc 必然产生病态框。
   边缘被裁的脸仍然保留。

回归：修复前 CLI/服务端每图稳定多出 4 个幽灵候选（`n=5`，`s1=0.99995 s3=0.999759`），
修复后同一张图 24 路并发只有一种结果；真实检测的 bbox/score 一字未变
（BruceLi `299.15,71.75,111.31,126.47 / 1.0`，2023mar 头 `312.66,68.34,98.14,135.52 / 0.9115`）。

#### 5.2 cpp-httplib 的 listen backlog 默认是 5（吞吐与尾延迟）

`httplib.h:117` `CPPHTTPLIB_LISTEN_BACKLOG 5`，且 `Server::listen()` 没有 backlog 参数，
只能在 include 前定宏。并发连接 >5 时 accept 队列溢出 → 内核丢 SYN → 客户端 1 s 后重传。

- 实测（cache off，Debug）：workers=4 → p99 1040 ms / max 1078 ms；workers=8 → 5/64 请求 ~1.08 s；
  `curl` 的分项计时显示 **`connect` = 1015 ms**，而服务端 `timing_ms.total` 只有 16 ms。
- 修复：`CMakeLists.txt` 加 `target_compile_definitions(main PRIVATE CPPHTTPLIB_LISTEN_BACKLOG=1024)`；
  `ss -lnt` 确认 Send-Q 列变成 1024（被 `net.core.somaxconn`=4096 截断）。
- 修复后（Debug，同脚本）：workers=4 max 67 ms、workers=8 max 126 ms，1.04 s 尾巴消失。

**第二个隐性延迟：** cpp-httplib 的 `CPPHTTPLIB_TCP_NODELAY` 默认是 `false`（httplib.h:90），
于是 accept 出来的每个连接都开着 Nagle。响应被写成两段（headers，然后 body），第二段是小包，
撞上客户端延迟 ACK 就等 ~41 ms。实测：裸 socket 打 `/healthz` 0.1 ms，http.client keep-alive 41 ms，
strace 里 body segment 被扣住 41.5 ms。修复：`src/server.cpp` 里 `svr.set_tcp_nodelay(true)`。
开后 `/healthz` 回到 0.1 ms，p50 identify 从 63 ms 掉到 ~25 ms。这条税率曾被计进所有记录的延迟
数字，但不计进吞吐（客户端在等 ACK 时还在算连接，所以 qps 数字本身没被低估，只是 p50/p99
全虚高）。

诚实的吞吐数字（**Release**，`--cache-entries 0`，`--cv-threads 1`，4×W 并发客户端，
**开 `TCP_NODELAY`**（否则每个响应多付 ~41 ms Nagle/延迟-ACK 税）：

| workers | p50 | p90 | max | qps | RSS |
|---|---|---|---|---|---|
| 4 | 24 ms | 26 ms | 800 ms | ≈160 | 112 MB |
| 8 | 33 ms | 46 ms | 1475 ms | ≈220 | 147 MB |
| 16（64 客户端，超订） | 35 ms | 42 ms | 2923 ms | ≈219 | 217 MB |

`max` 列是压测脚本单次请求耗时的上限（含连接重建、GIL 抖动），而非服务端 handler 时间。
`p50`/`p90` 已不含 Nagle 固定成本；开 `TCP_NODELAY` 前 p50 ≈ 63 ms、p99 ≈ 1 s。

#### 5.3 运维小修（本轮）

- 权重缺失时 `check_model_files(..., &hint)` 直接给出可粘贴的修法（cwd + `--set paths.root=build`），
  解析规则不变（相对路径仍挂 `paths.root`，默认 cwd，锚定是 opt-in）。
- 裸词参数不再被当成 demo source：`./main compaer` 现在报 `Unknown subcommand`（带子命令列表）；
  `/`、`://`、全数字、cwd 下存在的文件仍走演示路径。
- `CMakeLists.txt` 在没有 `CMAKE_BUILD_TYPE` 时默认 Release（Debug 需显式指定）。
- `--cv-threads N` knob（默认 1）：并发模型是"多实例并行"而不是"单 Net 内并行"。
- `enroll --yes` 之前只把 `dry_run` 置 false 就报 `ok`，**什么都不写**（注册接口上最坏的错）。
  现在真的把"含可用脸"的照片复制进 `data/<person>/`：`--name` 必须是单个安全路径分量
  （`../evil`、`a/b`、前导 `-` 全拒），空白按索引规则归一，同名不覆盖（`_1` 后缀），
  一张能用的脸都没有就报错退出；`written`/`skipped`/`would_write_to` 都是 JSON 字段。
  顺带修了一个括号错位导致 dry-run 输出 JSON 不合法的问题。
- MCP `list_people` 之前数的是**目录**：空的 `data/test/` 被说成"已注册"，而那种人永远不可能
  匹配（LLM 会去推断一个不存在的身份）。现在摘要与 JSON 同源，并把"目录存在但无可用照片"单独点出来。
- `person_id` / provenance 的空白归一只剩一处（`normalize_entry_id`，声明在 `gallery.hpp`），
  建库时就归一：否则 `data/John Doe/` 冷启动叫 `John Doe`、索引重载后叫 `John_Doe`，
  同一个人两个名字。

## 13. 决策记录（ADR，2026-11-07，v1.2.0 之后）

每条决策写清"选了什么 / 为什么 / 否决了什么 / 怎么验收"。改动这些决策要在本文件里改并写原因，
**不要**在代码注释里悄悄漂移。

### D1 审计日志：查询不留身份，写入必留身份 —— **已实现**

- **决策**：审计是**一行一个 JSON 对象**的请求日志（默认 stderr，`log.audit` 可改路径或 `off`）。
  查询类（identify/embed/compare/people/crop）记录 `req/caller/op/status/ms/faces/matched/cache/bytes`，
  **不记录** person_id、匹配到的名字、embedding、图像字节、bbox、内容哈希。
  写入类（enroll、gallery 重建、模型加载、鉴权拒绝）**必须**记录 person_id 与文件清单。
- **为什么**：审计的用途是"事后能回答谁在什么时候用了服务、用了多少"，而不是"再建一个可检索的
  人脸-身份数据库"。把一个请求和一个人的名字持久地连在一起，就是把 gallery 之外的追责面扩大到
  了每一个被拍到的路人。反过来，enroll 是不可逆写入，它必须可审计——那里 person_id 是必需品。
- **否决**：① 记录 person_id 到查询日志（等于给每次查询建索引，泄漏面大于收益）；② 记录
  `sha256(image)`（看着无害，实则是一个可用作 join key 的内容指纹：拿到图的人能反查"这图被查过几次"）；
  ③ 完全不记录（滥用无法发现，卷量、调用方、人脸数已经是足够的滥用信号）。
- **验收**：`grep -c person_id` 在对查询日志里为 0；鉴权失败即使 `--quiet` 也必出现。

### D2 平面 B（流式会话）**不进 `serve`**；要流式就是客户端或 sidecar

- **决策**：`serve` 只有平面 A（无状态、可水平扩展、可缓存）。会话语义（tracker、label 缓存、
  max_lost、MJPEG/SSE 出流）放在**客户端**或一个可选的独立进程 `facerec ingest`（内部调平面 A，
  复用 `FaceTracker` 作为库）。PLAN 早期 §4.3 的 `POST /v1/sessions` 设计作废。
- **为什么**：服务端持有每路流的状态，等于把"一个可水平扩展的无状态服务"变成"一个有粘性的媒体
  服务器"：要管解码、重连、背压、TTL/LRU、每进程会话归属，而 tracker 的收益（平滑、稳定 id、
  省掉重复 embed）本质上是**时间维度**的问题，谁拥有帧谁就该拥有它。手机/浏览器/NVR 都有帧；
  没有帧的客户端也不需要 id。
- **否决**：① 在 `serve` 里实现 sessions（把最重的状态塞进最需要无状态的地方）；② 直接删掉计划
  （流式是真需求，只是位置错了）。
- **验收（现在）**：README 明确"单帧 + 批量，流式由客户端或 sidecar 承担"；`demo` 子命令仍是
  唯一的 tracker 消费者，且它与 API 共享同一套 `Pipeline`/`FaceTracker`。

### D3 鉴权：进程内 bearer 表（哈希存盘）+ "非 loopback 无 token 就拒绝启动" —— **已实现**

- **决策**：`auth.tokens_file`（默认空）指向一个 `name:sha256:<64hex>` 的行式文件（`#` 注释）。
  请求带 `Authorization: Bearer <token>`，常量时间比对；命中则把 `name` 作为审计里的 `caller`。
  **`--host` 不是 loopback 且没有 tokens 文件 → 拒绝启动**（`/healthz`、`/readyz` 免鉴权，其余全要）。
  `./main token --name <id>` 生成随机 token 并打印可直接粘贴的那一行。
- **为什么**：TLS 与边缘身份交给反向代理（本仓库不引入 OpenSSL 依赖），但"进程自己不知道谁在调"
  会让审计无从归因、也让误配（`--host 0.0.0.0` 直接裸奔）成为默认路径。把安全做成**绑定约束**
  而不是一个容易忘的开关，是这里唯一优雅的做法。
- **否决**：① OAuth/JWT（要引 crypto 依赖，且没有 IdP 时只是自造协议）；② 明文 token 存配置
  （配置文件会被复制、被 dotfile 同步、被贴进 issue）；③ 只靠反向代理（MCP-over-HTTP 的客户端
  常常直连，且审计没有 caller 就只剩时间戳）。
- **验收**：无 token → 401 + `WWW-Authenticate`；错 token → 401；对 token → 200；`--host 0.0.0.0`
  无 tokens → 启动失败并指出修法；token 文件里 `strings` 不到任何可用凭据。

> 实现时额外发现：cpp-httplib 对 `Content-Type: application/x-www-form-urlencoded` 的请求体有
> 8KB 内建上限，而 `curl --data-binary @photo.jpg` 默认发的就是这个类型 —— 48KB 的照片直接 413。
> 更麻烦的是这个上限**是否生效取决于包含 `httplib.h` 的 TU 数量**（同一请求、同一 header：
> 2 个 TU 放行、3 个 TU 413，实测，可复现）。结论：不要把语义建在库的内部启发式上，
> CMake 里显式钉住 `CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH`，让 `--max-request-mb`
> 成为唯一体积闸门。

### D4 `gallery.idx` 是**生物特征库**，按这个身份对待 —— **已实现**

- **决策**：写入时显式 `0600`；载入时若权限比 `0640` 更宽则**警告并给出修法**（不硬失败，避免把
  已有部署打挂）；README/CHANGELOG 明写"索引 = 明文 512-d 模板 + 人名，备份/加密/销毁按生物特征
  数据对待"；`paths.gallery_index: ""` 是"只在内存里注册"的选项。
- **为什么**：一个 0644 的文件里存着每个人的 512-d 模板，等于把整个身份库暴露给同机任何用户；
  而模板不可撤销（不能像密码那样改），所以只能靠"最小可见性"。
- **验收**：新写入的索引 `stat -c %a` 为 600；手工 `chmod 644` 后重启会打印警告；README 有该段落。

### D5 扩展性靠**进程数**，SO_REUSEPORT 是**显式 opt-in**

- **决策**：默认单进程多 worker；扩容 = N 个进程监听同一端口，内核按连接做负载均衡（`--reuseport`）。
  默认**不开**：cpp-httplib 的 `default_socket_options` 在 Linux 上自带 SO_REUSEPORT（httplib.h:1867），
  这意味着两个进程绑同一端口不会报 EADDRINUSE，而是**静默地各接一半连接** —— 重启服务后老进程
  还在，一半流量打到旧二进制，表现为"时灵时不灵"，日志里什么都没有。默认关掉它，只开 SO_REUSEADDR，
  这样端口被占时立刻启动失败并说明原因；`--reuseport` 让管理员自己决定，并可和 systemd 的
  `ExecStart` 实例数配合（见部署样例）。
- **为什么**：每个进程有自己的 Net 集合，实测 8 worker 后已到 CPU 墙（≈220 qps），跨进程是唯一还有
  收益的方向。反代（nginx/caddy/haproxy）提供 TLS、超时、重试、连接数上限，与 SO_REUSEPORT
  互补而非互斥。
- **验收**：1/2/4 进程 × 4 worker 的聚合 qps ≈ 161/190/199（允许 ±15%），p99 随进程数
  下降（1501 ms → 109 ms，因为请求被更细地拆分）。文档给出该拓扑。

### D6 回归网：三个 harness 变成 `ctest`，而且**数字变成断言**

- **决策**：`make check`（CTest）固定三件事：① `PriorBox::decode` 的两道门（NaN 分数、
  退化几何 → 0 候选；正常张量 → 期望候选）；② tracker 的 6 个脚本化场景（AGENTS.md 里的配方）；
  ③ 服务端契约（鉴权 401/200、413、优雅停机、24 路并发 → 1 种结果）。
- **为什么**：v1.2.0 的所有保证目前都活在一次性 harness 里；最容易被后人"简化"掉的正是
  decode 的两道门和 tracker 的两道门。数字进断言，回归就从"读 commit message"变成"测试红了"。
- **验收**：`ctest` 全绿；把 `priorbox.cpp` 的分数门改回 `score < ignore_score` 必须让 ① 变红
  （测试必须真的能失败 —— 这是最有价值的一条）。

### D7 模型：把 per-level 导出扶正，fused 降级为 legacy（待 A/B）

- **决策**：per-level（2023mar, 320×320）跑完 augmentation harness 后，若 margin 不低于当前
  0.874 的同等条件，则改为默认；fused 导出保留但降级。**decode 的两道门不管换不换都留着**。
- **为什么**：一个"最后一级 6 个 anchor 会写出 NaN"的模型是永久性地雷，靠解码侧兜住是伤疤不是
  免疫。但换默认值必须有同样的测量（同 seed、同 warp、同身份集合）。
- **否决**：① 现在就换（没有 A/B 就是猜）；② 保留 fused 并删掉兜底（不可能）。
- **验收**：harness 报告 within/between/margin 三组数字；换默认后 24 路并发仍只有 1 种结果。

### D8 API 契约只增不减；破坏性变更走 `/v2`

- **决策**：`/v1/*` 响应字段只增不改语义；新增字段永远可选；MCP tool 的 JSON Schema 同样只增。
  破坏性变更 = 新前缀 + 旧前缀保留一个 release cycle。
- **为什么**：调用方是脚本和 LLM，两边都会缓存 schema 与字段名；静默改语义在 LLM 侧表现为
  "模型说了奇怪的话"，比 500 更难查。

## 9. 不变量与禁区（重写绝不能碰）

这些是 AGENTS.md 已实测确认的正确性，重构时必须原样保留，不能“简化”：

1. **注册与推理必须同一预处理**——`align.border` 挂在 `FeatureExtractor` 上，`FeatureDatabase` 构造前设好，否则 gallery 与查询特征不可比。
2. **`measured` 才 embed**；纯预测轨迹不识别，避免幽灵框冒出随机名字。
3. **无匹配必须写空名覆盖 `label_of` 缓存**，否则人走名赖在框上。
4. **`CosSimilarity::similarity` 假设 stored feature 已 unit-norm**；返回值是 `‖ref‖·cos`，不是 [0,1] 归一化 cosine。两边必须一起改。
5. **`align_detection` 的 ROI 是子矩阵，`copyMakeBorder` 必须 `BORDER_ISOLATED`**，否则 OpenCV 会静默读父图像素。
6. **检测头由模型层名推断，永不配置**（FusedHead vs PerLevelHead，第三种布局 = 新类，非新 knob）。
7. **YuNet 关键点顺序**：`[right_eye, left_eye, nose, mouth_left, mouth_right]`，offsets 8/9 是 **nose** 不是 mouth。
8. **`cv::Size` 顺序 `{width, height}`**；`img_shape={160,120}` = 160 宽，120 高。
9. **检测坐标永远在 detector-input 空间（160×120）**，`align_detection` 传同一图作为 `detection_shape` 的 pair。
10. **模型契约 knob（`detector.input_w/h`、`embedder.input_w/h`、`out_*`）启动期不可变**；阈值类 knob 可运行期调。

---

## 10. 验收标准（Definition of Done）

| 阶段 | 验收命令 | 预期 |
|---|---|---|
| P0 | `./main 0` / `./main 1` | 与当前行为一致：Loaded N person(s)，画框，ESC 退出，非摄像头源 `waitKey(0)` |
| P0 | `./main --dump-config` / `--set` | 默认值打印；未知 key 给最近名称建议；越界值硬失败并指名 `file:line` |
| P1 | `facerec identify --image data/BruceLi/*.jpeg --json` | 自匹配 top-1 = BruceLi，score ≈ 1.000；stdout 纯 JSON；退出码 0 |
| P1 | `facerec enroll --name test --dry-run` | 不写入磁盘，只打印计划；加 `--yes` 才真建库 |
| P2 | `wrk -t16 -c100 -d30s http://.../v1/identify` | P99 < 200 ms（单脸），QPS ≥ 单核估计值 × 核数 × 0.7；无 OOM |
| P3 | MCP Inspector / Claude Desktop | `identify_faces` 返回 JSON + 文本；`list_people` 不含 embedding；无 enroll tool |
| P4 | `./facerec serve &` + `kill -TERM` | 新请求拒绝，在途请求完成，进程干净退出 |

---

## 11. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|
| OpenCV 版本差异（4.13 vs 其他）导致 ONNX 导入失败 | 中 | 高 | 启动前 `check_model_files()` + sha256；`/readyz` 回报 `model_hash`；版本不匹配拒绝服务 |
| GPU 后端不可用（本机无 CUDA/OpenVINO） | 高（本机） | 中 | `DNNConfig::cuda()/openvino()` 在模型配置时即 abort，错误信息指名后端；默认 cpu 可用 |
| LLM 把 `top_k` 里的低分名字当真 | 高 | 高 | tool description 明确“score < 阈值不得推断身份”；返回 `matched:null` 时 top_k 仅供参考 |
| 多进程共享模型文件被意外覆写 | 低 | 中 | 模型文件只读挂载；启动时 sha256 校验，变更即重启 |
| 解压炸弹 / 巨图拖死 worker | 中 | 高 | 请求大小上限 + 像素上限 + `IMREAD_REDUCED_COLOR_2` 先读 header |

---

## 12. 下一步行动（按 D1–D8 排序的落地顺序）

1. ~~D3 鉴权 + D1 审计~~ **已完成**（token 表 / `token` 子命令 / 非 loopback 拒绝启动 /
   结构化审计；`/healthz` 与 `/readyz` 免鉴权，401 永远记审计且只记一行）。
2. ~~D4 索引权限~~ **已完成**（新建即 0600；更宽权限告警一次并给出修法）。
3. **D6 回归网**（进行中）：decode 两道门的单测（`tests/decode_gates.cpp`）和 tracker 六场景
   （`tests/tracker_scenarios.cpp`）已加入 `ctest`，`make check` 一键，2/2 绿；突变实验已验证
   两条门必须存在（关掉位移门 → "贴边翻转不拆轨迹" 变红）。服务端契约（401/413/优雅停机/
   24 路并发 → 1 种结果）目前以脚本化手工检查存在（`scripts/stress.py` + 同端口多实例验证），
   不适合写进 ctest（需要端口、权重文件、后台进程，容易留下僵尸）。
4. **D5 部署样例**：systemd unit + Dockerfile + 反代片段；实测 4 进程聚合 qps（预期 ≈4×135）。
5. **D7 模型 A/B**：per-level 导出跑 augmentation harness，再决定是否改默认。
6. 日志分级（audit 之外的运行日志）、`/metrics` 里 lease 等待与 503 的独立计数（现混在
   `request_errors_total`）。
7. 可选项：CLI 的 `compare`/`verify` 子命令（与 server/MCP 对齐）。

> 更新本文件时：任何架构决策变更（并发模型、MCP tool 列表、API 端点）必须在此同步，并记录变更原因，防止 compaction 后记忆丢失。