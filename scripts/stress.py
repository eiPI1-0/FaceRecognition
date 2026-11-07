#!/usr/bin/env python3
"""并发压测：N 个长连接 worker，每个反复发同一类请求，统计延迟分布与 qps。

用法: stress.py <port> <clients> <requests_per_client> <image|--healthz> [token]

刻意用**长连接**（keep-alive）而不是每请求新建连接：要量的是服务端处理能力，
不是 accept 队列（backlog 那条 1.0s 尾巴是另一件事，见 AGENTS.md）。

`--healthz` 代替图片路径时压的是**压测器自己**的上限 —— 这个数字必须远高于被测吞吐，
否则测到的是 Python 线程 + GIL 而不是服务端。它低于/接近被测吞吐时，绝对 qps 不可信，
只能用于相对比较（换 wrk/vegeta 或多机再谈绝对值）。
"""
import base64
import http.client
import json
import sys
import threading
import time


def worker(port, path, headers, body, n, sink):
    """一个 worker：一条长连接上连发 n 次，结果整体交给 sink（调用方给一个 list）。"""
    lat, errors, first_error = [], 0, None
    is_probe = path == "/healthz"
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            conn.request("GET" if is_probe else "POST", path,
                         body=None if is_probe else body, headers=headers)
            res = conn.getresponse()
            payload = res.read()
            dt = (time.perf_counter() - t0) * 1000.0
            ok = res.status == 200 and (
                is_probe or json.loads(payload).get("status") == "ok"
            )
            if ok:
                lat.append(dt)
            else:
                errors += 1
                if first_error is None:
                    first_error = "HTTP %d %s" % (res.status, payload[:120])
        except Exception as exc:  # noqa: BLE001 - 种类不影响结论，但要能看见第一条
            errors += 1
            if first_error is None:
                first_error = type(exc).__name__ + ": " + str(exc)[:120]
            try:
                conn.close()
            except Exception:
                pass
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    sink.append((lat, errors, first_error))


def main():
    port, clients, per = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    probe = sys.argv[4] == "--healthz"
    path = "/healthz" if probe else "/v1/identify"
    token = sys.argv[5] if len(sys.argv) > 5 else ""

    body = b""
    if not probe:
        with open(sys.argv[4], "rb") as fh:
            body = json.dumps({"image_b64": base64.b64encode(fh.read()).decode()}).encode()
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    if token:
        headers["Authorization"] = "Bearer " + token

    sink = []
    threads = [
        threading.Thread(target=worker, args=(port, path, headers, body, per, sink))
        for _ in range(clients)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    lat, errors, first_error = [], 0, None
    for part_lat, part_err, part_fe in sink:
        lat.extend(part_lat)
        errors += part_err
        first_error = first_error or part_fe

    if not lat:
        detail = "; first: " + first_error if first_error else ""
        print("no successful request (errors=%d)%s" % (errors, detail))
        sys.exit(1)

    lat.sort()

    def p(q):
        return lat[min(len(lat) - 1, int(q * len(lat)))]

    print(
        "clients=%-3d reqs=%-5d ok=%-5d err=%-3d qps=%6.1f  p50=%6.1f p90=%6.1f p99=%6.1f "
        "max=%6.1f ms"
        % (clients, clients * per, len(lat), errors, len(lat) / wall, p(0.50), p(0.90),
           p(0.99), lat[-1])
    )


if __name__ == "__main__":
    main()
