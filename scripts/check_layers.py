#!/usr/bin/env python3
"""源码布局守卫：include 只能指向同层或下层，且每个 .cpp 都必须在 CMakeLists.txt 里。

层序：common(0) < core(1) < app(2) < serve(3) < cli(4) < src/main.cpp(5)
规则：
  1. 项目 include 必须写成 "<层>/<文件>"（如 "core/engine.hpp"）；未限定即失败。
  2. 被 include 的层号不得大于包含者（同层允许）。
  3. 每个 src/**/*.cpp 的路径必须原样出现在 CMakeLists.txt 里（防"编过了但没跑"）。
  4. third_party 头用尖括号（<httplib.h>），不参与上面的规则。
退出码 0 = 通过；1 = 有违规（逐行打印 文件:行: 原因）。

用法：python3 scripts/check_layers.py [仓库根目录]
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
