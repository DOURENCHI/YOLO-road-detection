# -*- coding: utf-8 -*-
"""并发分块下载 ORT CUDA EP 缺失的 7 个运行库 wheel。

为什么不用 pip 直接装
    WSL 侧连 files.pythonhosted.org 不通（HTTP 000），清华 PyPI 源 403。
    由 Windows 侧下载好放到 _wheelhouse，再让 WSL 本地 pip 装。

为什么分块
    单连接被限速到 ~90 KB/s，16 线程 Range 分块可叠加到 ~1 MB/s（实测）。
"""
import io, os, re, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT = r"D:\VC Code\YOLO_Road_Detection\ros2\_wheelhouse"
os.makedirs(OUT, exist_ok=True)

# (包名, 目标版本) —— 版本按 ORT 1.22 要求的 CUDA 12.8 / cuDNN 9 选定
TARGETS = [
    ("nvidia-cuda-runtime-cu12", "12.8.90"),
    ("nvidia-cublas-cu12",       "12.8.5.5"),
    ("nvidia-cufft-cu12",        "11.4.1.4"),
    ("nvidia-curand-cu12",       "10.3.10.19"),
    ("nvidia-cuda-nvrtc-cu12",   "12.8.93"),
    ("nvidia-cudnn-cu12",        "9.9.0.52"),
]
CHUNK = 8 * 1024 * 1024
WORKERS = 16


def resolve(pkg, ver):
    """用 PyPI JSON 找到 manylinux x86_64 wheel 的真实 URL 和大小。"""
    d = json.load(urllib.request.urlopen("https://pypi.org/pypi/%s/%s/json" % (pkg, ver), timeout=40))
    for f in d["urls"]:
        if "manylinux" in f["filename"] and "x86_64" in f["filename"]:
            return f["filename"], f["url"], f["size"]
    raise RuntimeError("没有 manylinux x86_64 wheel: %s %s" % (pkg, ver))


def fetch_part(url, out, idx, start, end, total):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={
                "Range": "bytes=%d-%d" % (start, end),
                "User-Agent": "Mozilla/5.0",
            })
            data = urllib.request.urlopen(req, timeout=120).read()
            if len(data) != end - start + 1:
                raise IOError("short %d/%d" % (len(data), end - start + 1))
            with open(out, "r+b") as f:
                f.seek(start); f.write(data)
            return idx, len(data)
        except Exception as ex:
            if attempt == 3:
                return idx, -1
            time.sleep(2)


def download_one(name, url, size):
    out = os.path.join(OUT, name)
    if os.path.exists(out) and os.path.getsize(out) == size:
        try:
            with zipfile.ZipFile(out) as z:
                if z.testzip() is None:
                    print("  跳过（已完整）: %s" % name[:52]); return True
        except Exception:
            pass
    with open(out, "wb") as f:
        f.truncate(size)
    n = (size + CHUNK - 1) // CHUNK
    done, bad = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch_part, url, out, i, i*CHUNK,
                          min((i+1)*CHUNK, size)-1, size) for i in range(n)]
        for fu in as_completed(futs):
            i, v = fu.result()
            if v > 0:
                done += 1
                print("\r    %2d/%d 块  %.0f MB" % (done, n, done*CHUNK/1048576), end="", flush=True)
            else:
                bad.append(i)
    print()
    if bad:
        print("    失败块 %s" % bad[:5]); return False
    with zipfile.ZipFile(out) as z:
        ok = z.testzip() is None
    print("    %.1f MB | %.0fs | ZIP %s" % (size/1048576, time.time()-t0, "OK" if ok else "BAD"))
    return ok


import json, sys
only = sys.argv[1:] if len(sys.argv) > 1 else None
results = []
for pkg, ver in TARGETS:
    if only and not any(o.lower() in pkg.lower() for o in only):
        continue
    print("\n=== %s %s ===" % (pkg, ver))
    try:
        name, url, size = resolve(pkg, ver)
        print("  %s (%.1f MB)" % (name[:60], size/1048576))
        results.append((pkg, download_one(name, url, size)))
    except Exception as e:
        print("  失败: %s" % e)
        results.append((pkg, False))

print("\n" + "=" * 60)
for pkg, ok in results:
    print("  %-30s %s" % (pkg, "OK" if ok else "失败"))
print("\n全部成功" if all(x[1] for x in results) else "有失败包，需重跑本脚本")
