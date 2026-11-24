# -*- coding: utf-8 -*-
"""分块并发下载大 wheel（单连接被限速到 ~90KB/s，多连接可叠加）。"""
import io, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://mirrors.aliyun.com/pypi/packages/27/76/81de592072d6a41553b1523e15447f0ef94392e8f4cb98fda42909f24f9b/onnxruntime_gpu-1.22.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
OUT = r"D:\VC Code\YOLO_Road_Detection\ros2\_wheelhouse\onnxruntime_gpu-1.22.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
TOTAL = 283205327
CHUNK = 8 * 1024 * 1024
N = (TOTAL + CHUNK - 1) // CHUNK

os.makedirs(os.path.dirname(OUT), exist_ok=True)
# 预分配
with open(OUT, "wb") as f:
    f.truncate(TOTAL)

def get(i):
    s, e = i * CHUNK, min((i + 1) * CHUNK, TOTAL) - 1
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, headers={
                "Range": f"bytes={s}-{e}",
                "User-Agent": "Mozilla/5.0",
            })
            data = urllib.request.urlopen(req, timeout=90).read()
            if len(data) != e - s + 1:
                raise IOError("short %d/%d" % (len(data), e - s + 1))
            with open(OUT, "r+b") as f:
                f.seek(s); f.write(data)
            return i, len(data)
        except Exception as ex:
            if attempt == 2:
                return i, -str(ex)
            time.sleep(1)

t0 = time.time()
ok, bad = 0, []
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(get, i) for i in range(N)]
    for fu in as_completed(futs):
        i, v = fu.result()
        if isinstance(v, int) and v > 0:
            ok += 1
            print("\r  块 %2d/%d 完成（%.0f MB）" % (ok, N, ok*CHUNK/1048576), end="", flush=True)
        else:
            bad.append((i, v))

print()
print("耗时 %.1fs | 成功 %d/%d | 失败 %s" % (time.time()-t0, ok, N, bad[:3]))
size = os.path.getsize(OUT)
print("文件大小 %d（应为 %d）" % (size, TOTAL))

# 校验 zip 完整性
if not bad:
    import zipfile
    with zipfile.ZipFile(OUT) as z:
        print("ZIP 完整性:", "OK" if z.testzip() is None else "BAD")
        so = [n for n in z.namelist() if 'providers_cuda' in n or 'libonnxruntime.so' in n]
        for n in so[:8]:
            print("  ", n, "%.1f MB" % (z.getinfo(n).file_size/1048576))
