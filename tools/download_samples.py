"""下载拼豆算法测试样例图片到 samples/。

图源：
- picsum.photos/id/N  可复现的真实照片（风景/动物/人像/物体等）
- images.unsplash.com 特定 photo id（卡通/插画/静物等补充类别）

覆盖类别：动物、人像、风景、食物、物体、建筑、卡通/插画、几何图形，
以及不同复杂度（纯色块 vs 细节丰富），用于打磨像素化/量化算法。
"""
from __future__ import annotations

import ssl
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)

# 20 张图：文件名 -> (url, 类别)
IMAGES: list[tuple[str, str, str]] = [
    # 动物
    ("animal_dog_black", "https://picsum.photos/id/237/640/640", "动物-黑狗"),
    ("animal_deer", "https://picsum.photos/id/1003/640/640", "动物-鹿"),
    ("animal_corgi", "https://picsum.photos/id/1025/640/640", "动物-柯基"),
    ("animal_walrus", "https://picsum.photos/id/1084/640/640", "动物-海象"),
    ("animal_bird", "https://picsum.photos/id/1024/640/640", "动物-鸟"),
    ("animal_bull", "https://picsum.photos/id/200/640/640", "动物-牛"),
    # 人像
    ("portrait_01", "https://picsum.photos/id/1005/640/640", "人像-1"),
    ("portrait_02", "https://picsum.photos/id/1012/640/640", "人像-2"),
    ("portrait_03", "https://picsum.photos/id/1062/640/640", "人像-3"),
    ("portrait_04", "https://picsum.photos/id/1011/640/640", "人像-划船"),
    # 风景
    ("landscape_mountain", "https://picsum.photos/id/1018/640/640", "风景-山"),
    ("landscape_river", "https://picsum.photos/id/1015/640/640", "风景-河流"),
    ("landscape_canyon", "https://picsum.photos/id/1016/640/640", "风景-峡谷"),
    ("landscape_coast", "https://picsum.photos/id/1043/640/640", "风景-海岸"),
    ("landscape_forest", "https://picsum.photos/id/10/640/640", "风景-森林"),
    # 食物 / 物体
    ("food_strawberry", "https://picsum.photos/id/1080/640/640", "食物-草莓"),
    ("food_veg", "https://picsum.photos/id/292/640/640", "食物-蔬菜"),
    ("object_watch", "https://picsum.photos/id/30/640/640", "物体-手表"),
    # 建筑
    ("building_city", "https://picsum.photos/id/429/640/640", "建筑-城市"),
    ("building_arch", "https://picsum.photos/id/1039/640/640", "建筑-拱门"),
]

# 图源备用：若 picsum 某 id 失效，用 Unsplash 补位
FALLBACK = {
    "portrait_01": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=640",
    "food_strawberry": "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?w=640",
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=ctx)
)
opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]


def download(url: str, dest: Path, tries: int = 2) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url)
            with opener.open(req, timeout=30) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:  # noqa
            if i == tries - 1:
                print(f"  [失败] {dest.name}: {e}")
                return False
            time.sleep(2)
    return False


def main():
    ok = 0
    manifest = []
    for name, url, cat in IMAGES:
        dest = SAMPLES / f"{name}.jpg"
        if dest.exists():
            print(f"[跳过] {dest.name} 已存在")
            ok += 1
            manifest.append((name, cat, "已存在"))
            continue
        print(f"[下载] {cat}: {name} ...")
        if download(url, dest):
            ok += 1
            manifest.append((name, cat, "ok"))
        elif name in FALLBACK:
            print(f"  [回退] 使用 Unsplash 备用源 {name}")
            if download(FALLBACK[name], dest):
                ok += 1
                manifest.append((name, cat, "ok(fallback)"))
            else:
                manifest.append((name, cat, "FAILED"))
        else:
            manifest.append((name, cat, "FAILED"))

    print(f"\n=== 完成：{ok}/{len(IMAGES)} 张 ===\n")
    for name, cat, st in manifest:
        print(f"  {st:12s} {cat:14s} {name}")

    # 写 manifest
    with open(SAMPLES / "manifest.txt", "w", encoding="utf-8") as f:
        for name, cat, st in manifest:
            f.write(f"{st}\t{cat}\t{name}\n")
    print("\n已写 samples/manifest.txt")


if __name__ == "__main__":
    main()
