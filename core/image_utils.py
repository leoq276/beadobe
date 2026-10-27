"""
image_utils.py
图片预处理：等比缩放、base64 编码。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Tuple

from PIL import Image


def resize_keep_aspect(
    img: Image.Image,
    max_side: int,
) -> Image.Image:
    """等比缩放，长边不超过 max_side"""
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    # 卡通/线稿用 LANCZOS 更稳，照片 BICUBIC 也够
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_base64(
    img: Image.Image,
    fmt: str = "jpeg",
    quality: int = 85,
) -> Tuple[str, str]:
    """
    将 PIL Image 编码为 base64 字符串。

    :param img: PIL Image（建议已转 RGB）
    :param fmt: "jpeg" | "png" | "webp"
    :param quality: JPEG/WEBP 质量 1~100
    :return: (base64_str, mime_type)
    """
    fmt = fmt.lower()
    if fmt == "jpg":
        fmt = "jpeg"
    mime = f"image/{fmt}"
    buf = io.BytesIO()
    save_kwargs = {"format": fmt.upper()}
    if fmt in ("jpeg", "webp"):
        save_kwargs["quality"] = int(quality)
        if fmt == "jpeg":
            # 避免透明通道导致 JPEG 保存失败
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
    img.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii"), mime


def file_to_base64(
    path: str | Path,
    max_side: int = 1024,
    fmt: str = "jpeg",
    quality: int = 85,
) -> Tuple[str, str, Image.Image]:
    """
    一步完成：读文件 → 转 RGB → 缩放 → base64。

    :return: (base64_str, mime_type, 处理后的 PIL Image)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = resize_keep_aspect(img, max_side)
    b64, mime = image_to_base64(img, fmt=fmt, quality=quality)
    return b64, mime, img


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python image_utils.py <image>")
        sys.exit(1)
    b64, mime, img = file_to_base64(sys.argv[1], max_side=512)
    print(f"size={img.size} mime={mime} b64_len={len(b64)}")
