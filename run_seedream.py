"""
run_seedream.py
入口脚本：
  1. 读取上一阶段 LLM 生成的 prompt (cat_seedream_prompt_cn.txt)
  2. 用 SeedDream 5.0 Pro 对 cat_photo.jpg 做图生图
  3. 下载并保存卡通化结果
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_loader import load_config, get_image_config  # noqa: E402
from core.seedream import (  # noqa: E402
    SeedreamClient, SeedreamError, file_to_data_url, download_result_url,
)  # noqa: E402

SAMPLE_IMAGE = ROOT / "samples" / "cat_photo.jpg"
PROMPT_FILE = ROOT / "outputs" / "cat_seedream_prompt_cn.txt"
OUTPUTS_DIR = ROOT / "outputs"


def main() -> int:
    if not SAMPLE_IMAGE.exists():
        print(f"[ERROR] 找不到示例图片: {SAMPLE_IMAGE}")
        return 1
    if not PROMPT_FILE.exists():
        print(f"[ERROR] 找不到 prompt 文件: {PROMPT_FILE}")
        print("  请先跑 run_understand.py 生成 prompt")
        return 1

    cfg = load_config()
    img_cfg = get_image_config(cfg)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    print("=" * 60)
    print("【使用的 Prompt】")
    print("=" * 60)
    print(prompt)
    print()

    cli = SeedreamClient()
    cart = cfg["ark"]["cartoonize"]
    print(f"[INFO] 模型: {cart['model']}")
    print(f"[INFO] 相似度 (image_strength): {cart['image_strength']}")
    print(f"[INFO] 输出尺寸: {cart['size']}")
    print(f"[INFO] 端点: {cli.base_url}")
    print()

    print("[RUN] 编码原图为 data URL ...")
    data_url = file_to_data_url(
        SAMPLE_IMAGE,
        max_side=img_cfg["max_side"],
        quality=img_cfg["quality"],
    )
    print(f"  data URL 长度: {len(data_url)} 字符")

    print("\n[RUN] 调用 SeedDream 图生图 ...")
    try:
        resp = cli.image_to_image(
            prompt=prompt,
            image=data_url,
            image_strength=float(cart["image_strength"]),
            size=cart["size"],
            response_format="url",
            watermark=True,
        )
    except SeedreamError as e:
        print(f"[FAIL] {e}")
        return 1

    item = resp["data"][0]
    result_url = item.get("url")
    if not result_url:
        print(f"[FAIL] 响应里没有 url: {item}")
        return 1

    print(f"[OK] SeedDream 返回 URL: {result_url[:120]}...")

    out_png = OUTPUTS_DIR / "cat_seedream.png"
    download_result_url(result_url, out_png)
    print(f"[OK] 卡通化结果已保存: {out_png}")

    # 同时保存响应元数据（方便复现/调试）
    out_meta = OUTPUTS_DIR / "cat_seedream_response.json"
    import json
    out_meta.write_text(
        json.dumps({
            "model": resp.get("model", cart["model"]),
            "created": resp.get("created"),
            "usage": resp.get("usage"),
            "result_url": result_url,
            "prompt": prompt,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 响应元数据:     {out_meta}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
