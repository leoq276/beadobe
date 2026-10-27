"""
run_svg.py
入口脚本：对 samples/cat_photo.jpg 调用多模态 LLM，
让 LLM 直接设计一份"为拼豆网格量身定做"的几何描述，
再渲染为 SVG + PNG 预览。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_loader import load_config, get_image_config  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from core.svg_design import design_svg_file, SVGDesign  # noqa: E402
from core.svg_renderer import write_svg, save_preview  # noqa: E402


SAMPLE_IMAGE = ROOT / "samples" / "cat_photo.jpg"
OUTPUTS_DIR = ROOT / "outputs"


def main() -> int:
    if not SAMPLE_IMAGE.exists():
        print(f"[ERROR] 找不到示例图片: {SAMPLE_IMAGE}")
        return 1

    cfg = load_config()
    img_cfg = get_image_config(cfg)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    grid_w, grid_h = 58, 58

    print(f"[INFO] 输入图片: {SAMPLE_IMAGE}")
    print(f"[INFO] 目标网格: {grid_w} x {grid_h}")
    print(f"[INFO] 图片预处理: max_side={img_cfg['max_side']}")

    cli = LLMClient()
    print(f"[INFO] LLM model: {cli.model}")

    print("\n[RUN] 调用 LLM 设计拼豆几何描述 ...")
    design = design_svg_file(
        str(SAMPLE_IMAGE),
        preserve_features="不耐烦的神情（半眯眼、耳朵后压、嘴部紧抿、目光冷淡）",
        style_hint="日系平涂/扁平卡通，色块干净，线条简洁",
        grid_w=grid_w,
        grid_h=grid_h,
        max_side=img_cfg["max_side"],
        client=cli,
    )

    # 摘要
    print("\n" + "=" * 60)
    print("【几何设计结果摘要】")
    print("=" * 60)
    print(f"画布: {design.canvas.get('grid_w')}x{design.canvas.get('grid_h')}")
    print(f"调色板: {len(design.palette)} 色")
    for p in design.palette:
        print(f"  - {p.get('role'):<14} {p.get('hex'):<8} {p.get('note', '')}")
    print(f"形状数: {len(design.shapes)}")
    # 按 z 分组展示
    shapes_sorted = sorted(design.shapes, key=lambda s: int(s.get("z", 0)))
    print("形状清单 (按 z 排序):")
    for s in shapes_sorted:
        bbox = s.get("bbox") or s.get("points")
        print(f"  z={s.get('z'):>2}  {s.get('type'):<7}  "
              f"{s.get('id'):<14}  fill={s.get('fill_role'):<14}  geom={bbox}")

    if design.design_notes:
        print("\n【设计备注】")
        for n in design.design_notes:
            print(f"  - {n}")

    # 保存 JSON
    out_json = OUTPUTS_DIR / "cat_svg_design.json"
    out_json.write_text(
        json.dumps(design.raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] 设计 JSON: {out_json}")

    # 保存 SVG
    out_svg = OUTPUTS_DIR / "cat_cartoon.svg"
    write_svg(design, out_svg, px_per_cell=20, include_grid=True)
    print(f"[OK] SVG 文件:  {out_svg}")

    # 保存 PNG 预览
    out_png = OUTPUTS_DIR / "cat_cartoon_preview.png"
    save_preview(design, out_png, px_per_cell=16, include_grid=True)
    print(f"[OK] PNG 预览:  {out_png}")

    # 不带网格的纯净版（方便直接用）
    out_svg_clean = OUTPUTS_DIR / "cat_cartoon_clean.svg"
    write_svg(design, out_svg_clean, px_per_cell=20, include_grid=False)
    out_png_clean = OUTPUTS_DIR / "cat_cartoon_clean.png"
    save_preview(design, out_png_clean, px_per_cell=16, include_grid=False)
    print(f"[OK] 纯净 SVG:  {out_svg_clean}")
    print(f"[OK] 纯净 PNG:  {out_png_clean}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
