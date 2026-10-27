"""
run_understand.py
入口脚本：对 samples/cat_photo.jpg 调用多模态 LLM 做视觉理解，
输出结构化分析结果到 outputs/ 目录。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 允许从项目根目录直接 `python run_understand.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_loader import load_config, get_image_config  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from core.visual_understand import analyze_image_file, VisualAnalysis  # noqa: E402


SAMPLE_IMAGE = ROOT / "samples" / "cat_photo.jpg"
OUTPUTS_DIR = ROOT / "outputs"


def main() -> int:
    if not SAMPLE_IMAGE.exists():
        print(f"[ERROR] 找不到示例图片: {SAMPLE_IMAGE}")
        return 1

    cfg = load_config()
    img_cfg = get_image_config(cfg)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 输入图片: {SAMPLE_IMAGE}")
    print(f"[INFO] 图片预处理: max_side={img_cfg['max_side']}, "
          f"format={img_cfg['format']}, quality={img_cfg['quality']}")

    # 复用同一个 client（避免重复建立连接）
    cli = LLMClient()
    print(f"[INFO] LLM model: {cli.model}")
    print(f"[INFO] LLM base_url: {cli.client.base_url}")

    print("\n[RUN] 调用多模态 LLM 做视觉理解 ...")
    result = analyze_image_file(
        str(SAMPLE_IMAGE),
        preserve_features="不耐烦的神情（半眯眼、耳朵后压、嘴部紧抿）",
        style_hint="日系平涂/扁平卡通，色块干净，线条简洁，便于下采样到拼豆网格",
        max_side=img_cfg["max_side"],
        client=cli,
    )

    # 打印摘要
    print("\n" + "=" * 60)
    print("【视觉理解结果摘要】")
    print("=" * 60)
    print(result.short_summary())

    print("\n【中文 Cartoon Prompt（可直接喂给 SeedDream）】")
    print(result.cartoon_prompt_cn)

    print("\n【English Cartoon Prompt】")
    print(result.cartoon_prompt_en)

    if result.simplify_suggestions:
        print("\n【简化建议】")
        for s in result.simplify_suggestions:
            print(f"  - {s}")

    if result.risk_warnings:
        print("\n【风险提示】")
        for s in result.risk_warnings:
            print(f"  - {s}")

    # 保存完整 JSON
    out_json = OUTPUTS_DIR / "cat_analysis.json"
    out_json.write_text(
        json.dumps(result.raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] 完整分析已保存到: {out_json}")

    # 保存 prompt 到独立文件，方便后续 pipeline 直接读取
    out_prompt_cn = OUTPUTS_DIR / "cat_seedream_prompt_cn.txt"
    out_prompt_cn.write_text(result.cartoon_prompt_cn, encoding="utf-8")
    out_prompt_en = OUTPUTS_DIR / "cat_seedream_prompt_en.txt"
    out_prompt_en.write_text(result.cartoon_prompt_en, encoding="utf-8")
    print(f"[OK] 中文 prompt: {out_prompt_cn}")
    print(f"[OK] 英文 prompt: {out_prompt_en}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
