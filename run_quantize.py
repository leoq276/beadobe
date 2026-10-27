"""
run_quantize.py
入口脚本：对 outputs/cat_seedream.png 做 MARD 色板量化 + 网格化，
输出拼豆图纸（完整版 + 网格预览版 + JSON 描述 + 统计）。

支持命令行参数:
  --grid N         同时设置 grid_w = grid_h = N（默认 52）
  --grid-w N --grid-h M   分别指定宽高
  --analysis PATH  LLM 视觉理解结果 JSON（默认 outputs/cat_analysis.json）
  --no-detail      关闭细节优先模式（用最朴素投票）
  --keep-watermark 不影响 (SeedDream 阶段控制)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.palette_loader import load_palette  # noqa: E402
from core.quantize import quantize  # noqa: E402
from core.bead_chart import save_bead_chart, render_grid_only  # noqa: E402


# 关键特征默认 bbox 尺寸（归一化坐标，宽×高）
# 越小越精准，越大越宽松
FEATURE_DEFAULT_SIZE = {
    "eye":    (0.075, 0.045),   # 眼睛横长
    "ear":    (0.075, 0.085),   # 耳朵
    "mouth":  (0.060, 0.022),   # 嘴
    "nose":   (0.030, 0.025),
    "brow":   (0.060, 0.020),
    "whisker":(0.080, 0.015),
    "lid":    (0.060, 0.020),
    "head":   (0.55,  0.50),
    "face":   (0.50,  0.45),
    "cheek":  (0.18,  0.20),
    "muzzle": (0.16,  0.10),
    "chin":   (0.20,  0.10),
    "body":   (0.70,  0.60),
    "chest":  (0.40,  0.35),
    "paw":    (0.12,  0.10),
    "tail":   (0.25,  0.20),
}


def _infer_size(name: str) -> tuple[float, float]:
    n = name.lower()
    for kw, size in FEATURE_DEFAULT_SIZE.items():
        if kw in n:
            return size
    return (0.05, 0.04)  # fallback


def _find_dark_palette_codes(palette, l_threshold: float = 30.0) -> set:
    """找出 MARD 色板中 L (Lab 亮度) < 阈值 的所有色号"""
    return {c.code for c in palette if c.lab[0] < l_threshold}


def _build_key_features(analysis_path: Path) -> dict:
    """
    从 visual_understand 的 JSON 输出构造 {name: [x1,y1,x2,y2]} 字典。
    - 若 features[i].bbox 存在：直接用
    - 否则用 position + 默认尺寸生成
    """
    if not analysis_path.exists():
        print(f"  [WARN] LLM 分析文件不存在: {analysis_path}，关闭关键区域保护")
        return {}
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    out: dict = {}
    for f in data.get("key_features", []) or []:
        name = f.get("name") or f.get("label_cn") or ""
        if not name:
            continue
        # 优先用 bbox
        if f.get("bbox") and len(f["bbox"]) == 4:
            out[name] = list(f["bbox"])
            continue
        # 否则用 position + 默认 size
        pos = f.get("position")
        if not pos or len(pos) != 2:
            continue
        cx, cy = pos
        sw, sh = _infer_size(name)
        out[name] = [max(0, cx - sw / 2), max(0, cy - sh / 2),
                     min(1, cx + sw / 2), min(1, cy + sh / 2)]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="MARD 色板量化 + 拼豆图纸生成")
    parser.add_argument("--grid", type=int, default=52,
                        help="目标网格边长（grid_w = grid_h），默认 52")
    parser.add_argument("--grid-w", type=int, default=None, help="目标网格宽")
    parser.add_argument("--grid-h", type=int, default=None, help="目标网格高")
    parser.add_argument("--analysis", type=str,
                        default=str(ROOT / "outputs" / "cat_analysis.json"),
                        help="LLM 视觉理解结果 JSON 路径")
    parser.add_argument("--no-detail", action="store_true",
                        help="关闭细节优先模式（最朴素投票）")
    parser.add_argument("--outline-preserve", action="store_true",
                        help="启用轮廓/深色强制保留（保留眼缝/嘴缝/外轮廓/瞳孔等细线条）")
    parser.add_argument("--no-outline-preserve", action="store_true",
                        help="显式关闭轮廓保护（与默认相同）")
    parser.add_argument("--min-island-size", type=int, default=2,
                        help="小色块清理阈值（默认 2，清理 < 2 像素的孤立小色块）")
    parser.add_argument("--no-protect-outline", action="store_true",
                        help="不保护深色色号不被清理")
    parser.add_argument("--cleanup-min-contrast", type=float, default=20.0,
                        help="小色块清理的色差阈值 (默认 20)。"
                             "小色块与外部主色 ΔE >= 阈值 → 保留 (视为关键特征)。"
                             "如孤立黑色像素(ΔE~80)保留,米色杂点(ΔE~10)清理。"
                             "设为 0 或负数关闭色差判定。")
    parser.add_argument("--line-detect", action="store_true",
                        help="启用纹理线条检测 (独立于深浅, 深线浅线都识别)")
    parser.add_argument("--line-method", type=str, default="canny",
                        choices=["local", "multiscale", "canny", "frangi", "hybrid"],
                        help="线条检测方法 (默认 canny): "
                             "canny=#1 OpenCV Canny(标准滞后跟踪, 需 opencv, 推荐) / "
                             "hybrid=Canny ∪ Local(救胡须细线, 需 opencv) / "
                             "multiscale=#2多尺度Sobel+NMS(无opencv依赖, 备选) / "
                             "frangi=#4 Hessian/Frangi vesselness(显式线状度, 慢, 不适合黑线) / "
                             "local=#5局部对比度(最快, 老方案)")
    parser.add_argument("--line-threshold", type=float, default=88.0,
                        help="线条检测的分位数阈值 (默认 88, 越高越严格, 仅 local 模式)")
    parser.add_argument("--line-min-pct", type=float, default=0.05,
                        help="块内线条像素占比阈值 (默认 5%%)")
    parser.add_argument("--min-line-count", type=int, default=1,
                        help="细线兜底判据: 块内线条像素最少个数 (默认 1)")
    parser.add_argument("--min-line-contrast", type=float, default=30.0,
                        help="细线兜底判据的 ΔE 阈值 (默认 30)。"
                             "块内至少 --min-line-count 个线条像素, "
                             "且最暗线条 vs 块主体 ΔE > 此值 → 强制按线条处理。"
                             "用于救单根胡须/细线。设为 0 关闭。")
    parser.add_argument("--line-aspect-min", type=float, default=2.5,
                        help="hybrid 模式形状过滤: 连通域长宽比下限 (默认 2.5)。"
                             "长宽比 >= 此值的连通域保留 (认为是线条), "
                             "块状/斑驳的连通域 (如脸颊阴影) 被剔除。")
    parser.add_argument("--line-min-region-size", type=int, default=4,
                        help="hybrid 模式形状过滤: 连通域最小像素数 (默认 4)。"
                             "小于此值的连通域视为噪点剔除。")
    parser.add_argument("--outline-min-pct", type=float, default=0.015,
                        help="轮廓保护中深色像素的最小占比 (默认 0.015 = 1.5%%)")
    parser.add_argument("--outline-min-edge", type=float, default=0.04,
                        help="轮廓保护中边缘像素的最小占比 (默认 0.04 = 4%%)")
    parser.add_argument("--outline-center-l", type=float, default=85.0,
                        help="块中心像素的 L 阈值 (默认 85，< 该值就视为含描边)")
    parser.add_argument("--outline-darkest-l", type=float, default=None,
                        help="[已弃用] 请用 --outline-center-l")
    parser.add_argument("--supersample", type=int, default=1,
                        help="超采样倍率 (默认 1=不超采样，推荐 4)。"
                             "原理: 先量化到 N×倍率 高分辨率,再下采样回 N。"
                             "能完美保留 1 像素宽的细线。")
    parser.add_argument("--ss-min-dark", type=int, default=2,
                        help="超采样下采样的'深色子格数量阈值' (默认 2)。"
                             "1=纯 OR (最糊), 2=折中 (推荐), 3+=保守 (细线可能丢)")
    parser.add_argument("--ss-strategy", type=str, default="vote",
                        choices=["darkest", "vote", "line_aware"],
                        help="超采样下采样策略 (默认 vote, 干净但细线可能丢): "
                             "vote=频次最高(干净, 适合基础/GUI 默认) / "
                             "line_aware=线条保护+投票 (4x4子格里 ≥ ss_line_min_pixels 个"
                             "line_mask 像素 → 整块用线条色; 否则 vote。需 --line-detect) / "
                             "darkest=最暗优先(保留描边, 可能糊)")
    parser.add_argument("--ss-line-min-pixels", type=int, default=1,
                        help="line_aware 策略: 4x4 子格里至少几个 line_mask 像素才触发"
                             "线条覆盖 (默认 1)")
    parser.add_argument("--image", type=str, default=None,
                        help="输入卡通图路径（默认 outputs/cat_seedream.png）")
    parser.add_argument("--no-key-features", action="store_true",
                        help="不使用关键特征 bbox（关闭区域分级）")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="输出文件后缀（默认空 = 覆盖）")
    args = parser.parse_args()

    grid_w = args.grid_w or args.grid
    grid_h = args.grid_h or args.grid
    suffix = args.out_suffix

    image_path = Path(args.image) if args.image else (ROOT / "outputs" / "cat_seedream.png")
    if not image_path.exists():
        print(f"[ERROR] 找不到输入图片: {image_path}")
        print("  请先跑 run_seedream.py 生成卡通图")
        return 1

    print(f"[INFO] 输入:   {image_path.name}")
    print(f"[INFO] 网格:   {grid_w} x {grid_h}")

    print("[RUN] 加载 MARD 色板 ...")
    palette = load_palette()
    print(f"  色板色数: {len(palette)}")

    # 关键特征
    key_features: dict = {}
    if not args.no_key_features and not args.no_detail:
        kf = _build_key_features(Path(args.analysis))
        key_features = kf
        print(f"  关键特征: {len(key_features)} 个 (区域分级已启用)")
        for n, b in list(kf.items())[:6]:
            print(f"    {n:<14} bbox=[{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}]")
        if len(kf) > 6:
            print(f"    ... 还有 {len(kf) - 6} 个")
    elif args.no_detail:
        print("  [WARN] --no-detail 已开启，将用最朴素投票")

    # 轮廓保护开关
    outline_on = args.outline_preserve and not args.no_outline_preserve
    if args.supersample > 1:
        if args.ss_strategy == "vote":
            print(f"  超采样: {args.supersample}x + 投票下采样 (不会糊,可能丢细线)")
        elif args.ss_strategy == "line_aware":
            print(f"  超采样: {args.supersample}x + 线条保护下采样 "
                  f"(≥{args.ss_line_min_pixels}个line_mask像素/子区域 → 整块用线条色, 否则vote)")
        else:
            print(f"  超采样: {args.supersample}x + 最暗下采样 (阈值 ss_min_dark={args.ss_min_dark})")
        if args.ss_strategy == "line_aware" and not args.line_detect:
            print(f"  [WARN] --ss-strategy=line_aware 需要 --line-detect=True, 将自动降级到 vote")
    if outline_on:
        # 兼容旧的 --outline-darkest-l 参数
        center_l = args.outline_darkest_l if args.outline_darkest_l is not None else args.outline_center_l
        print(f"  轮廓保护: ON (中心像素+占比+边缘，center_l<{center_l}, min_pct={args.outline_min_pct}, min_edge={args.outline_min_edge})")
    if args.line_detect:
        if args.line_method == "multiscale":
            print(f"  纹理线条: ON (策略#2多尺度Sobel+NMS, min_pct={args.line_min_pct}, "
                  f"细线兜底≥{args.min_line_count}px ΔE≥{args.min_line_contrast})")
        elif args.line_method == "canny":
            print(f"  纹理线条: ON (策略#1 OpenCV Canny, 自适应median阈值, min_pct={args.line_min_pct}, "
                  f"细线兜底≥{args.min_line_count}px ΔE≥{args.min_line_contrast})")
        elif args.line_method == "hybrid":
            print(f"  纹理线条: ON (Hybrid Canny∪Local(形状过滤), min_pct={args.line_min_pct}, "
                  f"细线兜底≥{args.min_line_count}px ΔE≥{args.min_line_contrast}, "
                  f"长宽比≥{args.line_aspect_min} 像素数≥{args.line_min_region_size})")
        elif args.line_method == "frangi":
            print(f"  纹理线条: ON (策略#4 Hessian/Frangi vesselness, min_pct={args.line_min_pct}, "
                  f"细线兜底≥{args.min_line_count}px ΔE≥{args.min_line_contrast})")
        else:
            print(f"  纹理线条: ON (策略#5局部对比度, threshold={args.line_threshold}, min_pct={args.line_min_pct}, "
                  f"细线兜底≥{args.min_line_count}px ΔE≥{args.min_line_contrast})")

    # 受保护的色号（小色块清理时跳过）
    protected_codes: set = set()
    if outline_on and not args.no_protect_outline:
        # MARD 中所有深色（L < 30）的色号
        protected_codes = _find_dark_palette_codes(palette, l_threshold=30.0)
        print(f"  保护色号: {sorted(protected_codes)} (避免被清理)")

    print(f"  清理阈值: min_island_size={args.min_island_size}")
    if args.cleanup_min_contrast > 0:
        print(f"  色差保护: ΔE >= {args.cleanup_min_contrast} 的小色块保留 (保护描边)")
    if outline_on:
        center_l = args.outline_darkest_l if args.outline_darkest_l is not None else args.outline_center_l
        print(f"  轮廓阈值: center_l<{center_l}, min_pct={args.outline_min_pct}, min_edge={args.outline_min_edge}")

    print("\n[RUN] 色板量化 ...")
    result = quantize(
        image_path,
        grid_w=grid_w,
        grid_h=grid_h,
        palette=palette,
        min_island_size=args.min_island_size,
        cleanup=True,
        detail_mode=not args.no_detail,
        min_sub_pct=0.06,
        min_contrast=18.0,
        key_features=key_features if key_features else None,
        outline_preserve=outline_on,
        protected_codes=protected_codes if protected_codes else None,
        outline_min_pct=args.outline_min_pct,
        outline_min_edge_pct=args.outline_min_edge,
        outline_center_l=(args.outline_darkest_l if args.outline_darkest_l is not None
                          else args.outline_center_l),
        supersample=args.supersample,
        ss_min_dark_count=args.ss_min_dark,
        ss_strategy=args.ss_strategy,
        ss_line_min_pixels=args.ss_line_min_pixels,
        cleanup_min_contrast=(args.cleanup_min_contrast if args.cleanup_min_contrast > 0 else None),
        line_detect=args.line_detect,
        line_method=args.line_method,
        line_threshold_percentile=args.line_threshold,
        line_min_pct=args.line_min_pct,
        min_line_count=args.min_line_count,
        min_line_contrast=(args.min_line_contrast if args.min_line_contrast > 0 else 0.0),
        line_aspect_min=args.line_aspect_min,
        line_min_region_size=args.line_min_region_size,
    )
    print(f"  用色数: {result.color_count}")
    print(f"  总格数: {result.total_cells}")
    print(f"  清理小色块: {result.removed_islands} 格")

    # 打印 Top 12
    print("\n色号用量 TOP 12:")
    for code, cnt in sorted(result.stats.items(), key=lambda x: -x[1])[:12]:
        c = palette.by_code(code)
        hex_ = c.hex if c else "?"
        print(f"  {code:<6} {cnt:>5} 格  hex={hex_}")

    # ---------- 保存产物 ----------
    name = f"cat_bead_chart_{grid_w}x{grid_h}{suffix}.png"
    out_chart = OUTPUTS_DIR / name
    cell_px = max(8, min(24, 1800 // max(grid_w, grid_h)))  # 自适应格子大小
    save_bead_chart(
        result, out_chart,
        cell_px=cell_px,
        label_every=max(5, grid_w // 12),
        title=f"拼豆图纸 - 猫咪（不耐烦）· {grid_w}×{grid_h}",
        palette_name="MARD",
    )
    print(f"\n[OK] 完整图纸:  {out_chart}")

    name = f"cat_bead_grid_only_{grid_w}x{grid_h}{suffix}.png"
    out_grid = OUTPUTS_DIR / name
    grid_img = render_grid_only(result, cell_px=cell_px, with_grid=True)
    grid_img.save(out_grid)
    print(f"[OK] 网格预览:  {out_grid}")

    # JSON: 网格化色号矩阵
    name = f"cat_grid_{grid_w}x{grid_h}{suffix}.json"
    out_grid_json = OUTPUTS_DIR / name
    grid_json = {
        "grid_w": result.grid_w,
        "grid_h": result.grid_h,
        "palette": "MARD",
        "detail_mode": not args.no_detail,
        "key_features_used": list(key_features.keys()) if key_features else [],
        "cells": [
            [cell.code for cell in row]
            for row in result.grid
        ],
    }
    out_grid_json.write_text(
        json.dumps(grid_json, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"[OK] 网格 JSON:  {out_grid_json}")

    # JSON: 完整统计
    name = f"cat_bead_stats_{grid_w}x{grid_h}{suffix}.json"
    out_stats = OUTPUTS_DIR / name
    out_stats.write_text(
        json.dumps({
            "grid_w": result.grid_w,
            "grid_h": result.grid_h,
            "total_cells": result.total_cells,
            "color_count": result.color_count,
            "removed_islands": result.removed_islands,
            "stats": result.stats,
            "top12": sorted(
                [{"code": k, "count": v,
                  "hex": (palette.by_code(k).hex if palette.by_code(k) else "?"),
                  "name": (palette.by_code(k).name if palette.by_code(k) else "?")}
                 for k, v in result.stats.items()],
                key=lambda x: -x["count"],
            )[:12],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 统计 JSON:  {out_stats}")

    return 0


OUTPUTS_DIR = ROOT / "outputs"


if __name__ == "__main__":
    sys.exit(main())
