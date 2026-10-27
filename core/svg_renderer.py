"""
svg_renderer.py
把 SVGDesign 渲染为：
  1) SVG 字符串 / 文件
  2) PNG 位图预览（用 Pillow 自绘，不依赖 cairosvg）

坐标映射：归一化 (0~1) → 像素坐标（基于每格的 px_per_cell）
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from .svg_design import SVGDesign


SVG_NS = "http://www.w3.org/2000/svg"


# =============================================================================
# 工具：颜色查找
# =============================================================================
def _build_palette_lookup(palette: List[Dict[str, Any]]) -> Dict[str, str]:
    """role -> hex（统一成 #RRGGBB 大写）"""
    out: Dict[str, str] = {}
    for p in palette:
        role = p.get("role")
        hex_ = p.get("hex")
        if not role or not hex_:
            continue
        h = hex_.lstrip("#").upper()
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            out[role] = "#" + h
    return out


def _color_with_fallback(
    role: Optional[str],
    palette_lut: Dict[str, str],
    default: str = "#FFFFFF",
) -> str:
    if not role:
        return default
    return palette_lut.get(role, default)


# =============================================================================
# SVG 字符串生成
# =============================================================================
def _svg_attr(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def render_svg_string(
    design: SVGDesign,
    *,
    px_per_cell: int = 20,
    include_grid: bool = True,
) -> Tuple[str, int, int]:
    """
    输出 SVG 字符串。

    :param px_per_cell: 每个拼豆格在 SVG 里占多少像素（影响清晰度，不影响几何）
    :param include_grid: 是否画网格线（调试用）
    :return: (svg_text, canvas_w_px, canvas_h_px)
    """
    grid_w, grid_h = design.grid_size()
    canvas_w = grid_w * px_per_cell
    canvas_h = grid_h * px_per_cell

    palette_lut = _build_palette_lookup(design.palette)
    bg_hex = _color_with_fallback(
        design.canvas.get("bg_color_role"),
        palette_lut,
        design.canvas.get("bg_color_hex", "#FFFFFF"),
    )

    # root <svg>
    svg = ET.Element(
        "svg",
        {
            "xmlns": SVG_NS,
            "viewBox": f"0 0 {canvas_w} {canvas_h}",
            "width": str(canvas_w),
            "height": str(canvas_h),
            "shape-rendering": "crispEdges",
        },
    )

    # 背景
    ET.SubElement(
        svg,
        "rect",
        {
            "x": "0", "y": "0",
            "width": str(canvas_w), "height": str(canvas_h),
            "fill": bg_hex,
        },
    )

    # 按 z 排序
    shapes_sorted = sorted(design.shapes, key=lambda s: int(s.get("z", 0)))

    for shape in shapes_sorted:
        stype = shape.get("type")
        fill = _color_with_fallback(shape.get("fill_role"), palette_lut, "none")
        stroke = _color_with_fallback(shape.get("stroke_role"), palette_lut, "none")
        try:
            sw_norm = float(shape.get("stroke_width", 0) or 0)
        except (TypeError, ValueError):
            sw_norm = 0.0
        sw_px = max(0.0, sw_norm * canvas_w)

        attrs: Dict[str, str] = {
            "fill": fill if fill != "none" else "none",
            "stroke": stroke if stroke != "none" else "none",
        }
        if sw_px > 0 and stroke != "none":
            attrs["stroke-width"] = f"{sw_px:.2f}"
        # 闭路径描边默认无填充时给个透明填充避免某些渲染器跳过
        if attrs["fill"] == "none" and attrs["stroke"] == "none":
            continue

        if stype == "rect":
            bbox = shape.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            attrs["x"] = f"{x1 * canvas_w:.2f}"
            attrs["y"] = f"{y1 * canvas_h:.2f}"
            attrs["width"] = f"{(x2 - x1) * canvas_w:.2f}"
            attrs["height"] = f"{(y2 - y1) * canvas_h:.2f}"
            ET.SubElement(svg, "rect", attrs)

        elif stype == "ellipse":
            bbox = shape.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2 * canvas_w
            cy = (y1 + y2) / 2 * canvas_h
            rx = (x2 - x1) / 2 * canvas_w
            ry = (y2 - y1) / 2 * canvas_h
            attrs["cx"] = f"{cx:.2f}"
            attrs["cy"] = f"{cy:.2f}"
            attrs["rx"] = f"{rx:.2f}"
            attrs["ry"] = f"{ry:.2f}"
            ET.SubElement(svg, "ellipse", attrs)

        elif stype == "polygon":
            points = shape.get("points")
            if not points:
                continue
            pts_str = " ".join(
                f"{float(p[0]) * canvas_w:.2f},{float(p[1]) * canvas_h:.2f}"
                for p in points
            )
            attrs["points"] = pts_str
            ET.SubElement(svg, "polygon", attrs)

        else:
            # 未知类型跳过
            continue

    # 可选：网格线
    if include_grid:
        grid_attrs = {
            "stroke": "#00000022",
            "stroke-width": "1",
            "fill": "none",
        }
        # 竖线
        for i in range(grid_w + 1):
            x = i * px_per_cell
            ET.SubElement(
                svg, "line",
                {**grid_attrs, "x1": str(x), "y1": "0",
                 "x2": str(x), "y2": str(canvas_h)},
            )
        # 横线
        for j in range(grid_h + 1):
            y = j * px_per_cell
            ET.SubElement(
                svg, "line",
                {**grid_attrs, "x1": "0", "y1": str(y),
                 "x2": str(canvas_w), "y2": str(y)},
            )

    # 序列化
    ET.indent(svg, space="  ")
    body = ET.tostring(svg, encoding="unicode")
    # 补 xmlns（ET 不会自动加默认命名空间）
    if 'xmlns="' not in body.split(">", 1)[0]:
        body = body.replace("<svg ", f'<svg xmlns="{SVG_NS}" ', 1)
    return body, canvas_w, canvas_h


def write_svg(design: SVGDesign, path: str | Path, **kwargs) -> Path:
    body, w, h = render_svg_string(design, **kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# =============================================================================
# Pillow 光栅化（用于 PNG 预览，不依赖 cairosvg）
# =============================================================================
def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rasterize(
    design: SVGDesign,
    *,
    px_per_cell: int = 16,
    include_grid: bool = True,
) -> Image.Image:
    """
    用 Pillow 自绘 PNG 预览。
    """
    grid_w, grid_h = design.grid_size()
    W = grid_w * px_per_cell
    H = grid_h * px_per_cell
    palette_lut = _build_palette_lookup(design.palette)
    bg_hex = _color_with_fallback(
        design.canvas.get("bg_color_role"),
        palette_lut,
        design.canvas.get("bg_color_hex", "#FFFFFF"),
    )

    img = Image.new("RGB", (W, H), _hex_to_rgb(bg_hex))
    draw = ImageDraw.Draw(img, "RGBA")

    shapes_sorted = sorted(design.shapes, key=lambda s: int(s.get("z", 0)))
    for shape in shapes_sorted:
        stype = shape.get("type")
        fill_hex = _color_with_fallback(shape.get("fill_role"), palette_lut, "none")
        stroke_hex = _color_with_fallback(shape.get("stroke_role"), palette_lut, "none")
        try:
            sw_norm = float(shape.get("stroke_width", 0) or 0)
        except (TypeError, ValueError):
            sw_norm = 0.0
        sw_px = max(1, round(sw_norm * W))

        fill_rgb = _hex_to_rgb(fill_hex) if fill_hex != "none" else None
        stroke_rgb = _hex_to_rgb(stroke_hex) if stroke_hex != "none" else None
        if fill_rgb is None and stroke_rgb is None:
            continue

        if stype == "rect":
            bbox = shape.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            xa = round(x1 * W); ya = round(y1 * H)
            xb = round(x2 * W); yb = round(y2 * H)
            if fill_rgb is not None:
                draw.rectangle([xa, ya, xb, yb], fill=fill_rgb)
            if stroke_rgb is not None and sw_px > 0:
                draw.rectangle([xa, ya, xb, yb], outline=stroke_rgb, width=sw_px)

        elif stype == "ellipse":
            bbox = shape.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            xa = round(x1 * W); ya = round(y1 * H)
            xb = round(x2 * W); yb = round(y2 * H)
            if fill_rgb is not None:
                draw.ellipse([xa, ya, xb, yb], fill=fill_rgb)
            if stroke_rgb is not None and sw_px > 0:
                draw.ellipse([xa, ya, xb, yb], outline=stroke_rgb, width=sw_px)

        elif stype == "polygon":
            points = shape.get("points")
            if not points:
                continue
            pts = [(round(p[0] * W), round(p[1] * H)) for p in points]
            if fill_rgb is not None:
                draw.polygon(pts, fill=fill_rgb)
            if stroke_rgb is not None and sw_px > 0:
                draw.polygon(pts, outline=stroke_rgb)

    if include_grid:
        grid_color = (0, 0, 0, 60)
        for i in range(grid_w + 1):
            x = i * px_per_cell
            draw.line([(x, 0), (x, H)], fill=grid_color, width=1)
        for j in range(grid_h + 1):
            y = j * px_per_cell
            draw.line([(0, y), (W, y)], fill=grid_color, width=1)

    return img


def save_preview(
    design: SVGDesign,
    path: str | Path,
    *,
    px_per_cell: int = 16,
    include_grid: bool = True,
) -> Path:
    img = rasterize(design, px_per_cell=px_per_cell, include_grid=include_grid)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return p


__all__ = [
    "render_svg_string",
    "write_svg",
    "rasterize",
    "save_preview",
]
