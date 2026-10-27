"""
bead_chart.py
拼豆图纸渲染器。

输入: QuantizeResult
输出: 一张完整的拼豆图纸 PNG
  - 主体: 网格化色块 (grid_w x grid_h)
  - 网格线（淡色）
  - 色号标注（每隔 N 格标一次）
  - 右侧图例: 色块 + 色号 + 名称 + 用量 + 占比
  - 底部统计: 总格数、用色数、清理小色块数等
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .palette_loader import PaletteColor
from .quantize import QuantizeResult


# 中文字体路径（按需调整）
_CJK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]


def _load_cjk_font(size: int) -> ImageFont.FreeTypeFont:
    for p in _CJK_FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_color_for_bg(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """根据背景亮度自动选黑/白文字"""
    # 相对亮度 (Rec. 601)
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (0, 0, 0) if luma > 140 else (255, 255, 255)


def _highest_contrast_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """返回与给定颜色对比度最高的纯色（黑白或互补色极值）。

    在 RGB 立方体的 8 个顶点（黑/白/纯红绿蓝黄青品红）中选一个
    与给定颜色欧氏距离最大的，作为参考线/文字的高对比色。
    """
    candidates = [
        (255, 255, 255), (0, 0, 0),
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]
    best = None
    best_d = -1
    for c in candidates:
        d = (c[0] - rgb[0]) ** 2 + (c[1] - rgb[1]) ** 2 + (c[2] - rgb[2]) ** 2
        if d > best_d:
            best_d = d
            best = c
    return best


def _fit_font_for_text(
    text: str,
    cell_px: int,
    *,
    max_size: int = 9,
    min_size: int = 5,
    pad: int = 2,
) -> ImageFont.FreeTypeFont:
    """为给定文字在 cell_px 宽的格子内找到能完整显示的最大字号。

    从 max_size 向下递减，直到文字宽度 <= cell_px - pad。
    """
    size = max_size
    while size > min_size:
        f = _load_cjk_font(size)
        bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=f)
        w = bbox[2] - bbox[0]
        if w <= cell_px - pad:
            return f
        size -= 1
    return _load_cjk_font(min_size)


def _mix_color(color: Tuple[int, int, int], other: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """在 color 与 other 之间按 t(0~1) 线性插值。"""
    return (
        int(round(color[0] + (other[0] - color[0]) * t)),
        int(round(color[1] + (other[1] - color[1]) * t)),
        int(round(color[2] + (other[2] - color[2]) * t)),
    )


def _lighter(color: Tuple[int, int, int], t: float = 0.75) -> Tuple[int, int, int]:
    """向白色混合，得到浅色。"""
    return _mix_color(color, (255, 255, 255), t)


def _darker(color: Tuple[int, int, int], t: float = 0.5) -> Tuple[int, int, int]:
    """向黑色混合，得到深色。"""
    return _mix_color(color, (0, 0, 0), t)


# =============================================================================
# 渲染
# =============================================================================
def render_bead_chart(
    result: QuantizeResult,
    *,
    cell_px: int = 20,
    label_every: int = 5,
    title: str = "拼豆图纸",
    palette_name: str = "MARD",
    bg: Tuple[int, int, int] = (255, 255, 255),
    text: Tuple[int, int, int] = (40, 40, 40),
    label_all_cells: bool = False,
) -> Image.Image:
    """
    主入口：渲染拼豆图纸。

    :param cell_px: 每个格子的像素边长
    :param label_every: 每隔 N 格标注一次色号（1=全标，越大越稀疏）
    :param title: 顶部标题
    :param palette_name: 色板名（标题副标题用）
    :param label_all_cells: 强制每个格子都标注色号（拼装图纸需要）

    布局（从上到下）：标题 → 网格主体（含四边行列标注，浅蓝打底）→ 图例（色号+个数，
    换行，每行不超过图纸宽度）。
    """
    grid = result.grid
    gw, gh = result.grid_w, result.grid_h

    # ---------- 布局常量 ----------
    padding = 24
    title_h = 56
    # 行列标注：外扩一行/列，宽度与网格格子对齐（精确对应每个格子）
    axis_top = cell_px      # 顶部列号标注高度（= 一格高）
    axis_left = cell_px     # 左侧行号标注宽度（= 一格宽）
    axis_bottom = cell_px   # 底部列号标注高度
    axis_right = cell_px    # 右侧行号标注宽度
    border_w = 3            # 标注区外沿粗边框宽度

    grid_img_w = gw * cell_px
    grid_img_h = gh * cell_px

    # 图例参数
    legend_pad = 16
    legend_sw = 18          # 色块高度（色号写块内）
    legend_count_h = 18     # 下方个数文字高度
    legend_row_h = legend_sw + legend_count_h + 2  # 色块 + 个数 + 间距
    legend_gap_x = 18
    legend_gap_y = 10

    # ---------- 字体 ----------
    font_title = _load_cjk_font(22)
    font_sub = _load_cjk_font(14)
    font_legend_count = _load_cjk_font(12)  # 图例个数字体
    font_legend_code = _load_cjk_font(11)   # 图例色号字体（色块内）

    # ---------- 参考线颜色：基于图中用量最多色的对比色 ----------
    top_code = max(result.stats.items(), key=lambda kv: kv[1])[0] if result.stats else None
    top_rgb = None
    if top_code:
        _pc = _find_color_in_grid(grid, top_code)
        top_rgb = _pc.rgb if _pc else None
    # 网格线颜色：默认红；有最多色时取与之对比度最高的颜色
    grid_line_color = _highest_contrast_color(top_rgb) if top_rgb else (200, 40, 40)
    # 细线与粗线使用同色，粗线靠宽度区分
    thin_color = grid_line_color
    bold_color = grid_line_color
    # 标注区：与参考线相似但浅一点的打底色；外沿比参考线略深的粗边框
    axis_bg = _lighter(grid_line_color, 0.85)     # 浅色打底（偏白）
    axis_border = _darker(grid_line_color, 0.45)  # 略深粗边框
    axis_text = _darker(grid_line_color, 0.55)    # 深色标注文字

    # ---------- 主体网格起点 ----------
    grid_x0 = padding + axis_left
    grid_y0 = padding + title_h + axis_top

    # ---------- 计算图例尺寸（放在下方，宽度=网格宽度，可换行）----------
    legend_w = grid_img_w + axis_left + axis_right
    legend_items = sorted(
        result.stats.items(),
        key=lambda kv: -kv[1],
    )
    # 每个图例条目：长方形色块（色号写在内）+ 下方个数；色块宽度按色号文字自适应
    def _block_width(code):
        bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
            (0, 0), code, font=font_legend_code)
        w = bbox[2] - bbox[0]
        return max(w + 8, 24)  # 色块最小宽 24px，文字 + 内边距

    def _item_width(code):
        return _block_width(code) + legend_gap_x

    legend_lines: List[List[Tuple[str, int]]] = [[]]
    cur_w = 0
    for code, cnt in legend_items:
        w = _item_width(code)
        if cur_w + w > legend_w and legend_lines[-1]:
            legend_lines.append([])
            cur_w = 0
        legend_lines[-1].append((code, cnt))
        cur_w += w
    legend_h = len(legend_lines) * legend_row_h + (len(legend_lines) - 1) * legend_gap_y + legend_pad * 2

    # ---------- 画布 ----------
    canvas_w = padding * 2 + axis_left + grid_img_w + axis_right
    canvas_h = padding + title_h + axis_top + grid_img_h + axis_bottom + legend_h + padding

    img = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(img)

    # ---------- 标题 ----------
    draw.text((padding, padding), title, fill=text, font=font_title)
    sub = f"{palette_name} 色板 · {gw}×{gh} 网格 · {result.total_cells} 格 · 共用 {result.color_count} 色"
    draw.text((padding, padding + 28), sub, fill=(120, 120, 120), font=font_sub)

    # ---------- 主体网格色块 ----------
    for r in range(gh):
        for c in range(gw):
            cell = grid[r][c]
            xa = grid_x0 + c * cell_px
            ya = grid_y0 + r * cell_px
            xb = xa + cell_px - 1
            yb = ya + cell_px - 1
            if cell.code == "":  # 空白格子：留白 + 淡灰斜叉
                draw.rectangle([xa, ya, xb, yb], fill=(255, 255, 255))
                if cell_px >= 12:
                    draw.line([(xa + 2, ya + 2), (xb - 2, yb - 2)],
                              fill=(225, 225, 225), width=1)
                    draw.line([(xb - 2, ya + 2), (xa + 2, yb - 2)],
                              fill=(225, 225, 225), width=1)
                continue
            draw.rectangle([xa, ya, xb, yb], fill=cell.rgb)
            # 色号标注（自适应字号，保证 104 宽下字母+两位数完整显示）
            need_label = label_all_cells or (
                label_every >= 1 and (r % label_every == 0) and (c % label_every == 0)
            )
            if need_label and cell_px >= 10:
                txt = cell.code
                tc = _text_color_for_bg(cell.rgb)
                font_label = _fit_font_for_text(txt, cell_px)
                bbox = draw.textbbox((0, 0), txt, font=font_label)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = xa + (cell_px - tw) // 2 - bbox[0]
                ty = ya + (cell_px - th) // 2 - bbox[1]
                draw.text((tx, ty), txt, fill=tc, font=font_label)

    # ---------- 网格线（对比色参考线，每 5 格加粗）----------
    grid_bottom = grid_y0 + grid_img_h
    grid_right = grid_x0 + grid_img_w
    # 细线 1px，每 5 格粗线加粗（更明显）
    bold_w = 3 if cell_px >= 12 else (2 if cell_px >= 8 else 1)

    for c in range(gw + 1):
        x = grid_x0 + c * cell_px
        is_bold = (c % 5 == 0)
        draw.line([(x, grid_y0), (x, grid_bottom)],
                  fill=bold_color if is_bold else thin_color,
                  width=bold_w if is_bold else 1)
    for r in range(gh + 1):
        y = grid_y0 + r * cell_px
        is_bold = (r % 5 == 0)
        draw.line([(grid_x0, y), (grid_right, y)],
                  fill=bold_color if is_bold else thin_color,
                  width=bold_w if is_bold else 1)

    # ---------- 行列号标注（外扩一行/列，浅色打底 + 数字 + 深色粗边框）----------
    # 标注区边界
    axis_outer_x0 = grid_x0 - axis_left
    axis_outer_y0 = grid_y0 - axis_top
    axis_outer_x1 = grid_right + axis_right
    axis_outer_y1 = grid_bottom + axis_bottom

    # 1) 四个标注区浅色打底
    # 顶部列号区
    draw.rectangle([axis_outer_x0, axis_outer_y0, axis_outer_x1, grid_y0 - 1], fill=axis_bg)
    # 底部列号区
    draw.rectangle([axis_outer_x0, grid_bottom + 1, axis_outer_x1, axis_outer_y1], fill=axis_bg)
    # 左侧行号区
    draw.rectangle([axis_outer_x0, axis_outer_y0, grid_x0 - 1, axis_outer_y1], fill=axis_bg)
    # 右侧行号区
    draw.rectangle([grid_right + 1, axis_outer_y0, axis_outer_x1, axis_outer_y1], fill=axis_bg)

    # 2) 标注数字（每格居中，精确对齐网格格子）
    # 行列号自适应字号：最长数字（列号 gw 或行号 gh 中位数较大者）能放进 cell_px 宽
    longest_axis = str(max(gw, gh))
    font_axis = _fit_font_for_text(longest_axis, cell_px, max_size=12, min_size=6, pad=2)

    def _draw_centered_in_cell(cx0, cy0, cw, ch, txt, font, color):
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx0 + (cw - tw) // 2 - bbox[0]
        ty = cy0 + (ch - th) // 2 - bbox[1]
        draw.text((tx, ty), txt, fill=color, font=font)

    # 顶部/底部：列号（每个列号占 cell_px 宽，与网格列精确对齐）
    for c in range(gw):
        xc = grid_x0 + c * cell_px
        _draw_centered_in_cell(xc, axis_outer_y0, cell_px, axis_top, str(c + 1), font_axis, axis_text)
        _draw_centered_in_cell(xc, grid_bottom + 1, cell_px, axis_bottom, str(c + 1), font_axis, axis_text)
    # 左侧/右侧：行号（每个行号占 cell_px 高，与网格行精确对齐）
    for r in range(gh):
        yc = grid_y0 + r * cell_px
        _draw_centered_in_cell(axis_outer_x0, yc, axis_left, cell_px, str(r + 1), font_axis, axis_text)
        _draw_centered_in_cell(grid_right + 1, yc, axis_right, cell_px, str(r + 1), font_axis, axis_text)

    # 3) 标注区外沿深色粗边框（包围整个标注+网格区域）
    draw.rectangle(
        [axis_outer_x0 - border_w, axis_outer_y0 - border_w,
         axis_outer_x1 + border_w, axis_outer_y1 + border_w],
        outline=axis_border, width=border_w,
    )

    # ---------- 图例（下方：色块内写色号，下方写个数，换行）----------
    ly = grid_bottom + axis_bottom + legend_pad
    lx = padding + axis_left
    for line in legend_lines:
        cx = lx
        for code, cnt in line:
            pal_c = _find_color_in_grid(grid, code)
            rgb = pal_c.rgb if pal_c else (200, 200, 200)
            bw = _block_width(code)
            # 长方形色块，色号写在块内（用对比色文字）
            draw.rectangle([cx, ly, cx + bw, ly + legend_sw],
                           fill=rgb, outline=(180, 180, 180), width=1)
            code_tc = _text_color_for_bg(rgb)
            cb = draw.textbbox((0, 0), code, font=font_legend_code)
            cw = cb[2] - cb[0]
            ch = cb[3] - cb[1]
            draw.text((cx + (bw - cw) // 2 - cb[0],
                       ly + (legend_sw - ch) // 2 - cb[1]),
                      code, fill=code_tc, font=font_legend_code)
            # 个数写在色块下方
            nb = draw.textbbox((0, 0), str(cnt), font=font_legend_count)
            nw = nb[2] - nb[0]
            draw.text((cx + (bw - nw) // 2 - nb[0], ly + legend_sw + 2),
                      str(cnt), fill=text, font=font_legend_count)
            cx += bw + legend_gap_x
        ly += legend_row_h + legend_gap_y

    return img


def _find_color_in_grid(grid, code):
    for row in grid:
        for c in row:
            if c.code == code:
                return c
    return None


def save_bead_chart(
    result: QuantizeResult,
    path: str | Path,
    **kwargs,
) -> Path:
    img = render_bead_chart(result, **kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return p


def rebuild_result_stats(result: QuantizeResult) -> QuantizeResult:
    """编辑网格后重算统计（用色数 / 各色用量 / 总格数）。

    网格内容可能被手动编辑改变，`color_count`、`stats`、`total_cells`
    需要据此重算，否则图例与统计会错。空白格子（code 为空）不计入颜色。
    原地更新并返回同一对象。
    """
    from collections import Counter

    stats: Counter = Counter()
    for row in result.grid:
        for cell in row:
            if cell.code != "":  # 跳过空白格子
                stats[cell.code] += 1
    result.stats = dict(stats)
    result.color_count = len(stats)
    result.total_cells = result.grid_w * result.grid_h
    return result


def render_grid_only(
    result: QuantizeResult,
    *,
    cell_px: int = 24,
    with_grid: bool = True,
    with_axis: bool = True,
) -> Image.Image:
    """只画网格（不带图例），用于贴纸/小图预览。

    参考线取用量最多色的对比色、每 5 格加粗；外扩一行/列标注行列号
    （(1,1) 为左上原点），浅色打底 + 外沿深色粗边框。
    """
    grid = result.grid
    gw, gh = result.grid_w, result.grid_h

    # 外扩一行/列做标注区，与网格格子对齐
    axis_top = cell_px
    axis_left = cell_px
    axis_bottom = cell_px
    axis_right = cell_px
    border_w = 3

    grid_w = gw * cell_px
    grid_h = gh * cell_px
    grid_x0 = axis_left
    grid_y0 = axis_top

    W = grid_w + axis_left + axis_right + border_w * 2
    H = grid_h + axis_top + axis_bottom + border_w * 2
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # 平移原点，给外沿边框留空间
    grid_x0 += border_w
    grid_y0 += border_w

    # 参考线颜色：基于用量最多色的对比色
    top_code = max(result.stats.items(), key=lambda kv: kv[1])[0] if result.stats else None
    top_rgb = None
    if top_code:
        _pc = _find_color_in_grid(grid, top_code)
        top_rgb = _pc.rgb if _pc else None
    line_color = _highest_contrast_color(top_rgb) if top_rgb else (200, 40, 40)
    axis_bg = _lighter(line_color, 0.85)
    axis_border = _darker(line_color, 0.45)
    axis_text = _darker(line_color, 0.55)

    for r in range(gh):
        for c in range(gw):
            cell = grid[r][c]
            if cell.code == "":
                continue
            xa = grid_x0 + c * cell_px
            ya = grid_y0 + r * cell_px
            draw.rectangle(
                [xa, ya, xa + cell_px - 1, ya + cell_px - 1],
                fill=cell.rgb,
            )
    if with_grid:
        thin = line_color
        bold = line_color
        bold_w = 3 if cell_px >= 12 else (2 if cell_px >= 8 else 1)
        for c in range(gw + 1):
            x = grid_x0 + c * cell_px
            is_bold = (c % 5 == 0)
            draw.line([(x, grid_y0), (x, grid_y0 + grid_h)],
                      fill=bold if is_bold else thin,
                      width=bold_w if is_bold else 1)
        for r in range(gh + 1):
            y = grid_y0 + r * cell_px
            is_bold = (r % 5 == 0)
            draw.line([(grid_x0, y), (grid_x0 + grid_w, y)],
                      fill=bold if is_bold else thin,
                      width=bold_w if is_bold else 1)

    if with_axis:
        grid_bottom = grid_y0 + grid_h
        grid_right = grid_x0 + grid_w
        axis_outer_x0 = grid_x0 - axis_left
        axis_outer_y0 = grid_y0 - axis_top
        axis_outer_x1 = grid_right + axis_right
        axis_outer_y1 = grid_bottom + axis_bottom

        # 标注区浅色打底
        draw.rectangle([axis_outer_x0, axis_outer_y0, axis_outer_x1, grid_y0 - 1], fill=axis_bg)
        draw.rectangle([axis_outer_x0, grid_bottom + 1, axis_outer_x1, axis_outer_y1], fill=axis_bg)
        draw.rectangle([axis_outer_x0, axis_outer_y0, grid_x0 - 1, axis_outer_y1], fill=axis_bg)
        draw.rectangle([grid_right + 1, axis_outer_y0, axis_outer_x1, axis_outer_y1], fill=axis_bg)

        # 数字（自适应字号，每格居中）
        longest_axis = str(max(gw, gh))
        font_axis = _fit_font_for_text(longest_axis, cell_px, max_size=12, min_size=6, pad=2)

        def _centered(cx0, cy0, cw, ch, txt):
            bbox = draw.textbbox((0, 0), txt, font=font_axis)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((cx0 + (cw - tw) // 2 - bbox[0], cy0 + (ch - th) // 2 - bbox[1]),
                      txt, fill=axis_text, font=font_axis)

        for c in range(gw):
            xc = grid_x0 + c * cell_px
            _centered(xc, axis_outer_y0, cell_px, axis_top, str(c + 1))
            _centered(xc, grid_bottom + 1, cell_px, axis_bottom, str(c + 1))
        for r in range(gh):
            yc = grid_y0 + r * cell_px
            _centered(axis_outer_x0, yc, axis_left, cell_px, str(r + 1))
            _centered(grid_right + 1, yc, axis_right, cell_px, str(r + 1))

        # 外沿深色粗边框
        draw.rectangle(
            [axis_outer_x0 - border_w, axis_outer_y0 - border_w,
             axis_outer_x1 + border_w, axis_outer_y1 + border_w],
            outline=axis_border, width=border_w,
        )

    return img


def render_plain_image(
    result: QuantizeResult,
    *,
    cell_px: int = 24,
) -> Image.Image:
    """渲染不带网格线的纯像素化图像（模拟拼豆烫好后融化的样子）。

    空白格子留白，其余格子填色，无任何网格线。
    """
    grid = result.grid
    gw, gh = result.grid_w, result.grid_h
    W = gw * cell_px
    H = gh * cell_px
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for r in range(gh):
        for c in range(gw):
            cell = grid[r][c]
            if cell.code == "":
                continue
            xa = c * cell_px
            ya = r * cell_px
            draw.rectangle([xa, ya, xa + cell_px, ya + cell_px], fill=cell.rgb)
    return img


__all__ = [
    "render_bead_chart",
    "save_bead_chart",
    "render_grid_only",
    "render_plain_image",
    "rebuild_result_stats",
]
