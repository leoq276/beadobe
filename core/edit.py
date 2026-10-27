"""
edit.py
网格编辑的纯逻辑层（不依赖 Qt）。

提供对 QuantizeResult.grid 的批量操作：
  - paint_cells: 把一批格子改成指定色号
  - erase_cells: 把一批格子清空为空白（无珠子）
  - adjust_cells: 对一批格子做加深/变浅/对比度调整（Lab 空间操作后重映射色板）
  - select_by_codes: 按色号集合返回命中的格子坐标

颜色调整原理：
  每个 PaletteColor 有 lab=(L, a, b)。调整只在 L 通道上操作：
    - 加深 (darken):  L' = L - amount
    - 变浅 (lighten): L' = L + amount
    - 对比度:        L' = 50 + (L - 50) * factor  （以 50 为中性灰中心拉伸）
  a/b 保持不变（保留色相），然后取色板中 Lab 最近邻作为新色号。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .palette_loader import Palette, PaletteColor
from .quantize import QuantizeResult

# 网格坐标类型
CellCoord = Tuple[int, int]  # (row, col)

# 空白格子（被删除的珠子）。用特殊 PaletteColor 表示，code 为空串。
# rgb 用纯白，渲染时 GUI 会把它画成空白（带浅灰叉或纯白）。
BLANK = PaletteColor(
    code="",
    name="空白",
    hex="#FFFFFF",
    rgb=(255, 255, 255),
    lab=(100.0, 0.0, 0.0),
)


def is_blank(cell: PaletteColor) -> bool:
    """判断格子是否为空白。"""
    return cell is BLANK or cell.code == ""


def _cell_set(coords) -> Set[CellCoord]:
    return {(r, c) for r, c in coords}


def paint_cells(
    result: QuantizeResult,
    coords,
    target: PaletteColor,
) -> int:
    """把 coords 里的格子改成 target 色号。返回改动格子数。"""
    changed = 0
    for r, c in coords:
        if 0 <= r < result.grid_h and 0 <= c < result.grid_w:
            if result.grid[r][c].code != target.code:
                result.grid[r][c] = target
                changed += 1
    return changed


def erase_cells(
    result: QuantizeResult,
    coords,
) -> int:
    """把 coords 里的格子清空为空白。返回改动格子数。"""
    changed = 0
    for r, c in coords:
        if 0 <= r < result.grid_h and 0 <= c < result.grid_w:
            if not is_blank(result.grid[r][c]):
                result.grid[r][c] = BLANK
                changed += 1
    return changed


def adjust_cells(
    result: QuantizeResult,
    coords,
    *,
    mode: str,
    amount: float,
    palette: Palette,
) -> int:
    """
    对 coords 里的格子做颜色调整，返回改动格子数。

    :param mode: "darken" | "lighten" | "contrast"
    :param amount: 强度。
        - darken/lighten: L 通道增减量 (0~100，常用 5~25)
        - contrast: 对比度因子（>1 增强，<1 减弱，常用 1.1~1.5 或 0.7~0.9）
    """
    changed = 0
    for r, c in coords:
        if not (0 <= r < result.grid_h and 0 <= c < result.grid_w):
            continue
        cell = result.grid[r][c]
        if is_blank(cell):
            continue
        L, a, b = cell.lab
        if mode == "darken":
            L2 = max(0.0, L - amount)
        elif mode == "lighten":
            L2 = min(100.0, L + amount)
        elif mode == "contrast":
            L2 = 50.0 + (L - 50.0) * amount
            L2 = max(0.0, min(100.0, L2))
        else:
            raise ValueError(f"未知调整模式: {mode}")
        new_color = palette.find_nearest_lab((L2, a, b))
        if new_color.code != cell.code:
            result.grid[r][c] = new_color
            changed += 1
    return changed


def select_by_codes(
    result: QuantizeResult,
    codes: Set[str],
) -> List[CellCoord]:
    """返回所有色号属于 codes 的格子坐标列表。"""
    hits: List[CellCoord] = []
    for r in range(result.grid_h):
        for c in range(result.grid_w):
            if result.grid[r][c].code in codes:
                hits.append((r, c))
    return hits


def codes_in_grid(result: QuantizeResult) -> List[str]:
    """返回网格中实际出现过的色号（按出现频次降序）。"""
    cnt: Dict[str, int] = {}
    for row in result.grid:
        for cell in row:
            if is_blank(cell):
                continue
            cnt[cell.code] = cnt.get(cell.code, 0) + 1
    return [k for k, _ in sorted(cnt.items(), key=lambda kv: -kv[1])]


def codes_in_palette(palette: Palette) -> List[PaletteColor]:
    """返回完整色板（全部色号），用于"从没有的颜色中选"。"""
    return list(palette.colors)


def snapshot_grid(result: QuantizeResult) -> List[List[PaletteColor]]:
    """深拷贝当前网格（用于撤销/重做）。返回新的二维数组。"""
    return [list(row) for row in result.grid]


def restore_grid(result: QuantizeResult, grid: List[List[PaletteColor]]) -> None:
    """把快照还原回 result.grid（原地修改每格，保持外层引用不变）。

    注意：必须原地修改（result.grid[r][c] = ...），不能 `result.grid = [...]`
    替换外层引用。因为 GridEditWidget 持有 result.grid 的引用，替换引用会导致
    控件与数据脱钩，界面不刷新。
    """
    for r in range(result.grid_h):
        for c in range(result.grid_w):
            result.grid[r][c] = grid[r][c]

