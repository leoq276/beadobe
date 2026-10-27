"""
palette_loader.py
读取 MARD 色板 CSV (code,name,hex)，并把每个色号转换到 CIELAB 颜色空间。

提供 Lab 空间最近邻匹配 (CIE76 ΔE)，比 RGB 欧氏距离更符合人眼感知。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .paths import resource_path

# 色板 CSV 路径（只读资源：源码运行时在项目根 / palettes / mard.csv，
# 打包后在 _MEIPASS / palettes / mard.csv）
DEFAULT_PALETTE_CSV = resource_path("palettes", "mard.csv")


# =============================================================================
# RGB -> CIELAB (D65)
# =============================================================================
def _srgb_to_linear(c: float) -> float:
    """sRGB 反 gamma 校正"""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _f_lab(t: float) -> float:
    """CIE Lab f 函数"""
    delta = 6.0 / 29.0
    if t > delta ** 3:
        return t ** (1.0 / 3.0)
    return t / (3.0 * delta * delta) + 4.0 / 29.0


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """sRGB (0~255) -> CIE Lab (D65)"""
    # 1. RGB -> linear RGB
    rl = _srgb_to_linear(r / 255.0)
    gl = _srgb_to_linear(g / 255.0)
    bl = _srgb_to_linear(b / 255.0)
    # 2. linear RGB -> XYZ (D65)
    x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047
    y = (0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl) / 1.00000
    z = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) / 1.08883
    # 3. XYZ -> Lab
    L = 116.0 * _f_lab(y) - 16.0
    a = 500.0 * (_f_lab(x) - _f_lab(y))
    b_ = 200.0 * (_f_lab(y) - _f_lab(z))
    return (L, a, b_)


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_lab(h: str) -> Tuple[float, float, float]:
    r, g, b = hex_to_rgb(h)
    return rgb_to_lab(r, g, b)


def delta_e76(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    """CIE76 ΔE — 简单但够用"""
    return (
        (lab1[0] - lab2[0]) ** 2
        + (lab1[1] - lab2[1]) ** 2
        + (lab1[2] - lab2[2]) ** 2
    ) ** 0.5


# =============================================================================
# 色板数据类
# =============================================================================
@dataclass(frozen=True)
class PaletteColor:
    code: str
    name: str
    hex: str
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float]


@dataclass
class Palette:
    """完整色板，含 Lab 索引"""
    colors: List[PaletteColor]
    source_path: str

    def __len__(self) -> int:
        return len(self.colors)

    def __iter__(self):
        return iter(self.colors)

    def by_code(self, code: str) -> Optional[PaletteColor]:
        for c in self.colors:
            if c.code == code:
                return c
        return None

    def find_nearest_lab(self, lab: Sequence[float]) -> PaletteColor:
        """找 Lab 距离最近的色号"""
        best = self.colors[0]
        best_d = delta_e76(lab, best.lab)
        for c in self.colors[1:]:
            d = delta_e76(lab, c.lab)
            if d < best_d:
                best_d = d
                best = c
        return best

    def find_nearest_lab_batch(
        self,
        labs: Sequence[Sequence[float]],
    ) -> List[PaletteColor]:
        return [self.find_nearest_lab(l) for l in labs]


# =============================================================================
# 加载
# =============================================================================
def load_palette(path: str | Path | None = None) -> Palette:
    """
    加载色板 CSV。CSV 格式: code,name,hex (列名必须存在)。
    """
    p = Path(path) if path else DEFAULT_PALETTE_CSV
    if not p.exists():
        raise FileNotFoundError(f"色板文件不存在: {p}")

    colors: List[PaletteColor] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or "").strip()
            name = (row.get("name") or "").strip()
            hex_ = (row.get("hex") or "").strip()
            if not code or not hex_:
                continue
            rgb = hex_to_rgb(hex_)
            lab = rgb_to_lab(*rgb)
            colors.append(PaletteColor(
                code=code, name=name, hex=hex_.upper(), rgb=rgb, lab=lab,
            ))
    return Palette(colors=colors, source_path=str(p))


if __name__ == "__main__":
    pal = load_palette()
    print(f"已加载色板: {len(pal)} 色  <- {pal.source_path}")
    # 找几个色测一下
    for code in ("H7", "H1", "A4", "C8", "H2"):
        c = pal.by_code(code)
        if c:
            print(f"  {code}: hex={c.hex}  rgb={c.rgb}  lab=({c.lab[0]:.1f},{c.lab[1]:.1f},{c.lab[2]:.1f})")
    # 最近邻测试
    test = hex_to_lab("#FF0000")  # 纯红
    nearest = pal.find_nearest_lab(test)
    print(f"  纯红 -> {nearest.code} ({nearest.name}) ΔE={delta_e76(test, nearest.lab):.1f}")
