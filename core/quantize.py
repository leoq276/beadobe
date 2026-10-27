"""
quantize.py
色板量化核心：
  1. 把原图切成 grid_w × grid_h 块
  2. 每块用"频次投票"选主色（最不容易产生中间色）
  3. 可选：清理小色块（<N 像素的孤立连通区合并到邻域主色）
  4. 返回 grid (二维数组，每个元素是 PaletteColor)
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .palette_loader import Palette, PaletteColor, delta_e76, load_palette


# =============================================================================
# 0. 像素化前置方法（area / dpid / pixeloe）
# =============================================================================
def _apply_pixelize_preprocess(
    arr: np.ndarray,
    grid_w: int,
    grid_h: int,
    factor: int,
    pixelize_method: str,
    pixelize_params: dict,
) -> np.ndarray:
    """启用像素化方法时，把原图下采样到 (grid_w*factor) × (grid_h*factor)，返回 RGB uint8。

    像素化方法直接产生结果，不经过超采样/投票/细节链路：调用时 factor=1，
    即直接下采样到网格分辨率，之后 _block_vote_quantize 每块只有 1 像素，
    投票等价于直接映射（最近色号）。cleanup / 统计照常复用。
    """
    from .pixelize import pixelize
    factor = max(1, int(factor))
    out_w = grid_w * factor
    out_h = grid_h * factor
    params = dict(pixelize_params or {})
    method = (pixelize_method or "").lower()
    return pixelize(arr, out_w, out_h, method=method, **params)


@dataclass
class QuantizeResult:
    """量化结果"""
    grid: List[List[PaletteColor]]     # [row][col] = 色号
    stats: Dict[str, int]              # code -> 像素数
    color_count: int                   # 用到的色号数
    total_cells: int                   # 总格子数
    grid_w: int
    grid_h: int
    removed_islands: int               # 清理掉的孤立小色块数


# =============================================================================
# 1. 投票量化（numpy 向量化版）
# =============================================================================
# 预计算色板的 Lab 数组 (N, 3)
_PALETTE_LAB_CACHE: Dict[int, np.ndarray] = {}
_PALETTE_INDEX_CACHE: Dict[int, np.ndarray] = {}


def _palette_lab_arrays(palette: Palette):
    """获取色板 Lab 数组，按 id(palette) 缓存"""
    key = id(palette)
    if key not in _PALETTE_LAB_CACHE:
        labs = np.array([c.lab for c in palette.colors], dtype=np.float32)
        _PALETTE_LAB_CACHE[key] = labs
        _PALETTE_INDEX_CACHE[key] = np.array(
            [c.code for c in palette.colors], dtype=object,
        )
    return _PALETTE_LAB_CACHE[key], _PALETTE_INDEX_CACHE[key]


def _rgb_to_lab_batch(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    rgb_uint8: (..., 3) uint8 -> lab (..., 3) float32
    完整 sRGB -> linear -> XYZ (D65) -> Lab
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    # 反 gamma
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    # sRGB -> XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    # 归一化到 D65 白点
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883
    # f
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    f = np.where(y > delta3, np.cbrt(y), y / (3.0 * delta * delta) + 4.0 / 29.0)
    fx = np.where(x > delta3, np.cbrt(x), x / (3.0 * delta * delta) + 4.0 / 29.0)
    fz = np.where(z > delta3, np.cbrt(z), z / (3.0 * delta * delta) + 4.0 / 29.0)
    L = 116.0 * f - 16.0
    a = 500.0 * (fx - f)
    b_ = 200.0 * (f - fz)
    return np.stack([L, a, b_], axis=-1)


def _classify_region(
    r: int, c: int, grid_w: int, grid_h: int,
    key_features: Optional[Dict[str, List[float]]],
) -> str:
    """
    判断 (r, c) 块位于哪个区域。
    返回: "core" | "face" | "background"
      - core: 眼/嘴/鼻/耳/眉/胡须等关键五官
      - face: 脸/颊/嘴部/胸 等面部外围
      - background: 其余
    """
    if not key_features:
        return "background"
    cx = (c + 0.5) / grid_w
    cy = (r + 0.5) / grid_h
    core_kw = ("eye", "mouth", "nose", "ear", "brow", "whisker", "lid")
    face_kw = ("face", "head", "cheek", "muzzle", "chin", "body", "chest")

    in_any = False
    for name, bbox in key_features.items():
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            in_any = True
            name_lower = name.lower()
            for kw in core_kw:
                if kw in name_lower:
                    return "core"
            for kw in face_kw:
                if kw in name_lower:
                    return "face"
            # 命中任何 key_features 至少算 face
            return "face"
    if in_any:
        return "face"
    return "background"


def _region_thresholds(
    region: str,
    base_pct: float,
    base_contrast: float,
) -> Tuple[float, float]:
    """
    按区域返回 (min_sub_pct, min_contrast) 阈值。
    关键区域: 更激进（4% / 12），让小细节也胜出
    面部外围: 默认
    背景: 更保守（12% / 24），避免杂色
    """
    if region == "core":
        return (
            max(0.03, base_pct * 0.5),
            max(10.0, base_contrast * 0.65),
        )
    if region == "face":
        return (base_pct, base_contrast)
    return (
        min(0.15, base_pct * 1.8),
        min(28.0, base_contrast * 1.4),
    )


def _block_luminance(block_uint8: np.ndarray) -> np.ndarray:
    """块内每像素的 Rec.601 亮度 (0~255)"""
    r = block_uint8[:, 0].astype(np.float32)
    g = block_uint8[:, 1].astype(np.float32)
    b = block_uint8[:, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _detect_line_mask_local_contrast(
    img_array: np.ndarray,
    *,
    blur_radius: int = 1,
    threshold_percentile: float = 88.0,
) -> np.ndarray:
    """
    #5 局部对比度线条检测。
    每个像素的"局部对比度" = max(|pixel - 邻域均值|, |pixel - 整体均值|)。
    高于 N 分位数的像素视为"线条"。

    优点:
      - 独立于深浅 (深线浅线都识别)
      - 零新依赖, 纯 PIL + numpy
      - ~15 行代码

    参数:
      blur_radius: Pillow BoxBlur 半径 (1=3x3, 2=5x5)
      threshold_percentile: 高于此分位数的像素视为线条
    """
    gray = (
        0.299 * img_array[:, :, 0].astype(np.float32) +
        0.587 * img_array[:, :, 1].astype(np.float32) +
        0.114 * img_array[:, :, 2].astype(np.float32)
    )
    img_pil = Image.fromarray(gray.astype(np.uint8), mode="L")
    local_mean = np.asarray(
        img_pil.filter(ImageFilter.BoxBlur(blur_radius)),
        dtype=np.float32,
    )
    global_mean = float(gray.mean())
    diff_local = np.abs(gray - local_mean)
    diff_global = np.abs(gray - global_mean)
    contrast = np.maximum(diff_local, diff_global)
    threshold = float(np.percentile(contrast, threshold_percentile))
    return contrast > threshold


def _detect_line_mask_multiscale_sobel(
    img_array: np.ndarray,
    *,
    sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
    high_pct: float = 80.0,
    low_pct: float = 40.0,
) -> np.ndarray:
    """
    #2 多尺度 Sobel + 非极大值抑制 (Canny 思路)。

    流程:
      1. 灰度化
      2. 在 3 个尺度 (高斯 sigma) 下分别做 Sobel
      3. 取 3 个尺度的最大梯度响应
      4. NMS: 在梯度方向上保留局部最大值 (细化边缘到 1 像素)
      5. 双阈值 + 滞后跟踪: 强边缘必保留, 弱边缘只在连接强边缘时保留

    优点 (相比 #5):
      - "线条性" 判定更精准 (细长 vs 块状)
      - NMS 细化后边缘更干净
      - 多尺度兼顾粗细

    返回: 二值线条 mask (H×W bool)
    """
    gray = (
        0.299 * img_array[:, :, 0].astype(np.float32) +
        0.587 * img_array[:, :, 1].astype(np.float32) +
        0.114 * img_array[:, :, 2].astype(np.float32)
    )
    H, W = gray.shape

    # Sobel 算子
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = Kx.T

    # 多尺度梯度 (取 max 响应)
    mag_max = np.zeros_like(gray)
    dir_max = np.zeros_like(gray)  # 弧度
    img_pil_gray = Image.fromarray(gray.astype(np.uint8), mode="L")
    for sigma in sigmas:
        # 高斯模糊
        if sigma < 0.5:
            blurred = np.asarray(img_pil_gray, dtype=np.float32)
        else:
            blurred = np.asarray(
                img_pil_gray.filter(ImageFilter.GaussianBlur(sigma)),
                dtype=np.float32,
            )
        # Sobel (用 numpy 卷积)
        gx = _convolve2d_same(blurred, Kx)
        gy = _convolve2d_same(blurred, Ky)
        mag = np.sqrt(gx * gx + gy * gy)
        # 取最大响应
        better = mag > mag_max
        mag_max = np.where(better, mag, mag_max)
        dir_max = np.where(better, np.arctan2(gy, gx), dir_max)

    # 非极大值抑制 (NMS): 在量化方向上比较
    # 量化方向到 4 个: 0°, 45°, 90°, 135°
    # 用 (gx, gy) 符号判定方向
    # 0°: |gx| > |gy| 且 gx > 0
    # 90°: |gy| > |gx| 且 gy > 0
    # ... (略, 用 round(dir / 45°) 量化)
    dir_q = (np.round(dir_max / (np.pi / 4)).astype(np.int32)) % 4
    # NMS: 对每个像素, 在其梯度方向上的两个邻居中, 如果不是最大就抑制
    nms = np.zeros_like(mag_max, dtype=bool)
    # 4 方向对应的 (dy, dx) 偏移
    dir_offsets = {
        0: (0, 1),    # 0° (水平向右)
        1: (1, 1),    # 45°
        2: (1, 0),    # 90° (垂直向下)
        3: (1, -1),   # 135°
    }
    # Pad
    mag_padded = np.pad(mag_max, 1, mode="edge")
    for d, (dy, dx) in dir_offsets.items():
        mask_d = dir_q == d
        # 沿 d 方向的两个邻居: (cy+dy, cx+dx) 和 (cy-dy, cx-dx)
        n1 = mag_padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        n2 = mag_padded[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
        # 在该方向上, 中心 >= 邻居 → 保留
        local_max = (mag_max >= n1) & (mag_max >= n2) & mask_d
        nms |= local_max

    # 双阈值
    high_thr = float(np.percentile(mag_max[nms], high_pct)) if nms.any() else 50.0
    low_thr = high_thr * (low_pct / 100.0)

    strong = mag_max >= high_thr
    weak = (mag_max >= low_thr) & ~strong

    # 滞后跟踪: 弱边缘如果在 8 邻域内有强边缘, 升级为强
    # 简化: 直接把 strong + weak 都视为线条
    line_mask = strong | weak
    return line_mask


def _detect_line_mask_canny_opencv(
    img_array: np.ndarray,
    *,
    blur_ksize: int = 5,
    low: Optional[int] = None,
    high: Optional[int] = None,
    sigma_ratio: float = 0.33,
    l2gradient: bool = False,
) -> np.ndarray:
    """
    #1 Canny (OpenCV) 线条检测。

    完整 Canny 流程: 高斯模糊 -> Sobel -> NMS -> 双阈值滞后。
    比 #2 多尺度 Sobel 更标准 (滞后跟踪真正剔除了孤立弱响应)。

    参数:
      blur_ksize: 高斯模糊核大小 (奇数, 默认 5)
      low / high: Canny 双阈值 (None 则基于图像中位数自适应)
        自适应公式 (OpenCV 官方推荐):
          median = np.median(gray)
          low  = max(0,  (1 - sigma_ratio) * median)
          high = min(255,(1 + sigma_ratio) * median)
      sigma_ratio: 自适应阈值比例 (默认 0.33 → 高中位比 ~2:1)
      l2gradient: 是否用 L2 范数算梯度 (False = L1, 快一倍)

    返回: 二值线条 mask (H×W bool)
    """
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "_detect_line_mask_canny_opencv 需要 opencv-python，"
            "请 conda activate NDP 或 pip install opencv-python"
        ) from e

    # 灰度
    if img_array.ndim == 3:
        gray = (
            0.299 * img_array[:, :, 0].astype(np.float32) +
            0.587 * img_array[:, :, 1].astype(np.float32) +
            0.114 * img_array[:, :, 2].astype(np.float32)
        ).astype(np.uint8)
    else:
        gray = img_array.astype(np.uint8)

    # 自适应阈值
    if low is None or high is None:
        median = float(np.median(gray))
        low = int(max(0, (1.0 - sigma_ratio) * median))
        high = int(min(255, (1.0 + sigma_ratio) * median))
        # 保底: low 不能太低 (会噪点泛滥)
        low = max(low, 20)
        high = max(high, low * 2)

    # Canny
    edges = cv2.Canny(
        gray, low, high,
        apertureSize=3,           # Sobel 核大小
        L2gradient=l2gradient,
    )
    return edges > 0


def _hessian_2d(gray: np.ndarray, sigma: float) -> tuple:
    """
    2D Hessian 矩阵: H = [[Ixx, Ixy], [Ixy, Iyy]]
    通过对灰度图做高斯平滑后的二阶偏导得到。

    返回: Ixx, Ixy, Iyy (H×W float32)
    """
    from PIL import ImageFilter
    img_pil = Image.fromarray(gray.astype(np.uint8), mode="L")
    smoothed = np.asarray(
        img_pil.filter(ImageFilter.GaussianBlur(sigma)),
        dtype=np.float32,
    )
    # 二阶偏导核 (sigma 不影响归一化，导数符号是固定的)
    Kxx = np.array([[1, -2, 1]], dtype=np.float32)         # (1, 3)
    Kyy = Kxx.T                                            # (3, 1)
    Kxy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
                   dtype=np.float32) * 0.25  # 等价 (1/4)*[[1,0,-1],[0,0,0],[-1,0,1]]
    Ixx = _convolve2d_same(smoothed, Kxx)
    Iyy = _convolve2d_same(smoothed, Kyy)
    Ixy = _convolve2d_same(smoothed, Kxy)
    return Ixx, Ixy, Iyy


def _frangi_response_2d(
    img_array: np.ndarray,
    *,
    sigmas: tuple = (1.0, 2.0, 4.0),
    beta: float = 0.5,
    black_ridges: bool = True,
) -> np.ndarray:
    """
    #4 Frangi 2D vesselness / lineness 响应。

    原理:
      1. 每个尺度下求 Hessian 矩阵
      2. 求特征值 λ1, λ2 (|λ1| ≤ |λ2|)
      3. 计算两个判别:
         Rb = λ1/λ2   → 0 表示线状 (管/线), 1 表示块状
         S  = ||H||_F = sqrt(λ1² + λ2²) → 越大越不是噪点
      4. 响应 = exp(-Rb²/(2β²)) × (1 - exp(-S²/(2c²)))
         (c 自适应 = Hessian norm 中位数)
      5. 多尺度取 max 响应

    参数:
      sigmas: 多尺度高斯平滑
      beta: 控制 Rb 的灵敏度 (越小越倾向"线状才算")
      black_ridges: True=检测"暗线 on 亮背景" (我们的 cartoon), False=亮线

    返回: 响应图 (H×W float32)，0~1 之间
    """
    if img_array.ndim == 3:
        gray = (
            0.299 * img_array[:, :, 0].astype(np.float32) +
            0.587 * img_array[:, :, 1].astype(np.float32) +
            0.114 * img_array[:, :, 2].astype(np.float32)
        )
    else:
        gray = img_array.astype(np.float32)

    resp_max = np.zeros_like(gray, dtype=np.float32)

    for sigma in sigmas:
        Ixx, Ixy, Iyy = _hessian_2d(gray, sigma)
        # 2D Hessian 特征值:
        # λ1,2 = (Ixx + Iyy ± sqrt((Ixx - Iyy)² + 4·Ixy²)) / 2
        # 较小/较大记为 λ1/λ2 (|λ1| ≤ |λ2|)
        diff = Ixx - Iyy
        disc = np.sqrt(np.maximum(diff * diff + 4.0 * Ixy * Ixy, 0.0))
        lambda1 = (Ixx + Iyy - disc) * 0.5
        lambda2 = (Ixx + Iyy + disc) * 0.5

        # 排序: |λ1| ≤ |λ2| (Frangi 论文定义)
        swap_mask = np.abs(lambda1) > np.abs(lambda2)
        if swap_mask.any():
            l1_tmp = lambda1.copy()
            lambda1 = np.where(swap_mask, lambda2, lambda1)
            lambda2 = np.where(swap_mask, l1_tmp, lambda2)

        eps = 1e-10
        Rb = lambda1 / (lambda2 + eps)             # 0=线状, 1=块状
        S = np.sqrt(lambda1 * lambda1 + lambda2 * lambda2)  # Frobenius 范数

        # c 自适应: 取该尺度 Hessian 范数的中位数作为"背景结构强度"
        c = max(float(np.median(S)), 1.0)

        # 暗线/亮线筛选:
        # 2D 中, 暗线 (黑 on 白) 的 λ2 < 0 (主方向曲率为负)
        # 亮线 (白 on 黑) 的 λ2 > 0
        if black_ridges:
            line_mask = lambda2 < 0
        else:
            line_mask = lambda2 > 0

        # Frangi 响应
        resp = np.exp(-(Rb * Rb) / (2.0 * beta * beta)) * \
               (1.0 - np.exp(-(S * S) / (2.0 * c * c)))
        resp = np.where(line_mask, resp, 0.0)

        # 多尺度: 取 max
        resp_max = np.maximum(resp_max, resp)

    return resp_max


def _connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8 邻域连通域标记（等价 scipy.ndimage.label，纯 numpy 实现）。

    用 BFS 扫描，返回 (labeled, n_features)。labeled 中背景为 0，
    各连通域编号从 1 到 n_features。用于替换 scipy 依赖，减小打包体积。
    """
    H, W = mask.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    visited = np.zeros((H, W), dtype=bool)
    label_id = 0
    # 8 邻域偏移
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1)]
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue
            label_id += 1
            # BFS 扩展当前连通域
            q = deque([(r, c)])
            visited[r, c] = True
            labeled[r, c] = label_id
            while q:
                cr, cc = q.popleft()
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            labeled[nr, nc] = label_id
                            q.append((nr, nc))
    return labeled, label_id


def _shape_filter_line_mask(
    mask: np.ndarray,
    *,
    aspect_min: float = 2.5,
    min_size: int = 4,
) -> np.ndarray:
    """
    对 line_mask 做形状过滤: 只保留"线状"连通域。

    剔除条件 (满足任一即剔除):
      1. 连通域面积 < min_size (孤立噪点)
      2. 长宽比 < aspect_min (块状/斑驳, 而非线条)
         长宽比 = max(h, w) / max(min(h, w), 1)

    主要用于 hybrid 模式: 过滤 local contrast 在脸颊阴影的误识别
    (阴影块状 vs 胡须线状, 长宽比差异明显)

    依赖: 自实现的 _connected_components（纯 numpy，替代 scipy.ndimage.label）
    """
    if not mask.any():
        return mask

    # 8 邻域连通
    labeled, n_features = _connected_components(mask)
    if n_features == 0:
        return mask

    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n_features + 1):
        comp = (labeled == i)
        size = int(comp.sum())
        if size < min_size:
            continue
        ys, xs = np.where(comp)
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        ratio = max(h, w) / max(min(h, w), 1)
        if ratio >= aspect_min:
            out |= comp
    return out


def _detect_line_mask_frangi(
    img_array: np.ndarray,
    *,
    sigmas: tuple = (1.0, 2.0, 4.0),
    threshold_percentile: float = 92.0,
    beta: float = 0.5,
    black_ridges: bool = True,
) -> np.ndarray:
    """
    #4 Frangi 线条检测 (多尺度 Hessian + vesselness 响应 + 阈值)。

    流程:
      1. 多尺度 Frangi 响应
      2. 取高分位数为阈值
      3. 二值化

    优点 (相比 Sobel/Canny):
      - 显式区分"线状 vs 块状" (用特征值比 Rb)
      - 多尺度天然适配粗细
      - 对"管状/脊状"结构响应强

    缺点:
      - 比 Canny 慢 (~2-3x, 自己实现无 opencv contrib)
      - 对噪点也响应, 阈值要小心
    """
    resp = _frangi_response_2d(
        img_array, sigmas=sigmas, beta=beta, black_ridges=black_ridges,
    )
    threshold = float(np.percentile(resp, threshold_percentile))
    return resp > threshold


def _convolve2d_same(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """same 模式 2D 卷积 (reflect padding, 纯 numpy, 无新依赖)"""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="reflect")
    H, W = arr.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            k_ij = kernel[i, j]
            if k_ij == 0:
                continue
            out += k_ij * padded[i:i + H, j:j + W]
    return out


def _find_line_color_adaptive(
    block: np.ndarray,
    pal_lab: np.ndarray,
    pal_codes: np.ndarray,
    palette: Palette,
    line_mask_in_block: np.ndarray,
) -> "Optional[PaletteColor]":
    """
    块内"线条像素"群中,找和"外部主色"ΔE 最大的像素,
    取色板最近邻作为线条色。

    关键: 线条色 = 块内最显著的对比色 (深线用深色, 浅线用浅色, 自适应)
    """
    if not line_mask_in_block.any():
        return None
    # 找外部 4 邻域色号 (用块边缘像素)
    H, W = line_mask_in_block.shape
    edge_pixels = np.concatenate([
        line_mask_in_block[0, :], line_mask_in_block[-1, :],
        line_mask_in_block[:, 0], line_mask_in_block[:, -1],
    ])
    # 块内"非线条"像素作为"主体"
    bg_mask = ~line_mask_in_block
    if not bg_mask.any():
        return None
    bg_pixels = block.reshape(H, W, 3)[bg_mask]
    bg_lab = _rgb_to_lab_batch(bg_pixels)
    diff_bg = bg_lab[:, None, :] - pal_lab[None, :, :]
    d2_bg = np.einsum("npd,npd->np", diff_bg, diff_bg)
    bg_votes = np.bincount(np.argmin(d2_bg, axis=1), minlength=len(pal_codes))
    bg_idx = int(np.argmax(bg_votes))
    bg_lab_arr = pal_lab[bg_idx]

    # 找线条像素群中"和 bg_lab ΔE 最大"的像素
    line_pixels = block.reshape(H, W, 3)[line_mask_in_block]
    line_labs = _rgb_to_lab_batch(line_pixels)
    # ΔE
    diff = line_labs - bg_lab_arr
    d2 = np.einsum("nd,nd->n", diff, diff)
    far_local = int(np.argmax(d2))
    far_lab = line_labs[far_local]
    # 找色板最近邻
    diff_pal = pal_lab - far_lab
    d2_pal = np.einsum("pd,pd->p", diff_pal, diff_pal)
    return palette.by_code(pal_codes[int(np.argmin(d2_pal))])


def _full_image_edge_strength(img_array: np.ndarray) -> np.ndarray:
    """
    对整图做简易 Sobel 边缘检测，返回每像素的边缘强度 (H×W float)。
    实现: 用相邻像素差分的最大值近似梯度幅值
    """
    gray = (0.299 * img_array[:, :, 0].astype(np.float32) +
            0.587 * img_array[:, :, 1].astype(np.float32) +
            0.114 * img_array[:, :, 2].astype(np.float32))
    H, W = gray.shape
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    dx[:, :-1] = np.abs(gray[:, 1:] - gray[:, :-1])
    dy[:-1, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return np.maximum(dx, dy)


def _block_center_pixel_index(n: int) -> int:
    """块内中心像素在 1D 数组中的索引（假设块近似方形）"""
    side = int(round(n ** 0.5))
    if side * side != n:
        return n // 2
    ch = cw = side // 2
    return ch * side + cw


def _find_outline_color(
    block: np.ndarray,
    pal_lab: np.ndarray,
    pal_codes: np.ndarray,
    palette: Palette,
    block_edge_pct: float = 0.0,
    *,
    main_l_threshold: float = 80.0,
    outline_l_threshold: float = 70.0,
    center_l_threshold: float = 85.0,
    min_outline_pct: float = 0.01,
    min_edge_pct: float = 0.025,
    contrast_threshold: float = 10.0,
) -> Optional[PaletteColor]:
    """
    检测块内"深色像素 / 强边缘像素"，
    若满足任一阈值条件且主色是浅色 + 与深色 ΔE ≥ 阈值，
    返回深色色板色号，否则返回 None。

    判定条件（OR）:
      ★ 块"中心像素" L < center_l_threshold (默认 85) — **精准定位，避免糊片**
        1 像素宽的描边只覆盖"中心被描边穿过"的块
      - 深色像素占比 ≥ min_outline_pct (默认 1%) — 兜底
      - 强边缘像素占比 ≥ min_edge_pct (默认 2.5%) — 兜底

    "糊片"修复说明:
      旧逻辑: 块内"任何深色像素" → 整块变黑
      → 1 像素描边横跨 N 个块，每块独立判定，导致 N 行都变黑
      新逻辑: 只有描边"穿过块中心"的块才变黑
      → 描边在网格上"细到 1 像素宽"
    """
    N = len(block)
    L = _block_luminance(block)
    center_idx = _block_center_pixel_index(N)
    center_l = float(L[center_idx])
    outline_mask = L < outline_l_threshold
    outline_pct = float(outline_mask.sum()) / N

    # 判定 1: 块中心像素 L 够低 → 精准定位
    # 判定 2/3: 占比阈值（兜底，处理描边粗到跨块的情况）
    passes_center = center_l < center_l_threshold
    passes_pct = outline_pct >= min_outline_pct or block_edge_pct >= min_edge_pct
    if not (passes_center or passes_pct):
        return None

    # 块主色（投票）
    labs = _rgb_to_lab_batch(block)
    diff = labs[:, None, :] - pal_lab[None, :, :]
    d2 = np.einsum("npd,npd->np", diff, diff)
    idx = np.argmin(d2, axis=1)
    votes = np.bincount(idx, minlength=len(pal_codes))
    main_idx = int(np.argmax(votes))
    main_lab = pal_lab[main_idx]

    # 主色本身就是深色（不需要覆盖）
    if main_lab[0] < main_l_threshold:
        return None

    # 找代表色：优先用"中心像素"（精准），其次用"最深像素"（兜底）
    if passes_center:
        target_lab = labs[center_idx]
    elif outline_pct > 0:
        outline_labs = labs[outline_mask]
        darkest_local = int(np.argmin(outline_labs[:, 0]))
        target_lab = outline_labs[darkest_local]
    else:
        target_lab = labs[int(np.argmin(L))]

    diff_out = pal_lab - target_lab
    d2_out = np.einsum("pd,pd->p", diff_out, diff_out)
    outline_idx = int(np.argmin(d2_out))

    contrast = float(np.sqrt(np.sum(
        (pal_lab[main_idx] - pal_lab[outline_idx]) ** 2
    )))
    if contrast < contrast_threshold:
        return None
    return palette.by_code(pal_codes[outline_idx])


def _block_vote_quantize(
    img_array: np.ndarray,
    palette: Palette,
    grid_w: int,
    grid_h: int,
    *,
    detail_mode: bool = False,
    min_sub_pct: float = 0.06,
    min_contrast: float = 18.0,
    key_features: Optional[Dict[str, List[float]]] = None,
    outline_preserve: bool = False,
    edge_threshold: float = 30.0,
    outline_min_pct: float = 0.015,
    outline_min_edge_pct: float = 0.04,
    outline_center_l: float = 85.0,
    line_mask: Optional[np.ndarray] = None,
    line_min_pct: float = 0.05,
    min_line_count: int = 1,
    min_line_contrast: float = 30.0,
) -> List[List[PaletteColor]]:
    """
    块投票量化，三层决策（优先级从高到低）:

    1. 轮廓保护（outline_preserve=True 时）:
       同时检测"深色像素"和"强边缘像素"，任一满足阈值即保留

    2. detail_mode:
       投票 + 次色覆盖。阈值按区域（core/face/background）动态调整

    3. 朴素投票:
       每块取频次最高色号

    线条识别两个判据 (任一满足即当线条块):
      A. 块内 line_mask 占比 >= line_min_pct (默认 5%, 适用于粗描边)
      B. 块内 line_mask 至少 min_line_count 像素 (默认 1) 且
         最暗线条像素 vs 块内主体 ΔE >= min_line_contrast (默认 30,
         适用于单根细线/胡须等稀疏细线)
    """
    H, W, _ = img_array.shape
    cell_w = W / grid_w
    cell_h = H / grid_h

    pal_lab, pal_codes = _palette_lab_arrays(palette)

    # 整图边缘强度（仅 outline_preserve 启用时计算）
    edge_strength: Optional[np.ndarray] = None
    edge_mask: Optional[np.ndarray] = None
    if outline_preserve:
        edge_strength = _full_image_edge_strength(img_array)
        edge_mask = edge_strength > edge_threshold

    grid: List[List[PaletteColor]] = [[None] * grid_w for _ in range(grid_h)]

    for r in range(grid_h):
        y0 = int(round(r * cell_h))
        y1 = int(round((r + 1) * cell_h))
        y1 = max(y1, y0 + 1)
        y1 = min(y1, H)
        for c in range(grid_w):
            x0 = int(round(c * cell_w))
            x1 = int(round((c + 1) * cell_w))
            x1 = max(x1, x0 + 1)
            x1 = min(x1, W)

            block = img_array[y0:y1, x0:x1].reshape(-1, 3)
            labs = _rgb_to_lab_batch(block)

            # ===== 第 0 层: 纹理线条 (独立于深浅) =====
            if line_mask is not None:
                block_line = line_mask[y0:y1, x0:x1]
                block_line_count = int(block_line.sum())
                block_line_pct = block_line_count / block_line.size

                # 判据 A: 块内线条像素占比 >= line_min_pct (默认 5%)
                # 判据 B: 块内 line_mask >= min_line_count (默认 1 像素) 且
                #         最暗像素 vs 块内主体 ΔE >= min_line_contrast (默认 30)
                # 判据 B 用来救"单根细线/胡须" — 占比不够但对比度够的细线
                use_line_block = False
                if block_line_pct >= line_min_pct:
                    use_line_block = True
                elif block_line_count >= min_line_count and min_line_contrast > 0:
                    # 算块内"最暗 vs 主体"的 ΔE
                    block_2d_for_check = img_array[y0:y1, x0:x1]
                    H_b, W_b = block_2d_for_check.shape[:2]
                    bg_mask = ~block_line
                    if bg_mask.any() and block_line.any():
                        bg_pixels = block_2d_for_check[bg_mask]
                        line_pixels = block_2d_for_check[block_line]
                        bg_lab = _rgb_to_lab_batch(bg_pixels)
                        line_lab = _rgb_to_lab_batch(line_pixels)
                        bg_mean = bg_lab.mean(axis=0)
                        # 取"和 bg_mean ΔE 最大"的线条像素
                        diff = line_lab - bg_mean
                        d2 = np.einsum("nd,nd->n", diff, diff)
                        max_contrast = float(np.sqrt(d2.max()))
                        if max_contrast >= min_line_contrast:
                            use_line_block = True

                if use_line_block:
                    # 找"自适应线条色": 块内和外部主体 ΔE 最大的像素
                    block_2d = img_array[y0:y1, x0:x1]
                    line_color = _find_line_color_adaptive(
                        block_2d, pal_lab, pal_codes, palette,
                        block_line,
                    )
                    if line_color is not None:
                        grid[r][c] = line_color
                        continue

            # ===== 第 1 层: 轮廓保护 =====
            if outline_preserve and edge_mask is not None:
                block_edge = edge_mask[y0:y1, x0:x1]
                block_edge_pct = float(block_edge.sum()) / block_edge.size
                outline_color = _find_outline_color(
                    block, pal_lab, pal_codes, palette,
                    block_edge_pct=block_edge_pct,
                    min_outline_pct=outline_min_pct,
                    min_edge_pct=outline_min_edge_pct,
                    center_l_threshold=outline_center_l,
                )
                if outline_color is not None:
                    grid[r][c] = outline_color
                    continue

            # ===== 正常投票 =====
            diff = labs[:, None, :] - pal_lab[None, :, :]
            d2 = np.einsum("npd,npd->np", diff, diff)
            idx = np.argmin(d2, axis=1)
            N = len(idx)
            votes = np.bincount(idx, minlength=len(pal_codes))
            order = np.argsort(-votes)
            main_idx = int(order[0])
            if not detail_mode or len(order) < 2:
                grid[r][c] = palette.by_code(pal_codes[main_idx])
                continue

            # ===== 第 2 层: 细节覆盖（按区域分级） =====
            sub_idx = int(order[1])
            sub_pct = float(votes[sub_idx]) / N
            contrast = float(np.sqrt(np.sum(
                (pal_lab[main_idx] - pal_lab[sub_idx]) ** 2
            )))

            region = _classify_region(r, c, grid_w, grid_h, key_features)
            th_pct, th_contrast = _region_thresholds(region, min_sub_pct, min_contrast)

            if sub_pct >= th_pct and contrast >= th_contrast:
                grid[r][c] = palette.by_code(pal_codes[sub_idx])
            else:
                grid[r][c] = palette.by_code(pal_codes[main_idx])

    return grid


# =============================================================================
# 2. 小色块清理（连通区域法）
# =============================================================================
def _remove_small_islands(
    grid: List[List[PaletteColor]],
    min_size: int = 3,
    protected_codes: Optional[set] = None,
    min_contrast: Optional[float] = None,
) -> Tuple[List[List[PaletteColor]], int]:
    """
    把 <min_size 像素的连通区域合并到 4 邻域主色。

    双重判定 (任一命中则保留不清理):
      1. protected_codes 白名单: 该色号的小连通区不清理
      2. min_contrast 色差阈值: 小色块与外部主色 ΔE >= 阈值,说明是关键特征
         → 不清理 (例如孤立黑色像素,周围是白色,但 ΔE 很大,说明是描边)

    旧逻辑只判定大小,导致孤立的描边像素被误清;
    新逻辑综合大小+色差,既清杂点又保描边。
    """
    H = len(grid)
    W = len(grid[0])
    code_to_obj: Dict[str, PaletteColor] = {}
    for row in grid:
        for c in row:
            code_to_obj.setdefault(c.code, c)

    visited = [[False] * W for _ in range(H)]
    removed = 0

    for r0 in range(H):
        for c0 in range(W):
            if visited[r0][c0]:
                continue
            # BFS 找连通区（同色号 + 4 邻接）
            queue = deque([(r0, c0)])
            visited[r0][c0] = True
            cells: List[Tuple[int, int]] = []
            code = grid[r0][c0].code
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] \
                            and grid[nr][nc].code == code:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

            if len(cells) >= min_size:
                continue

            # 找外部邻域的色号分布
            cells_set = set(cells)
            external: Counter = Counter()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cells_set:
                        external[grid[nr][nc].code] += 1
            if not external:
                continue
            target_code, _ = external.most_common(1)[0]
            target_obj = code_to_obj[target_code]

            # 判定 1: protected_codes 白名单
            if protected_codes and code in protected_codes and len(cells) >= 2:
                continue

            # 判定 2: 色差阈值 (ΔE >= min_contrast → 视为关键特征,不清理)
            if min_contrast is not None:
                from .palette_loader import delta_e76
                cell_lab = grid[r0][c0].lab
                dE = delta_e76(cell_lab, target_obj.lab)
                if dE >= min_contrast:
                    continue

            # 合并
            for r, c in cells:
                grid[r][c] = target_obj
            removed += len(cells)
    return grid, removed


# =============================================================================
# 3. 主入口
# =============================================================================
def _rgb_uint8_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    from .palette_loader import rgb_to_lab
    return rgb_to_lab(r, g, b)


def _downsample_vote(
    high_grid: List[List[PaletteColor]],
    factor: int,
) -> List[List[PaletteColor]]:
    """
    频次最高下采样: factor×factor 子区域里投票取众数。
    - 不会糊 (多数派胜出)
    - 但细线 (1-2 子格) 可能丢 (不够多数)
    """
    from collections import Counter
    H = len(high_grid)
    W = len(high_grid[0])
    new_H = H // factor
    new_W = W // factor
    out: List[List[PaletteColor]] = [[None] * new_W for _ in range(new_H)]
    for r in range(new_H):
        for c in range(new_W):
            cells: List[PaletteColor] = []
            for dr in range(factor):
                rr = r * factor + dr
                if rr >= H:
                    break
                for dc in range(factor):
                    cc = c * factor + dc
                    if cc >= W:
                        break
                    cells.append(high_grid[rr][cc])
            if not cells:
                continue
            votes = Counter(x.code for x in cells)
            best_code, _ = votes.most_common(1)[0]
            out[r][c] = next(x for x in cells if x.code == best_code)
    return out


def _downsample_darkest(
    high_grid: List[List[PaletteColor]],
    factor: int,
    min_dark_count: int = 2,
    dark_l_threshold: float = 80.0,
) -> List[List[PaletteColor]]:
    """
    下采样策略: 按"最暗优先" + 数量阈值。

    流程:
      1. 取 factor×factor 子区域的所有子格
      2. 统计"深色子格"数 (L < dark_l_threshold)
      3. 若 ≥ min_dark_count: 子区域视为"深色区",选最暗子格作为输出
         否则: 走投票 (少数深色被忽略)

    min_dark_count 调参:
      1 = 纯 OR (最激进,容易糊,比如眼睛内部被染黑)
      2 = 折中 (推荐): 1 根细线穿过 (2 子格) 保留,孤立深色被过滤
      3 = 保守: 仅当密集描边时才覆盖,细线可能丢失
    """
    from collections import Counter
    H = len(high_grid)
    W = len(high_grid[0])
    new_H = H // factor
    new_W = W // factor
    out: List[List[PaletteColor]] = [[None] * new_W for _ in range(new_H)]
    for r in range(new_H):
        for c in range(new_W):
            cells: List[PaletteColor] = []
            for dr in range(factor):
                rr = r * factor + dr
                if rr >= H:
                    break
                for dc in range(factor):
                    cc = c * factor + dc
                    if cc >= W:
                        break
                    cells.append(high_grid[rr][cc])
            if not cells:
                continue
            dark_cells = [x for x in cells if x.lab[0] < dark_l_threshold]
            if len(dark_cells) >= min_dark_count:
                out[r][c] = min(dark_cells, key=lambda x: x.lab[0])
            else:
                # 走投票: 多数派胜出 (少数深色被忽略,避免糊)
                votes = Counter(x.code for x in cells)
                best_code, _ = votes.most_common(1)[0]
                out[r][c] = next(x for x in cells if x.code == best_code)
    return out


def _downsample_line_aware(
    high_grid: List[List[PaletteColor]],
    line_mask_high: np.ndarray,
    factor: int,
    img_array: np.ndarray,
    palette: "Palette",
    pal_lab: np.ndarray,
    pal_codes: np.ndarray,
    *,
    min_line_pixels: int = 1,
) -> List[List[PaletteColor]]:
    """
    下采样策略: 线条保护 + 投票 (line_aware)。

    核心思路:
      - 下采样时, 4×factor 子区域里只要有 >= min_line_pixels 个 line_mask 像素
        → 整块视为"线条块", 用"线条色"覆盖 vote 结果
      - 否则走 vote (多数派胜出)

    优势 vs 纯 vote:
      - 线条不会被周围主体色"投掉"
      - 单根胡须 (1 像素) 也能在 4×4 子格里被救回来 (line_mask 直接传达)

    优势 vs 纯 darkest:
      - 不会把孤立深色噪点错误染黑 (min_line_pixels 兜底)
      - 不会把眼睛内部染糊 (走 vote 路径)

    线条色选择:
      - 从 4×factor 子区域里所有 line_mask=1 的原图像素里
        选"和周围主体色 ΔE 最大"的颜色
      - 取色板最近邻
    """
    H = len(high_grid)
    W = len(high_grid[0])
    new_H = H // factor
    new_W = W // factor
    out: List[List[PaletteColor]] = [[None] * new_W for _ in range(new_H)]

    # line_mask 在高分辨率网格上, 我们要按高分辨率网格切
    for r in range(new_H):
        for c in range(new_W):
            # 子区域在高分辨率上的范围
            y0 = r * factor
            y1 = min(y0 + factor, H)
            x0 = c * factor
            x1 = min(x0 + factor, W)

            # 取 line_mask 块
            mask_block = line_mask_high[y0:y1, x0:x1]
            n_line = int(mask_block.sum())

            if n_line >= min_line_pixels:
                # 线条块: 从原图对应区域选"线条色"
                # 原图也要按比例缩放: 高分辨率 y0:y1 对应原图 y0*scale : y1*scale
                scale_y = img_array.shape[0] / H
                scale_x = img_array.shape[1] / W
                iy0 = int(y0 * scale_y)
                iy1 = int(y1 * scale_y)
                ix0 = int(x0 * scale_x)
                ix1 = int(x1 * scale_x)
                block_img = img_array[iy0:iy1, ix0:ix1]

                # 缩放 line_mask 到 block 大小
                from PIL import Image as _PILImage
                m_pil = _PILImage.fromarray(mask_block.astype(np.uint8) * 255)
                m_resized = np.asarray(
                    m_pil.resize((ix1 - ix0, iy1 - iy0), _PILImage.NEAREST),
                    dtype=bool,
                )
                if m_resized.shape != block_img.shape[:2]:
                    # 兜底: 跳过这步
                    pass

                line_color = _find_line_color_adaptive(
                    block_img, pal_lab, pal_codes, palette, m_resized,
                )
                if line_color is not None:
                    out[r][c] = line_color
                    continue

            # 否则走 vote
            from collections import Counter
            cells: List[PaletteColor] = []
            for rr in range(y0, y1):
                for cc in range(x0, x1):
                    cells.append(high_grid[rr][cc])
            if not cells:
                continue
            votes = Counter(x.code for x in cells)
            best_code, _ = votes.most_common(1)[0]
            out[r][c] = next(x for x in cells if x.code == best_code)
    return out


def quantize(
    image: Image.Image | str,
    *,
    grid_w: int = 58,
    grid_h: int = 58,
    palette: Optional[Palette] = None,
    min_island_size: int = 3,
    cleanup: bool = True,
    detail_mode: bool = False,
    min_sub_pct: float = 0.06,
    min_contrast: float = 18.0,
    key_features: Optional[Dict[str, List[float]]] = None,
    outline_preserve: bool = False,
    protected_codes: Optional[set] = None,
    outline_min_pct: float = 0.015,
    outline_min_edge_pct: float = 0.04,
    outline_center_l: float = 85.0,
    supersample: int = 1,
    ss_min_dark_count: int = 2,
    ss_strategy: str = "darkest",   # "darkest" | "vote" | "line_aware"
    ss_line_min_pixels: int = 1,
    cleanup_min_contrast: Optional[float] = None,
    line_detect: bool = False,
    line_method: str = "local",   # "local" | "multiscale" | "canny"
    line_threshold_percentile: float = 88.0,
    line_min_pct: float = 0.05,
    min_line_count: int = 1,
    min_line_contrast: float = 30.0,
    line_aspect_min: float = 2.5,
    line_min_region_size: int = 4,
    pixelize_method: str = "none",   # none | area | dpid | pixeloe
    pixelize_params: Optional[dict] = None,
) -> QuantizeResult:
    """
    主入口：把图片量化到 grid_w x grid_h 的色板网格。

    三层决策（优先级高→低）:
      1. 轮廓保护 (outline_preserve=True) — 强制保留深色/边缘像素群
      2. 细节优先 (detail_mode=True) — 投票 + 次色覆盖（按区域分级）
      3. 朴素投票 — 频次最高

    :param image: PIL.Image 或文件路径
    :param grid_w: 目标网格宽
    :param grid_h: 目标网格高
    :param palette: 色板，None 则自动加载 MARD
    :param min_island_size: 小于该像素数的连通区会被合并（默认 3）
                            设 1 可保留更多细线
    :param cleanup: 是否做小色块清理
    :param detail_mode: 是否启用细节优先
    :param min_sub_pct: 细节模式下，第二名色号需要达到的最小占比（按区域分级）
    :param min_contrast: 细节模式下，主色与次色的最小 Lab 距离（按区域分级）
    :param key_features: 关键特征 bbox 字典 {name: [x1,y1,x2,y2] 归一化坐标}
    :param outline_preserve: 是否强制保留深色/边缘像素群（轮廓/线条/瞳孔等）
    :param protected_codes: 不被清理的色号集合（如 {"H7","H16"} 保护黑色描边）
    :param supersample: 超采样倍率。
        >1 时先量化到 (grid_w*factor) × (grid_h*factor) 高分辨率，
        再下采样回原尺寸。推荐 4。
    :param ss_min_dark_count: 超采样下采样的"深色子格数量阈值"(仅 darkest 策略)。
        子区域里至少 N 个深色子格才视为深色区,否则走投票。
        1=纯 OR (最糊), 2=折中 (推荐), 3+=保守 (细线可能丢)
    :param ss_strategy: 超采样下采样策略。
        "darkest"    = 最暗优先 (适合保留描边,可能糊)
        "vote"       = 频次最高 (不会糊,可能丢细线)
        "line_aware" = 线条保护 + 投票 (默认推荐, 需 line_detect=True)
                      4×4 子区域里 ≥ ss_line_min_pixels 个 line_mask 像素
                      → 整块用线条色; 否则走 vote
    :param ss_line_min_pixels: line_aware 策略的最小线条像素数 (默认 1)
    :param cleanup_min_contrast: 小色块清理的色差阈值。
        若小色块与外部主色 ΔE >= 阈值,视为关键特征,保留不清理。
        例如孤立黑色像素周围白色,ΔE ~ 80,被保留 → 描边不丢。
        米色/灰色杂点 ΔE ~ 10,被清理 → 杂点消除。
        None = 不启用色差判定 (仅按大小清理)
    :param line_detect: 是否启用"独立于深浅的纹理线条检测"
    :param line_method: 线条检测方法
        "local" = #5 局部对比度 (简单快速)
        "multiscale" = #2 多尺度 Sobel + NMS (精准但慢)
    :param line_threshold_percentile: 局部对比度策略的分位数阈值
        (高于此百分位的对比度像素视为线条,默认 88, 仅 local 模式有效)
    :param line_min_pct: 块内"线条像素"占比阈值
        (>= 阈值则该块强制用线条色,默认 5%)
    :param min_line_count: 块内"线条像素"最少个数 (用于细线兜底判据,默认 1)
    :param min_line_contrast: 细线兜底判据的 ΔE 阈值
        (块内至少 min_line_count 个线条像素,且最暗线条 vs 主体 ΔE > 此值,默认 30。
         设为 0 关闭细线兜底,只用 line_min_pct 判据)
    :param line_aspect_min: 连通域长宽比下限 (用于 hybrid 模式形状过滤,默认 2.5)
    :param line_min_region_size: 连通域最小像素数 (用于 hybrid 模式形状过滤,默认 4)
    :param pixelize_method: 像素化前置方法
        "none"    = 不启用（默认，走原生块投票）
        "area"    = cv2.INTER_AREA 面积插值
        "dpid"    = DPID 细节保留下采样（参数 lam）
        "pixeloe" = PixelOE 对比度感知像素化（参数 mode/thickness/contrast/...）
    :param pixelize_params: 像素化方法参数字典，键与方法对应：
        area: 无参数
        dpid: lam(float, 默认 1.0)
        pixeloe: mode(str), thickness(int), contrast(float), saturation(float),
                 colors(int|None)
    """
    if isinstance(image, (str, Path, bytes)):
        img = Image.open(image)
    else:
        img = image
    if img.mode != "RGB":
        img = img.convert("RGB")

    if palette is None:
        palette = load_palette()

    arr = np.asarray(img, dtype=np.uint8)

    # 像素化前置方法：直接把原图下采样到网格分辨率（每块 1 像素），
    # 完全跳过超采样/投票/次色覆盖/线条检测等细节链路。
    # 只有 SuperSample（pixelize_method="none"）才走超采样 + 投票。
    pm = (pixelize_method or "none").lower()
    if pm not in ("none", ""):
        arr = _apply_pixelize_preprocess(
            arr, grid_w, grid_h, 1, pm, pixelize_params,
        )
        # 下采样后每块 1 像素，强制走 else 分支（不超采样、不投票细节链路）
        supersample = 1

    # Canny 需要 opencv：若指定但环境没有，自动降级到 multiscale + WARN
    if line_detect and line_method == "canny":
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("[WARN] --line-method canny 需要 opencv-python，但当前环境没装。\n"
                  "       自动降级到 multiscale (无需 opencv)。\n"
                  "       想用 Canny 请: pip install opencv-python 或 conda install -n <env> opencv")
            line_method = "multiscale"

    if supersample > 1:
        # 超采样: 先量化到高分辨率,再下采样
        high_w = grid_w * supersample
        high_h = grid_h * supersample
        # 纹理线条检测: 在高分辨率上做更精准
        line_mask_h: Optional[np.ndarray] = None
        if line_detect:
            if line_method == "multiscale":
                line_mask_h = _detect_line_mask_multiscale_sobel(arr)
            elif line_method == "canny":
                line_mask_h = _detect_line_mask_canny_opencv(arr)
            elif line_method == "frangi":
                line_mask_h = _detect_line_mask_frangi(arr)
            elif line_method == "hybrid":
                # Canny (粗线/外轮廓) ∪ Local(形状过滤后, 细线/胡须)
                # 互补: Canny 抓强边缘精准; Local 抓灰度差无模糊 -> 能救胡须
                # 形状过滤: 剔掉 local 在脸颊阴影的块状误识别
                m_canny = _detect_line_mask_canny_opencv(arr)
                m_local_raw = _detect_line_mask_local_contrast(
                    arr, threshold_percentile=line_threshold_percentile,
                )
                m_local = _shape_filter_line_mask(
                    m_local_raw,
                    aspect_min=line_aspect_min,
                    min_size=line_min_region_size,
                )
                line_mask_h = m_canny | m_local
            else:
                line_mask_h = _detect_line_mask_local_contrast(
                    arr, threshold_percentile=line_threshold_percentile,
                )
        high_grid = _block_vote_quantize(
            arr, palette, high_w, high_h,
            detail_mode=detail_mode,
            min_sub_pct=min_sub_pct,
            min_contrast=min_contrast,
            key_features=None,  # 高分辨率下关键特征 bbox 太小,先关
            outline_preserve=False,
            outline_min_pct=outline_min_pct,
            outline_min_edge_pct=outline_min_edge_pct,
            line_mask=line_mask_h,
            line_min_pct=line_min_pct,
            min_line_count=min_line_count,
            min_line_contrast=min_line_contrast,
        )
        if ss_strategy == "vote":
            grid = _downsample_vote(high_grid, supersample)
        elif ss_strategy == "line_aware":
            if line_mask_h is None:
                print("[WARN] ss_strategy=line_aware 需要 line_detect=True，自动降级到 vote")
                grid = _downsample_vote(high_grid, supersample)
            else:
                # line_mask 从 2048 原图缩到 208 高分辨率网格
                from PIL import Image as _PILImage
                m_pil = _PILImage.fromarray(line_mask_h.astype(np.uint8) * 255)
                m_resized = np.asarray(
                    m_pil.resize((high_w, high_h), _PILImage.NEAREST),
                    dtype=bool,
                )
                # 构建 pal_lab / pal_codes
                pal_lab = np.array([c.lab for c in palette.colors], dtype=np.float32)
                pal_codes = np.array([c.code for c in palette.colors], dtype=object)
                grid = _downsample_line_aware(
                    high_grid, m_resized, supersample, arr,
                    palette, pal_lab, pal_codes,
                    min_line_pixels=ss_line_min_pixels,
                )
        else:  # "darkest"
            grid = _downsample_darkest(
                high_grid, supersample,
                min_dark_count=ss_min_dark_count,
            )
    else:
        # 纹理线条检测 (独立于深浅)
        line_mask: Optional[np.ndarray] = None
        if line_detect:
            if line_method == "multiscale":
                line_mask = _detect_line_mask_multiscale_sobel(arr)
            elif line_method == "canny":
                line_mask = _detect_line_mask_canny_opencv(arr)
            elif line_method == "frangi":
                line_mask = _detect_line_mask_frangi(arr)
            elif line_method == "hybrid":
                m_canny = _detect_line_mask_canny_opencv(arr)
                m_local_raw = _detect_line_mask_local_contrast(
                    arr, threshold_percentile=line_threshold_percentile,
                )
                m_local = _shape_filter_line_mask(
                    m_local_raw,
                    aspect_min=line_aspect_min,
                    min_size=line_min_region_size,
                )
                line_mask = m_canny | m_local
            else:
                line_mask = _detect_line_mask_local_contrast(
                    arr, threshold_percentile=line_threshold_percentile,
                )
        grid = _block_vote_quantize(
            arr, palette, grid_w, grid_h,
            detail_mode=detail_mode,
            min_sub_pct=min_sub_pct,
            min_contrast=min_contrast,
            key_features=key_features,
            outline_preserve=outline_preserve,
            outline_min_pct=outline_min_pct,
            outline_min_edge_pct=outline_min_edge_pct,
            outline_center_l=outline_center_l,
            line_mask=line_mask,
            line_min_pct=line_min_pct,
            min_line_count=min_line_count,
            min_line_contrast=min_line_contrast,
        )

    removed = 0
    if cleanup:
        grid, removed = _remove_small_islands(
            grid, min_size=min_island_size,
            protected_codes=protected_codes,
            min_contrast=cleanup_min_contrast,
        )

    # 统计
    stats: Counter = Counter()
    for row in grid:
        for c in row:
            stats[c.code] += 1

    return QuantizeResult(
        grid=grid,
        stats=dict(stats),
        color_count=len(stats),
        total_cells=grid_w * grid_h,
        grid_w=grid_w,
        grid_h=grid_h,
        removed_islands=removed,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python quantize.py <image> [grid_w] [grid_h]")
        sys.exit(1)
    img_path = sys.argv[1]
    gw = int(sys.argv[2]) if len(sys.argv) > 2 else 58
    gh = int(sys.argv[3]) if len(sys.argv) > 3 else gw
    res = quantize(img_path, grid_w=gw, grid_h=gh)
    print(f"网格: {res.grid_w}x{res.grid_h} = {res.total_cells} 格")
    print(f"用色: {res.color_count} 种")
    print(f"清理小色块: {res.removed_islands} 格")
    print("\n色号用量 TOP 10:")
    for code, cnt in sorted(res.stats.items(), key=lambda x: -x[1])[:10]:
        pal_c = None
        # 找对象
        for row in res.grid:
            for c in row:
                if c.code == code:
                    pal_c = c
                    break
            if pal_c:
                break
        print(f"  {code:<5} {cnt:>4} 格  hex={pal_c.hex if pal_c else '?'}")
