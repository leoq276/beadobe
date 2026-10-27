"""
pixelize.py
三种像素化（下采样）方法，作为量化的前置预处理。

统一约定：
  - 输入: RGB uint8 numpy 数组 (H, W, 3)
  - 输出: RGB uint8 numpy 数组 (out_h, out_w, 3)
  - 每种方法独立、无副作用，供 quantize() 与 GUI 复用。

方法：
  1. area       —— cv2.INTER_AREA 面积插值（简单、快、经典）
  2. dpid       —— DPID（SIGGRAPH Asia 2016 "Rapid, Detail-Preserving
                   Image Downscaling"），核心参数 lambda 控制细节保留强度
  3. pixeloe    —— PixelOE（KohakuBlueleaf，对比度感知轮廓扩展 + 对比度感知
                   下采样），纯 numpy 实现，支持 mode / thickness / contrast /
                   saturation / palette 量化
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


# =============================================================================
# 1. INTER_AREA
# =============================================================================
def pixelize_area(
    img: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """cv2.INTER_AREA 面积插值下采样。

    INTER_AREA 对缩小图像是首选：每个输出像素取对应输入区域的平均，
    抗锯齿、无摩尔纹，速度快。细节会因平均而柔和。
    """
    if not _HAS_CV2:  # pragma: no cover
        raise RuntimeError("pixelize_area 需要 opencv-python")
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)


# =============================================================================
# 2. DPID
# =============================================================================
def pixelize_dpid(
    img: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    lam: float = 1.0,
) -> np.ndarray:
    """DPID 细节保持下采样（numpy 移植自官方 Matlab 版 dpid.m）。

    原理：
      1. 先算每个输出块的面积加权平均色（avgImage）
      2. 对每个输出像素，取 3x3 邻域的高斯加权平均作为"参考色"
      3. 块内每个像素的权重 f = (||pixel - 参考色|| / (255*sqrt(3)))^lam
         —— 与参考色差异越大的像素（细节/边缘）权重越大，从而保留细节
      4. 加权求和得到输出色

    :param lam: 细节保留强度。0 = 纯平均（等价 box 下采样）；越大越强调
        细节/边缘，越"锐"。论文默认 1.0。
    """
    H, W = img.shape[:2]
    ch = img.shape[2] if img.ndim == 3 else 1
    img = img.reshape(H, W, ch).astype(np.float64)

    pw = W / out_w
    ph = H / out_h

    # ---- 第 1 步：块面积加权平均 ----
    avg = np.zeros((out_h, out_w, ch), dtype=np.float64)
    for py in range(out_h):
        sy = py * ph
        ey = min((py + 1) * ph, H)
        syr = int(np.floor(sy))
        eyr = int(np.ceil(ey))
        for px in range(out_w):
            sx = px * pw
            ex = min((px + 1) * pw, W)
            sxr = int(np.floor(sx))
            exr = int(np.ceil(ex))
            acc = np.zeros(ch, dtype=np.float64)
            fsum = 0.0
            for iy in range(syr, eyr):
                fy = 1.0
                if iy < sy:
                    fy = 1.0 - (sy - iy)
                if (iy + 1) > ey:
                    fy = 1.0 - ((iy + 1) - ey)
                for ix in range(sxr, exr):
                    f = fy
                    if ix < sx:
                        f *= 1.0 - (sx - ix)
                    if (ix + 1) > ex:
                        f *= 1.0 - ((ix + 1) - ex)
                    acc += img[iy, ix] * f
                    fsum += f
            avg[py, px] = acc / fsum if fsum > 0 else acc

    # ---- 第 2 步：邻域高斯加权参考色 + 细节权重 ----
    # 3x3 高斯核 [1 2 1; 2 4 2; 1 2 1] / 16
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
    out = np.zeros((out_h, out_w, ch), dtype=np.float64)

    norm_const = np.sqrt(255.0 ** 2 * 3.0)

    for py in range(out_h):
        sy = py * ph
        ey = min((py + 1) * ph, H)
        syr = int(np.floor(sy))
        eyr = int(np.ceil(ey))
        for px in range(out_w):
            sx = px * pw
            ex = min((px + 1) * pw, W)
            sxr = int(np.floor(sx))
            exr = int(np.ceil(ex))

            # 3x3 邻域高斯平均（边界截断）
            ref = np.zeros(ch, dtype=np.float64)
            wsum = 0.0
            for ky in range(3):
                ny = py + ky - 1
                if ny < 0 or ny >= out_h:
                    continue
                for kx in range(3):
                    nx = px + kx - 1
                    if nx < 0 or nx >= out_w:
                        continue
                    ref += avg[ny, nx] * kernel[ky, kx]
                    wsum += kernel[ky, kx]
            if wsum > 0:
                ref /= wsum

            acc = np.zeros(ch, dtype=np.float64)
            fsum = 0.0
            for iy in range(syr, eyr):
                fy = 1.0
                if iy < sy:
                    fy = 1.0 - (sy - iy)
                if (iy + 1) > ey:
                    fy = 1.0 - ((iy + 1) - ey)
                for ix in range(sxr, exr):
                    if lam == 0.0:
                        f = 1.0
                    else:
                        d = float(np.linalg.norm(ref - img[iy, ix]))
                        f = (d / norm_const) ** lam
                    if ix < sx:
                        f *= 1.0 - (sx - ix)
                    if (ix + 1) > ex:
                        f *= 1.0 - ((ix + 1) - ex)
                    f *= fy
                    acc += img[iy, ix] * f
                    fsum += f
            if fsum > 0:
                out[py, px] = acc / fsum
            else:
                out[py, px] = ref

    out = np.clip(out, 0, 255).astype(np.uint8)
    if ch == 1:
        out = out[:, :, 0]
    return out


# =============================================================================
# 3. PixelOE（纯 numpy 实现，忠实移植官方 torch 管线）
#
# 约定：内部用 numpy float32 数组，布局 (B, C, H, W)，值域 [0,1]。
# torch 的 F.unfold / F.fold / cdist / sigmoid 均用 numpy 等价实现。
# =============================================================================
def _to_tensor(img: np.ndarray):
    # (H, W, 3) uint8 -> (1, 3, H, W) float32 [0,1]
    t = img.astype(np.float32) / 255.0
    return t.transpose(2, 0, 1)[None, ...]


def _from_tensor(t) -> np.ndarray:
    t = np.clip(t, 0, 1)[0].transpose(1, 2, 0)
    return np.round(t * 255.0).astype(np.uint8)


def _rgb_to_lab(img):
    """(B,3,H,W) [0,1] RGB -> Lab（L 0~100, a/b 约 -128~127）。"""
    rgb = np.clip(img, 0, 1)
    t = 0.04045
    lin = np.where(rgb > t, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    eps = 0.008856
    kap = 7.787
    off = 16 / 116
    fx = np.where(x > eps, np.clip(x, 0, None) ** (1 / 3), kap * x + off)
    fy = np.where(y > eps, np.clip(y, 0, None) ** (1 / 3), kap * y + off)
    fz = np.where(z > eps, np.clip(z, 0, None) ** (1 / 3), kap * z + off)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.concatenate([L, a, b], axis=1)


def _lab_to_rgb(lab):
    """Lab -> (B,3,H,W) [0,1] RGB。"""
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    eps = 0.008856
    kap = 7.787
    off = 16 / 116
    xr = np.where(fx ** 3 > eps, fx ** 3, (fx - off) / kap)
    yr = np.where(L > 7.9996, fy ** 3, L / 903.3)
    zr = np.where(fz ** 3 > eps, fz ** 3, (fz - off) / kap)
    x = xr * 0.95047
    y = yr
    z = zr * 1.08883
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb = np.concatenate([r, g, b], axis=1)
    t = 0.0031308
    lin2srgb = np.where(rgb > t,
                        1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055,
                        rgb * 12.92)
    return np.clip(lin2srgb, 0, 1)


def _replicate_pad(x, pad_h, pad_w):
    """等价 torch F.pad(mode='replicate')，边界像素复制填充。"""
    return np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                  mode="edge")


def _unfold_patches(x, kernel, stride):
    """等价 torch F.unfold(x, kernel_size=kernel, stride=stride, padding=0)。

    返回 (B, C*k*k, L)，其中 L 为输出位置数，按 C 逐通道、逐像素展开。
    注意：torch 的 unfold 输出布局是 C 外层、像素内层，与
    sliding_window_view 的 (B,C,oh,ow,kh,kw) 需 reshape 对齐。
    """
    B, C, H, W = x.shape
    kh = kw = kernel
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    # (B, C, oh, ow, kh, kw) —— 步长切片后按块抽取
    out = np.empty((B, C, oh, ow, kh, kw), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i * stride:i * stride + kh,
                                j * stride:j * stride + kw]
    # 重排为 (B, C*kh*kw, oh*ow)，torch unfold 顺序：先 C 后 kh*kw，位置行优先
    out = out.transpose(0, 1, 4, 5, 2, 3).reshape(B, C * kh * kw, oh * ow)
    return out


def _fold_patches(patches, kernel, output_size, stride):
    """等价 torch F.fold(patches, output_size, kernel, stride, padding=0)。

    对重叠区域求和（不归一化）。patches 形状 (B, C*k*k, L)。
    """
    B, Ck, L = patches.shape
    kh = kw = kernel
    out_h, out_w = output_size
    oh = (out_h - kh) // stride + 1
    ow = (out_w - kw) // stride + 1
    C = Ck // (kh * kw)
    p = patches.reshape(B, C, kh, kw, oh * ow)
    p = p.transpose(0, 1, 4, 2, 3)  # (B, C, L, kh, kw)
    out = np.zeros((B, C, out_h, out_w), dtype=np.float32)
    n = 0
    for i in range(oh):
        for j in range(ow):
            out[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw] \
                += p[:, :, n]
            n += 1
    return out


def _local_stat(tensor, kernel, stride, stat="median"):
    """局部统计（median/max/min），用 unfold+fold 向量化。

    注意：手动 replicate padding（边界复制），避免 zero padding 在图像
    左上边缘混入黑色 0 像素，导致边缘统计异常、产生深色边框。
    """
    B, _, H, W = tensor.shape
    p = kernel // 2
    pad = _replicate_pad(tensor, p, p)
    patches = _unfold_patches(pad, kernel, stride)
    Ck = patches.shape[1]
    if stat == "median":
        # torch.median 对偶数个元素返回"下中位数"（排序后第 n/2 个，即
        # 索引 (len-1)//2），而 np.median 返回上下中位数均值。这里用
        # partition 精确复现 torch 语义，避免边界像素的微小差异。
        kth = (Ck - 1) // 2
        part = np.partition(patches, kth, axis=1)
        v = part[:, kth:kth + 1]
    elif stat == "max":
        v = np.max(patches, axis=1, keepdims=True)
    elif stat == "min":
        v = np.min(patches, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported local stat: {stat}")
    vals = np.repeat(v, Ck, axis=1)
    out_h, out_w = H + 2 * p, W + 2 * p
    div = _fold_patches(np.ones_like(vals), kernel, (out_h, out_w), stride)
    out = _fold_patches(vals, kernel, (out_h, out_w), stride)
    out = out / (div + 1e-8)
    return out[:, :, p:p + H, p:p + W]


def _circle_kernel(r: float):
    """圆形形态学核（PixelOE 原版 circle_kernel）。"""
    real_r = r
    r = int(r)
    kernel = np.zeros((2 * r + 1, 2 * r + 1))
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            pts = np.array([[i, j]] * 8) + np.array(
                [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5],
                 [0, 0.5], [0, -0.5], [0.5, 0], [-0.5, 0]])
            dists = np.linalg.norm(pts - r, axis=1)
            if np.max(dists) <= real_r:
                kernel[i, j] = 1
            elif np.min(dists) <= real_r:
                b = (real_r - np.min(dists)) / (np.max(dists) - np.min(dists))
                kernel[i, j] = b
    return kernel.astype(np.float32)


_KERNELS = {
    1: _circle_kernel(1),
    2: _circle_kernel(1.5),
    3: _circle_kernel(2)[1:4, 1:4].copy(),
    4: _circle_kernel(2.5),
    5: _circle_kernel(3)[1:6, 1:6].copy(),
    6: _circle_kernel(3.5),
}


def _morph(img, kernel, iterations, op):
    """通用形态学（dilate/erode），逐 batch 逐通道处理，replicate 边界。"""
    B, C, H, W = img.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    kflat = kernel.reshape(1, 1, kh * kw).astype(np.float32)
    x = img
    for _ in range(iterations):
        xp = _replicate_pad(x, ph, pw)
        # (B, C, oh, ow, kh, kw)
        xp = xp.reshape(B * C, 1, xp.shape[2], xp.shape[3])
        # 用 unfold 抽取块
        pat = _unfold_patches(xp, kernel=kh, stride=1)  # (B*C, kh*kw, H*W)
        pat = pat.reshape(B, C, kh * kw, H * W)
        if op == "dilate":
            pat = pat + kflat[..., None] - 1.0
            x = pat.max(axis=2).reshape(B, C, H, W)
        else:  # erode
            pat = pat - kflat[..., None] + 1.0
            x = pat.min(axis=2).reshape(B, C, H, W)
    return np.clip(x, 0, 1)


def _dilate_cont(img, kernel, iterations=1):
    return _morph(img, kernel, iterations, "dilate")


def _erode_cont(img, kernel, iterations=1):
    return _morph(img, kernel, iterations, "erode")


def _outline_expansion(img_t, thickness=3, contrast=1.0, saturation=1.0):
    """PixelOE 第 1 步：对比度感知轮廓扩展（忠实移植官方 outline_expansion）。

    权重 w = sigmoid((l_med-0.5)*10 + (dark_dist-bright_dist)*3)，全局归一化。
    输出 = erode * w + dilate * (1-w)，再做开闭清理。
    """
    k = max(int(thickness) * 4, 4)  # 局部统计窗口（原版默认 16，随厚度缩放）
    stride = max(k // 2, 1)

    lab = _rgb_to_lab(img_t)
    lum = lab[:, 0:1] / 100.0  # (B,1,H,W) [0,1]

    l_med = _local_stat(lum, k * 2, stride, stat="median")
    l_min = _local_stat(lum, k, stride, stat="min")
    l_max = _local_stat(lum, k, stride, stat="max")
    bright_dist = l_max - l_med
    dark_dist = l_med - l_min

    # 官方 current_weight_mapping: sigmoid(polarity_score)
    avg_scale, dist_scale = 10.0, 3.0
    score = (l_med - 0.5) * avg_scale + (dark_dist - bright_dist) * dist_scale
    w = 1.0 / (1.0 + np.exp(-(score * contrast)))  # sigmoid
    # 全局归一化到 0~1
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)

    it = max(1, int(thickness))
    it = min(it, 6)
    e = _erode_cont(img_t, _KERNELS[it], 1)
    d = _dilate_cont(img_t, _KERNELS[it], 1)

    out = e * w + d * (1.0 - w)

    # 开闭清理
    oc = max(it - 1, 1)
    out = _erode_cont(out, _KERNELS[oc], 1)
    out = _dilate_cont(out, _KERNELS[oc], 2)
    out = _erode_cont(out, _KERNELS[oc], 1)

    # 饱和度调整
    if saturation != 1.0:
        g = out.mean(axis=1, keepdims=True)
        out = g + (out - g) * saturation

    return np.clip(out, 0, 1)


def _find_pixel_luminance(chunk):
    """对比度下采样的 L 通道选代表像素（官方 find_pixel_luminance）。

    chunk 形状 (B, 1, L)，L 为块内像素数（沿 patch 展开）。
    """
    mid_idx = chunk.shape[2] // 2
    mid = chunk[:, :, mid_idx][..., None]
    med = np.median(chunk, axis=2, keepdims=True)
    mu = chunk.mean(axis=2, keepdims=True)
    maxi = chunk.max(axis=2, keepdims=True)
    mini = chunk.min(axis=2, keepdims=True)
    out = mid.copy()
    cond1 = (med < mu) & ((maxi - med) > (med - mini))
    cond2 = (med > mu) & ((maxi - med) < (med - mini))
    out[cond1] = mini[cond1]
    out[cond2] = maxi[cond2]
    return out


def _contrast_downscale(img_t, patch_size):
    """PixelOE 对比度下采样：L 通道自适应选代表像素，AB 通道取中位数。"""
    N, C, H, W = img_t.shape
    lab = _rgb_to_lab(img_t)
    L, A, B = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

    pl = _unfold_patches(L, patch_size, patch_size)  # (B, ps*ps, L)
    pa = _unfold_patches(A, patch_size, patch_size)
    pb = _unfold_patches(B, patch_size, patch_size)

    # unfold 输出 (B, Ck, L) -> 需要 (B, L, Ck) 才能按块取统计
    pl = pl.transpose(0, 2, 1)
    pa = pa.transpose(0, 2, 1)
    pb = pb.transpose(0, 2, 1)

    rl = _find_pixel_luminance(pl)  # (B, L, 1)
    ra = np.median(pa, axis=2, keepdims=True)
    rb = np.median(pb, axis=2, keepdims=True)

    out_h = H // patch_size
    # (B, L, 1) -> (B, 1, out_h, out_w)
    rl = rl.reshape(N, 1, out_h, -1)
    ra = ra.reshape(N, 1, out_h, -1)
    rb = rb.reshape(N, 1, out_h, -1)

    out_lab = np.concatenate([rl, ra, rb], axis=1)
    return _lab_to_rgb(out_lab)


def _kcentroid_downscale(img_t, out_w, out_h, centroids_k=2):
    """PixelOE k-centroid 下采样：每块 k-means 聚 centroids_k 类取代表色。"""
    import math
    B, C, H, W = img_t.shape
    h_factor = H / out_h
    w_factor = W / out_w
    ks = (int(math.ceil(h_factor)), int(math.ceil(w_factor)))
    eh = out_h * ks[0]
    ew = out_w * ks[1]
    dh, dw = eh - H, ew - W
    img = _replicate_pad(img_t, dh // 2, dw // 2)
    # 补齐可能因整除产生的右侧/下侧缺口
    if img.shape[2] < eh:
        img = np.pad(img, ((0, 0), (0, 0), (0, eh - img.shape[2]), (0, 0)),
                     mode="edge")
    if img.shape[3] < ew:
        img = np.pad(img, ((0, 0), (0, 0), (0, 0), (0, ew - img.shape[3])),
                     mode="edge")

    patches = _unfold_patches(img, ks[0], ks[0])  # 注意：此处需方形核
    # torch 版用 kernel_size=ks 元组，但 ks[0]==ks[1] 在多数情况下成立；
    # 为忠实，这里按方核处理，矩形核时退化（与 torch F.unfold 允许矩形不同）
    # 重建 (B, N, kh*kw, C)
    kh = kw = ks[0]
    patches = patches.reshape(B, C, -1, patches.shape[-1])  # (B, C, ks2, N)
    patches = patches.transpose(0, 3, 2, 1)  # (B, N, ks2, C)
    num_patches = patches.shape[1]
    patches = patches.reshape(B * num_patches, -1, C)  # (B*N, ks2, C)

    # 确定性 k-means 初始化 + 迭代
    maxv = patches.max(axis=1, keepdims=True)
    minv = patches.min(axis=1, keepdims=True)
    interp = np.linspace(0, 1, centroids_k)[None, :, None]
    centroids = interp * minv + (1 - interp) * maxv  # (B*N, K, C)
    for _ in range(max(2 * int(centroids_k ** 0.5), 4)):
        d = np.linalg.norm(patches[:, :, None, :] - centroids[:, None, :, :],
                           axis=3)  # (B*N, ks2, K)
        labels = d.argmin(axis=2)  # (B*N, ks2)
        new_c = centroids.copy()
        for j in range(centroids_k):
            mask = (labels == j)
            cnt = mask.sum(axis=1, keepdims=True)
            cnt = np.clip(cnt, 1, None)
            new_c[:, j] = (patches * mask[..., None]).sum(axis=1) / cnt
        if np.abs(new_c - centroids).max() < 1 / 256:
            centroids = new_c
            break
        centroids = new_c

    # 每块用质心均值代表块色（与原版一致）
    out = centroids.mean(axis=1)  # (B*N, C)
    out = out.reshape(B, num_patches, C).transpose(0, 2, 1)
    out = out.reshape(B, C, out_h, out_w)
    return np.clip(out, 0, 1)


def _resize_area(img_t, out_w, out_h):
    """等价 torch F.interpolate(mode='area')，即 adaptive average pooling。

    每个输出像素取输入对应区间 [floor(i*H/oh), ceil((i+1)*H/oh)) 的平均，
    区间为空时（缩小时可能）退化为取左端点单点。这与 cv2.INTER_AREA
    的面积加权插值不同，为忠实移植 torch 语义需自行实现。
    """
    B, C, H, W = img_t.shape
    out = np.empty((B, C, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        y0 = int(i * H / out_h)
        y1 = int(np.ceil((i + 1) * H / out_h))
        y1 = max(y1, y0 + 1) if y1 > y0 else y0 + 1
        y1 = min(y1, H)
        if y1 <= y0:
            y1 = y0 + 1
        for j in range(out_w):
            x0 = int(j * W / out_w)
            x1 = int(np.ceil((j + 1) * W / out_w))
            if x1 <= x0:
                x1 = x0 + 1
            x1 = min(x1, W)
            block = img_t[:, :, y0:y1, x0:x1]
            out[:, :, i, j] = block.mean(axis=(2, 3))
    return out


def _resize_nearest(img_t, out_w, out_h):
    """等价 torch F.interpolate(mode='nearest')：src = floor(i*in/out)。"""
    B, C, H, W = img_t.shape
    sy = np.floor(np.arange(out_h) * H / out_h).astype(int)
    sx = np.floor(np.arange(out_w) * W / out_w).astype(int)
    return img_t[:, :, sy][:, :, :, sx]


def _resize_batch(img_t, out_w, out_h, mode):
    """等价 torch F.interpolate（batch 图），逐 batch 缩放。"""
    B, C, H, W = img_t.shape
    if mode == "area":
        return _resize_area(img_t, out_w, out_h)
    if mode == "nearest":
        return _resize_nearest(img_t, out_w, out_h)
    interp = {
        "bicubic": cv2.INTER_CUBIC,
    }[mode]
    out = np.empty((B, C, out_h, out_w), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            out[b, c] = cv2.resize(img_t[b, c], (out_w, out_h),
                                   interpolation=interp)
    return out


def _downscale_mode(img_t, out_w, out_h, mode):
    """按模式做下采样到 (out_w, out_h)。"""
    _, _, H, W = img_t.shape
    if mode == "contrast":
        patch = max(1, int(np.ceil(H / out_h)))
        nh = (H // patch) * patch
        nw = (W // patch) * patch
        if nh == 0 or nw == 0:
            return _resize_batch(img_t, out_w, out_h, "area")
        crop = img_t[:, :, :nh, :nw]
        down = _contrast_downscale(crop, patch)
        return _resize_batch(down, out_w, out_h, "nearest")
    if mode == "k-centroid":
        patch = max(1, int(np.ceil(H / out_h)))
        nh = (H // patch) * patch
        nw = (W // patch) * patch
        crop = img_t[:, :, :nh, :nw]
        down = _kcentroid_downscale(crop, out_h, out_w, centroids_k=2)
        return _resize_batch(down, out_w, out_h, "nearest")
    if mode == "bicubic":
        return _resize_batch(img_t, out_w, out_h, "bicubic")
    if mode == "nearest":
        return _resize_batch(img_t, out_w, out_h, "nearest")
    # center（默认）
    return _resize_batch(img_t, out_w, out_h, "area")


def pixelize_pixeloe(
    img: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    mode: str = "center",      # center | contrast | k-centroid | bicubic | nearest
    thickness: int = 3,
    contrast: float = 1.0,
    saturation: float = 1.0,
    patch_size: Optional[int] = None,
    colors: Optional[int] = None,
) -> np.ndarray:
    """PixelOE：对比度感知轮廓扩展 + 对比度感知下采样（纯 numpy 移植）。

    :param mode: 下采样模式（center/contrast/k-centroid/bicubic/nearest）
    :param thickness: 轮廓扩展形态学核半径（1~6，越大轮廓越粗）
    :param contrast: 对比度权重强度（>0，越大越强调高对比细节）
    :param saturation: 饱和度（1=不变，>1 更艳，<1 更灰）
    :param patch_size: 对比度下采样的块边长（None 自动）
    :param colors: 调色板颜色数（None 不量化；>0 用 k-means 限色）
    """
    img = np.ascontiguousarray(img, dtype=np.uint8)
    t = _to_tensor(img)

    if thickness > 0:
        t = _outline_expansion(t, thickness=thickness, contrast=contrast,
                               saturation=saturation)

    t = _downscale_mode(t, out_w, out_h, mode=mode)

    if colors and colors > 0:
        t = _kmeans_quantize(t, colors)

    return _from_tensor(t)


def _kmeans_quantize(img_t, k: int):
    """对 batch 图做 k-means 限色（用于 PixelOE 调色板量化）。"""
    B, C, H, W = img_t.shape
    out = img_t.copy()
    for b in range(B):
        pixels = img_t[b].reshape(C, -1).transpose(1, 0)  # (N, C)
        n = pixels.shape[0]
        idx = np.linspace(0, n - 1, k).astype(int)
        centroids = pixels[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(10):
            d = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
            labels = d.argmin(axis=1)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids[j] = pixels[mask].mean(axis=0)
        out[b] = centroids[labels].reshape(C, H, W)
    return out


# =============================================================================
# 统一入口
# =============================================================================
def pixelize(
    img: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    method: str = "area",
    **kwargs,
) -> np.ndarray:
    """统一入口：按 method 调用对应像素化方法。

    :param method: "area" | "dpid" | "pixeloe"
    """
    m = (method or "area").lower()
    if m == "area":
        return pixelize_area(img, out_w, out_h)
    if m == "dpid":
        return pixelize_dpid(img, out_w, out_h, lam=kwargs.get("lam", 1.0))
    if m == "pixeloe":
        return pixelize_pixeloe(
            img, out_w, out_h,
            mode=kwargs.get("mode", "center"),
            thickness=kwargs.get("thickness", 3),
            contrast=kwargs.get("contrast", 1.0),
            saturation=kwargs.get("saturation", 1.0),
            colors=kwargs.get("colors", None),
        )
    raise ValueError(f"未知像素化方法: {method}")


__all__ = [
    "pixelize",
    "pixelize_area",
    "pixelize_dpid",
    "pixelize_pixeloe",
]
