# 拼豆图纸生成器 (Bead Studio)

> 把照片 / 卡通图转换成 **MARD 色板** 的拼豆（拼豆图纸 / perler bead pattern）生成器。
> 支持命令行与 PySide6 桌面 GUI 两种使用方式，内置 AI 卡通化（多模态 LLM 理解 + 图生图）与纯 numpy 像素化算法。

**📖 图文操作手册（新用户从这里开始）：[docs/manual.html](docs/manual.html)**

**核心量化零重依赖**：纯 `numpy + PIL`，无需 torch / opencv / scipy 即可跑通完整链路；AI 卡通化为可选增强。

---

## 目录

1. [项目概览](#1-项目概览)
2. [目录结构](#2-目录结构)
3. [完整工作流（5 阶段）](#3-完整工作流5-阶段)
4. [核心算法详解](#4-核心算法详解)
5. [命令行用法与参数表](#5-命令行用法与参数表)
6. [桌面 GUI](#6-桌面-gui)
7. [配置文件说明](#7-配置文件说明)
8. [实测结果数据](#8-实测结果数据)
9. [已知限制与踩坑记录](#9-已知限制与踩坑记录)
10. [后续路线图](#10-后续路线图)
11. [AI 工具声明](#11-ai-工具声明)
12. [算法引用与致谢](#12-算法引用与致谢)

---

## 1. 项目概览

### 目标

输入一张图片，输出：
- **拼豆图纸**（带色号标注的完整网格图）
- **网格预览**（纯色块 + 网格线，用于照着摆豆子）
- **JSON 描述**（二维色号矩阵，可供二次开发 / 打印程序消费）

### 核心约束

| 约束 | 说明 |
|------|------|
| 色板固定 | MARD 221 色，`palettes/mard.csv`，格式 `code,name,hex` |
| 网格尺寸可变 | 16–104，默认 52（≈ 一块实体拼豆板 52×52） |
| 轮廓必须保留 | 描边/眼线/胡须不能被投票吞掉 |
| 误识别必须低 | 脸颊阴影等渐变区不能被当成线条 |
| 零重依赖 | 核心量化只用 numpy + PIL；opencv/scipy 为可选增强 |

### 依赖

```bash
# 必需（核心量化，纯 numpy）
pip install numpy pillow

# 可选（桌面 GUI）
pip install PySide6

# 可选（AI 卡通化，OpenAI 兼容客户端）
pip install requests openai pyyaml
```

缺失可选依赖时程序**不会崩溃**，会弹窗说明哪个功能不可用，其余功能照常工作。

> 说明：早期版本曾用 PyTorch（pixeloe 官方实现）与 scipy（连通域分析），
> 现已**全部移除**——像素化算法改为纯 numpy 重写（见第 4/12 节），
> 连通域分析改为手写 BFS，因此核心量化对 torch/scipy/cv2 零依赖。

---

## 2. 目录结构

```
├── bead_studio.py            # ★ 桌面 GUI 主程序 (PySide6)
├── run_quantize.py           # ★ 量化入口 (CLI)
├── run_understand.py         # LLM 视觉理解入口
├── run_seedream.py           # SeedDream 图生图（卡通化）入口
├── run_svg.py                # LLM 直接设计 SVG（备选路线）入口
├── tools/                    # 工具/实验脚本
│   ├── run_balanced.py       # 折中版量化实验
│   └── verify_outputs.py     # 产物检查工具
│
├── bead_studio.spec          # PyInstaller 打包配置
├── config.yaml               # API 配置（ARK / LLM / 图像预处理），Key 用 env: 引用
├── .env.example              # .env 模板（可安全提交）
├── .gitignore                # 忽略 .env / trash/ / outputs/ / build / dist 等
├── icon_transparent.png      # 应用图标（窗口 + exe）
├── icon_bead.ico             # exe 多尺寸图标
│
├── core/                     # 核心库
│   ├── quantize.py           # ★ 量化核心（全部量化算法在这）
│   ├── pixelize.py           # ★ 像素化方法（pixeloe/area/dpid，纯 numpy 实现）
│   ├── bead_chart.py         # 拼豆图纸渲染 (PNG)
│   ├── edit.py               # 网格编辑（涂色/框选/批量改色/撤销重做）
│   ├── palette_loader.py     # 色板加载 + sRGB→Lab + ΔE76
│   ├── pipeline.py           # Stage 0/1 管线（LLM 理解 + SeedDream 卡通化）
│   ├── visual_understand.py  # LLM 视觉理解
│   ├── svg_design.py         # LLM SVG 设计
│   ├── svg_renderer.py       # SVG 渲染
│   ├── seedream.py           # SeedDream 客户端
│   ├── llm_client.py         # 通用 LLM 客户端（OpenAI 兼容，多厂商）
│   ├── config_loader.py      # YAML 配置读写 + .env 加载 + key 检测
│   ├── settings_dialog.py    # API 设置对话框（多厂商 + 异步测试连接）
│   ├── style.py              # 全局 QSS 样式表
│   ├── paths.py              # 资源定位（源码 / PyInstaller _MEIPASS）
│   └── image_utils.py        # 图片预处理（缩放/编码）
│
├── palettes/
│   └── mard.csv              # MARD 221 色色板
│
├── samples/                  # 样例图（动物/食物/风景/人像等）
├── docs/
│   ├── manual.html           # 图文操作手册（使用者入门）
│   └── assets/               # 手册配图（界面截图 + 流程演示）
│
└── outputs/                  # 产物目录（被 .gitignore 忽略）
```

---

## 3. 完整工作流（5 阶段）

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 0  [可选] LLM 视觉理解                                      │
│  run_understand.py → outputs/cat_analysis.json                    │
│  输出: 关键特征 bbox (眼/耳/嘴/鼻/胡须) + 卡通风格描述            │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1  [可选] SeedDream 图生图（照片 → 卡通）                  │
│  run_seedream.py → outputs/cat_seedream.png                       │
│  模型: doubao-seedream-5-0-pro  /  image_strength=0.65  /  2K     │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2  MARD 色板量化  ★ 核心                                   │
│  run_quantize.py → 52×52 色号网格                                 │
│  ├─ 4x 超采样: 2048² → 208×208 → vote 下采样 → 52×52            │
│  ├─ 分层决策: 线条 → 轮廓 → 细节 → 投票                          │
│  └─ 小色块清理 (ΔE 感知)                                          │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 3  渲染拼豆图纸                                             │
│  core/bead_chart.py                                               │
│  ├─ render_bead_chart: 带色号 + 坐标 + 图例 + 标题                │
│  └─ render_grid_only: 纯色块 + 网格线                            │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 4  输出产物 (PNG ×2 + JSON ×1)                             │
└──────────────────────────────────────────────────────────────────┘
```

**Stage 0/1 已集成进 GUI**（点「卡通化」按钮即可走完「LLM 理解 → SeedDream 卡通化 → 量化」全流程，也可直接用现成卡通图量化）。

---

## 4. 核心算法详解

全部在 `core/quantize.py`。

### 4.1 量化管线总览

```
原图 (2048×2048 RGB)
   │
   ├─ [可选] 线条检测 → line_mask (2048×2048 bool)
   │             5 种策略，见 4.2
   │
   ├─ 4x 超采样量化 → 208×208 高分辨率色号网格
   │             每块 10×10 像素，走「分层决策」（见 4.3）
   │
   ├─ 下采样 208×208 → 52×52
   │             3 种策略，见 4.4
   │
   ├─ 小色块清理（ΔE 感知），见 4.5
   │
   └─ QuantizeResult { grid, stats, color_count, ... }
```

### 4.2 线条检测策略（`--line-method`）

目标：**独立于深浅**地识别线条——不是"所有深色都是线"，而是"不管深线浅线，只要形状像线就抓"。

| 策略 | 实现 | 依赖 | 线条覆盖率 | 实测评价 |
|------|------|------|-----------|---------|
| `canny` | OpenCV Canny（高斯 → Sobel → NMS → 双阈值滞后） | opencv | 4.1% | 🥇 **错检最少**，描边干净 |
| `multiscale` | 自研多尺度 Sobel + NMS（3 sigma: 1/2/4） | 无 | 67.7% | 🥈 稍粗，脸颊有零散黑点 |
| `local` | 局部对比度 `max(\|p-邻域均值\|, \|p-全局均值\|)` | 无 | 12.0% | 🥉 抓细线强但误识多 |
| `hybrid` | `canny ∪ local(形状过滤后)` | opencv+scipy | ~13% | 胡须友好，需配 shape filter |
| `frangi` | 多尺度 Hessian + Frangi vesselness 响应 | 无 | 8.0% | ❌ **不适合黑线**（见下） |

#### Frangi 为什么不行

Frangi 用 `Rb = λ1/λ2`（Hessian 两个特征值之比）判别"线状 vs 块状"：
- 血管类管状结构：一个方向强曲率、另一个弱 → Rb ≈ 0 → 强响应
- **纯黑细描边**：两个方向都强曲率（\|λ1\| ≈ \|λ2\|）→ Rb ≈ 1 → **被当成块状压制**

这是设计本意，**改参数救不回来**。Canny 抓不到胡须也是同理：高斯模糊 + NMS 把 1 像素宽、灰度差仅 120 的胡须衰减掉了。`local` 不做模糊，反而能抓到。

#### 关键修复：形状过滤（`_shape_filter_line_mask`）

`local` 策略在脸颊阴影处有大量误识别（阴影块状 vs 胡须线状）。用 scipy 连通域分析按形状过滤：

```
保留条件:
  连通域面积 ≥ min_size (默认 4)
  长宽比 = max(h,w)/min(h,w) ≥ aspect_min (默认 2.5)
```

实测：H7（纯黑误识别）从 **131 → 0**，胡须仍保留。

### 4.3 分层决策（`_block_vote_quantize`）

每个块的选色按优先级从上到下：

| 层级 | 判据 | 作用 |
|------|------|------|
| **第 0 层** | 线条检测 | 块内 `line_mask` 占比 ≥ `line_min_pct`（默认 5%）**或**（≥ `min_line_count` 像素 **且** 最暗线条 vs 主体 ΔE ≥ `min_line_contrast`）→ 用**自适应线条色** |
| **第 1 层** | 轮廓保护 | 块中心像素 L < `outline_center_l`（默认 85）**或** 深色像素占比 ≥ 1.5% **或** 边缘像素占比 ≥ 4% → 用轮廓色 |
| **第 2 层** | 细节优先 | 投票 + 次色覆盖。阈值按区域分级（core / face / background）动态调整 |
| **第 3 层** | 朴素投票 | 取频次最高色号 |

**自适应线条色**（`_find_line_color_adaptive`）：不写死"线条=黑色"。做法是——
1. 找块内"非线条像素"的主色（背景色）
2. 在"线条像素"里找**和背景色 ΔE 最大**的那个像素
3. 取它的色板最近邻

→ 深线得到深色，浅线得到浅色，同一套参数适配两种情况。

### 4.4 下采样策略（`--ss-strategy`）

4x 超采样后从 208×208 压回 52×52，每块对应 4×4 个高分辨率子格。

| 策略 | 逻辑 | 优点 | 缺点 |
|------|------|------|------|
| `vote` | 16 个子格多数派胜出 | 干净，不糊 | **细线被投掉**（胡须/描边丢失） |
| `darkest` | 深色子格数 ≥ `ss_min_dark_count` → 取最暗；否则 vote | 保留描边 | 孤立深色噪点会把眼睛染糊 |
| `line_aware` | 4×4 子区域里 ≥ `ss_line_min_pixels` 个 `line_mask` 像素 → **整块用线条色**；否则 vote | 线条不被抹除，又不糊 | 需配 `--line-detect`；阈值太松会误识 |

`line_aware` 的关键是**把 line_mask 直接传递到最终网格**，绕开投票的信息损失。线条色从原图对应区域按"ΔE 最大"原则选，不依赖高分辨率量化结果。

### 4.5 小色块清理（`_remove_small_islands`）

孤立小色块会干扰实际拼豆（多一板豆子、视觉脏），但**不能一刀切**——孤立的黑色像素往往正是眼线/瞳孔。

判定逻辑：

```
对每个连通域（面积 < min_island_size）:
  计算它与外部主色的 ΔE76
  ΔE ≥ cleanup_min_contrast (默认 20) → 保留（视为关键特征）
  否则 → 合并到邻域主色
```

效果：
- 孤立黑色像素（ΔE ≈ 80）→ 保留 ✓
- 米色/灰色杂点（ΔE ≈ 10）→ 清理 ✓

另有 `protected_codes` 机制：MARD 中 L < 30 的深色色号（`B22 B23 C12 C18 D10 D15 D22 D4 F11 H16 H6 H7`）直接跳过清理。

### 4.6 色彩空间

统一用 **CIE Lab**（`core/palette_loader.py`）：
- sRGB → 线性 → XYZ → Lab（D65）
- 色差用 **ΔE76**（CIE 1976）——够用且快，未用 ΔE2000
- 深色判定用 Lab 的 L 通道

---

## 5. 命令行用法与参数表

### 5.1 最小命令

```powershell
conda activate NDP
cd D:\26-27\PD

# 用默认卡通图，52×52 网格
python run_quantize.py --grid 52
```

### 5.2 完整管线

```powershell
# Stage 0: LLM 视觉理解
python run_understand.py

# Stage 1: SeedDream 卡通化
python run_seedream.py

# Stage 2+3: 量化 + 渲染
python run_quantize.py --grid 52
```

### 5.3 全部参数

#### 网格与基础

| 参数 | 默认 | 说明 |
|------|------|------|
| `--grid N` | 52 | 网格边长（正方形） |
| `--grid-w N` / `--grid-h M` | — | 分别指定宽高（覆盖 `--grid`） |
| `--image PATH` | `outputs/cat_seedream.png` | 输入图 |
| `--analysis PATH` | `outputs/cat_analysis.json` | LLM 分析 JSON（提供 key_features） |
| `--no-key-features` | off | 关闭关键特征 bbox |
| `--no-detail` | off | 关闭细节优先（最朴素投票） |

#### 超采样 / 下采样

| 参数 | 默认 | 说明 |
|------|------|------|
| `--supersample N` | 1 | 超采样倍率，推荐 4 |
| `--ss-strategy` | `vote` | `vote` / `darkest` / `line_aware` |
| `--ss-min-dark N` | 2 | darkest 策略的深色子格阈值（1=纯OR 最糊，3+=保守） |
| `--ss-line-min-pixels N` | 1 | line_aware 策略的最小线条像素数 |

#### 量化细节

| 参数 | 默认 | 说明 |
|------|------|------|
| `--min-island-size N` | 2 | 小色块清理阈值 |
| `--cleanup-min-contrast F` | 20.0 | 小色块色差阈值，≤0 关闭 |
| `--outline-preserve` | off | 启用轮廓强制保留 |
| `--no-protect-outline` | off | 不保护深色色号 |
| `--outline-min-pct F` | 0.015 | 轮廓保护：深色像素最小占比 |
| `--outline-min-edge F` | 0.04 | 轮廓保护：边缘像素最小占比 |
| `--outline-center-l F` | 85.0 | 轮廓保护：中心像素 L 阈值 |
| `--outline-darkest-l F` | — | **已弃用**，用 `--outline-center-l` |

#### 线条检测

| 参数 | 默认 | 说明 |
|------|------|------|
| `--line-detect` | off | 启用线条检测 |
| `--line-method` | `canny` | `canny` / `multiscale` / `local` / `hybrid` / `frangi` |
| `--line-threshold F` | 88.0 | local 模式的分位数阈值 |
| `--line-min-pct F` | 0.05 | 块内线条像素占比阈值 |
| `--min-line-count N` | 1 | 细线兜底：最少线条像素数 |
| `--min-line-contrast F` | 30.0 | 细线兜底：ΔE 阈值，≤0 关闭 |
| `--line-aspect-min F` | 2.5 | hybrid 形状过滤：长宽比下限 |
| `--line-min-region-size N` | 4 | hybrid 形状过滤：最小连通域像素数 |

#### 输出

| 参数 | 默认 | 说明 |
|------|------|------|
| `--out-suffix S` | `""` | 输出文件名后缀（用于区分不同参数的结果） |

### 5.4 常用组合

```powershell
# A. 干净基础版（GUI 默认，推荐）
python run_quantize.py --grid 52

# B. 干净 + 显式清理参数
python run_quantize.py --grid 52 --supersample 4 --ss-strategy vote `
    --cleanup-min-contrast 20 --min-island-size 2

# C. 抢救胡须/描边（line_aware）
python run_quantize.py --grid 52 --supersample 4 --ss-strategy line_aware `
    --ss-line-min-pixels 3 --line-detect --line-method hybrid --line-threshold 90 `
    --line-aspect-min 2.5 --line-min-region-size 4 `
    --cleanup-min-contrast 20 --min-island-size 2

# D. 无 opencv 依赖环境
python run_quantize.py --grid 52 --line-detect --line-method multiscale

# E. 换网格尺寸
python run_quantize.py --grid 64 --out-suffix _g64
```

---

## 6. 桌面 GUI

### 6.1 启动

```powershell
conda activate NDP
cd D:\26-27\PD
python bead_studio.py
```

### 6.2 界面结构

```
┌─────────────────────────────────────────────────────────────────┐
│ 工具栏: 打开图片 | 生成拼豆图 | 导出结果 | 关于        │
├──────────────────────┬──────────────────────────────────────────┤
│ 左侧参数面板 (滚动)   │  右侧预览区（两个可拖拽工作区）            │
│ ┌──────────────────┐ │  ┌────────────────────────────────────┐  │
│ │输入图片        │ │  │ 工作区1「源图」                      │  │
│ ├──────────────────┤ │  │ [原图 | 卡通图 | 转化缓冲区]   │  │
│ │卡通化          │ │  │   (原图/卡通图可框选；缓冲区放裁剪)  │  │
│ │  LLM理解+SeedDream│ │  ├──────────── 拖拽分隔条 ────────────┤  │
│ │  +自定义Prompt    │ │  │ 工作区2「结果」                      │  │
│ ├──────────────────┤ │  │ [网格预览 | 拼豆图纸]            │  │
│ │基础参数        │ │  ├──────────── 拖拽分隔条 ────────────┤  │
│ │超采样          │ │  │统计 (用色数/格数/TOP12/产物路径) │  │
│ │  倍率 / 策略      │ │  └────────────────────────────────────┘  │
│ │  min_dark/min_px │ └──────────────────────────────────────────┘
│ ├──────────────────┤  状态栏: 提示文字 + 进度条                 │
│ │量化细节        │                                          │
│ │  detail/outline  │                                          │
│ │  island/ΔE       │                                          │
│ ├──────────────────┤                                          │
│ │线条检测        │                                          │
│ │  开关/方法/阈值   │                                          │
│ │  min_pct/count    │                                          │
│ │  contrast/aspect  │                                          │
│ ├──────────────────┤                                          │
│ │输出目录        │                                          │
│ ├──────────────────┤                                          │
│ │  生成拼豆图纸  │                                          │
│ └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 特性

- **拖拽支持**：直接把图片拖进窗口
- **不卡 UI**：量化/卡通化跑在 `QThread` 子线程，通过信号回传进度
- **卡通化全流程**：一键跑 LLM 视觉理解 + SeedDream 卡通化，自动把关键特征 bbox 传给量化（细节优先 + 区域分级）
- **自定义 Prompt**：卡通化分组里可填写自定义 prompt，覆盖 LLM 自动生成的描述
- **框选区域像素化**：在原图/卡通图上拖拽框选区域，只对选中区域裁剪量化（留空=整图）
- **选区可拖边角调整**：框选后拖动 8 个控制点（4 边 + 4 角）调整选区位置/大小，拖动框内可整体移动，无需重新框选
- **图片缩放/平移**：`Ctrl + 鼠标滚轮` 缩放图片，鼠标中键或拖动平移图片
- **原图/卡通图分离**：卡通化后原图保留在「原图」Tab，卡通图独立显示在「卡通图」Tab；生成拼豆图时根据当前 Tab 自动选择量化源
- **双工作区布局**：预览区分为「源图区」（原图/卡通图/转化缓冲区）和「结果区」（网格预览/拼豆图纸）两个工作区，中间是可拖拽分隔条，可自由调整上下比例（类似 MATLAB 页面布局），方便对照原图/卡通图手动修改网格
- **转化缓冲区**：源图区的「转化缓冲区」Tab 实时显示框选后的裁剪图（将被像素化的源），未框选时显示整图
- **敏感词自动重试**：SeedDream 遇到敏感词审核拦截时，自动改写 prompt 重试（最多 3 次），而非盲目重发
- **快捷键**：`Ctrl+O` 打开 / `Ctrl+R` 生成 / `Ctrl+S` 保存
- **依赖自检**：启动时检查 opencv/scipy/openai，缺失则弹窗说明并禁用相关选项
- **图片校验**：格式错误 / 太小 / 太大都有友好提示
- **四视图 + 双工作区**：原图 / 卡通图 / 转化缓冲区 在「源图区」，网格预览 / 拼豆图纸 在「结果区」，两个工作区可拖拽分隔条调整比例（卡通图 Tab 仅在卡通化后启用）
- **统计面板**：用色数、总格数、清理数、色号 TOP12、产物路径

### 6.4 当前限制

- 卡通化依赖 `openai` / `requests`（OpenAI 兼容客户端），缺失时 GUI 自动降级为仅量化模式
- 窗口里的中文在 offscreen 渲染下显示为方框（缺中文字体），真实 Windows 桌面正常
- 打包产物体积约 256 MB（含 Qt 运行时）；已在 `.gitignore` 中排除 `build/` / `dist/`

---

## 7. 配置文件说明

`config.yaml` 里的 **API Key 已脱敏**：真实密钥不存这里，而是放在项目根目录的 `.env`（已被 `.gitignore` 忽略）。`config.yaml` 中对应字段写成 `env:ARK_API_KEY` / `env:LLM_API_KEY`，`core/config_loader.py` 在加载时自动从 `.env` 读取真实值。

> 首次运行：复制 `.env.example` 为 `.env`，填入你自己的两个 Key（火山方舟 + LLM）。
> `.env.example` 是模板（不含真实密钥，可安全提交）；`.env` 是本地私有文件（勿提交）。

### `ark` — 火山方舟（豆包）

| 键 | 当前值 | 说明 |
|----|--------|------|
| `api_key` | `env:ARK_API_KEY` | 方舟 API Key（真实值存于 `.env`） |
| `base_url` | `https://ark.cn-beijing.volces.com/api/v3/images/generations` | 图像生成端点 |
| `timeout` | 120 | 请求超时（秒） |
| `max_retries` | 3 | 失败重试 |
| `proxy` | `""` | HTTP 代理，留空直连 |
| `pipeline.enable_cartoonize` | true | 是否启用卡通化 |
| `pipeline.enable_inpaint` | false | 是否启用局部重绘（需蒙版） |
| `cartoonize.model` | `doubao-seedream-5-0-pro-260628` | 卡通化模型 |
| `cartoonize.image_strength` | 0.65 | 图生图相似度（0.50 夸张 ~ 0.75 保留姿态） |
| `cartoonize.size` | `1K` | 输出尺寸（最低分辨率，省时省 token；支持 1K/1.5K/2K） |
| `inpaint.model` | `doubao-seededit-3.0-t` | 局部重绘模型 |
| `inpaint.mask_convention` | `black_is_inpaint` | 黑色=重绘区 |

> **敏感词防护**：方舟 SeedDream 会对 prompt 做敏感内容审核，负面情绪词（如"不耐烦""冷淡""下压嘴角"）会触发 `InputTextSensitiveContentDetected`。已做双重防护：① `visual_understand.py` 的 system prompt 要求 LLM 用中性措辞；② `core/pipeline.py` 的 `sanitize_prompt()` 在发送前兜底替换敏感词。

### `llm` — 多模态 LLM（OpenAI 兼容）

| 键 | 当前值 |
|----|--------|
| `provider` | `minimax` |
| `api_key` | `env:LLM_API_KEY`（真实值存于 `.env`） |
| `base_url` | `https://api.minimax.chat/v1`（国内直连，不受网络环境影响） |
| `model` | `MiniMax-M3`（原生多模态，支持图片输入） |
| `temperature` | 0.7 |
| `max_tokens` | 4096 |
| `disable_thinking` | true（关闭 M3 思考块，直接返回 content） |

> 也可切换到其他 OpenAI 兼容端点（如 `codexcn`），只需改 `provider` / `base_url` / `model`。

### `image` — 送 API 前的预处理

| 键 | 当前值 | 说明 |
|----|--------|------|
| `max_side` | 1024 | 最大边长（省 token） |
| `format` | `jpeg` | `jpeg` / `png` |
| `quality` | 85 | JPEG 质量 |

> ✅ **安全提示**：API Key 已脱敏——真实密钥存于 `.env`（被 `.gitignore` 忽略），
> `config.yaml` 只保留 `env:` 引用，可安全提交到版本库。

---

## 8. 实测结果数据

测试素材：`outputs/cat_seedream.png`（2048×2048，SeedDream 卡通化的猫）

### 8.1 各方案对比（52×52）

| 方案 | 用色数 | H16 (深棕黑) | H7 (纯黑) | 清理数 | 胡须 | 脸颊误识 |
|------|-------|--------------|------------|-------|------|---------|
| `canny` + vote | 31 | 38 | 0 | 31 | 3-4 条 | 无 |
| `multiscale` + vote | 32 | 56 | 36 | 71 | 少 | 有 |
| `frangi` + vote | 35 | 0 | 0 | 39 | 无 | 无（但描边全丢） |
| `hybrid` (无形状过滤) + vote | 31 | 0 | 131 | 27 | 5-6 条 | **有** |
| `hybrid` + 形状过滤 + vote | 31 | 35 | 0 | 33 | 4-5 条 | 无 |
| `hybrid` + 形状过滤 + **line_aware(3)** | 33 | 37 | 45 | 36 | **5-6 条** | 无 |
| **朴素 vote (当前默认)** | **32** | **71** | **0** | 33 | 少 | 无 |

### 8.2 不同网格尺寸（canny + vote）

| Grid | 用色数 | 清理数 | 观察 |
|------|-------|-------|------|
| 40×40 | 31 | 30 | 轮廓分明，细节少 |
| 52×52 | 31 | 31 | **推荐**，细节与规模平衡 |
| 64×64 | — | — | 描边更细腻，胡须可表达 |

### 8.3 性能

- 52×52 + 4x 超采样 + vote：约 3-5 秒（2048² 输入）
- Canny 线条检测：38ms（512² 缩图）/ 全分辨率稍慢
- Frangi：约 2x 于 Canny（纯 numpy 实现）
- GUI 全流程（含渲染）：约 8-12 秒

---

## 9. 已知限制与踩坑记录

### 9.1 算法层面

1. **52×52 网格物理上装不下完整胡须**。胡须最细 1 像素宽，压缩后必然丢失或断续。要表达更多细节需提高 grid（64/80），但超出单块实体板尺寸。
2. **描边粗细与"干净度"是对立的**。`line_aware` 能加粗描边但会引入更多黑色像素（H7 0→45）。用户实际拼豆时，干净的色块面往往比多一根胡须更有价值——这是当前选择朴素 `vote` 的原因。
3. **Frangi 永远抓不到黑线**（见 4.2），不建议再投入。
4. **阴影 vs 细线**在纯图像处理上难以完美区分。当前用形状过滤（长宽比）近似，不是语义理解。
5. **`detail_mode` 的区域分级**依赖 LLM 输出的 bbox。bbox 不准时分级失效。
6. **ΔE76** 在高饱和色区域区分度不如 ΔE2000，暂未替换。

### 9.2 工程层面

1. **Windows GBK 终端编码**：`print()` 里带 emoji/✓ 会抛 `UnicodeEncodeError`。脚本输出请写文件或用 ASCII。
2. **`conda run -n XXX python -c "多行脚本"`** 不支持换行，需写成文件再执行。
3. **PySide6 offscreen 渲染缺中文字体**：截图里中文是方框，真实桌面正常。
4. **`mavis-trash` 依赖 node**，当前环境无 node，删除文件会失败。
5. ~~临时调试文件堆积~~ ✅ 已清理（归档至 `trash/`，确认无用后可整个删除）。
6. ~~`config.yaml` 明文 API Key~~ ✅ 已脱敏：真实密钥移至 `.env`（被 `.gitignore` 忽略），`config.yaml` 用 `env:ARK_API_KEY` / `env:LLM_API_KEY` 引用，`config_loader.py` 自动从 `.env` 加载。

### 9.3 未完成

- ~~Stage 0/1 未接入 GUI~~ ✅ 已集成（Phase 3）
- 无 `.exe` 打包
- 无 README 的安装脚本 / 首次运行引导
- ~~`run_balanced.py` / `verify_outputs.py` 临时脚本未整理~~ ✅ 已归档至 `tools/`

---

## 10. 后续路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | PySide6 GUI 骨架 + 全部参数暴露 + 后台线程 + 三视图 | ✅ 完成 |
| Phase 2 | 错误处理 + 依赖自检 + 图片校验 | ✅ 完成（Phase 1 内） |
| Phase 3 | 集成 Stage 0/1（LLM 理解 + SeedDream 卡通化）到 GUI | ✅ 完成 |
| Phase 4 | PyInstaller 打包成 `.exe` | ✅ 完成（`dist/BeadStudio/BeadStudio.exe`） |
| Phase 5 | README + 图文操作手册 + 首次运行引导 | ✅ 完成（`docs/manual.html`） |
| Phase 6 | 清理临时文件 + 目录重构 + API Key 脱敏 | ✅ 完成 |
| Phase 7 | 页面美化 + API 设置对话框 + 批量改色交互 | ✅ 完成 |

### 技术选型备忘

- **GUI 框架**：PySide6（LGPL，许可比 PyQt6 友好）
- **打包**：PyInstaller 目录模式（`bead_studio.spec`），排除 FFmpeg DLL / 未用 Qt 组件，产物约 256 MB
- **可选依赖降级**：core 层已做 cv2/scipy/torch 缺失时的 fallback；像素化与连通域已改为纯 numpy 实现，核心量化零重依赖

---

## 附录 A：核心 API

```python
from core.palette_loader import load_palette
from core.quantize import quantize

palette = load_palette()                    # MARD 221 色
result = quantize(
    "path/to/image.png",
    grid_w=52, grid_h=52,
    palette=palette,
    min_island_size=2,                      # 小色块清理
    cleanup=True,
    detail_mode=True,                       # 细节优先
    supersample=4,                          # 4x 超采样
    ss_strategy="vote",                     # 下采样策略
    cleanup_min_contrast=20.0,              # ΔE 感知清理
    line_detect=False,
    line_method="canny",
)
# result.grid[r][c] -> PaletteColor
# result.stats -> {code: count}
```

## 附录 B：MARD 色板格式

```csv
code,name,hex
A1,Mard_A1,#FAF4C8
A2,Mard_A2,#FFFFD5
...
```

深色色号（Lab L < 30，清理时受保护）：
`B22 B23 C12 C18 D10 D15 D22 D4 F11 H16 H6 H7`

---

## 11. AI 工具声明

本项目在开发过程中使用了以下 AI 工具辅助编码、设计与文档撰写。核心算法（量化、像素化、渲染）均为自研实现或基于开源算法的手写移植，AI 工具主要用于开发提效与生成式图像处理。

### 11.1 开发辅助（编码 / 设计 / 文档）

| 工具 / 模型 | 用途 |
|-------------|------|
| WorkBuddy（腾讯） | 主开发环境：代码生成与重构、GUI 美化、打包排障、文档撰写 |
| DeepSeek-V4-Pro | 代码逻辑与算法对齐的推理辅助（如纯 numpy 复现 torch 语义） |
| MiniMax Code | 编码辅助、代码片段生成与 review |

### 11.2 运行时生成式能力（应用内 AI 功能）

| 模型 / 服务 | 用途 | 说明 |
|-------------|------|------|
| MiniMax-M3 | 多模态视觉理解 | 分析照片关键特征（眼睛/耳朵/花纹位置）与生成卡通风格描述；OpenAI 兼容端点 |
| 豆包 SeedDream（火山方舟） | 图生图卡通化 | 照片 → 干净色块的卡通风格（`doubao-seedream-5-0-pro`） |
| 豆包 SeedEdit（火山方舟） | 局部重绘（可选） | 蒙版消除水印 / 杂物（`doubao-seededit-3.0-t`） |

> 其余 OpenAI 兼容端点（codexcn / openai / anthropic / gemini / qwen）可在「设置」中一键切换，
> 见 `core/settings_dialog.py` 的多厂商预设。

> **隐私说明**：所有 API Key 均存于本地 `.env`（已 `.gitignore` 忽略），仓库内仅提供 `.env.example` 模板；
> 图片仅在用户主动点击「卡通化」时才上传至对应云端 API。

---

## 12. 算法引用与致谢

### 12.1 PixelOE（像素化方法）

「像素化方法」中的 **pixeloe** 选项移植自开源项目 **PixelOE**：

> **PixelOE: Detail-Oriented Pixelization based on Contrast-Aware Outline Expansion**
> 作者：[KohakuBlueleaf](https://github.com/KohakuBlueleaf)
> 仓库：https://github.com/KohakuBlueleaf/PixelOE
> 许可：Apache License 2.0

PixelOE 的两阶段核心思路——**对比度感知轮廓扩展**（先膨胀高对比边缘再下采样）与
**对比度感知下采样**（按局部亮度分布选代表像素）——在本项目中被**忠实地用纯 numpy 重写**
（`core/pixelize.py`），语义与官方 torch 实现逐像素对齐（默认参数端到端色号差异 0 格）。

> 原实现为 PyTorch 版，本项目为降低打包体积与运行时依赖，将 outline expansion、
> contrast downscale、unfold/fold 等算子用 numpy 等价实现，特此致谢原作者。

### 12.2 其他参考

| 来源 | 用途 |
|------|------|
| MARD 221 色板 | 拼豆色号体系，见 `palettes/mard.csv` |
| OpenCV（可选依赖） | Canny 线条检测（未装时自动降级为自研 multiscale Sobel） |
| scipy（已移除） | 早期连通域分析，现已改为手写 BFS（`core/quantize.py::_connected_components`） |

---

*README 最后更新：2026-10-27*
