# 拼豆图纸生成器 (Beadobe)

> [**English**](README.en.md) · **中文**

> 把照片 / 卡通图转换成 **MARD 色板** 的拼豆图纸（perler bead pattern）生成器。
> 桌面 GUI 一键操作，内置 AI 卡通化与纯 numpy 像素化算法，核心量化零重依赖。

> 💡 **关于命名**：本项目原名 *Bead Studio* (AI 给起的)，因同名项目过多而改名为 **Beadobe**。代码、可执行文件与配置目录等内部标识仍保留原名（`BeadStudio.exe`、`bead_studio.py`、`%APPDATA%\BeadStudio` 等），属正常现象，不影响使用。

**🎯 图文操作手册**（二选一，内容相同）：
- [manual.md](docs/manual.md) —— 可在 GitHub 直接在线阅读
- [manual.html](docs/manual.html) —— 下载后双击浏览器打开，样式更精美

以下是整体流程一图流：

![完整流程演示](docs/assets/demo_pipeline.png)

---

## 功能特性

- **一键生成**：图片 → 拼豆图纸（带色号）+ 网格预览 + JSON 色号矩阵
- **AI 卡通化**：多模态 LLM 视觉理解 + 豆包 SeedDream 图生图，照片秒变干净卡通图
- **手动编辑**：涂色 / 擦除 / 取色 / 框选 / 批量改色 / 撤销重做
- **纯 numpy 核心**：量化与像素化算法零 torch / opencv / scipy 依赖
- **三种像素化方法**：PixelOE（对比度感知）/ DPID（细节保持）/ AREA（面积插值）

---

## 快速开始

### 方式一：下载打包版（推荐，无需装环境）

1. 前往 [Releases](https://github.com/leoq276/beadobe/releases) 下载最新的 `BeadStudio-v1.0.0-windows-x64.zip`
2. 解压后双击 `BeadStudio.exe` 即可运行，无需安装 Python

### 方式二：源码运行

```powershell
pip install numpy pillow PySide6 requests openai pyyaml

# 启动桌面 GUI
python bead_studio.py
```

> AI 卡通化为可选功能，需在「⚙️ 设置」中配置 API Key（详见 `docs/manual.html` 第 3 节）。

---

## 目录结构

```
├── bead_studio.py            # 桌面 GUI 主程序 (PySide6)
├── run_quantize.py           # 量化入口 (CLI)
├── core/                     # 核心库
│   ├── quantize.py           # 量化核心（分层决策 + 小色块清理）
│   ├── pixelize.py           # 像素化方法（pixeloe / dpid / area）
│   ├── bead_chart.py         # 拼豆图纸渲染 (PNG)
│   ├── edit.py               # 网格编辑
│   ├── palette_loader.py     # 色板加载 + sRGB→Lab + ΔE76
│   ├── pipeline.py           # LLM 理解 + SeedDream 卡通化管线
│   ├── seedream.py           # SeedDream 客户端
│   ├── llm_client.py         # 通用 LLM 客户端（OpenAI 兼容，多厂商）
│   ├── settings_dialog.py    # API 设置对话框
│   └── ...
├── palettes/mard.csv         # MARD 221 色色板
├── samples/                  # 样例图
├── docs/
│   ├── manual.html           # 图文操作手册
│   └── assets/               # 手册配图
└── outputs/                  # 产物目录（被 .gitignore 忽略）
```

---

## 命令行用法

```powershell
# 最小命令（默认 52×52）
python run_quantize.py --grid 52

# 换网格尺寸 + 输出后缀
python run_quantize.py --grid 64 --out-suffix _g64
```

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--grid N` | 52 | 网格边长（16–104） |
| `--image PATH` | — | 输入图片 |
| `--supersample N` | 1 | 超采样倍率，推荐 4 |
| `--ss-strategy` | `vote` | 下采样策略：`vote` / `darkest` / `line_aware` |
| `--line-detect` | off | 启用线条检测 |
| `--line-method` | `canny` | `canny` / `multiscale` / `local` / `hybrid` / `frangi` |
| `--min-island-size N` | 2 | 小色块清理阈值 |
| `--cleanup-min-contrast F` | 20.0 | 小色块色差阈值（ΔE） |

> 完整参数与算法详解见下方「算法说明」或直接运行 `python run_quantize.py --help`。

---

## 配置 API Key（启用 AI 卡通化）

AI 卡通化是可选项——不配置也能正常生成拼豆图纸。若想用「照片 → 卡通」功能，按以下步骤操作：

1. 打开软件，点工具栏「**⚙️ 设置**」按钮
2. 在弹窗中填入两个 Key：
   - **对话型 LLM**：用于理解照片关键特征（默认 MiniMax-M3，国内直连）
   - **方舟生图**：用于照片转卡通（豆包 SeedDream）
3. 点「测试连接」验证（会消耗少量 token），通过后即可使用「卡通化当前图片」

> 密钥只保存在你自己的电脑本地，不会上传；其他 LLM 服务（openai / anthropic / gemini / qwen 等）也可在设置里一键切换。详见操作手册第 3 节。

---

## 算法说明

### 像素化方法（`core/pixelize.py`）

| 方法 | 原理 | 特点 |
|------|------|------|
| **PixelOE** | 对比度感知轮廓扩展 + 对比度感知下采样 | 描边清晰、细节保留最好（纯 numpy 移植，见引用） |
| **DPID** | 细节保持下采样（SIGGRAPH Asia 2016） | 按与参考色差异加权，锐利、保细节 |
| **AREA** | 面积插值（cv2.INTER_AREA） | 简单快速、抗锯齿、柔和 |

### 量化管线（`core/quantize.py`）

```
原图 → [线条检测] → 4x 超采样量化 → 分层决策选色 → 下采样 → 小色块清理 → 结果
```

- **分层决策**：线条 → 轮廓 → 细节 → 朴素投票，逐层优先选色
- **自适应线条色**：不写死「线条=黑色」，按与背景色 ΔE 最大原则取色
- **小色块清理**：ΔE 感知，孤立黑点（眼线/瞳孔）保留，杂点合并
- **色彩空间**：统一 CIE Lab + ΔE76

---

## 已知限制

- **52×52 网格装不下完整胡须**：最细 1 像素宽的细节(如胡须)压缩后将大概率丢失，需提高 grid 到 64/80（但超出单块实体板）
- **描边粗细与「干净度」对立**：`line_aware` 加粗描边会引入更多黑色像素
- **Frangi 抓不到黑线**（算法设计本意），不建议用于本场景
- **打包体积约 256 MB**（含 Qt 运行时）

---

## AI 工具声明

本项目开发过程中使用了以下 AI 工具辅助编码、设计与文档撰写。核心算法均为自研实现或基于开源算法的手写移植。

### 开发辅助

| 工具 / 模型 | 用途 |
|-------------|------|
| WorkBuddy（腾讯） | 主开发环境：代码生成重构、GUI 美化、打包排障、文档撰写 |
| DeepSeek-V4-Pro | 代码逻辑与算法对齐的推理辅助（如纯 numpy 复现 torch 语义） |
| MiniMax Code | 编码辅助、代码片段生成与 review |

### 运行时生成式能力

| 模型 / 服务 | 用途 |
|-------------|------|
| MiniMax-M3 | 多模态视觉理解（关键特征 + 卡通风格描述） |
| 豆包 SeedDream（火山方舟） | 图生图卡通化 |
| 豆包 SeedEdit（火山方舟） | 局部重绘（可选，消除水印/杂物） |

> **隐私说明**：所有 API Key 存于本地 `.env`（已忽略），图片仅在用户主动点击「卡通化」时才上传至云端。

---

## 算法引用与致谢

| 算法 / 来源 | 用途 | 许可 |
|-------------|------|------|
| **PixelOE** — [KohakuBlueleaf/PixelOE](https://github.com/KohakuBlueleaf/PixelOE) | 像素化（pixeloe），纯 numpy 移植，与官方 torch 实现逐像素对齐 | Apache-2.0 |
| **DPID** — Weber et al., *Rapid, Detail-Preserving Image Downscaling*, SIGGRAPH Asia 2016（[mergian/dpid](https://github.com/mergian/dpid)） | 像素化（dpid），numpy 移植自官方 Matlab 版 | 学术论文 |
| **cv2.INTER_AREA** — [OpenCV](https://opencv.org/) | 像素化（area），面积插值下采样 | Apache-2.0 |
| **MARD 221 色板** | 拼豆色号体系 | — |

> PixelOE 的对比度感知轮廓扩展与对比度感知下采样在本项目中用纯 numpy 忠实重写（`core/pixelize.py`），
> DPID 从官方 Matlab 版 `dpid.m` 移植，特此致谢原作者。

---

*README 最后更新：2026-10-27 · 英文版见 [README.en.md](README.en.md)*
