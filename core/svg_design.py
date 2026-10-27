"""
svg_design.py
让多模态 LLM "看懂"原图后，直接输出一份为拼豆网格量身定做的卡通几何描述。

设计原则:
- LLM 不直接吐 SVG 字符串(易错)，而是吐结构化 JSON。
- 坐标采用归一化 (0~1)，由 renderer 精确映射到目标网格。
- 每个形状用一个语义 role（fur_white / eye / nose / ...）+ 可选 hex 近似色，
  后续可被 MARD 色板量化模块替换成实际色号。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient


SYSTEM_PROMPT = """你是一名擅长把照片转译为"扁平卡通拼豆图纸"的几何设计师。

【拼豆工艺硬约束】
- 画布会被等分为 GRID_W x GRID_H 的网格（默认 58x58）。
- 画布坐标系: 左上角 (0,0)，右下角 (1,1)。所有形状 bbox 必须落在 [0,1]。
- 每个形状的"最小可识别尺寸"建议:
    眼睛 >= 4x4 格  → bbox 边长 >= 4/GRID_W
    鼻 / 嘴 >= 2x2 格 → 边长 >= 2/GRID_W
    耳朵主体 >= 6x6 格
- 避免在同一语义区域里出现 2 个仅相差几个像素的小色块（会形成杂色）。
- 颜色块越少越好，但必须保留识别度。

【输出严格 JSON Schema】
__SCHEMA_PLACEHOLDER__

【字段说明】
- type: rect | ellipse | polygon
- bbox: [x1, y1, x2, y2]，左上→右下
- polygon 不写 bbox，写 "points": [[x,y], [x,y], ...]
- z 小的先画（背景 0, 主体 5~10, 五官 15~20, 描边 25）
- fill_role / stroke_role 必须是 palette 中定义过的 role，否则视为 background
- 所有数值保留 3 位小数以内

【保留特征】
{preserve_features}

【风格倾向】
{style_hint}

【目标网格】
{grid_w} x {grid_h}

只输出 JSON，不要任何解释或 markdown 围栏。
"""


# JSON Schema 单独定义，避免与 str.format 冲突
SCHEMA_EXAMPLE = """
{
  "canvas": {
    "grid_w": 58,
    "grid_h": 58,
    "bg_color_role": "background",
    "bg_color_hex": "#D7DCE0"
  },
  "palette": [
    {"role": "fur_white",     "hex": "#F5F1E8", "note": "主体白毛"},
    {"role": "fur_shadow",    "hex": "#C9C2B0", "note": "米灰阴影"},
    {"role": "ear_inner",     "hex": "#F2C4C4", "note": "耳腔淡粉"},
    {"role": "eye_fill",      "hex": "#5E6B3A", "note": "灰绿虹膜"},
    {"role": "nose_mouth",    "hex": "#B98A78", "note": "粉棕鼻口"},
    {"role": "outline",       "hex": "#3A3631", "note": "深色描边/眼缝"},
    {"role": "background",    "hex": "#D7DCE0", "note": "背景蓝灰"}
  ],
  "shapes": [
    {
      "id": "bg",
      "z": 0,
      "type": "rect",
      "bbox": [0.0, 0.0, 1.0, 1.0],
      "fill_role": "background",
      "stroke_role": null,
      "stroke_width": 0
    },
    {
      "id": "body",
      "z": 1,
      "type": "ellipse",
      "bbox": [0.10, 0.32, 0.92, 0.96],
      "fill_role": "fur_white",
      "stroke_role": "fur_shadow",
      "stroke_width": 0.006
    }
  ],
  "design_notes": [
    "白猫正面坐姿，头部居中略偏上",
    "保留不耐烦神情: 双眼半眯成横向眼缝，耳朵后压"
  ]
}
""".strip()


USER_PROMPT_TEMPLATE = """请分析这张照片，按 system 里的 schema 输出"为拼豆设计的几何描述 JSON"。

【必须保留】
{preserve_features}

【风格】
{style_hint}

【构图建议】
- 主体居中，占据画面 50%~85% 高度
- 头部比例适当放大（卡通化），眼睛/鼻/嘴符合工艺硬约束
- 背景纯色铺底，必要时加 1~2 个低饱和辅助色块作"地面"
- 描边只用在关键轮廓（眼缝、嘴缝、鼻轮廓、耳朵外缘），不要到处描
- 同一语义区域只用 1 种主色 + 最多 1 种阴影色

只输出 JSON。
"""


@dataclass
class SVGDesign:
    """LLM 返回的结构化几何描述"""
    raw: Dict[str, Any]
    canvas: Dict[str, Any] = field(default_factory=dict)
    palette: List[Dict[str, Any]] = field(default_factory=list)
    shapes: List[Dict[str, Any]] = field(default_factory=list)
    design_notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SVGDesign":
        return cls(
            raw=d,
            canvas=d.get("canvas", {}) or {},
            palette=d.get("palette", []) or [],
            shapes=d.get("shapes", []) or [],
            design_notes=d.get("design_notes", []) or [],
        )

    def grid_size(self) -> tuple[int, int]:
        w = int(self.canvas.get("grid_w", 58))
        h = int(self.canvas.get("grid_h", 58))
        return w, h


def design_svg(
    image_b64: str,
    image_mime: str,
    *,
    preserve_features: str = "不耐烦的神情（半眯眼、耳朵后压、嘴部紧抿）",
    style_hint: str = "日系平涂/扁平卡通，色块干净，线条简洁",
    grid_w: int = 58,
    grid_h: int = 58,
    client: Optional[LLMClient] = None,
) -> SVGDesign:
    cli = client or LLMClient()
    system = SYSTEM_PROMPT.format(
        preserve_features=preserve_features,
        style_hint=style_hint,
        grid_w=grid_w,
        grid_h=grid_h,
    ).replace("__SCHEMA_PLACEHOLDER__", SCHEMA_EXAMPLE)
    user = USER_PROMPT_TEMPLATE.format(
        preserve_features=preserve_features,
        style_hint=style_hint,
    )
    text = cli.chat_with_image(
        text_prompt=user,
        image_b64=image_b64,
        image_mime=image_mime,
        system=system,
        response_format_json=True,
    )
    parsed = LLMClient.parse_json(text)
    return SVGDesign.from_dict(parsed)


def design_svg_file(
    image_path: str,
    *,
    preserve_features: str = "不耐烦的神情（半眯眼、耳朵后压、嘴部紧抿）",
    style_hint: str = "日系平涂/扁平卡通，色块干净，线条简洁",
    grid_w: int = 58,
    grid_h: int = 58,
    max_side: int = 1024,
    client: Optional[LLMClient] = None,
) -> SVGDesign:
    from .image_utils import file_to_base64
    b64, mime, _img = file_to_base64(
        image_path,
        max_side=max_side,
        fmt="jpeg",
        quality=85,
    )
    return design_svg(
        b64,
        mime,
        preserve_features=preserve_features,
        style_hint=style_hint,
        grid_w=grid_w,
        grid_h=grid_h,
        client=client,
    )


__all__ = ["SVGDesign", "design_svg", "design_svg_file"]
