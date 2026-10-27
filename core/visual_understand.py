"""
visual_understand.py
让多模态 LLM "看懂" 原图，输出结构化分析：
- 主体、姿态、神情
- 关键特征及位置（眼睛、嘴、耳朵等，归一化坐标）
- 配色
- 为拼豆网格量身定做的卡通 prompt（中英双语）
- 推荐网格尺寸
- 简化建议

这是把"重设计"这一步交给 LLM 的关键模块。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient


SYSTEM_PROMPT = """你是一名同时精通摄影、卡通插画与拼豆工艺的视觉设计师。
你的任务：看用户提供的照片，输出一份为「拼豆图纸」量身定做的视觉分析报告。

【拼豆工艺约束（你必须遵守）】
- 拼豆网格典型尺寸 16~100，推荐 80~100 网格（约 7~9cm 实物大小）。
- 每个网格格子里只能放 1 种颜色（来自固定色板），相邻格子颜色可以不同。
- 卡通化必须考虑最低可识别尺寸：眼睛建议至少 6x6 格，嘴/鼻至少 2x2 格。
- 颜色块越少越干净，但必须保留识别度（不要简化掉关键特征）。
- 关键：所有面部特征（眼缝、嘴缝、鼻轮廓、耳朵外缘）必须用**深色（黑/深灰）画清晰线条**。
  这是因为量化阶段会做"块投票 + 细节优先"算法，只有"少数但强对比"的色会胜出。
  如果五官没有深色硬边，会被主色稀释、细节丢失。

【极简图案约束（核心：线条纹理越少越好、图案越简单越好）】
- 目标是把照片转成"最干净的扁平色块卡通"，而不是写实插画。
- 只保留可识别的**关键轮廓线**（脸/身体外轮廓 + 眼缝/嘴缝/鼻轮廓），
  其余一切**装饰性细节一律去除**：
  * 不画毛发纹理、皮毛条纹、绒毛、发丝（主体毛发用整块纯色填充）
  * 不画衣物褶皱、面料纹理、装饰花纹、LOGO、文字
  * 不画背景细节（背景铺 1~2 个纯色块，最多一个简单色块）
  * 不画阴影渐变、高光渐变、环境反射（阴影最多 1 层，用整块纯色）
  * 不画多余的短线、点缀、噪点、斑纹
- 颜色用**大块纯色平涂**，每种颜色尽可能连成整片区域，减少颜色种类。
- 配色控制在 6~10 个主色以内（不含黑色描边色）。
- 判别标准：如果某个细节在 80~100 网格下小于 2x2 格、看不出是什么，
  就直接删掉它，不要画。

【措辞约束（重要：避免触发图片生成平台的敏感词审核）】
- 描述神情/情绪时，用**客观、中性的姿态描写**，不要使用负面情绪词。
- 禁止在 cartoon_prompt 里出现以下词或近似表述：不耐烦、冷淡、嫌弃、厌恶、凶狠、生气、
  愤怒、下压嘴角、紧抿、紧绷、敌对、警惕、疏离等。
- 改用中性表达，例如把"不耐烦的神情"改写成"平静而专注的表情，眼睛半闭呈放松状态，
  耳朵自然向两侧展开，嘴部自然闭合"。
- 目标是让图生图模型画出**同样的五官姿态**，但用词正面、中性、客观。

【你的输出】
严格按下方 JSON 结构输出，不要包含任何 JSON 之外的文字或 markdown 围栏。
"""


USER_PROMPT_TEMPLATE = """请分析这张照片，输出一份「拼豆用」视觉分析 JSON。

【必须保留的特征】
{preserve_features}

【可选的风格倾向】
{style_hint}

【拼豆量化阶段的硬约束（必须在 cartoon_prompt 中体现）】
- 眼睛、嘴缝、鼻轮廓、耳朵外缘**必须用深色（黑色/深灰）**画清晰线条
  → 量化时这些"少数但对比强"的色才能在投票中胜出
- 五官不能"嵌入"主色：瞳孔用深色、虹膜用饱和的中等色、眼缝用纯黑
- 眼睛整体占网格面积要大（最终眼睛在 100x100 网格下至少 6x6 格）
- 嘴缝要清晰画一条深色短线，不要让它融在嘴部主色里
- 避免半透明、避免渐变——所有边缘都是硬边

【线条与图案极简约束（必须在 cartoon_prompt 中明确写出）】
- 整体走"最简扁平色块"风格：大块纯色平涂，无纹理、无渐变、无阴影细节
- 只保留关键轮廓线（五官 + 主体外轮廓），删除一切装饰性线条：
  毛发纹理、皮毛条纹、衣物褶皱、背景细节、高光、噪点统统不要
- 主体毛发/衣物用整块纯色填充，不要画发丝、绒毛、条纹
- 背景用 1~2 个纯色块（或纯白），不要画任何背景物件或细节
- 阴影最多 1 层整块纯色，不做渐变
- 配色控制在 6~10 个主色（不含黑色描边），颜色越少越好
- 任何在 100x100 网格下小于 2x2 格、无法辨认的细节，一律删除不画
- cartoon_prompt 里要显式写上"flat solid colors, minimal outlines, no texture, no gradient, simple clean shapes"之类字样

【输出 JSON Schema（严格遵守字段名）】
{{
  "subject": "一句话主体描述（如：白色短毛猫）",
  "expression": "神情描述（必须详尽，包括眼神、嘴型、耳朵姿态、体态语言）",
  "pose": "姿态描述",
  "key_features": [
    {{
      "name": "left_eye",
      "label_cn": "左眼",
      "position": [0.42, 0.35],
      "importance": "high",
      "design_tip": "建议占 6x6 网格以上，半眯眼，深色眼缝+饱和虹膜+小高光"
    }}
  ],
  "palette": [
    {{"role": "fur_white", "color_cn": "纯白", "hex_approx": "#F5F1E8"}},
    {{"role": "fur_shadow", "color_cn": "米灰阴影", "hex_approx": "#C9C2B0"}}
  ],
  "cartoon_prompt_cn": "中文 prompt，给图生图模型用，重点保留「{preserve_features}」+ 五官深色硬边 + 极简扁平色块（无纹理无渐变无多余线条）",
  "cartoon_prompt_en": "English prompt for image-to-image model, emphasizing the expression to preserve, bold dark outlines only on key facial features, flat solid colors, minimal outlines, no texture, no gradient, simple clean shapes",
  "recommended_grid": {{"w": 100, "h": 100, "reason": "为何这个尺寸最合适"}},
  "simplify_suggestions": [
    "把毛发纹理全部去除，主体用整块纯色填充",
    "背景铺 1 个纯色块，删除所有背景细节",
    "只保留眼缝/嘴缝/外轮廓等关键描边，其余线条删除",
    "阴影最多保留 1 层纯色，不做渐变"
  ],
  "risk_warnings": [
    "耳朵内部粉色若网格过小会糊成单色，建议至少 6x6"
  ]
}}

position 字段必须是 [x, y] 归一化坐标（0~1，左上角为原点）。
importance 取值: high | medium | low。
output ONLY the JSON, no commentary。
"""


@dataclass
class VisualAnalysis:
    """LLM 返回的结构化分析"""
    raw: Dict[str, Any]
    subject: str = ""
    expression: str = ""
    pose: str = ""
    key_features: List[Dict[str, Any]] = field(default_factory=list)
    palette: List[Dict[str, Any]] = field(default_factory=list)
    cartoon_prompt_cn: str = ""
    cartoon_prompt_en: str = ""
    recommended_grid: Dict[str, Any] = field(default_factory=dict)
    simplify_suggestions: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualAnalysis":
        return cls(
            raw=d,
            subject=d.get("subject", ""),
            expression=d.get("expression", ""),
            pose=d.get("pose", ""),
            key_features=d.get("key_features", []) or [],
            palette=d.get("palette", []) or [],
            cartoon_prompt_cn=d.get("cartoon_prompt_cn", ""),
            cartoon_prompt_en=d.get("cartoon_prompt_en", ""),
            recommended_grid=d.get("recommended_grid", {}) or {},
            simplify_suggestions=d.get("simplify_suggestions", []) or [],
            risk_warnings=d.get("risk_warnings", []) or [],
        )

    def short_summary(self) -> str:
        """给人看的简明摘要"""
        lines = [
            f"主体: {self.subject}",
            f"神情: {self.expression}",
            f"姿态: {self.pose}",
            f"推荐网格: {self.recommended_grid.get('w')}x{self.recommended_grid.get('h')}",
            f"关键特征数: {len(self.key_features)}",
            f"调色板色数: {len(self.palette)}",
        ]
        return "\n".join(lines)


def analyze_image(
    image_b64: str,
    image_mime: str,
    *,
    preserve_features: str = "不耐烦的神情（半眯眼、耳朵后压、嘴部紧抿）",
    style_hint: str = "极简扁平卡通：大块纯色平涂、线条越少越好、无纹理无渐变无阴影细节，仅保留关键轮廓描边",
    client: Optional[LLMClient] = None,
) -> VisualAnalysis:
    """
    主入口：分析图片，返回结构化结果。

    :param image_b64: base64 字符串
    :param image_mime: image/jpeg 或 image/png
    :param preserve_features: 必须保留的特征描述
    :param style_hint: 风格倾向
    :param client: 可复用 LLMClient
    """
    cli = client or LLMClient()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        preserve_features=preserve_features,
        style_hint=style_hint,
    )
    text = cli.chat_with_image(
        text_prompt=user_prompt,
        image_b64=image_b64,
        image_mime=image_mime,
        system=SYSTEM_PROMPT,
        response_format_json=True,
    )
    parsed = LLMClient.parse_json(text)
    return VisualAnalysis.from_dict(parsed)


def analyze_image_file(
    image_path: str,
    *,
    preserve_features: str = "不耐烦的神情（半眯眼、耳朵后压、嘴部紧抿）",
    style_hint: str = "极简扁平卡通：大块纯色平涂、线条越少越好、无纹理无渐变无阴影细节，仅保留关键轮廓描边",
    max_side: int = 1024,
    client: Optional[LLMClient] = None,
) -> VisualAnalysis:
    """
    便捷入口：直接传文件路径。
    """
    from .image_utils import file_to_base64
    b64, mime, _img = file_to_base64(
        image_path,
        max_side=max_side,
        fmt="jpeg",
        quality=85,
    )
    return analyze_image(
        b64,
        mime,
        preserve_features=preserve_features,
        style_hint=style_hint,
        client=client,
    )


__all__ = [
    "VisualAnalysis",
    "analyze_image",
    "analyze_image_file",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
]
