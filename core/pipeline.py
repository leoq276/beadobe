"""
pipeline.py
Stage 0/1 的可复用管线逻辑，供 CLI 与 GUI 共用：

  Stage 0  run_understand()   多模态 LLM 视觉理解 -> VisualAnalysis + 关键特征 bbox
  Stage 1  run_cartoonize()   SeedDream 图生图（照片转卡通）-> 本地 PNG

关键点：把 LLM 返回的 key_features（{name, position:[cx,cy]}）转换成
quantize() 需要的 bbox 格式（{name: [x1,y1,x2,y2]} 归一化）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_loader import load_config, get_image_config
from .llm_client import LLMClient
from .visual_understand import analyze_image_file, VisualAnalysis
from .seedream import SeedreamClient, SeedreamError, file_to_data_url, download_result_url

# 敏感内容错误码（来自 seedream 模块），用于判断是否需要改写 prompt
_SENSITIVE_CODES = {
    "InputTextSensitiveContentDetected",
    "InputImageSensitiveContentDetected",
    "InputVideoSensitiveContentDetected",
    "InputAudioSensitiveContentDetected",
    "OutputTextSensitiveContentDetected",
    "OutputImageSensitiveContentDetected",
    "OutputVideoSensitiveContentDetected",
    "OutputAudioSensitiveContentDetected",
    "SensitiveContentDetected",
    "SensitiveContentDetected.SevereViolation",
    "SensitiveContentDetected.Violence",
    "InputTextSensitiveContentDetected.PolicyViolation",
    "InputImageSensitiveContentDetected.PolicyViolation",
    "OutputImageSensitiveContentDetected.PolicyViolation",
    "InputImageSensitiveContentDetected.PrivacyInformation",
    "InputTextRiskDetection",
    "InputImageRiskDetection",
    "OutputTextRiskDetection",
    "OutputImageRiskDetection",
}


# 各关键特征默认占图比例 (宽, 高)，用于把中心点扩展成 bbox
FEATURE_DEFAULT_SIZE: Dict[str, tuple] = {
    "eye": (0.075, 0.045),
    "ear": (0.075, 0.085),
    "mouth": (0.060, 0.022),
    "nose": (0.030, 0.025),
    "brow": (0.060, 0.020),
    "whisker": (0.080, 0.015),
    "lid": (0.060, 0.020),
    "head": (0.55, 0.50),
    "face": (0.50, 0.45),
}


# 敏感词 → 中性替换（兜底清洗，避免触发方舟 SeedDream 的敏感内容审核）
_SENSITIVE_REPLACEMENTS: Dict[str, str] = {
    "不耐烦": "平静",
    "冷淡": "平静",
    "嫌弃": "平静",
    "厌恶": "平静",
    "凶狠": "温和",
    "生气": "平静",
    "愤怒": "平静",
    "下压嘴角": "嘴角自然",
    "紧抿": "自然闭合",
    "紧绷": "放松",
    "敌对": "平静",
    "警惕": "专注",
    "疏离": "平静",
    "催促": "注视",
    "不想被打扰": "安静独处",
}

# 更强的负面词 → 中性替换（敏感词报错后的激进改写兜底）
_AGGRESSIVE_REPLACEMENTS: Dict[str, str] = {
    # 情绪/神情负面词
    "不耐烦": "平静专注",
    "烦躁": "平静",
    "冷淡": "温和",
    "冷漠": "温和",
    "嫌弃": "平静",
    "厌恶": "平静",
    "凶狠": "温和",
    "凶": "温和",
    "生气": "平静",
    "愤怒": "平静",
    "发怒": "平静",
    "敌对": "友好",
    "警惕": "专注",
    "疏离": "平和",
    "戒备": "放松",
    "催促": "注视",
    "不耐烦地": "平静地",
    # 嘴部负面
    "下压嘴角": "嘴角自然",
    "嘴角下压": "嘴角自然",
    "撇嘴": "嘴角自然",
    "紧抿": "自然闭合",
    "抿嘴": "自然闭合",
    "噘嘴": "自然闭合",
    # 身体/情绪紧张
    "紧绷": "放松",
    "紧张": "放松",
    "僵直": "自然",
    "僵硬": "自然",
    "不想被打扰": "安静独处",
    "被打扰": "独处",
    # 眼神负面
    "瞪": "注视",
    "斜视": "注视",
    "翻白眼": "注视",
    "怒视": "注视",
}


def sanitize_prompt(text: str, *, aggressive: bool = False) -> str:
    """把 prompt 里可能触发图生图平台审核的负面情绪词替换为中性表述。

    :param aggressive: True 时使用更激进的替换表（敏感词报错后重试用）
    """
    out = text
    table = _AGGRESSIVE_REPLACEMENTS if aggressive else _SENSITIVE_REPLACEMENTS
    for bad, good in table.items():
        out = out.replace(bad, good)
    return out


def _rewrite_prompt_after_sensitive(prompt: str, attempt: int) -> str:
    """敏感词报错后改写 prompt。

    策略（逐次加强）：
      1. 激进替换负面词
      2. 再追加一段明确的中性声明，要求用正面词汇
    """
    out = sanitize_prompt(prompt, aggressive=True)
    if attempt >= 1:
        out = (
            "请以平静、友好、自然的卡通风格重新绘制这张图片，"
            "保留主体姿态与五官特征，情绪表达积极正面。\n"
            + out
        )
    if attempt >= 2:
        out = (
            "生成一张清新可爱的卡通插画，主体安静平和，"
            "五官线条清晰，色彩干净明快。\n"
            + out
        )
    return out


@dataclass
class PipelineResult:
    """Stage 0/1 完整结果"""
    analysis: Optional[VisualAnalysis] = None
    cartoon_png: Optional[Path] = None
    key_features: Optional[Dict[str, List[float]]] = None


def features_to_bboxes(analysis: VisualAnalysis) -> Dict[str, List[float]]:
    """把 LLM 的 key_features（中心点）转成 quantize 需要的 bbox 字典。"""
    out: Dict[str, List[float]] = {}
    for f in analysis.key_features:
        name = f.get("name") or ""
        if not name:
            continue
        pos = f.get("position")
        if not pos or len(pos) < 2:
            continue
        cx, cy = float(pos[0]), float(pos[1])
        sw, sh = (0.05, 0.04)  # 默认
        name_lower = name.lower()
        for kw, sz in FEATURE_DEFAULT_SIZE.items():
            if kw in name_lower:
                sw, sh = sz
                break
        out[name] = [
            max(0.0, cx - sw / 2),
            max(0.0, cy - sh / 2),
            min(1.0, cx + sw / 2),
            min(1.0, cy + sh / 2),
        ]
    return out


def run_understand(
    image_path: str | Path,
    *,
    preserve_features: str = "不耐烦的神情（半眯眼、耳朵后压、嘴部紧抿）",
    style_hint: str = "极简扁平卡通：大块纯色平涂、线条越少越好、无纹理无渐变无阴影细节，仅保留关键轮廓描边，便于下采样到拼豆网格",
    progress_cb: Optional[callable] = None,
) -> VisualAnalysis:
    """Stage 0：调用多模态 LLM 做视觉理解。"""
    def _p(msg: str):
        if progress_cb:
            progress_cb(msg)

    cfg = load_config()
    img_cfg = get_image_config(cfg)

    _p("建立 LLM 连接...")
    cli = LLMClient()

    _p("多模态 LLM 视觉理解中（约 10-30 秒）...")
    return analyze_image_file(
        str(image_path),
        preserve_features=preserve_features,
        style_hint=style_hint,
        max_side=img_cfg["max_side"],
        client=cli,
    )


def run_cartoonize(
    image_path: str | Path,
    prompt_cn: str,
    out_png: str | Path,
    *,
    progress_cb: Optional[callable] = None,
    max_prompt_retries: int = 3,
) -> Path:
    """Stage 1：用 SeedDream 图生图，返回保存路径。

    遇到敏感词审核错误（InputTextSensitiveContentDetected 等）时，
    自动改写 prompt 并重试（最多 max_prompt_retries 次），而非盲目重发同样的 prompt。
    """
    def _p(msg: str):
        if progress_cb:
            progress_cb(msg)

    cfg = load_config()
    img_cfg = get_image_config(cfg)
    cart = cfg["ark"]["cartoonize"]

    # 兜底清洗：替换负面情绪词，避免触发方舟敏感内容审核
    prompt_cn = sanitize_prompt(prompt_cn)

    _p("编码原图为 data URL...")
    data_url = file_to_data_url(
        str(image_path),
        max_side=img_cfg["max_side"],
        quality=img_cfg["quality"],
    )

    _p(f"SeedDream 图生图（模型 {cart['model']}，约 10-60 秒）...")
    cli = SeedreamClient()
    current_prompt = prompt_cn
    resp = None

    for attempt in range(max_prompt_retries + 1):
        try:
            resp = cli.image_to_image(
                prompt=current_prompt,
                image=data_url,
                image_strength=float(cart["image_strength"]),
                size=cart["size"],
                response_format="url",
                watermark=True,
            )
            break  # 成功
        except SeedreamError as e:
            # 敏感词错误：改写 prompt 后重试（而非重发同样的内容）
            if e.code in _SENSITIVE_CODES and attempt < max_prompt_retries:
                _p(f"⚠ 敏感词审核拦截（{e.code}），第 {attempt+1}/{max_prompt_retries} 次改写 prompt 重试...")
                current_prompt = _rewrite_prompt_after_sensitive(current_prompt, attempt)
                continue
            # 其他错误：直接抛出
            raise

    if resp is None:
        raise SeedreamError(
            f"改写 prompt 重试 {max_prompt_retries} 次后仍触发敏感词审核，"
            f"请手动检查并调整 prompt 内容"
        )

    item = resp["data"][0]
    result_url = item.get("url")
    if not result_url:
        raise SeedreamError(f"响应里没有 url: {item}")

    _p("下载卡通化结果...")
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    download_result_url(result_url, out)

    # 保存响应元数据
    meta = Path(out_png).with_suffix(".response.json")
    meta.write_text(
        json.dumps({
            "model": resp.get("model", cart["model"]),
            "created": resp.get("created"),
            "usage": resp.get("usage"),
            "result_url": result_url,
            "prompt": current_prompt,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def run_full_pipeline(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    stem: Optional[str] = None,
    preserve_features: str = "不耐烦的神情（半眯眼、耳朵后压、嘴部紧抿）",
    style_hint: str = "极简扁平卡通：大块纯色平涂、线条越少越好、无纹理无渐变无阴影细节，仅保留关键轮廓描边，便于下采样到拼豆网格",
    custom_prompt: Optional[str] = None,
    progress_cb: Optional[callable] = None,
) -> PipelineResult:
    """一次性跑完 Stage 0 + Stage 1，返回结构化结果。

    :param custom_prompt: 若提供，则跳过 LLM 自动生成的 cartoon_prompt_cn，
        直接用这段自定义 prompt 做卡通化（Stage 0 仍会跑，用于关键特征 bbox）。
    """
    img = Path(image_path)
    stem = stem or img.stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stage 0: 理解
    analysis = run_understand(
        img,
        preserve_features=preserve_features,
        style_hint=style_hint,
        progress_cb=progress_cb,
    )

    # 保存分析 JSON 和 prompt
    (out / f"{stem}_analysis.json").write_text(
        json.dumps(analysis.raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if progress_cb:
        progress_cb("已保存视觉分析 JSON")

    # 决定最终使用的卡通化 prompt：自定义优先，否则用 LLM 生成的
    final_prompt = (custom_prompt or "").strip() or analysis.cartoon_prompt_cn

    # Stage 1: 卡通化
    cartoon_png = run_cartoonize(
        img,
        final_prompt,
        out / f"{stem}_seedream.png",
        progress_cb=progress_cb,
    )

    # 转换 key_features
    key_features = features_to_bboxes(analysis)

    return PipelineResult(
        analysis=analysis,
        cartoon_png=cartoon_png,
        key_features=key_features,
    )


__all__ = [
    "PipelineResult",
    "FEATURE_DEFAULT_SIZE",
    "features_to_bboxes",
    "run_understand",
    "run_cartoonize",
    "run_full_pipeline",
]
