"""
seedream.py
豆包 SeedDream 图生图客户端（方舟 ARK）。

API 字段以最新 curl 为准:
  model, prompt, image (URL or data URL), response_format, size, stream, watermark
  i2i 额外支持 image_strength
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .config_loader import get_ark_config


class SeedreamError(Exception):
    """SeedDream 调用失败（含服务端返回的错误码）。"""

    def __init__(self, message: str, *, code: str = "", status: int = 0):
        super().__init__(message)
        self.code = code          # 火山方舟错误码（如 InputTextSensitiveContentDetected）
        self.status = status      # HTTP 状态码


# 火山方舟错误码分类（参考官方 error-codes 文档）
# 敏感内容类错误：重试同样的 prompt 无效，必须改写输入内容
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
# 可退避重试的瞬时错误（限流 / 服务过载 / 内部错误）
_RETRYABLE_CODES = {
    "RateLimitExceeded.EndpointRPMExceeded",
    "RateLimitExceeded.EndpointTPMExceeded",
    "RateLimitExceeded.EndpointFlexTPMExceeded",
    "ModelAccountRpmRateLimitExceeded",
    "ModelAccountTpmRateLimitExceeded",
    "ModelAccountFlexTpmRateLimitExceeded",
    "APIAccountRpmRateLimitExceeded",
    "ModelAccountIpmRateLimitExceeded",
    "ServerOverloaded",
    "RequestBurstTooFast",
    "RequestTimeout",
    "InternalServiceError",
    "ModelLoading",
    "ServiceConnectionRefused",
    "NoAvailableService",
    "ContentSecurityDetectionError",
}


def _extract_error_code(r_text: str) -> tuple[str, int]:
    """从错误响应文本里提取 (code, http_status)。"""
    code = ""
    try:
        data = __import__("json").loads(r_text)
        code = (data.get("error") or {}).get("code", "") or ""
    except Exception:
        pass
    return code


class SeedreamClient:
    """豆包 SeedDream 图生图"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        ark = cfg or get_ark_config()
        if not ark:
            raise SeedreamError("config.yaml 中没有 ark 节点")
        if not ark.get("api_key"):
            raise SeedreamError("ark.api_key 为空，请填入方 API Key")

        self.api_key = ark["api_key"]
        self.base_url = ark.get("base_url", "https://ark.cn-beijing.volces.com/api/v3/images/generations")
        self.timeout = float(ark.get("timeout", 120))
        self.max_retries = int(ark.get("max_retries", 3))
        proxy = (ark.get("proxy") or "").strip()
        self.session = requests.Session()
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 节点参数
        cart = ark.get("cartoonize") or {}
        self.default_model = cart.get("model", "doubao-seedream-5-0-pro-260628")
        self.default_strength = float(cart.get("image_strength", 0.65))
        self.default_size = cart.get("size", "2K")

    # ------------------------------------------------------------------
    # 单次图生图
    # ------------------------------------------------------------------
    def image_to_image(
        self,
        prompt: str,
        image: str,
        *,
        model: Optional[str] = None,
        image_strength: Optional[float] = None,
        size: Optional[str] = None,
        response_format: str = "url",
        watermark: bool = True,
    ) -> Dict[str, Any]:
        """
        发起一次图生图。

        :param prompt: 文字提示词
        :param image: 输入图片（https URL 或 data:image/...;base64,...）
        :param model: 覆盖默认模型
        :param image_strength: 0~1
        :param size: "1K" | "2K" | "4K" | "WxH"
        :param response_format: "url" | "b64_json"
        :param watermark: 是否加水印
        :return: 原始响应 JSON（data[0] 里有 url 或 b64_json）
        """
        payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "prompt": prompt,
            "image": image,
            "response_format": response_format,
            "size": size or self.default_size,
            "stream": False,
            "watermark": watermark,
        }
        if image_strength is not None:
            payload["image_strength"] = float(image_strength)

        return self._post_with_retry(payload)

    # ------------------------------------------------------------------
    # 内部：单次 POST（不重试，把错误码透传给上层）
    # ------------------------------------------------------------------
    def _post_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        r = self.session.post(
            self.base_url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        elapsed = time.time() - t0
        if r.status_code != 200:
            code = _extract_error_code(r.text)
            raise SeedreamError(
                f"HTTP {r.status_code} ({elapsed:.1f}s): {r.text[:500]}",
                code=code,
                status=r.status_code,
            )
        data = r.json()
        if "data" not in data or not data["data"]:
            raise SeedreamError(f"响应里没有 data 字段: {data}")
        return data

    # ------------------------------------------------------------------
    # 内部：带重试的 POST（只对瞬时错误退避重试，敏感词等硬错误直接抛）
    # ------------------------------------------------------------------
    def _post_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._post_once(payload)
            except SeedreamError as e:
                last_err = e
                # 敏感词/参数/鉴权等硬错误：重试无效，直接抛出
                if e.code and (e.code in _SENSITIVE_CODES or e.status < 500):
                    if e.code not in _RETRYABLE_CODES:
                        raise e
                print(f"  [Seedream] 第 {attempt}/{self.max_retries} 次失败: {e}")
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))
            except requests.RequestException as e:
                last_err = e
                print(f"  [Seedream] 第 {attempt}/{self.max_retries} 次网络失败: {e}")
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))
        raise SeedreamError(f"重试 {self.max_retries} 次仍失败: {last_err}")


# =============================================================================
# 工具：把本地图片转成 base64 data URL
# =============================================================================
def file_to_data_url(
    path: str | Path,
    max_side: int = 1024,
    quality: int = 85,
) -> str:
    """
    读本地图片 → 缩放 → 编码为 data URL。
    data URL 形式: data:image/jpeg;base64,xxxxx
    """
    from PIL import Image

    p = Path(path)
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # 缩放
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def download_result_url(url: str, save_path: str | Path, timeout: int = 120) -> Path:
    """下载 SeedDream 返回的图片到本地"""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(r.content)
    return p


__all__ = ["SeedreamClient", "SeedreamError", "file_to_data_url", "download_result_url"]
