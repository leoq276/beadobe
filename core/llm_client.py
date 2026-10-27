"""
llm_client.py
OpenAI 兼容 API 客户端（支持 codexcn 等代理）。
基于官方 openai SDK。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config_loader import get_llm_config


class LLMClient:
    """OpenAI 兼容 API 的薄封装（支持 codexcn / minimax 等）"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or get_llm_config()
        self.cfg = cfg
        # openai SDK 自身支持 base_url + 任何 OpenAI 兼容端点
        client_kwargs: Dict[str, Any] = {
            "api_key": cfg["api_key"],
            "base_url": cfg.get("base_url") or None,
            "max_retries": int(cfg.get("max_retries", 3)),
            "timeout": float(cfg.get("timeout", 120)),
        }
        proxy = (cfg.get("proxy") or "").strip()
        if proxy:
            # openai SDK 底层使用 httpx，httpx 支持 proxy 参数
            client_kwargs["http_client"] = _build_httpx_client(proxy)
        self.client = OpenAI(**client_kwargs)
        self.model = cfg["model"]
        self.temperature = float(cfg.get("temperature", 0.7))
        self.max_tokens = int(cfg.get("max_tokens", 4096))

        # MiniMax M3 默认开启 thinking，需显式关闭才能拿到干净 content
        self.provider = (cfg.get("provider") or "").lower()
        self.disable_thinking = cfg.get("disable_thinking", self.provider == "minimax")

    # ------------------------------------------------------------------
    # 基础调用
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format_json: bool = False,
    ) -> str:
        """
        发起一次对话。返回 assistant 文本。
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = self.temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        resp = self.client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # 多模态便捷方法
    # ------------------------------------------------------------------
    def chat_with_image(
        self,
        text_prompt: str,
        image_b64: str,
        image_mime: str = "image/jpeg",
        *,
        system: Optional[str] = None,
        response_format_json: bool = False,
    ) -> str:
        """
        单图 + 单文本 prompt。返回 assistant 文本。
        """
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime};base64,{image_b64}",
                    },
                },
                {"type": "text", "text": text_prompt},
            ],
        })
        return self.chat(
            messages,
            response_format_json=response_format_json,
        )

    # ------------------------------------------------------------------
    # 强 JSON 解析（容错：剥 ```json 围栏、找首个大括号）
    # ------------------------------------------------------------------
    @staticmethod
    def parse_json(text: str) -> Any:
        """
        从模型输出中尽量鲁棒地解析 JSON。
        失败时抛出 ValueError。
        """
        s = text.strip()
        # 去掉 ```json ... ``` 或 ``` ... ``` 围栏
        fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
        if fence:
            s = fence.group(1).strip()
        # 直接尝试
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # 找首个 { 或 [ 到末尾
        for opener, closer in (("{", "}"), ("[", "]")):
            i = s.find(opener)
            j = s.rfind(closer)
            if i != -1 and j != -1 and j > i:
                candidate = s[i:j + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        # 兜底：修复"字符串值里含裸换行"的常见 LLM 输出问题
        # （把字符串值内部未转义的换行/制表符转义掉）
        fixed = _repair_json_newlines(s)
        if fixed is not None:
            return fixed
        raise ValueError(f"无法从模型输出中解析 JSON:\n{s[:500]}")


# =============================================================================
# 辅助
# =============================================================================
def _repair_json_newlines(s: str):
    """修复 JSON 字符串值里未转义的裸换行/制表符。

    某些 LLM（如 MiniMax M3）偶发在字符串值内直接输出换行，导致 json.loads 失败。
    做法：逐字符扫描，在字符串值（引号内）范围内把裸的 \\n / \\r / \\t 转义掉。
    无法安全修复时返回 None。
    """
    out = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
            elif ch == "\\":
                out.append(ch)
                escape = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            if ch == '"':
                out.append(ch)
                in_str = True
            else:
                out.append(ch)
    fixed = "".join(out)
    try:
        import json as _json
        return _json.loads(fixed)
    except Exception:
        return None


def _build_httpx_client(proxy_url: str):
    """构造一个带代理的 httpx.Client，注入到 openai SDK"""
    import httpx
    return httpx.Client(
        proxy=proxy_url,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )


if __name__ == "__main__":
    # 自检
    cli = LLMClient()
    print("model =", cli.model)
    print("base_url =", cli.client.base_url)
