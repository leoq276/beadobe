"""
config_loader.py
读取并解析 config.yaml，暴露给上层代码使用。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .paths import resource_path, user_config_path, ensure_user_config_dir


# 用户可写配置：config.yaml 与 .env 放到用户配置目录（打包后 _MEIPASS 只读，不能写）
# 首次运行若用户目录还没有 config.yaml，则从打包资源/项目根目录复制默认配置过去。
_CONFIG_PATH = user_config_path("config.yaml")
_ENV_PATH = user_config_path(".env")

# 源码运行时项目根目录的 .env（旧版把 key 放在这里，读取时回退兼容）
_LEGACY_ENV_PATH = resource_path(".env")

# 默认配置模板（打包资源里的 config.yaml，源码运行时即项目根目录的 config.yaml）
_DEFAULT_CONFIG_RESOURCE = resource_path("config.yaml")

# env: 前缀标记，表示该字段的真实值应从环境变量读取
_ENV_PREFIX = "env:"


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取：把 KEY=VALUE 行注入 os.environ（不覆盖已存在的变量）。"""
    # 兼容旧版：源码运行时项目根目录的 .env 也一并读取（用户目录优先）
    for p in (path, _LEGACY_ENV_PATH):
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _read_dotenv_values(path: Path) -> Dict[str, str]:
    """读取 .env 里所有 KEY=VALUE（原样，不去引号、不注入环境）。

    兼容旧版：用户目录没有的 key，从源码项目根目录 .env 回退补齐。
    """
    out: Dict[str, str] = {}
    # 先读旧版（低优先级），再用用户目录覆盖
    for p in (_LEGACY_ENV_PATH, path):
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                out[key] = value
    return out


def _write_dotenv(path: Path, values: Dict[str, str]) -> None:
    """把 KEY=VALUE 写回 .env，保留原文件里未知的键与注释尽量不动。

    简单实现：读取现有行，逐行匹配 key 更新；不存在的 key 追加到末尾。
    """
    ensure_user_config_dir()
    existing = _read_dotenv_values(path)
    merged = {**existing, **values}

    # 保留原始行顺序 + 注释：逐行重写
    lines: list[str] = []
    handled: set[str] = set()
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                lines.append(raw)
                continue
            key = line.partition("=")[0].strip()
            if key in merged:
                lines.append(f"{key}={merged[key]}")
                handled.add(key)
            else:
                lines.append(raw)
    # 追加新键
    for key, value in merged.items():
        if key not in handled:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_env_refs(data: Any) -> Any:
    """递归解析 env:VAR 引用：把字符串 'env:ARK_API_KEY' 替换为环境变量值。"""
    if isinstance(data, dict):
        return {k: _resolve_env_refs(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_refs(v) for v in data]
    if isinstance(data, str) and data.startswith(_ENV_PREFIX):
        var = data[len(_ENV_PREFIX):].strip()
        val = os.environ.get(var, "")
        if not val:
            raise ConfigError(
                f"环境变量 {var} 未设置。请复制 .env.example 为 .env 并填入真实值，"
                f"或直接 export {var}=..."
            )
        return val
    return data


class ConfigError(Exception):
    """配置文件缺失或字段错误"""
    pass


def _ensure_user_config() -> None:
    """确保用户配置目录里的 config.yaml 存在；不存在则从资源/项目根目录复制默认值。"""
    if _CONFIG_PATH.exists():
        return
    ensure_user_config_dir()
    if _DEFAULT_CONFIG_RESOURCE.exists():
        _CONFIG_PATH.write_text(
            _DEFAULT_CONFIG_RESOURCE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    # 否则 load_config 会抛 ConfigError，由调用方决定是否用内置默认值


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    加载 yaml 配置。

    :param path: 配置文件路径，留空则使用用户配置目录的 config.yaml
    :return: 嵌套 dict
    :raises ConfigError: 文件不存在 / 解析失败
    """
    if path:
        p = Path(path)
    else:
        _ensure_user_config()
        p = _CONFIG_PATH
    if not p.exists():
        raise ConfigError(f"配置文件不存在: {p}")

    # 先加载用户配置目录 .env（若存在），供 env: 引用解析
    _load_dotenv(_ENV_PATH)

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # 解析 env:VAR 引用为真实环境变量值
    return _resolve_env_refs(data)


def save_config(data: Dict[str, Any], path: str | Path | None = None) -> Path:
    """
    把配置 dict 写回 config.yaml（用户配置目录）。返回写入的路径。

    注意：只做整体覆盖写回；env:VAR 引用需调用方在 data 里保留为 "env:VAR" 字符串。
    """
    p = Path(path) if path else _CONFIG_PATH
    ensure_user_config_dir()
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return p


def save_env(values: Dict[str, str], path: str | Path | None = None) -> Path:
    """把 KEY=VALUE 写回 .env（用户配置目录）。返回写入的路径。"""
    p = Path(path) if path else _ENV_PATH
    _write_dotenv(p, values)
    # 同步注入当前进程环境，便于后续 env: 引用立即生效
    for k, v in values.items():
        os.environ[k] = v
    return p


def get_env_value(key: str, path: str | Path | None = None) -> str:
    """读取 .env 里某个 key 的原始值（不解析 env: 引用）。"""
    p = Path(path) if path else _ENV_PATH
    return _read_dotenv_values(p).get(key, "")


def get_env_path() -> Path:
    """返回当前使用的 .env 路径。"""
    return _ENV_PATH


def get_config_path() -> Path:
    """返回当前使用的 config.yaml 路径。"""
    return _CONFIG_PATH


def get_llm_config(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    取出 llm 节点。
    如果未配置（空字典或缺失必要字段），抛 ConfigError。
    """
    cfg = cfg or load_config()
    llm = cfg.get("llm")
    if not llm:
        raise ConfigError("config.yaml 中没有 llm 节点，请先启用对话型 LLM 配置")
    missing = [k for k in ("provider", "api_key", "model") if not llm.get(k)]
    if missing:
        raise ConfigError(f"llm 节点缺少必填字段: {missing}")
    return llm


def get_ark_config(cfg: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    """取出 ark 节点（可能为空）"""
    cfg = cfg or load_config()
    return cfg.get("ark") or None


def get_image_config(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """取出图片预处理配置（带默认值）"""
    cfg = cfg or load_config()
    img = cfg.get("image") or {}
    return {
        "max_side": int(img.get("max_side", 1024)),
        "format": img.get("format", "jpeg"),
        "quality": int(img.get("quality", 85)),
    }


def load_config_raw(path: str | Path | None = None) -> Dict[str, Any]:
    """
    宽松加载：不解析 env:VAR 引用，缺失字段不抛错。
    用于「检测是否已配置 key」等只读场景（避免 load_config 因缺 key 抛 ConfigError）。
    """
    if path:
        p = Path(path)
    else:
        _ensure_user_config()
        p = _CONFIG_PATH
    if not p.exists():
        return {}
    _load_dotenv(_ENV_PATH)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_env_ref_value(value: Any) -> str:
    """若值是 'env:VAR' 引用，返回对应环境变量值；否则返回原字符串。"""
    if isinstance(value, str) and value.startswith(_ENV_PREFIX):
        var = value[len(_ENV_PREFIX):].strip()
        return os.environ.get(var, "")
    return value or ""


def has_llm_key() -> bool:
    """检测对话型 LLM 的 API Key 是否已配置（宽松，不抛错）。"""
    cfg = load_config_raw()
    llm = cfg.get("llm") or {}
    return bool(get_env_ref_value(llm.get("api_key")))


def has_ark_key() -> bool:
    """检测方舟（豆包生图）的 API Key 是否已配置（宽松，不抛错）。"""
    cfg = load_config_raw()
    ark = cfg.get("ark") or {}
    return bool(get_env_ref_value(ark.get("api_key")))


if __name__ == "__main__":
    # 简单自检
    cfg = load_config()
    print("llm provider =", (cfg.get("llm") or {}).get("provider"))
    print("ark model    =", ((cfg.get("ark") or {}).get("cartoonize") or {}).get("model"))
    print("has_llm_key  =", has_llm_key())
    print("has_ark_key  =", has_ark_key())
    print("config path  =", get_config_path())
