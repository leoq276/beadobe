"""
paths.py
统一资源路径定位，兼容「源码运行」与「PyInstaller 打包运行」两种场景。

设计约定：
  - 只读资源（色板 CSV 等打包时作为 data 收集的）：打包后位于 sys._MEIPASS，
    源码运行时位于项目根目录。
  - 用户可写文件（config.yaml / .env）：不能写进 _MEIPASS（只读临时目录），
    统一放到用户数据目录（Windows %APPDATA%）。源码运行时也放这里，保证
    打包前后行为一致、配置不丢失。

对外主要接口：
  - resource_path(*parts)      只读资源路径
  - user_config_path(*parts)   用户可写配置路径（config.yaml / .env）
  - ensure_user_config_dir()   确保用户配置目录存在
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 打包后的应用名（用于用户配置目录）
_APP_NAME = "BeadStudio"


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包产物中。"""
    return getattr(sys, "frozen", False)


def _project_root() -> Path:
    """源码运行时的项目根目录（core 的上一级）。"""
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """
    只读资源的绝对路径。

    源码运行：项目根目录 / parts
    打包运行：sys._MEIPASS / parts（PyInstaller 解压出的临时目录）
    """
    base = Path(getattr(sys, "_MEIPASS", _project_root()))
    return base.joinpath(*parts)


def user_config_dir() -> Path:
    """
    用户可写配置目录。

    优先级：
      1. 环境变量 BEADSTUDIO_CONFIG_DIR（便于测试/便携模式）
      2. Windows %APPDATA%/BeadStudio
      3. 兜底：用户主目录 ~/.beadstudio
    """
    env = os.environ.get("BEADSTUDIO_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / _APP_NAME
    return Path.home() / ".beadstudio"


def user_config_path(*parts: str) -> Path:
    """用户可写配置文件的绝对路径（config.yaml / .env 等）。"""
    return user_config_dir().joinpath(*parts)


def ensure_user_config_dir() -> Path:
    """确保用户配置目录存在，返回其路径。"""
    d = user_config_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d
