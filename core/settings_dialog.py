"""
settings_dialog.py
API 设置对话框（PySide6）。

包含：
  - LLM_PROVIDERS：对话型多模态 LLM 各厂商的默认 base_url / model 预设
  - SettingsDialog：设置对话框，两个分组（对话型 LLM / 方舟生图），
    支持 provider 切换、测试连接、写回 config.yaml 与 .env。

所有 API Key 一律写入 .env（ARK_API_KEY / LLM_API_KEY），config.yaml 里
保留 "env:XXX" 引用，避免明文入库。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QComboBox, QPushButton, QMessageBox, QDialogButtonBox,
)

from . import config_loader as cfg
from .style import APP_QSS


# =============================================================================
# 多厂商预设：provider -> {label, base_url, model, api_key_env, doc}
# =============================================================================
# 对话型 LLM（OpenAI 兼容端点，openai SDK 的 base_url 机制已覆盖大半）
LLM_PROVIDERS: Dict[str, Dict[str, str]] = {
    "minimax": {
        "label": "MiniMax（国内直连）",
        "base_url": "https://api.minimax.chat/v1",
        "model": "MiniMax-M3",
        "api_key_env": "LLM_API_KEY",
        "doc": "MiniMax 开放平台 -> API Keys 管理",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key_env": "LLM_API_KEY",
        "doc": "platform.openai.com -> API keys",
    },
    "qwen": {
        "label": "通义千问 Qwen（DashScope 兼容）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "api_key_env": "LLM_API_KEY",
        "doc": "阿里云百炼 -> API-KEY 管理",
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-3-5-sonnet-latest",
        "api_key_env": "LLM_API_KEY",
        "doc": "console.anthropic.com -> API Keys",
    },
    "gemini": {
        "label": "Google Gemini（兼容端点）",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "api_key_env": "LLM_API_KEY",
        "doc": "aistudio.google.com -> API Key",
    },
    "codexcn": {
        "label": "CodexCN 代理",
        "base_url": "https://api.codexcn.com/v1",
        "model": "gpt-4o",
        "api_key_env": "LLM_API_KEY",
        "doc": "codexcn 控制台 -> API Key",
    },
}

# 方舟（豆包）生图模型预设
ARK_MODELS: List[str] = [
    "doubao-seedream-5-0-pro-260628",
    "doubao-seedream-4.5",
    "doubao-seedream-3.0-t",
]


# =============================================================================
# 后台测试线程：把网络请求放到 QThread，避免阻塞 UI
# =============================================================================
class _TestWorker(QThread):
    """通用后台测试线程，跑 fn() 并通过信号返回结果。"""

    done = Signal(bool, str)  # (ok, 消息)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            ok, msg = self._fn()
            self.done.emit(bool(ok), str(msg))
        except Exception as e:
            self.done.emit(False, f"{e}")


def _test_llm_impl(provider, base_url, model, api_key) -> tuple[bool, str]:
    """后台执行：LLM 测试连接。"""
    from .llm_client import LLMClient
    client = LLMClient({
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout": 20,
        "max_retries": 1,
    })
    reply = client.chat([{"role": "user", "content": "回复 OK 两个字"}], max_tokens=8)
    return True, f"连接成功 ✓ 回复: {reply[:40]}"


def _test_ark_impl(base_url, model, api_key) -> tuple[bool, str]:
    """后台执行：方舟生图测试连接。"""
    import requests
    resp = requests.post(
        base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": "a simple red circle",
            "response_format": "url",
            "size": "1K",
            "watermark": False,
        },
        timeout=60,
    )
    if resp.status_code == 200:
        return True, "连接成功 ✓（鉴权通过）"
    return False, f"HTTP {resp.status_code}: {resp.text[:120]}"


class SettingsDialog(QDialog):
    """API 设置对话框。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("API 设置")
        self.setMinimumWidth(560)
        self.setStyleSheet(APP_QSS)

        self._test_worker: Optional[_TestWorker] = None

        self._load_current()

        root = QVBoxLayout(self)

        # ---- 对话型 LLM 分组 ----
        llm_box = QGroupBox("对话型多模态 LLM（视觉理解 / prompt 生成）")
        llm_form = QFormLayout(llm_box)

        self.llm_provider = QComboBox()
        for key, meta in LLM_PROVIDERS.items():
            self.llm_provider.addItem(meta["label"], key)
        self.llm_provider.currentIndexChanged.connect(self._on_llm_provider_changed)

        self.llm_base_url = QLineEdit()
        self.llm_base_url.setPlaceholderText("OpenAI 兼容 base_url")
        self.llm_model = QLineEdit()
        self.llm_model.setPlaceholderText("模型名")
        self.llm_key = QLineEdit()
        self.llm_key.setEchoMode(QLineEdit.Password)
        self.llm_key.setPlaceholderText("API Key（留空则不修改）")

        llm_form.addRow("服务商:", self.llm_provider)
        llm_form.addRow("Base URL:", self.llm_base_url)
        llm_form.addRow("模型:", self.llm_model)
        llm_form.addRow("API Key:", self.llm_key)

        self.llm_doc = QLabel("")
        self.llm_doc.setStyleSheet("color: #888;")
        self.llm_doc.setWordWrap(True)
        llm_form.addRow("", self.llm_doc)

        self.btn_test_llm = QPushButton("测试连接")
        self.btn_test_llm.clicked.connect(self._on_test_llm)
        self.llm_test_result = QLabel("")
        self.llm_test_result.setWordWrap(True)
        llm_test_row = QHBoxLayout()
        llm_test_row.addWidget(self.btn_test_llm)
        llm_test_row.addWidget(self.llm_test_result, 1)
        llm_form.addRow("", llm_test_row)
        llm_token_hint = QLabel("提示：测试连接会向模型发送一条测试请求，可能消耗少量 token。")
        llm_token_hint.setWordWrap(True)
        llm_token_hint.setStyleSheet("color: #999; font-size: 12px;")
        llm_form.addRow("", llm_token_hint)

        root.addWidget(llm_box)

        # ---- 方舟生图分组 ----
        ark_box = QGroupBox("方舟生图（豆包 SeedDream，照片转卡通）")
        ark_form = QFormLayout(ark_box)

        self.ark_key = QLineEdit()
        self.ark_key.setEchoMode(QLineEdit.Password)
        self.ark_key.setPlaceholderText("API Key（留空则不修改）")
        self.ark_base_url = QLineEdit()
        self.ark_base_url.setPlaceholderText("方舟图像生成端点")
        self.ark_model = QComboBox()
        self.ark_model.setEditable(True)
        self.ark_model.addItems(ARK_MODELS)

        ark_form.addRow("API Key:", self.ark_key)
        ark_form.addRow("Base URL:", self.ark_base_url)
        ark_form.addRow("模型:", self.ark_model)

        self.ark_doc = QLabel("获取：火山引擎控制台 -> 火山方舟 -> API Key 管理")
        self.ark_doc.setStyleSheet("color: #888;")
        self.ark_doc.setWordWrap(True)
        ark_form.addRow("", self.ark_doc)

        self.btn_test_ark = QPushButton("测试连接")
        self.btn_test_ark.clicked.connect(self._on_test_ark)
        self.ark_test_result = QLabel("")
        self.ark_test_result.setWordWrap(True)
        ark_test_row = QHBoxLayout()
        ark_test_row.addWidget(self.btn_test_ark)
        ark_test_row.addWidget(self.ark_test_result, 1)
        ark_form.addRow("", ark_test_row)
        ark_token_hint = QLabel("提示：测试连接会生成一张测试图片，可能消耗少量 token / 费用。")
        ark_token_hint.setWordWrap(True)
        ark_token_hint.setStyleSheet("color: #999; font-size: 12px;")
        ark_form.addRow("", ark_token_hint)

        root.addWidget(ark_box)

        # ---- 底部 ----
        self.lbl_path = QLabel("")
        self.lbl_path.setStyleSheet("color: #888;")
        self.lbl_path.setWordWrap(True)
        root.addWidget(self.lbl_path)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Save).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._fill_controls()
        self._refresh_doc()
        self._refresh_path_label()

    # ------------------------------------------------------------------
    # 加载当前配置
    # ------------------------------------------------------------------
    def _load_current(self) -> None:
        raw = cfg.load_config_raw()
        llm = raw.get("llm") or {}
        ark = raw.get("ark") or {}
        cart = ark.get("cartoonize") or {}

        self._cur_llm = dict(llm)
        self._cur_ark = dict(ark)
        self._cur_cart = dict(cart)

        # 当前 provider
        self._cur_provider = (llm.get("provider") or "minimax").lower()
        # 当前 key（从 .env 读真实值）
        self._cur_llm_key = cfg.get_env_value("LLM_API_KEY")
        self._cur_ark_key = cfg.get_env_value("ARK_API_KEY")

    # ------------------------------------------------------------------
    # UI 初始化：把当前值填进控件
    # ------------------------------------------------------------------
    def _fill_controls(self) -> None:
        # LLM
        idx = self.llm_provider.findData(self._cur_provider)
        if idx < 0:
            idx = self.llm_provider.findData("minimax")
        self.llm_provider.setCurrentIndex(idx)
        self.llm_base_url.setText(
            self._cur_llm.get("base_url")
            or LLM_PROVIDERS.get(self._cur_provider, {}).get("base_url", "")
        )
        self.llm_model.setText(
            self._cur_llm.get("model")
            or LLM_PROVIDERS.get(self._cur_provider, {}).get("model", "")
        )
        self.llm_key.setText(self._cur_llm_key)

        # ARK
        self.ark_key.setText(self._cur_ark_key)
        self.ark_base_url.setText(
            self._cur_ark.get("base_url", "https://ark.cn-beijing.volces.com/api/v3/images/generations")
        )
        cart_model = self._cur_cart.get("model", ARK_MODELS[0])
        if self.ark_model.findText(cart_model) < 0:
            self.ark_model.addItem(cart_model)
        self.ark_model.setCurrentText(cart_model)

    def _on_llm_provider_changed(self):
        key = self.llm_provider.currentData()
        meta = LLM_PROVIDERS.get(key, {})
        # 切换 provider 时，若 base_url / model 为空或仍是上一家的默认值，则填入新预设
        self.llm_base_url.setText(meta.get("base_url", ""))
        self.llm_model.setText(meta.get("model", ""))
        self._refresh_doc()

    def _refresh_doc(self):
        key = self.llm_provider.currentData()
        meta = LLM_PROVIDERS.get(key, {})
        self.llm_doc.setText(meta.get("doc", ""))

    def _refresh_path_label(self):
        self.lbl_path.setText(
            f"配置文件：{cfg.get_config_path()}\n"
            f"密钥文件：{cfg.get_env_path()}"
        )

    # ------------------------------------------------------------------
    # 测试连接
    # ------------------------------------------------------------------
    def _on_test_llm(self):
        key = self.llm_key.text().strip() or self._cur_llm_key
        if not key:
            self._set_test_result(self.llm_test_result, False, "请先填写 API Key")
            return
        provider = self.llm_provider.currentData()
        base_url = self.llm_base_url.text().strip()
        model = self.llm_model.text().strip()
        self._start_test(
            self.btn_test_llm, self.llm_test_result,
            lambda: _test_llm_impl(provider, base_url, model, key),
        )

    def _on_test_ark(self):
        key = self.ark_key.text().strip() or self._cur_ark_key
        if not key:
            self._set_test_result(self.ark_test_result, False, "请先填写 API Key")
            return
        base_url = self.ark_base_url.text().strip()
        model = self.ark_model.currentText().strip()
        self._start_test(
            self.btn_test_ark, self.ark_test_result,
            lambda: _test_ark_impl(base_url, model, key),
        )

    def _start_test(self, btn: QPushButton, label: QLabel, fn):
        """启动后台测试线程，期间禁用按钮并显示等待提示。"""
        btn.setEnabled(False)
        btn.setText("测试中...")
        label.setText("测试中，请稍候...")
        label.setStyleSheet("color: #c90;")

        self._test_worker = _TestWorker(fn)
        self._test_worker.done.connect(
            lambda ok, msg, l=label, b=btn: self._on_test_done(l, b, ok, msg)
        )
        self._test_worker.finished.connect(self._test_worker.deleteLater)
        self._test_worker.start()

    def _on_test_done(self, label: QLabel, btn: QPushButton, ok: bool, msg: str):
        btn.setEnabled(True)
        btn.setText("测试连接")
        self._set_test_result(label, ok, msg)

    @staticmethod
    def _set_test_result(label: QLabel, ok: bool, msg: str):
        label.setText(msg)
        label.setStyleSheet(f"color: {'#2a7' if ok else '#c33'};")

    def closeEvent(self, event):
        """关闭时若测试线程仍在跑，先安全断开信号再等它结束，避免访问已销毁控件。"""
        if self._test_worker and self._test_worker.isRunning():
            # 断开信号，避免结果回调到已关闭的控件
            try:
                self._test_worker.done.disconnect()
            except Exception:
                pass
            self._test_worker.wait(3000)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def _on_save(self):
        # 收集 LLM
        provider = self.llm_provider.currentData()
        llm_base_url = self.llm_base_url.text().strip()
        llm_model = self.llm_model.text().strip()
        llm_key = self.llm_key.text().strip()

        # 收集 ARK
        ark_key = self.ark_key.text().strip()
        ark_base_url = self.ark_base_url.text().strip()
        ark_model = self.ark_model.currentText().strip()

        # 写 .env（只写非空 key，避免清掉已有 key）
        env_updates: Dict[str, str] = {}
        if llm_key:
            env_updates["LLM_API_KEY"] = llm_key
        if ark_key:
            env_updates["ARK_API_KEY"] = ark_key
        if env_updates:
            cfg.save_env(env_updates)

        # 写 config.yaml（保留 env: 引用，更新 provider/base_url/model）
        raw = cfg.load_config_raw()
        llm = raw.get("llm") or {}
        llm["provider"] = provider
        llm["base_url"] = llm_base_url
        llm["model"] = llm_model
        # 保持 api_key 为 env: 引用
        if "api_key" not in llm:
            llm["api_key"] = "env:LLM_API_KEY"
        # MiniMax 需关 thinking
        if provider == "minimax":
            llm.setdefault("disable_thinking", True)
        raw["llm"] = llm

        ark = raw.get("ark") or {}
        ark["base_url"] = ark_base_url
        if "api_key" not in ark:
            ark["api_key"] = "env:ARK_API_KEY"
        cart = ark.get("cartoonize") or {}
        cart["model"] = ark_model
        ark["cartoonize"] = cart
        raw["ark"] = ark

        try:
            cfg.save_config(raw)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入 config.yaml 失败：\n{e}")
            return

        self.accept()
