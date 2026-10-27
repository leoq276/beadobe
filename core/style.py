"""
style.py
全局样式表（简约高级感：低饱和蓝灰 + 白卡片 + 细边框圆角）。

供 bead_studio.py 与 settings_dialog.py 共用，保证主窗口与对话框视觉统一。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# spinbox 箭头图片：QSS 不支持 CSS border 三角技巧，箭头必须用 image 提供。
# 运行时生成极小的上下箭头 PNG（放系统临时目录），返回可供 QSS url() 用的路径。
# ---------------------------------------------------------------------------
_ARROW_DIR = Path(tempfile.gettempdir()) / "beadstudio_arrows"


def _write_arrow(name: str, up: bool) -> str:
    """生成 10x6 的灰色三角箭头 PNG，返回 posix 风格路径。"""
    import struct
    import zlib

    _ARROW_DIR.mkdir(parents=True, exist_ok=True)
    w, h = 10, 6
    gray = (0x7A, 0x82, 0x8E)
    transparent = (0, 0, 0, 0)
    fg = (*gray, 255)
    rows = []
    for y in range(h):
        row = bytearray([0])  # filter type 0
        if up:
            span = 2 * (y + 1)          # 顶点向下展开：2,4,...,12->clip 10
        else:
            span = 2 * (h - y)
        left = (w - min(span, w)) // 2
        right = left + min(span, w)
        for x in range(w):
            px = fg if left <= x < right else transparent
            row.extend(px)
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    path = _ARROW_DIR / name
    path.write_bytes(png)
    return path.as_posix()


def _spin_arrow_urls() -> tuple[str, str]:
    try:
        up = _write_arrow("up.png", up=True)
        down = _write_arrow("down.png", up=False)
        return up, down
    except Exception:
        return "", ""


_UP_URL, _DOWN_URL = _spin_arrow_urls()


def _maybe_img(p: str) -> str:
    return f"image: url({p});" if p else ""


APP_QSS = """
* {
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: 13px;
    color: #2b3440;
}
QMainWindow, QDialog {
    background: #f4f6f8;
}
/* 分组框：白卡片，标题骑在边框线上（白底遮线） */
QGroupBox {
    background: #ffffff;
    border: 1px solid #e3e7ec;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 12px 10px 12px;
    font-weight: 600;
    color: #4a5560;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 4px;
    padding: 0 4px;
    background: #ffffff;
    color: #5a6572;
    font-weight: 600;
    font-size: 12px;
}
/* 不可用分组/控件：整体灰化 + 降透明度，与可操作参数形成明显区分 */
QGroupBox:disabled {
    background: #fafbfc;
    border-color: #eef0f3;
}
QGroupBox:disabled::title {
    color: #b3bac2;
}
QWidget:disabled {
    color: #b3bac2;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled, QTextEdit:disabled, QLabel:disabled {
    color: #b3bac2;
    background: #f5f6f8;
    border-color: #eef0f3;
}
QComboBox:disabled {
    color: #b3bac2;
}
QSlider::sub-page:disabled {
    background: #d7dce2;
}
/* 按钮 */
QPushButton {
    background: #ffffff;
    border: 1px solid #d5dbe3;
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 18px;
}
QPushButton:hover {
    background: #f0f4f9;
    border-color: #b8c6d6;
}
QPushButton:pressed {
    background: #e6edf5;
}
QPushButton:disabled {
    background: #f3f5f7;
    color: #b0b8c0;
    border-color: #e8ebef;
}
QPushButton:checked {
    background: #e3ecf5;
    border-color: #7fa4c7;
    color: #2f5f8a;
    font-weight: 600;
}
/* 输入控件 */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    background: #ffffff;
    border: 1px solid #d5dbe3;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: #b9d3ea;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QTextEdit:focus {
    border: 1px solid #7fa4c7;
}
/* 数值调节框：显式保留上下箭头按钮 */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border: none;
    border-left: 1px solid #e3e7ec;
    background: #f2f5f8;
    border-top-right-radius: 5px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border: none;
    border-left: 1px solid #e3e7ec;
    background: #f2f5f8;
    border-bottom-right-radius: 5px;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 10px;
    height: 6px;
    %s
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 10px;
    height: 6px;
    %s
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #e6edf5;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: #d6e1ee;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #d5dbe3;
    selection-background-color: #e3ecf5;
    selection-color: #2b3440;
}
/* 复选框 */
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #c4ccd6;
    border-radius: 4px;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #3b6ea5;
    border-color: #3b6ea5;
}
/* 滑块 */
QSlider::groove:horizontal {
    height: 5px;
    background: #e3e7ec;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #7fa4c7;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: #ffffff;
    border: 1px solid #b8c6d6;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    border-color: #7fa4c7;
}
/* 标签页 */
QTabWidget::pane {
    border: 1px solid #e3e7ec;
    border-radius: 6px;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 16px;
    color: #7a828e;
}
QTabBar::tab:selected {
    color: #2f5f8a;
    border-bottom: 2px solid #3b6ea5;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    color: #4a5560;
}
/* 滚动条 */
QScrollArea {
    border: none;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #ccd3db;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #b0bac6;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #ccd3db;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
/* 工具栏 */
QToolBar {
    background: #ffffff;
    border: none;
    border-bottom: 1px solid #e3e7ec;
    spacing: 4px;
    padding: 4px 8px;
}
QToolBar QToolButton {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
}
QToolBar QToolButton:hover {
    background: #f0f4f9;
}
QToolBar QToolButton:pressed {
    background: #e6edf5;
}
/* 状态栏 */
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #e3e7ec;
    color: #7a828e;
}
/* 分隔条 */
QSplitter::handle {
    background: #e8ebef;
}
QSplitter::handle:horizontal {
    width: 4px;
}
QSplitter::handle:vertical {
    height: 4px;
}
/* 消息框按钮 */
QMessageBox QPushButton, QDialogButtonBox QPushButton {
    min-width: 72px;
}
""" % (_maybe_img(_UP_URL), _maybe_img(_DOWN_URL))
