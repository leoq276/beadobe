"""
bead_studio.py
拼豆图纸生成器 - 桌面 GUI (PySide6)

用法:
    conda activate NDP
    python bead_studio.py

功能:
    - 拖入/选择图片
    - 调参 (grid size, supersample, line detect, ss strategy, cleanup 等)
    - 一键生成拼豆图纸
    - 显示原图 / 拼豆图 / 网格预览 / 统计数据
    - 保存产物 (PNG 拼豆图 / 网格预览 / JSON 描述)
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Optional

# 让 core 包可导入
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QCheckBox, QComboBox,
    QFileDialog, QGroupBox, QScrollArea, QStatusBar, QProgressBar,
    QMessageBox, QSplitter, QFrame, QSizePolicy, QTabWidget, QDoubleSpinBox,
    QDialog, QLineEdit, QDialogButtonBox,
)

from core.palette_loader import load_palette
from core.paths import resource_path
from core.style import APP_QSS
from core.quantize import quantize, QuantizeResult
from core.bead_chart import (
    save_bead_chart, render_grid_only, render_plain_image, rebuild_result_stats,
)
from core.edit import (
    paint_cells, erase_cells, adjust_cells, select_by_codes,
    BLANK, is_blank, snapshot_grid, restore_grid,
)

# Stage 0/1（卡通化）依赖 openai + requests，缺失时降级为仅量化模式
try:
    from core.pipeline import run_full_pipeline, PipelineResult
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
    run_full_pipeline = None
    PipelineResult = None


# 检测可选依赖 (cv2 用于 Canny/Hybrid 线条检测)
try:
    import cv2
    HAS_CV2 = True
    CV2_VERSION = cv2.__version__
except ImportError:
    HAS_CV2 = False
    CV2_VERSION = None


# =============================================================================
# 可框选图片控件：拖拽框选区域，只对选中区域像素化
# =============================================================================

class SelectableImageLabel(QLabel):
    """可缩放、可平移、可框选、可拖边角调整选区的图片标签。

    功能：
      - Ctrl + 滚轮：缩放图片（以光标为中心）
      - 鼠标拖拽（空白处/图片上无选区时）：平移图片
      - 鼠标拖拽（图片上无选区时）：框选区域
      - 已有选区时：拖动选区内部=移动整个框；拖动边/角=调整大小
    用 `set_image(path)` 载入图片，`region` 属性返回归一化 bbox [x1,y1,x2,y2]。
    """

    region_changed = Signal(object)  # 发出归一化 bbox [x1,y1,x2,y2] 或 None

    # 控制点命中类型
    _HIT_NONE = 0
    _HIT_MOVE = 1        # 框内部（移动整框）
    _HIT_L = 2
    _HIT_R = 3
    _HIT_T = 4
    _HIT_B = 5
    _HIT_TL = 6
    _HIT_TR = 7
    _HIT_BL = 8
    _HIT_BR = 9

    _HANDLE_RADIUS = 8  # 控制点命中半径（像素）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        # 关键：阻止 pixmap 的 sizeHint 反向撑大父布局（工作区尺寸固定）
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._pixmap: Optional[QPixmap] = None
        self._zoom = 1.0                                # 缩放倍率
        self._offset = QtCore.QPointF(0, 0)             # 平移偏移（控件内）
        self._region: Optional[list] = None             # 归一化 bbox
        self._draw_rect = None                          # 控件内选区矩形 (QRect)
        # 交互状态
        self._mode = "none"                             # none|pan|draw|resize|move
        self._drag_start = None
        self._drag_current = None
        self._pan_last = None
        self._hit = self._HIT_NONE
        self._resize_anchor = None                      # 调整大小时固定的对角点
        self._empty_text = ""
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ================= 载入 / 选区 =================
    def set_image(self, path: str):
        """载入图片并重设选区与缩放。"""
        self._pixmap = QPixmap(path)
        self._region = None
        self._draw_rect = None
        self._drag_start = None
        self._drag_current = None
        self._zoom = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._empty_text = ""
        self._clamp_offset()
        self.update()

    def setText(self, text: str):
        # 覆盖 setText 用于占位提示（图片为空时显示）
        self._empty_text = text
        self.update()

    def clear_region(self):
        self._region = None
        self._draw_rect = None
        self.update()
        self.region_changed.emit(None)

    @property
    def region(self) -> Optional[list]:
        """返回归一化 bbox [x1,y1,x2,y2]，无选区时返回 None。"""
        return self._region

    def has_region(self) -> bool:
        return self._region is not None

    # ================= 缩放 / 显示 =================
    def _fit_size(self) -> QtCore.QSize:
        """图片适配到控件尺寸（保持宽高比）后的逻辑尺寸。"""
        if self._pixmap is None or self._pixmap.isNull():
            return QtCore.QSize(0, 0)
        return self._pixmap.size().scaled(
            self.size(), Qt.KeepAspectRatio
        )

    def _scaled_size(self) -> QtCore.QSize:
        """缩放后图片的像素尺寸。"""
        fit = self._fit_size()
        if fit.width() <= 0 or fit.height() <= 0:
            return QtCore.QSize(0, 0)
        w = max(1, int(fit.width() * self._zoom))
        h = max(1, int(fit.height() * self._zoom))
        return QtCore.QSize(w, h)

    def _clamp_offset(self):
        """把偏移钳制到合理范围，避免图片完全拖出视野。

        放大后图片比控件大，offset 可在 [-(iw-cw)/2, +(iw-cw)/2] 内移动，
        图片边缘可贴到控件边缘，保证所有内容都能滚动看到。
        """
        if self._pixmap is None or self._pixmap.isNull():
            return
        iw, ih = self._scaled_size().width(), self._scaled_size().height()
        cw, ch = self.width(), self.height()
        maxx = max(0.0, (iw - cw) / 2.0)
        maxy = max(0.0, (ih - ch) / 2.0)
        self._offset.setX(max(-maxx, min(maxx, self._offset.x())))
        self._offset.setY(max(-maxy, min(maxy, self._offset.y())))

    def zoom(self, factor: float, anchor: QtCore.QPointF):
        """以 anchor（控件内坐标）为中心缩放 factor 倍。"""
        new_zoom = max(0.1, min(8.0, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        # 缩放锚点调整：保持 anchor 处的图像点不动
        # 记录缩放前 anchor 对应的图像点，缩放后反推新 offset 使该点仍落在 anchor
        r = self._image_rect()
        if r is not None and r.width() > 0 and r.height() > 0:
            fx = (anchor.x() - r.x()) / r.width()
            fy = (anchor.y() - r.y()) / r.height()
        else:
            fx = fy = 0.5
        self._zoom = new_zoom
        # 缩放后新图像矩形
        newr = self._image_rect()
        if newr is not None:
            # 期望 anchor 处的图像点保持不动：
            # anchor.x = new_x + fx * new_w  ->  new_x = anchor.x - fx*new_w
            # 而 new_x = (cw - new_w)/2 + offset.x
            cw, ch = self.width(), self.height()
            new_w = newr.width()
            new_h = newr.height()
            self._offset.setX(anchor.x() - fx * new_w - (cw - new_w) / 2)
            self._offset.setY(anchor.y() - fy * new_h - (ch - new_h) / 2)
        self._clamp_offset()
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._clamp_offset()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_offset()
        self.update()

    # ================= 坐标换算 =================
    def _image_rect(self):
        """图片（缩放+平移后）在控件内的实际显示矩形。"""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        sz = self._scaled_size()
        w, h = sz.width(), sz.height()
        x = (self.width() - w) / 2 + self._offset.x()
        y = (self.height() - h) / 2 + self._offset.y()
        return QtCore.QRectF(x, y, w, h)

    def _to_norm(self, pt) -> tuple:
        """控件内坐标 -> 归一化 (0~1)。"""
        r = self._image_rect()
        if r is None:
            return (0.0, 0.0)
        nx = (pt.x() - r.x()) / max(1.0, r.width())
        ny = (pt.y() - r.y()) / max(1.0, r.height())
        return (max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))

    def _from_norm(self, nx, ny) -> QtCore.QPointF:
        """归一化 -> 控件内坐标。"""
        r = self._image_rect()
        if r is None:
            return QtCore.QPointF(0, 0)
        return QtCore.QPointF(r.x() + nx * r.width(), r.y() + ny * r.height())

    # ================= 控制点命中检测 =================
    def _hit_test(self, pos: QtCore.QPointF) -> int:
        """检测 pos 落在选区的哪个控制点/区域。"""
        if self._draw_rect is None:
            return self._HIT_NONE
        r = self._draw_rect
        R = self._HANDLE_RADIUS
        def near(p: QtCore.QPointF, q: QtCore.QPointF) -> bool:
            return abs(p.x() - q.x()) <= R and abs(p.y() - q.y()) <= R
        tl = QtCore.QPointF(r.x(), r.y())
        tr = QtCore.QPointF(r.x() + r.width(), r.y())
        bl = QtCore.QPointF(r.x(), r.y() + r.height())
        br = QtCore.QPointF(r.x() + r.width(), r.y() + r.height())
        if near(pos, tl):
            return self._HIT_TL
        if near(pos, tr):
            return self._HIT_TR
        if near(pos, bl):
            return self._HIT_BL
        if near(pos, br):
            return self._HIT_BR
        # 边
        if abs(pos.x() - r.x()) <= R and r.y() <= pos.y() <= r.y() + r.height():
            return self._HIT_L
        if abs(pos.x() - (r.x() + r.width())) <= R and r.y() <= pos.y() <= r.y() + r.height():
            return self._HIT_R
        if abs(pos.y() - r.y()) <= R and r.x() <= pos.x() <= r.x() + r.width():
            return self._HIT_T
        if abs(pos.y() - (r.y() + r.height())) <= R and r.x() <= pos.x() <= r.x() + r.width():
            return self._HIT_B
        # 内部
        if r.contains(pos):
            return self._HIT_MOVE
        return self._HIT_NONE

    def _cursor_for_hit(self, hit: int):
        if hit in (self._HIT_TL, self._HIT_BR):
            return Qt.SizeFDiagCursor
        if hit in (self._HIT_TR, self._HIT_BL):
            return Qt.SizeBDiagCursor
        if hit in (self._HIT_L, self._HIT_R):
            return Qt.SizeHorCursor
        if hit in (self._HIT_T, self._HIT_B):
            return Qt.SizeVerCursor
        if hit == self._HIT_MOVE:
            return Qt.SizeAllCursor
        return Qt.ArrowCursor

    # ================= 鼠标事件 =================
    def mousePressEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull():
            return
        pos = event.position()
        if event.button() == Qt.LeftButton:
            hit = self._hit_test(pos)
            if hit != self._HIT_NONE:
                # 已有选区：移动或调整大小
                self._mode = "move" if hit == self._HIT_MOVE else "resize"
                self._hit = hit
                self._drag_start = pos
                self._drag_current = pos
                self._resize_anchor = self._resize_opposite_anchor(hit)
            else:
                # 无选区命中：开始框选
                self._mode = "draw"
                self._drag_start = pos
                self._drag_current = pos
                self._region = None
                self._draw_rect = None
                self.update()
                self.region_changed.emit(None)
        elif event.button() == Qt.MiddleButton:
            # 中键平移
            self._mode = "pan"
            self._pan_last = pos
            self._drag_start = pos
        self.setCursor(self._cursor_for_hit(self._hit))

    def _resize_opposite_anchor(self, hit: int) -> QtCore.QPointF:
        """返回调整大小时应固定的对角点（控件内坐标）。

        注意：仅角控制点使用"对角锚点"；边控制点由 _apply_resize 单独处理
        （锁定垂直轴），这里不再为边返回锚点。
        """
        r = self._draw_rect
        if hit == self._HIT_TL:
            return QtCore.QPointF(r.x() + r.width(), r.y() + r.height())
        if hit == self._HIT_TR:
            return QtCore.QPointF(r.x(), r.y() + r.height())
        if hit == self._HIT_BL:
            return QtCore.QPointF(r.x() + r.width(), r.y())
        if hit == self._HIT_BR:
            return QtCore.QPointF(r.x(), r.y())
        # 边控制点：返回对边中点作为"锚点"仅用于占位，实际由 _apply_resize 锁定轴
        if hit == self._HIT_L:
            return QtCore.QPointF(r.x() + r.width(), r.y() + r.height() / 2)
        if hit == self._HIT_R:
            return QtCore.QPointF(r.x(), r.y() + r.height() / 2)
        if hit == self._HIT_T:
            return QtCore.QPointF(r.x() + r.width() / 2, r.y() + r.height())
        if hit == self._HIT_B:
            return QtCore.QPointF(r.x() + r.width() / 2, r.y())
        return QtCore.QPointF(r.x(), r.y())

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._mode == "draw":
            self._drag_current = pos
            self.update()
        elif self._mode == "move":
            # 移动整框
            delta = pos - self._drag_start
            self._apply_move_delta(delta)
            self._drag_start = pos
            self.update()
        elif self._mode == "resize":
            self._drag_current = pos
            self._apply_resize(pos)
            self.update()
        elif self._mode == "pan":
            if self._pan_last is not None:
                delta = pos - self._pan_last
                self._offset += delta
                self._clamp_offset()
                self.update()
            self._pan_last = pos
        else:
            # 悬停：更新光标
            hit = self._hit_test(pos)
            self.setCursor(self._cursor_for_hit(hit))

    def _apply_move_delta(self, delta: QtCore.QPointF):
        """按控件内位移移动选区。"""
        if self._draw_rect is None:
            return
        r = self._image_rect()
        if r is None:
            return
        new_rect = QtCore.QRectF(self._draw_rect)
        new_rect.translate(delta.x(), delta.y())
        # 钳制到图片内
        if new_rect.left() < r.left():
            new_rect.moveLeft(r.left())
        if new_rect.top() < r.top():
            new_rect.moveTop(r.top())
        if new_rect.right() > r.right():
            new_rect.moveRight(r.right())
        if new_rect.bottom() > r.bottom():
            new_rect.moveBottom(r.bottom())
        self._draw_rect = new_rect
        self._sync_region_from_rect()

    def _apply_resize(self, pos: QtCore.QPointF):
        """根据拖到的位置 + 控制点类型更新选区。

        - 角控制点：以对角为锚，两轴同时缩放
        - 边控制点：只沿该边法向缩放，锁定另一轴（对边不动）
        """
        if self._draw_rect is None or self._resize_anchor is None:
            return
        r = self._image_rect()
        if r is None:
            return
        hit = self._hit
        px = max(r.left(), min(pos.x(), r.right()))
        py = max(r.top(), min(pos.y(), r.bottom()))

        if hit in (self._HIT_TL, self._HIT_TR, self._HIT_BL, self._HIT_BR):
            # 角：对角锚点，两轴缩放
            anchor = self._resize_anchor
            x1 = min(anchor.x(), px)
            y1 = min(anchor.y(), py)
            x2 = max(anchor.x(), px)
            y2 = max(anchor.y(), py)
            self._draw_rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
        elif hit == self._HIT_L:
            # 左边：只改 left，right/top/bottom 不变
            rect = self._draw_rect
            new_left = min(px, rect.right() - 1)  # 保证宽度 >= 1
            self._draw_rect = QtCore.QRectF(
                new_left, rect.top(), rect.right() - new_left, rect.height()
            )
        elif hit == self._HIT_R:
            rect = self._draw_rect
            new_right = max(px, rect.left() + 1)
            self._draw_rect = QtCore.QRectF(
                rect.left(), rect.top(), new_right - rect.left(), rect.height()
            )
        elif hit == self._HIT_T:
            rect = self._draw_rect
            new_top = min(py, rect.bottom() - 1)
            self._draw_rect = QtCore.QRectF(
                rect.left(), new_top, rect.width(), rect.bottom() - new_top
            )
        elif hit == self._HIT_B:
            rect = self._draw_rect
            new_bottom = max(py, rect.top() + 1)
            self._draw_rect = QtCore.QRectF(
                rect.left(), rect.top(), rect.width(), new_bottom - rect.top()
            )
        else:
            return
        self._sync_region_from_rect()

    def _sync_region_from_rect(self):
        """从控件内矩形反推归一化 bbox 并发出信号。"""
        r = self._image_rect()
        if r is None or self._draw_rect is None:
            return
        rect = self._draw_rect
        nx1 = (rect.left() - r.left()) / max(1.0, r.width())
        ny1 = (rect.top() - r.top()) / max(1.0, r.height())
        nx2 = (rect.right() - r.left()) / max(1.0, r.width())
        ny2 = (rect.bottom() - r.top()) / max(1.0, r.height())
        self._region = [
            max(0.0, min(1.0, nx1)),
            max(0.0, min(1.0, ny1)),
            max(0.0, min(1.0, nx2)),
            max(0.0, min(1.0, ny2)),
        ]
        self.region_changed.emit(self._region)

    def mouseReleaseEvent(self, event):
        pos = event.position()
        if self._mode == "draw":
            self._drag_current = pos
            r = self._image_rect()
            if r is None:
                self._drag_start = None
                self._mode = "none"
                return
            x1 = min(self._drag_start.x(), self._drag_current.x())
            y1 = min(self._drag_start.y(), self._drag_current.y())
            x2 = max(self._drag_start.x(), self._drag_current.x())
            y2 = max(self._drag_start.y(), self._drag_current.y())
            x1 = max(r.left(), min(x1, r.right()))
            y1 = max(r.top(), min(y1, r.bottom()))
            x2 = max(r.left(), min(x2, r.right()))
            y2 = max(r.top(), min(y2, r.bottom()))
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                self._region = None
                self._draw_rect = None
                self.update()
                self.region_changed.emit(None)
            else:
                self._draw_rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
                self._sync_region_from_rect()
                self.update()
        elif self._mode in ("move", "resize"):
            self._sync_region_from_rect()
            self.update()
        self._mode = "none"
        self._drag_start = None
        self._pan_last = None
        self._hit = self._HIT_NONE
        self.setCursor(Qt.ArrowCursor)

    # ================= 滚轮缩放 =================
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.zoom(factor, event.position())
            event.accept()
        else:
            super().wheelEvent(event)

    # ================= 绘制 =================
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        # 背景
        painter.fillRect(self.rect(), QtGui.QColor(245, 245, 245))
        if self._pixmap is None or self._pixmap.isNull():
            if self._empty_text:
                painter.setPen(QtGui.QColor(150, 150, 150))
                painter.drawText(self.rect(), Qt.AlignCenter, self._empty_text)
            painter.end()
            return
        # 缩放后尺寸
        fit = self._fit_size()
        w = max(1, int(fit.width() * self._zoom))
        h = max(1, int(fit.height() * self._zoom))
        x = (self.width() - w) / 2 + self._offset.x()
        y = (self.height() - h) / 2 + self._offset.y()
        target = QtCore.QRectF(x, y, w, h)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._pixmap, QtCore.QRectF(self._pixmap.rect()))
        painter.end()

        # 选区绘制（在原图上叠加）
        rect = self._draw_rect
        if rect is None and self._mode == "draw" and self._drag_start is not None and self._drag_current is not None:
            x1 = min(self._drag_start.x(), self._drag_current.x())
            y1 = min(self._drag_start.y(), self._drag_current.y())
            x2 = max(self._drag_start.x(), self._drag_current.x())
            y2 = max(self._drag_start.y(), self._drag_current.y())
            rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
        if rect is not None:
            painter = QtGui.QPainter(self)
            pen = QtGui.QPen(QtGui.QColor(30, 160, 255), 2)
            painter.setPen(pen)
            painter.drawRect(rect)
            fill = QtGui.QColor(30, 160, 255, 40)
            painter.fillRect(rect, fill)
            # 画 8 个控制点手柄
            painter.setBrush(QtGui.QColor(255, 255, 255))
            handle = self._HANDLE_RADIUS // 2
            pts = [
                QtCore.QPointF(rect.left(), rect.top()),
                QtCore.QPointF(rect.left() + rect.width() / 2, rect.top()),
                QtCore.QPointF(rect.right(), rect.top()),
                QtCore.QPointF(rect.right(), rect.top() + rect.height() / 2),
                QtCore.QPointF(rect.right(), rect.bottom()),
                QtCore.QPointF(rect.left() + rect.width() / 2, rect.bottom()),
                QtCore.QPointF(rect.left(), rect.bottom()),
                QtCore.QPointF(rect.left(), rect.top() + rect.height() / 2),
            ]
            for p in pts:
                painter.drawRect(
                    QtCore.QRectF(p.x() - handle, p.y() - handle, handle * 2, handle * 2)
                )
            painter.end()


class ZoomableImageLabel(QLabel):
    """可缩放、可平移的图片标签（无框选），用于拼豆图纸 / 网格预览等只读视图。

    功能：
      - Ctrl + 滚轮：缩放图片（以光标为中心）
      - 鼠标左键拖拽 或 中键拖拽：平移图片（图片超出控件时可见范围随之移动）
      - 双击：复位到适配视图

    实现要点：
      - 用 paintEvent 手动绘制 pixmap（按 zoom 缩放 + offset 平移），
        而不是 setPixmap —— 这样 pixmap 可以任意超出控件边界，
        拖动平移即改变 offset，能完整看到放大后缺失的部分。
      - 控件尺寸固定（QSizePolicy.Ignored），缩放不会反向撑大所属工作区。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        # 关键：阻止 pixmap 的 sizeHint 反向撑大父布局（工作区尺寸固定）
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._pixmap: Optional[QPixmap] = None
        self._zoom = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._pan_last = None
        self._panning = False
        self._empty_text = ""

    def set_image(self, path: str):
        """载入图片并复位视图。"""
        self._pixmap = QPixmap(path)
        self._zoom = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._empty_text = ""
        self._clamp_offset()
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._clamp_offset()
        self.update()

    def clear_image(self):
        self._pixmap = None
        self._empty_text = ""
        self.update()

    def setText(self, text: str):
        # 覆盖 setText 用于占位提示（图片为空时显示）
        self._empty_text = text
        self.update()

    # ================= 尺寸 / 显示 =================
    def _fit_size(self) -> QtCore.QSize:
        """图片适配到控件尺寸（保持宽高比）后的逻辑尺寸。"""
        if self._pixmap is None or self._pixmap.isNull():
            return QtCore.QSize(0, 0)
        return self._pixmap.size().scaled(
            self.size(), Qt.KeepAspectRatio
        )

    def _scaled_size(self) -> QtCore.QSize:
        """缩放后图片的像素尺寸。"""
        fit = self._fit_size()
        if fit.width() <= 0 or fit.height() <= 0:
            return QtCore.QSize(0, 0)
        w = max(1, int(fit.width() * self._zoom))
        h = max(1, int(fit.height() * self._zoom))
        return QtCore.QSize(w, h)

    def _clamp_offset(self):
        """把偏移钳制到合理范围，避免图片完全拖出视野。

        注意：图片放大后比控件大，允许 offset 在 [-(iw-cw)/2, +(iw-cw)/2]
        范围内移动（即图片边缘对齐控件边缘），保证所有内容都可滚动看到。
        """
        if self._pixmap is None or self._pixmap.isNull():
            return
        iw, ih = self._scaled_size().width(), self._scaled_size().height()
        cw, ch = self.width(), self.height()
        # 允许图片边缘贴到控件边缘（无 margin），确保任何部分都能滚动到视野内
        maxx = max(0.0, (iw - cw) / 2.0)
        maxy = max(0.0, (ih - ch) / 2.0)
        self._offset.setX(max(-maxx, min(maxx, self._offset.x())))
        self._offset.setY(max(-maxy, min(maxy, self._offset.y())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_offset()
        self.update()

    # ================= 绘制 =================
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        # 背景
        painter.fillRect(self.rect(), QtGui.QColor(245, 245, 245))
        if self._pixmap is None or self._pixmap.isNull():
            if self._empty_text:
                painter.setPen(QtGui.QColor(150, 150, 150))
                painter.drawText(self.rect(), Qt.AlignCenter, self._empty_text)
            painter.end()
            return
        # 缩放后的尺寸（保持宽高比）
        fit = self._fit_size()
        w = max(1, int(fit.width() * self._zoom))
        h = max(1, int(fit.height() * self._zoom))
        # 图片中心对齐 + offset 平移
        x = (self.width() - w) / 2 + self._offset.x()
        y = (self.height() - h) / 2 + self._offset.y()
        target = QtCore.QRectF(x, y, w, h)
        # 直接从原始 pixmap 动态缩放绘制（放大不失真）
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._pixmap, QtCore.QRectF(self._pixmap.rect()))
        painter.end()

    # ================= 鼠标事件 =================
    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_last is not None:
            delta = event.position() - self._pan_last
            self._offset += delta
            self._clamp_offset()
            self._pan_last = event.position()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.reset_view()
            event.accept()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self._zoom = max(0.1, min(8.0, self._zoom * factor))
            self._clamp_offset()
            self.update()
            event.accept()
        else:
            super().wheelEvent(event)


# =============================================================================
# 可编辑网格控件：点击/拖动改色、框选、按颜色高亮
# =============================================================================

class GridEditWidget(QWidget):
    """可交互编辑的拼豆网格预览。

    在结果网格上直接涂色/框选/按色选中，支持缩放平移。数据源是
    QuantizeResult.grid（List[List[PaletteColor]]），编辑后通过信号
    `grid_changed` 回传，由主窗口重渲染网格预览与图纸。

    交互模式（由主窗口通过 set_tool 切换）：
      - "paint":  左键点击/拖动涂色；Ctrl+点击 = 取色器（吸管）
      - "select": 左键拖动=方形框选；拖动轨迹=路径选择；单击=切换单格；
                  右键=清空选中
      - "pick":   吸管取色（点击格子取其颜色设为画笔）
      - "erase":  橡皮擦（点击/拖动清空格子为空白）
      - "pan":    拖拽平移 / Ctrl+滚轮缩放（任意模式下 Ctrl+滚轮都缩放）

    支持：
      - 悬浮试色（hover 时临时预览画笔色，移开恢复）
      - 空白格子渲染（淡灰斜叉）
    """
    # 信号
    edit_started = Signal()          # 一次编辑操作即将开始（按下时发出，供主窗口记录撤销快照）
    grid_changed = Signal()          # 网格被编辑（供主窗口重渲染）
    selection_changed = Signal()     # 选中变化
    brush_picked = Signal(str)       # 取色器取到颜色（code）
    cell_clicked = Signal(int, int)  # 单击某个格子（供"图上按颜色选"）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(200, 160)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._result: Optional[object] = None   # QuantizeResult
        self._grid = None                        # List[List[PaletteColor]]
        self._gw = 0
        self._gh = 0

        self._zoom = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._tool = "paint"                     # paint | select | pick | erase | pan
        self._brush_code: Optional[str] = None   # 当前画笔色号

        # 高亮色号集合（按颜色选中时轮廓，需求3改为不再高亮，仅选中）
        self._highlight_codes: set = set()
        # 选中的格子集合
        self._selected_cells: set = set()
        # 交互状态
        self._panning = False
        self._pan_last = None
        self._selecting = False          # 正在框选
        self._select_start = None        # (row,col) 框选起点
        self._select_current = None
        self._select_mode = "rect"       # rect=方形 | path=拖动轨迹
        self._path_cells = set()         # 拖动轨迹经过的格子（本次）
        self._path_add = True            # 本次路径拖动的方向（True=加选，False=减选）
        self._painting = False
        self._erasing = False
        # 悬浮试色
        self._hover_cell = None          # 当前悬浮的格子 (row,col)
        self._hover_show_brush = False   # 悬浮时是否临时显示画笔色
        # 预览（色板悬浮试色）：{(row,col)} 格子临时显示 preview_code
        self._preview_cells: set = set()
        self._preview_code: Optional[str] = None
        self._empty_text = ""
        # 编辑许可（False=只读，如停在参数区时）
        self._editable = True
        # Ctrl 键按下状态（Ctrl+悬停显示格色号）
        self._ctrl_pressed = False

    # ================= 数据载入 =================
    def set_result(self, result):
        """载入量化结果并复位视图。"""
        self._result = result
        self._grid = result.grid
        self._gw = result.grid_w
        self._gh = result.grid_h
        self._selected_cells = set()
        self._highlight_codes = set()
        self._zoom = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._hover_cell = None
        self.update()
        self.selection_changed.emit()

    def clear_result(self):
        self._result = None
        self._grid = None
        self._selected_cells = set()
        self._highlight_codes = set()
        self.update()

    def setText(self, text: str):
        self._empty_text = text
        self.update()

    # ================= 工具 / 状态 =================
    def set_tool(self, tool: str):
        self._tool = tool
        self._panning = False
        self._selecting = False
        self._painting = False
        self._erasing = False
        self._path_cells = set()

    def set_editable(self, editable: bool):
        """设置是否可编辑。不可编辑时（如停在参数区），鼠标只用于平移/缩放，
        不响应涂色/擦除/取色/框选。"""
        self._editable = editable
        if not editable:
            self._panning = False
            self._selecting = False
            self._painting = False
            self._erasing = False

    def is_editable(self) -> bool:
        return self._editable

    def set_brush_code(self, code: Optional[str]):
        self._brush_code = code

    def set_highlight_codes(self, codes: set):
        self._highlight_codes = set(codes)
        self.update()

    def set_selected_cells(self, cells: set):
        self._selected_cells = set(cells)
        self.update()
        self.selection_changed.emit()

    def selected_cells(self) -> set:
        return set(self._selected_cells)

    def clear_selection(self):
        self._selected_cells = set()
        self.update()
        self.selection_changed.emit()

    def select_all_by_codes(self, codes: set):
        """按色号集合选中所有命中的格子。"""
        if self._grid is None:
            return
        hits = select_by_codes(self._result, set(codes))
        self._selected_cells = set(hits)
        self.update()
        self.selection_changed.emit()

    def toggle_cells_by_codes(self, codes: set, *, add: bool = True) -> bool:
        """按色号集合切换选中状态（叠加/取消）。

        对每个 code：
          - add=True：把命中的格子加入选中（叠加，不清除已有选中）
          - add=False：把命中的格子从选中移除
        返回：是否有格子被实际改变。
        """
        if self._grid is None:
            return False
        hits = set(select_by_codes(self._result, set(codes)))
        if add:
            before = len(self._selected_cells)
            self._selected_cells |= hits
            changed = len(self._selected_cells) != before
        else:
            before = len(self._selected_cells)
            self._selected_cells -= hits
            changed = len(self._selected_cells) != before
        if changed:
            self.update()
            self.selection_changed.emit()
        return changed

    def cells_for_codes_fully_selected(self, codes: set) -> bool:
        """判断 codes 对应的所有格子是否已全部处于选中状态。"""
        if self._grid is None:
            return False
        hits = select_by_codes(self._result, set(codes))
        if not hits:
            return False
        return all((r, c) in self._selected_cells for (r, c) in hits)

    def pick_color_at(self, r: int, c: int) -> Optional[str]:
        """返回 (r,c) 格子的色号（取色器用），空白返回 None。"""
        if self._grid is None:
            return None
        if 0 <= r < self._gh and 0 <= c < self._gw:
            cell = self._grid[r][c]
            return None if is_blank(cell) else cell.code
        return None

    # ================= 几何换算 =================
    def _cell_px(self) -> float:
        if self._gw <= 0 or self._gh <= 0:
            return 0.0
        avail_w = max(1, self.width() - 4)
        avail_h = max(1, self.height() - 4)
        base = min(avail_w / self._gw, avail_h / self._gh)
        return base * self._zoom

    def _grid_origin(self) -> QtCore.QPointF:
        cp = self._cell_px()
        gw_px = cp * self._gw
        gh_px = cp * self._gh
        x = (self.width() - gw_px) / 2 + self._offset.x()
        y = (self.height() - gh_px) / 2 + self._offset.y()
        return QtCore.QPointF(x, y)

    def _cell_at(self, pos: QtCore.QPointF):
        cp = self._cell_px()
        if cp <= 0:
            return None
        o = self._grid_origin()
        c = int((pos.x() - o.x()) / cp)
        r = int((pos.y() - o.y()) / cp)
        if 0 <= r < self._gh and 0 <= c < self._gw:
            return (r, c)
        return None

    def _cell_rect(self, r: int, c: int) -> QtCore.QRectF:
        cp = self._cell_px()
        o = self._grid_origin()
        return QtCore.QRectF(o.x() + c * cp, o.y() + r * cp, cp, cp)

    def _clamp_offset(self):
        cp = self._cell_px()
        gw_px = cp * self._gw
        gh_px = cp * self._gh
        maxx = max(0.0, (gw_px - self.width()) / 2 + 40)
        maxy = max(0.0, (gh_px - self.height()) / 2 + 40)
        self._offset.setX(max(-maxx, min(maxx, self._offset.x())))
        self._offset.setY(max(-maxy, min(maxy, self._offset.y())))

    # ================= 绘制 =================
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(245, 245, 245))
        if self._grid is None:
            if self._empty_text:
                painter.setPen(QtGui.QColor(150, 150, 150))
                painter.drawText(self.rect(), Qt.AlignCenter, self._empty_text)
            painter.end()
            return

        cp = self._cell_px()
        if cp < 1.5:
            painter.setPen(QtGui.QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "缩放过小，请 Ctrl+滚轮放大")
            painter.end()
            return
        o = self._grid_origin()

        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        # 绘制每个格子（含空白、悬浮试色）
        for r in range(self._gh):
            for c in range(self._gw):
                cell = self._grid[r][c]
                rect = QtCore.QRectF(o.x() + c * cp, o.y() + r * cp, cp, cp)
                if is_blank(cell):
                    painter.fillRect(rect, QtGui.QColor(255, 255, 255))
                    # 淡灰斜叉标记空白
                    if cp >= 6:
                        x0, y0 = rect.x(), rect.y()
                        x1, y1 = rect.right(), rect.bottom()
                        pen = QtGui.QPen(QtGui.QColor(228, 228, 228), 1)
                        painter.setPen(pen)
                        painter.drawLine(
                            QtCore.QPointF(x0 + cp*0.2, y0 + cp*0.2),
                            QtCore.QPointF(x1 - cp*0.2, y1 - cp*0.2),
                        )
                        painter.drawLine(
                            QtCore.QPointF(x1 - cp*0.2, y0 + cp*0.2),
                            QtCore.QPointF(x0 + cp*0.2, y1 - cp*0.2),
                        )
                else:
                    # 悬浮试色：hover 格且画笔模式时临时显示画笔色
                    if (self._hover_cell == (r, c) and self._hover_show_brush
                            and self._brush_code is not None
                            and self._tool == "paint"):
                        pc = self._result_palette_color(self._brush_code)
                        if pc is not None:
                            painter.fillRect(rect, QtGui.QColor(*pc.rgb))
                        else:
                            painter.fillRect(rect, QtGui.QColor(*cell.rgb))
                    # 色板悬浮预览：选中的格子临时显示目标色
                    elif (self._preview_code is not None
                            and (r, c) in self._preview_cells):
                        pc = self._result_palette_color(self._preview_code)
                        if pc is not None:
                            painter.fillRect(rect, QtGui.QColor(*pc.rgb))
                        else:
                            painter.fillRect(rect, QtGui.QColor(*cell.rgb))
                    else:
                        painter.fillRect(rect, QtGui.QColor(*cell.rgb))

        # 高亮轮廓（按颜色选中时，仅轮廓不覆盖，避免影响观察颜色）
        if self._highlight_codes:
            hl_pen = QtGui.QPen(QtGui.QColor(255, 80, 80), max(1.0, cp * 0.12))
            for r in range(self._gh):
                for c in range(self._gw):
                    if self._grid[r][c].code in self._highlight_codes:
                        rect = QtCore.QRectF(o.x() + c * cp, o.y() + r * cp, cp, cp)
                        painter.setPen(hl_pen)
                        painter.setBrush(QtCore.Qt.NoBrush)
                        painter.drawRect(rect)

        # 选中格子：蓝色轮廓（只画选中区域外边界，相邻选中格共享边不重复绘制）
        if self._selected_cells:
            sel_pen = QtGui.QPen(QtGui.QColor(30, 120, 255), max(2.0, cp * 0.20))
            sel_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(sel_pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            sel = self._selected_cells
            # 先画竖直边：该格被选中、且其左侧格未选中时，画左边界
            for (r, c) in sel:
                if not (0 <= r < self._gh and 0 <= c < self._gw):
                    continue
                rect = QtCore.QRectF(o.x() + c * cp, o.y() + r * cp, cp, cp)
                if (r, c - 1) not in sel:
                    painter.drawLine(rect.topLeft(), rect.bottomLeft())
                if (r, c + 1) not in sel:
                    painter.drawLine(rect.topRight(), rect.bottomRight())
            # 再画水平边
            for (r, c) in sel:
                if not (0 <= r < self._gh and 0 <= c < self._gw):
                    continue
                rect = QtCore.QRectF(o.x() + c * cp, o.y() + r * cp, cp, cp)
                if (r - 1, c) not in sel:
                    painter.drawLine(rect.topLeft(), rect.topRight())
                if (r + 1, c) not in sel:
                    painter.drawLine(rect.bottomLeft(), rect.bottomRight())

        # 网格线
        grid_pen = QtGui.QPen(QtGui.QColor(210, 210, 210), 1)
        painter.setPen(grid_pen)
        for c in range(self._gw + 1):
            x = o.x() + c * cp
            painter.drawLine(QtCore.QPointF(x, o.y()), QtCore.QPointF(x, o.y() + self._gh * cp))
        for r in range(self._gh + 1):
            y = o.y() + r * cp
            painter.drawLine(QtCore.QPointF(o.x(), y), QtCore.QPointF(o.x() + self._gw * cp, y))

        # 框选中的虚线框 / 路径选择高亮
        sel_pen = QtGui.QPen(QtGui.QColor(30, 120, 255), 2, Qt.DashLine)
        painter.setPen(sel_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        if self._selecting:
            if self._select_mode == "rect" and self._select_start is not None \
                    and self._select_current is not None:
                r1, c1 = self._select_start
                r2, c2 = self._select_current
                rmin, rmax = min(r1, r2), max(r1, r2)
                cmin, cmax = min(c1, c2), max(c1, c2)
                rect = QtCore.QRectF(
                    o.x() + cmin * cp, o.y() + rmin * cp,
                    (cmax - cmin + 1) * cp, (rmax - rmin + 1) * cp,
                )
                painter.drawRect(rect)
            elif self._select_mode == "path":
                # 路径经过的格子画实线框
                path_pen = QtGui.QPen(QtGui.QColor(30, 120, 255), 2, Qt.SolidLine)
                painter.setPen(path_pen)
                for (r, c) in self._path_cells:
                    rect = QtCore.QRectF(o.x() + c * cp, o.y() + r * cp, cp, cp)
                    painter.drawRect(rect)

        # Ctrl 悬停：显示鼠标停留格子的色号（替代通用说明）
        if self._ctrl_pressed and self._hover_cell is not None:
            r, c = self._hover_cell
            cell = self._grid[r][c]
            if is_blank(cell):
                code_text = "空白"
            else:
                code_text = cell.code
            # 在鼠标附近的格子上方绘制色号提示浮层
            rect = self._cell_rect(r, c)
            text_rect = QtCore.QRectF(
                rect.x() + cp * 0.5, rect.y() - cp * 0.6, cp * 3.0, cp * 0.6
            )
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(40, 40, 40, 220))
            painter.drawRoundedRect(text_rect, 3, 3)
            painter.setPen(QtGui.QColor(255, 255, 255))
            font = painter.font()
            font.setPointSizeF(max(7.0, cp * 0.45))
            painter.setFont(font)
            painter.drawText(
                text_rect, Qt.AlignCenter, f"{code_text}"
            )

        painter.end()

    # ================= 鼠标事件 =================
    def mousePressEvent(self, event):
        if self._grid is None:
            return
        pos = event.position()
        # 右键：清空选中（任意工具）
        if event.button() == Qt.RightButton:
            self.clear_selection()
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_last = pos
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            return

        # 不可编辑（如停在参数区）：左键仅用于平移，不响应任何编辑操作
        if not self._editable:
            self._panning = True
            self._pan_last = pos
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        cell = self._cell_at(pos)
        # Ctrl+点击 = 取色器（在 paint/select 模式下都可用）
        if event.modifiers() & Qt.ControlModifier:
            if cell is not None:
                code = self.pick_color_at(*cell)
                if code is not None:
                    self.brush_picked.emit(code)
            event.accept()
            return

        if self._tool == "paint":
            self._painting = True
            self.edit_started.emit()
            if cell is not None:
                self._paint_cell(cell)
            event.accept()
        elif self._tool == "erase":
            self._erasing = True
            self.edit_started.emit()
            if cell is not None:
                self._erase_cell(cell)
            event.accept()
        elif self._tool == "pick":
            if cell is not None:
                code = self.pick_color_at(*cell)
                if code is not None:
                    self.brush_picked.emit(code)
            event.accept()
        elif self._tool == "select":
            self._selecting = True
            self._select_start = cell if cell else (0, 0)
            self._select_current = self._select_start
            # Shift+拖动 = 方形框选；普通拖动 = 路径选择（画笔式经过即选/反选）
            if event.modifiers() & Qt.ShiftModifier:
                self._select_mode = "rect"
                self._path_add = True
            else:
                self._select_mode = "path"
                # 决定本次拖动的方向：从已选中格开始=减选（反选），从未选中格开始=加选
                self._path_add = cell is None or cell not in self._selected_cells
            self._path_cells = set()
            if cell is not None:
                self._path_cells.add(cell)
            self.update()
            event.accept()
        elif self._tool == "pan":
            self._panning = True
            self._pan_last = pos
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._grid is None:
            return
        pos = event.position()
        # 实时跟踪 Ctrl 状态（不依赖焦点，悬停提示用）
        ctrl = bool(event.modifiers() & Qt.ControlModifier)
        if ctrl != self._ctrl_pressed:
            self._ctrl_pressed = ctrl
            self.update()
        if self._panning:
            delta = pos - self._pan_last
            self._offset += delta
            self._clamp_offset()
            self._pan_last = pos
            self.update()
            event.accept()
            return
        if self._painting and self._tool == "paint":
            cell = self._cell_at(pos)
            if cell is not None:
                self._paint_cell(cell)
            event.accept()
            return
        if self._erasing and self._tool == "erase":
            cell = self._cell_at(pos)
            if cell is not None:
                self._erase_cell(cell)
            event.accept()
            return
        if self._selecting and self._tool == "select":
            cell = self._cell_at(pos)
            if cell is not None:
                self._select_current = cell
                # 路径模式：经过的格子都加入 path_cells
                if self._select_mode == "path":
                    self._path_cells.add(cell)
            self.update()
            event.accept()
            return
        # 悬浮试色（paint 模式且未按下时）
        if self._tool == "paint":
            cell = self._cell_at(pos)
            if cell != self._hover_cell:
                self._hover_cell = cell
                self._hover_show_brush = (cell is not None)
                self.update()
            event.accept()
            return
        else:
            # 其他模式：更新 hover 但不试色
            cell = self._cell_at(pos)
            if cell != self._hover_cell:
                self._hover_cell = cell
                self.update()

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        if self._painting:
            self._painting = False
            self.grid_changed.emit()
        if self._erasing:
            self._erasing = False
            self.grid_changed.emit()
        if self._selecting and self._tool == "select":
            self._selecting = False
            self._finish_select()
        event.accept()

    def leaveEvent(self, event):
        """鼠标离开控件：清除悬浮试色。"""
        self._hover_cell = None
        self._hover_show_brush = False
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        """跟踪 Ctrl 键：按下时悬停格子显示色号。"""
        if event.key() == Qt.Key_Control:
            self._ctrl_pressed = True
            self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self._ctrl_pressed = False
            self.update()
        super().keyReleaseEvent(event)

    def _paint_cell(self, cell):
        """把单个格子改成当前画笔色。"""
        if self._brush_code is None:
            return
        r, c = cell
        target = self._result_palette_color(self._brush_code)
        if target is None:
            return
        if self._grid[r][c].code != target.code:
            self._grid[r][c] = target
            self.update()

    def _erase_cell(self, cell):
        """把单个格子清空为空白。"""
        r, c = cell
        if not is_blank(self._grid[r][c]):
            self._grid[r][c] = BLANK
            self.update()

    def _result_palette_color(self, code):
        # 从主窗口传入的 palette 查找（由主窗口 set_palette 设置）
        palette = getattr(self, "_palette", None)
        if palette is None:
            return None
        return palette.by_code(code)

    def set_palette(self, palette):
        self._palette = palette

    def set_preview(self, cells: set, code: Optional[str]):
        """临时预览：把 cells 里的格子临时显示为 code 色（不写入 grid）。

        code=None 表示清除预览。用于色板悬浮试色。
        """
        self._preview_cells = set(cells) if cells else set()
        self._preview_code = code
        self.update()

    def clear_preview(self):
        self.set_preview(set(), None)

    def _finish_select(self):
        """框选结束：处理方形框选 / 路径选择 / 点击切换。"""
        if self._select_start is None or self._select_current is None:
            return
        r1, c1 = self._select_start
        r2, c2 = self._select_current

        # 路径选择模式（普通拖动）
        if self._select_mode == "path":
            if len(self._path_cells) <= 1 and (r1, c1) in self._path_cells:
                # 单击：切换该格选中状态（点击框选 / 反选）
                self._toggle_cell(r1, c1)
            else:
                # 拖动：按本次方向（加选/减选）处理所有经过的格子
                for (r, c) in self._path_cells:
                    if self._path_add:
                        self._selected_cells.add((r, c))
                    else:
                        self._selected_cells.discard((r, c))
            self.update()
            self.selection_changed.emit()
            self._select_mode = "rect"
            self._path_cells = set()
            return

        # 方形框选（Shift+拖动）
        rmin, rmax = min(r1, r2), max(r1, r2)
        cmin, cmax = min(c1, c2), max(c1, c2)
        # Shift+单击（起止同一格）→ 切换该格选中状态
        if rmin == rmax and cmin == cmax:
            self._toggle_cell(rmin, cmin)
        else:
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    self._selected_cells.add((r, c))
        self.update()
        self.selection_changed.emit()

    def _toggle_cell(self, r, c):
        """切换单格选中状态（反选）。"""
        key = (r, c)
        if key in self._selected_cells:
            self._selected_cells.discard(key)
        else:
            self._selected_cells.add(key)

    def select_all(self):
        """全选：选中网格中所有格子（含空白）。"""
        if self._grid is None:
            return
        self._selected_cells = {
            (r, c) for r in range(self._gh) for c in range(self._gw)
        }
        self.update()
        self.selection_changed.emit()

    def invert_selection(self):
        """反选：当前未选中的格子变为选中，选中的取消。"""
        if self._grid is None:
            return
        all_cells = {(r, c) for r in range(self._gh) for c in range(self._gw)}
        self._selected_cells = all_cells - self._selected_cells
        self.update()
        self.selection_changed.emit()

    # ================= 缩放 / 平移 =================
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self._zoom = max(0.05, min(20.0, self._zoom * factor))
            self._clamp_offset()
            self.update()
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._zoom = 1.0
            self._offset = QtCore.QPointF(0, 0)
            self.update()
            event.accept()

    def reset_view(self):
        self._zoom = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_offset()
        self.update()


# =============================================================================
# 色板色块按钮：支持悬浮试色（enter/leave 信号）
# =============================================================================

class SwatchButton(QPushButton):
    """色板色块按钮，带 hover 信号（用于悬浮试色）。"""

    hovered = Signal(str)   # 进入时发出色号
    left = Signal()          # 离开时发出

    def __init__(self, code: str, parent=None):
        super().__init__(parent)
        self._code = code

    def enterEvent(self, event):
        self.hovered.emit(self._code)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.left.emit()
        super().leaveEvent(event)


# =============================================================================
# 导出对话框：选择导出哪些图 + 目标文件夹 + 文件名基名
# =============================================================================

class ExportDialog(QDialog):
    """导出选项对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出拼豆产物")
        self.setMinimumWidth(420)
        v = QVBoxLayout(self)

        tip = QLabel("勾选要导出的文件（基于当前图纸，含手动编辑）:")
        tip.setStyleSheet("color: #666;")
        v.addWidget(tip)

        self.chk_chart = QCheckBox("拼豆图纸（带色号标注）")
        self.chk_chart.setChecked(True)
        self.chk_grid = QCheckBox("网格预览（带网格线）")
        self.chk_grid.setChecked(True)
        self.chk_plain = QCheckBox("像素化预览（无网格线）")
        self.chk_plain.setChecked(True)
        self.chk_json = QCheckBox("JSON 色号矩阵")
        self.chk_json.setChecked(True)
        v.addWidget(self.chk_chart)
        v.addWidget(self.chk_grid)
        v.addWidget(self.chk_plain)
        v.addWidget(self.chk_json)

        v.addSpacing(8)
        # 文件名基名
        h_name = QHBoxLayout()
        h_name.addWidget(QLabel("文件名基名:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如 my_bead（各文件自动加 _chart/_grid/_plain 后缀）")
        h_name.addWidget(self.name_edit)
        v.addLayout(h_name)

        # 导出文件夹
        h_dir = QHBoxLayout()
        h_dir.addWidget(QLabel("导出目录:"))
        self.dir_edit = QLineEdit()
        h_dir.addWidget(self.dir_edit)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_dir)
        h_dir.addWidget(btn_browse)
        v.addLayout(h_dir)

        # 按钮
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("导出")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "选择导出目录", self.dir_edit.text() or str(Path.home()))
        if d:
            self.dir_edit.setText(d)

    def _on_accept(self):
        # 校验：至少选一个 + 目录非空 + 基名非空
        if not any([self.chk_chart.isChecked(), self.chk_grid.isChecked(),
                    self.chk_plain.isChecked(), self.chk_json.isChecked()]):
            QMessageBox.warning(self, "未选择", "请至少勾选一个要导出的文件。")
            return
        if not self.dir_edit.text().strip():
            QMessageBox.warning(self, "未选择目录", "请选择导出目录。")
            return
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "未填写文件名", "请填写文件名基名。")
            return
        self.accept()


# =============================================================================
# 后台线程：跑 quantize() 不卡 UI
# =============================================================================

class QuantizeWorker(QThread):
    """后台线程跑 quantize()。发信号到主线程更新进度和结果。"""

    progress = Signal(str)            # 进度文字
    finished_ok = Signal(object, str, str, str)  # result, chart_path, grid_path, json_path
    failed = Signal(str)              # 错误信息

    def __init__(
        self,
        image_path: str,
        grid: int,
        # 高级参数
        supersample: int,
        ss_strategy: str,
        ss_min_dark_count: int,
        ss_line_min_pixels: int,
        detail_mode: bool,
        outline_preserve: bool,
        min_island_size: int,
        cleanup_min_contrast: float,
        line_detect: bool,
        line_method: str,
        line_threshold_percentile: float,
        line_min_pct: float,
        min_line_count: int,
        min_line_contrast: float,
        line_aspect_min: float,
        line_min_region_size: float,
        out_dir: str,
        pixelize_method: str = "none",
        pixelize_params: Optional[dict] = None,
        chart_name: Optional[str] = None,
    ):
        super().__init__()
        self.image_path = image_path
        self.grid = grid
        self.supersample = supersample
        self.ss_strategy = ss_strategy
        self.ss_min_dark_count = ss_min_dark_count
        self.ss_line_min_pixels = ss_line_min_pixels
        self.detail_mode = detail_mode
        self.outline_preserve = outline_preserve
        self.min_island_size = min_island_size
        self.cleanup_min_contrast = cleanup_min_contrast
        self.line_detect = line_detect
        self.line_method = line_method
        self.line_threshold_percentile = line_threshold_percentile
        self.line_min_pct = line_min_pct
        self.min_line_count = min_line_count
        self.min_line_contrast = min_line_contrast
        self.line_aspect_min = line_aspect_min
        self.line_min_region_size = line_min_region_size
        self.out_dir = out_dir
        self.pixelize_method = pixelize_method
        self.pixelize_params = pixelize_params or {}
        self.chart_name = (chart_name or "").strip() or None
        self.key_features = None  # 由外部（卡通化流程）注入
        self.region = None        # 归一化 bbox [x1,y1,x2,y2]，仅对该区域量化
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.progress.emit("加载 MARD 色板...")
            palette = load_palette()

            # 处理框选区域：裁剪出子图后量化
            image_to_quantize = self.image_path
            if self.region is not None:
                from PIL import Image as _PILImage
                self.progress.emit("裁剪框选区域...")
                im = _PILImage.open(self.image_path).convert("RGB")
                w, h = im.size
                x1, y1, x2, y2 = self.region
                box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
                box = (
                    max(0, box[0]), max(0, box[1]),
                    min(w, box[2]), min(h, box[3]),
                )
                if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                    self.failed.emit("框选区域太小，请重新框选")
                    return
                crop = im.crop(box)
                # 保存裁剪子图到临时文件
                stem = Path(self.image_path).stem
                crop_path = Path(self.out_dir) / f"{stem}_region_crop.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(str(crop_path))
                image_to_quantize = str(crop_path)

            self.progress.emit(f"读取图片: {Path(image_to_quantize).name}")
            # 复制参数以防被改
            grid = self.grid
            # 关键特征：由卡通化流程注入（若有）；区域量化时不套用全局 bbox
            key_features = self.key_features if self.region is None else None

            self.progress.emit(f"色板量化到 {grid}×{grid}...")
            result = quantize(
                image_to_quantize,
                grid_w=grid,
                grid_h=grid,
                palette=palette,
                min_island_size=self.min_island_size,
                cleanup=True,
                detail_mode=self.detail_mode,
                min_sub_pct=0.06,
                min_contrast=18.0,
                key_features=key_features,
                outline_preserve=self.outline_preserve,
                protected_codes=None,
                outline_min_pct=0.015,
                outline_min_edge_pct=0.04,
                outline_center_l=85.0,
                supersample=self.supersample,
                ss_min_dark_count=self.ss_min_dark_count,
                ss_strategy=self.ss_strategy,
                ss_line_min_pixels=self.ss_line_min_pixels,
                cleanup_min_contrast=(self.cleanup_min_contrast if self.cleanup_min_contrast > 0 else None),
                line_detect=self.line_detect,
                line_method=self.line_method,
                line_threshold_percentile=self.line_threshold_percentile,
                line_min_pct=self.line_min_pct,
                min_line_count=self.min_line_count,
                min_line_contrast=(self.min_line_contrast if self.min_line_contrast > 0 else 0.0),
                line_aspect_min=self.line_aspect_min,
                line_min_region_size=self.line_min_region_size,
                pixelize_method=self.pixelize_method,
                pixelize_params=self.pixelize_params,
            )

            if self._cancel:
                self.failed.emit("已取消")
                return

            # 生成产物文件
            out_dir = Path(self.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = self.chart_name or Path(self.image_path).stem
            suffix = "_region" if self.region is not None else ""
            chart_path = out_dir / f"{stem}{suffix}_bead_chart_{grid}x{grid}.png"
            grid_path = out_dir / f"{stem}{suffix}_bead_grid_{grid}x{grid}.png"
            json_path = out_dir / f"{stem}{suffix}_bead_grid_{grid}x{grid}.json"

            self.progress.emit("渲染拼豆图纸...")
            cell_px = max(8, min(24, 1800 // grid))
            save_bead_chart(
                result, str(chart_path),
                cell_px=cell_px,
                label_every=max(5, grid // 12),
                title=stem,
                palette_name="MARD",
            )

            self.progress.emit("渲染网格预览...")
            grid_img = render_grid_only(result, cell_px=cell_px, with_grid=True)
            grid_img.save(str(grid_path))

            self.progress.emit("保存 JSON 描述...")
            grid_json = {
                "grid_w": result.grid_w,
                "grid_h": result.grid_h,
                "palette": "MARD",
                "cells": [[cell.code for cell in row] for row in result.grid],
            }
            json_path.write_text(
                json.dumps(grid_json, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

            self.progress.emit("完成 ✓")
            self.finished_ok.emit(result, str(chart_path), str(grid_path), str(json_path))
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n\n{tb}")


# =============================================================================
# 后台线程：跑 Stage 0 (LLM 理解) + Stage 1 (SeedDream 卡通化)
# =============================================================================

class CartoonizeWorker(QThread):
    """后台线程跑 LLM 视觉理解 + SeedDream 卡通化，不卡 UI。"""

    progress = Signal(str)                                   # 进度文字
    finished_ok = Signal(object, str)                        # PipelineResult, 卡通图路径
    failed = Signal(str)                                     # 错误信息

    def __init__(self, image_path: str, out_dir: str, custom_prompt: Optional[str] = None):
        super().__init__()
        self.image_path = image_path
        self.out_dir = out_dir
        self.custom_prompt = custom_prompt
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = run_full_pipeline(
                self.image_path,
                self.out_dir,
                custom_prompt=self.custom_prompt,
                progress_cb=self.progress.emit,
            )
            if self._cancel:
                self.failed.emit("已取消")
                return
            self.progress.emit("卡通化完成 ✓")
            self.finished_ok.emit(result, str(result.cartoon_png))
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n\n{tb}")


# =============================================================================
# 主窗口
# =============================================================================

class BeadStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("拼豆图纸生成器 · Beadobe")
        self.resize(1180, 780)
        self.setAcceptDrops(True)
        # 窗口/应用图标（透明底图标，随打包一起收集为只读资源）
        try:
            icon_path = resource_path("icon_transparent.png")
            if icon_path.exists():
                self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        except Exception:
            pass

        # State
        self.image_path: Optional[str] = None
        self.worker: Optional[QuantizeWorker] = None
        self.cartoon_worker: Optional[CartoonizeWorker] = None
        self.result = None
        self.last_chart_path: Optional[str] = None
        self.last_grid_path: Optional[str] = None
        self.last_plain_path: Optional[str] = None
        self.last_json_path: Optional[str] = None
        # Stage 0/1 产物
        self.key_features = None          # {name: [x1,y1,x2,y2]}
        self.analysis = None              # VisualAnalysis
        self.region = None                # 框选区域归一化 bbox [x1,y1,x2,y2]
        self.cartoon_path = None          # 卡通化结果路径（不覆盖原图）
        self._active_source = "orig"      # 最近激活的源图 Tab: orig | cartoon
        # 撤销/重做栈
        self._undo_stack = []             # List[List[List[PaletteColor]]] 网格快照
        self._redo_stack = []
        self._pick_on_canvas_active = False
        self._adjust_base_grid = None
        self._adjust_base_cells = None
        self._color_bubble = None
        self._replace_bubble = None
        self._replace_target_code = ""  # "" 表示空白（删除格子）

        self.palette = load_palette()

        self._build_ui()
        self._connect_signals()
        self._update_param_state()
        self._check_deps()
        # 初始停在参数区，网格只读
        self._on_left_tab_changed(self.left_tabs.currentIndex())

    # -------- UI 构建 --------

    def _build_ui(self):
        # 全局样式
        self.setStyleSheet(APP_QSS)
        # 工具栏
        tb = self.addToolBar("主工具栏")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.act_open = QAction("打开图片", self)
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_gen = QAction("生成拼豆图", self)
        self.act_gen.setShortcut("Ctrl+R")
        self.act_gen.setEnabled(False)
        self.act_save = QAction("导出", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.setEnabled(False)
        self.act_import = QAction("导入 JSON", self)
        self.act_import.setShortcut("Ctrl+I")
        self.act_settings = QAction("设置", self)
        self.act_about = QAction("关于", self)
        tb.addAction(self.act_open)
        tb.addAction(self.act_gen)
        tb.addAction(self.act_save)
        tb.addAction(self.act_import)
        tb.addSeparator()
        tb.addAction(self.act_settings)
        tb.addAction(self.act_about)

        # 中央: 左侧参数 + 右侧预览
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # 左: 参数面板（参数 / 编辑 共享位置，用 Tab 切换）
        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        # 右: 预览
        right_panel = self._build_preview_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([380, 800])

        # 状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(200)
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("就绪 · 拖入图片或点「打开图片」")

        # 快捷键：撤销 / 重做
        self._shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._shortcut_undo.activated.connect(self._on_undo)
        self._shortcut_redo = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self._shortcut_redo.activated.connect(self._on_redo)
        self._shortcut_select_all = QShortcut(QKeySequence("Ctrl+A"), self)
        self._shortcut_select_all.activated.connect(self._on_select_all)

    def _build_left_panel(self) -> QWidget:
        """左侧面板：参数区 / 绘图（编辑）区共享同一位置，用 Tab 切换。

        因为量化参数和手动编辑不会同时使用，用 QTabWidget 在两者间切换，
        节省左侧空间。
        """
        tabs = QTabWidget()
        self.left_tabs = tabs
        # Tab 1: 量化参数
        tabs.addTab(self._build_param_panel(), "参数")
        # Tab 2: 手动编辑（绘图）
        tabs.addTab(self._build_edit_panel(), "编辑")
        # 生成后自动切到编辑 Tab 的提示
        tabs.setTabToolTip(0, "设置量化参数后点击「生成拼豆图纸」")
        tabs.setTabToolTip(1, "生成后可在此手动编辑网格（涂色/框选/批量改色）")
        return tabs

    def _build_edit_panel(self) -> QWidget:
        """手动编辑面板：色板选择 + 工具 + 批量操作。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        form = QVBoxLayout(inner)
        form.setContentsMargins(4, 4, 4, 4)
        form.setSpacing(8)

        # 状态提示
        self.lbl_edit_status = QLabel("（尚未生成图纸，请先在「参数」页生成）")
        self.lbl_edit_status.setWordWrap(True)
        self.lbl_edit_status.setStyleSheet("color: #888; font-size: 12px;")
        form.addWidget(self.lbl_edit_status)

        # === 撤销 / 重做 ===
        gb_undo = QGroupBox("撤销 / 重做")
        vu = QHBoxLayout(gb_undo)
        self.btn_undo = QPushButton("撤销 (Ctrl+Z)")
        self.btn_redo = QPushButton("重做 (Ctrl+Shift+Z)")
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)
        vu.addWidget(self.btn_undo)
        vu.addWidget(self.btn_redo)
        form.addWidget(gb_undo)

        # === 工具切换 ===
        gb_tool = QGroupBox("编辑工具")
        v = QVBoxLayout(gb_tool)
        h = QHBoxLayout()
        self.btn_tool_paint = QPushButton("🖌 画笔")
        self.btn_tool_paint.setCheckable(True)
        self.btn_tool_paint.setChecked(True)
        self.btn_tool_erase = QPushButton("🧽 擦除")
        self.btn_tool_erase.setCheckable(True)
        self.btn_tool_pick = QPushButton("💧 取色")
        self.btn_tool_pick.setCheckable(True)
        self.btn_tool_select = QPushButton("⬚ 框选")
        self.btn_tool_select.setCheckable(True)
        self.btn_tool_pan = QPushButton("✋ 平移")
        self.btn_tool_pan.setCheckable(True)
        grp = QtWidgets.QButtonGroup(self)
        grp.setExclusive(True)
        for b in (self.btn_tool_paint, self.btn_tool_erase, self.btn_tool_pick,
                  self.btn_tool_select, self.btn_tool_pan):
            grp.addButton(b)
        for b in (self.btn_tool_paint, self.btn_tool_erase, self.btn_tool_pick,
                  self.btn_tool_select, self.btn_tool_pan):
            h.addWidget(b)
        v.addLayout(h)
        tip = QLabel(
            "🖌 画笔：点击/拖动涂色（悬停试色）\n"
            "🧽 擦除：点击/拖动清空格子为空白\n"
            "💧 取色：点击格子取其颜色为画笔\n"
            "⬚ 框选：拖动=路径选择（经过即选），Shift+拖动=方形框选，"
            "单击=反选单格，右键=清空\n"
            "✋ 平移：拖动或 Ctrl+滚轮缩放\n"
            "任意模式 Ctrl+点击 = 取色器"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #666; font-size: 11px;")
        v.addWidget(tip)
        form.addWidget(gb_tool)

        # === 画笔颜色（色块 + 色号代码） ===
        gb_brush = QGroupBox("画笔颜色（点击选中）")
        vb = QVBoxLayout(gb_brush)
        # 当前画笔色指示器（色块 + 文字）
        h_brush_cur = QHBoxLayout()
        self.lbl_brush_swatch = QLabel()
        self.lbl_brush_swatch.setFixedSize(26, 26)
        self.lbl_brush_swatch.setStyleSheet(
            "background-color: #cccccc; border: 2px solid #666; border-radius: 3px;"
        )
        h_brush_cur.addWidget(self.lbl_brush_swatch)
        self.lbl_brush_cur = QLabel("当前画笔：（未选）")
        self.lbl_brush_cur.setStyleSheet("font-weight: bold;")
        h_brush_cur.addWidget(self.lbl_brush_cur)
        h_brush_cur.addStretch(1)
        vb.addLayout(h_brush_cur)
        brush_scroll = QScrollArea()
        brush_scroll.setWidgetResizable(True)
        brush_scroll.setFrameShape(QFrame.NoFrame)
        brush_scroll.setMinimumHeight(200)
        self.palette_swatch = QWidget()
        self.palette_layout = QGridLayout(self.palette_swatch)
        self.palette_layout.setSpacing(1)
        self.palette_layout.setContentsMargins(2, 2, 2, 2)
        brush_scroll.setWidget(self.palette_swatch)
        vb.addWidget(brush_scroll)
        hf = QHBoxLayout()
        self.chk_brush_used_only = QCheckBox("只显示图纸用到的色")
        self.chk_brush_used_only.setChecked(False)
        hf.addWidget(self.chk_brush_used_only)
        hf.addStretch(1)
        vb.addLayout(hf)
        form.addWidget(gb_brush)

        # === 选中 & 批量改色 ===
        gb_sel = QGroupBox("选中 & 批量改色")
        vs = QVBoxLayout(gb_sel)
        self.lbl_sel_info = QLabel("已选中 0 格")
        self.lbl_sel_info.setStyleSheet("color: #2a7; font-weight: bold;")
        vs.addWidget(self.lbl_sel_info)
        # 按颜色选中（气泡菜单）
        h0 = QHBoxLayout()
        self.btn_select_by_color = QPushButton("按颜色选中…")
        self.btn_select_by_color.clicked.connect(self._show_color_bubble)
        h0.addWidget(self.btn_select_by_color)
        self.btn_pick_on_canvas = QPushButton("图上点选同色")
        self.btn_pick_on_canvas.setToolTip(
            "点击后回到网格预览，点击某个格子即可选中所有同色格子"
        )
        self.btn_pick_on_canvas.setCheckable(True)
        h0.addWidget(self.btn_pick_on_canvas)
        vs.addLayout(h0)
        # 反选 / 清空
        h1 = QHBoxLayout()
        self.btn_invert_sel = QPushButton("反选")
        self.btn_clear_sel = QPushButton("清除选中")
        h1.addWidget(self.btn_invert_sel)
        h1.addWidget(self.btn_clear_sel)
        vs.addLayout(h1)
        # 批量改色目标
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("改为:"))
        self.btn_replace_target = QPushButton("空白（删除格子）")
        self.btn_replace_target.setIconSize(QtCore.QSize(16, 16))
        self.btn_replace_target.setToolTip("点击选择要改成的颜色（色块 + 色号）")
        self.btn_replace_target.clicked.connect(self._show_replace_bubble)
        h2.addWidget(self.btn_replace_target, 1)
        self.btn_apply_replace = QPushButton("应用")
        h2.addWidget(self.btn_apply_replace)
        vs.addLayout(h2)
        form.addWidget(gb_sel)

        # === 颜色调整（三个滑条，实时） ===
        gb_adjust = QGroupBox("颜色调整（实时）")
        va = QVBoxLayout(gb_adjust)
        # 加深滑条
        va.addWidget(QLabel("加深:"))
        self.slider_darken = QSlider(Qt.Horizontal)
        self.slider_darken.setRange(0, 50)
        self.slider_darken.setValue(0)
        self.lbl_darken_val = QLabel("0")
        hd = QHBoxLayout()
        hd.addWidget(self.slider_darken, 1)
        hd.addWidget(self.lbl_darken_val)
        va.addLayout(hd)
        # 变浅滑条
        va.addWidget(QLabel("变浅:"))
        self.slider_lighten = QSlider(Qt.Horizontal)
        self.slider_lighten.setRange(0, 50)
        self.slider_lighten.setValue(0)
        self.lbl_lighten_val = QLabel("0")
        hl = QHBoxLayout()
        hl.addWidget(self.slider_lighten, 1)
        hl.addWidget(self.lbl_lighten_val)
        va.addLayout(hl)
        # 对比度滑条
        va.addWidget(QLabel("对比度 (1.0=原样):"))
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(50, 300)  # 0.5x ~ 3.0x
        self.slider_contrast.setValue(100)
        self.lbl_contrast_val = QLabel("1.00")
        hc = QHBoxLayout()
        hc.addWidget(self.slider_contrast, 1)
        hc.addWidget(self.lbl_contrast_val)
        va.addLayout(hc)
        self.btn_reset_adjust = QPushButton("重置滑条")
        va.addWidget(self.btn_reset_adjust)
        form.addWidget(gb_adjust)

        form.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_param_panel(self) -> QWidget:
        """左侧参数面板：滚动 + 多个 GroupBox"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        form = QVBoxLayout(inner)
        form.setContentsMargins(4, 4, 4, 4)
        form.setSpacing(8)

        # === 输入 ===
        gb_input = QGroupBox("输入图片")
        v = QVBoxLayout(gb_input)
        self.lbl_image = QLabel("（未选择）")
        self.lbl_image.setStyleSheet("color: #888;")
        self.lbl_image.setWordWrap(True)
        v.addWidget(self.lbl_image)
        h = QHBoxLayout()
        btn_browse = QPushButton("选择图片...")
        btn_browse.clicked.connect(self._on_open)
        h.addWidget(btn_browse)
        h.addStretch(1)
        v.addLayout(h)
        form.addWidget(gb_input)

        # === 卡通化 (Stage 0 + 1) ===
        gb_cartoon = QGroupBox("卡通化（LLM 理解 + SeedDream）")
        vc = QVBoxLayout(gb_cartoon)
        tip = QLabel("照片 → 卡通：先 LLM 理解关键特征，再 SeedDream 图生图。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #666; font-size: 12px;")
        vc.addWidget(tip)
        tip_key = QLabel("⚠ 需先在「设置」中配置 API Key（对话型 LLM + 方舟生图），否则无法使用卡通化。")
        tip_key.setWordWrap(True)
        tip_key.setStyleSheet("color: #c90; font-size: 12px;")
        vc.addWidget(tip_key)
        self.lbl_cartoon_status = QLabel("（未运行）")
        self.lbl_cartoon_status.setWordWrap(True)
        self.lbl_cartoon_status.setStyleSheet("color: #888;")
        vc.addWidget(self.lbl_cartoon_status)
        # 自定义 prompt（可选，覆盖 LLM 自动生成的卡通化 prompt）
        vc.addWidget(QLabel("自定义 Prompt（可选，留空则用 LLM 生成）:"))
        self.prompt_edit = QtWidgets.QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "例：清新可爱的卡通猫，白色短毛，半眯眼，深色眼缝，干净色块…"
        )
        self.prompt_edit.setMaximumHeight(90)
        self.prompt_edit.setToolTip(
            "填写后卡通化将直接使用这段 prompt，跳过 LLM 自动生成的描述。"
        )
        vc.addWidget(self.prompt_edit)
        self.btn_cartoon = QPushButton("卡通化当前图片")
        self.btn_cartoon.setEnabled(False)
        self.btn_cartoon.clicked.connect(self._on_cartoonize)
        vc.addWidget(self.btn_cartoon)
        if not HAS_PIPELINE:
            self.btn_cartoon.setToolTip("卡通化依赖 openai 库，未安装")
            self.lbl_cartoon_status.setText("（需安装 openai 库）")
        form.addWidget(gb_cartoon)

        # === 基础参数 ===
        gb_basic = QGroupBox("基础参数")
        grid = QGridLayout(gb_basic)
        # Grid size（滑块 + 输入框双向联动，上限 104）
        grid.addWidget(QLabel("Grid size (16-104):"), 0, 0)
        self.grid_slider = QSlider(Qt.Horizontal)
        self.grid_slider.setRange(16, 104)
        self.grid_slider.setValue(52)
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(16, 104)
        self.grid_spin.setValue(52)
        self.grid_spin.setMinimumWidth(64)
        self.grid_slider.valueChanged.connect(self.grid_spin.setValue)
        self.grid_spin.valueChanged.connect(self.grid_slider.setValue)
        grid.addWidget(self.grid_slider, 0, 1)
        grid.addWidget(self.grid_spin, 0, 2)
        # 图纸命名（替换自动生成的名字，并作为导出默认文件名）
        grid.addWidget(QLabel("图纸命名:"), 1, 0)
        self.chart_name_edit = QLineEdit()
        self.chart_name_edit.setPlaceholderText("留空则用图片文件名")
        grid.addWidget(self.chart_name_edit, 1, 1)
        form.addWidget(gb_basic)

        # === 超采样 ===
        gb_ss = QGroupBox("超采样 (SuperSample)")
        self.gb_ss = gb_ss
        v = QVBoxLayout(gb_ss)
        h = QHBoxLayout()
        h.addWidget(QLabel("倍率:"))
        self.ss_spin = QSpinBox()
        self.ss_spin.setRange(1, 8)
        self.ss_spin.setValue(4)
        h.addWidget(self.ss_spin)
        h.addWidget(QLabel("  策略:"))
        self.ss_combo = QComboBox()
        self.ss_combo.addItems(["vote", "darkest", "line_aware"])
        h.addWidget(self.ss_combo)
        h.addStretch(1)
        v.addLayout(h)
        # SS 高级
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("darkest min_dark:"))
        self.ss_dark_spin = QSpinBox()
        self.ss_dark_spin.setRange(1, 8)
        self.ss_dark_spin.setValue(2)
        h2.addWidget(self.ss_dark_spin)
        h2.addWidget(QLabel("  line_aware min_px:"))
        self.ss_line_spin = QSpinBox()
        self.ss_line_spin.setRange(1, 16)
        self.ss_line_spin.setValue(3)
        h2.addWidget(self.ss_line_spin)
        h2.addStretch(1)
        v.addLayout(h2)
        form.addWidget(gb_ss)

        # === 量化细节 ===
        gb_detail = QGroupBox("量化细节")
        v = QVBoxLayout(gb_detail)
        self.chk_detail = QCheckBox("Detail mode (次色覆盖 + 关键特征)")
        self.chk_detail.setChecked(True)
        v.addWidget(self.chk_detail)
        h = QHBoxLayout()
        h.addWidget(QLabel("Min island size:"))
        self.island_spin = QSpinBox()
        self.island_spin.setRange(0, 10)
        self.island_spin.setValue(2)
        h.addWidget(self.island_spin)
        h.addWidget(QLabel("  Cleanup ΔE:"))
        self.deltae_spin = QDoubleSpinBox()
        self.deltae_spin.setRange(0, 60)
        self.deltae_spin.setValue(20.0)
        h.addWidget(self.deltae_spin)
        h.addStretch(1)
        v.addLayout(h)
        form.addWidget(gb_detail)

        # === 像素化方法 ===
        gb_px = QGroupBox("像素化方法 (Pixelize)")
        v = QVBoxLayout(gb_px)
        h = QHBoxLayout()
        h.addWidget(QLabel("方法:"))
        self.px_combo = QComboBox()
        self.px_combo.addItems(["SuperSample", "area", "dpid", "pixeloe"])
        self.px_combo.setCurrentText("pixeloe")
        h.addWidget(self.px_combo)
        h.addStretch(1)
        v.addLayout(h)
        # DPID 参数（仅 dpid 显示）
        self.px_dpid_row = QWidget()
        h2 = QHBoxLayout(self.px_dpid_row)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(QLabel("DPID λ:"))
        self.px_dpid_lam = QDoubleSpinBox()
        self.px_dpid_lam.setRange(0.0, 4.0)
        self.px_dpid_lam.setSingleStep(0.1)
        self.px_dpid_lam.setValue(1.0)
        h2.addWidget(self.px_dpid_lam)
        h2.addStretch(1)
        v.addWidget(self.px_dpid_row)
        # PixelOE 参数（仅 pixeloe 显示）
        self.px_poe_row = QWidget()
        pv = QVBoxLayout(self.px_poe_row)
        pv.setContentsMargins(0, 0, 0, 0)
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("POE mode:"))
        self.px_poe_mode = QComboBox()
        self.px_poe_mode.addItems(["center", "contrast", "k-centroid", "bicubic", "nearest"])
        self.px_poe_mode.setCurrentText("nearest")
        h3.addWidget(self.px_poe_mode)
        h3.addWidget(QLabel(" thickness:"))
        self.px_poe_thick = QSpinBox()
        self.px_poe_thick.setRange(0, 6)
        self.px_poe_thick.setValue(3)
        h3.addWidget(self.px_poe_thick)
        h3.addStretch(1)
        pv.addLayout(h3)
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("contrast:"))
        self.px_poe_contrast = QDoubleSpinBox()
        self.px_poe_contrast.setRange(0.1, 4.0)
        self.px_poe_contrast.setSingleStep(0.1)
        self.px_poe_contrast.setValue(1.0)
        h4.addWidget(self.px_poe_contrast)
        h4.addWidget(QLabel("  saturation:"))
        self.px_poe_sat = QDoubleSpinBox()
        self.px_poe_sat.setRange(0.0, 3.0)
        self.px_poe_sat.setSingleStep(0.1)
        self.px_poe_sat.setValue(1.0)
        h4.addWidget(self.px_poe_sat)
        h4.addStretch(1)
        pv.addLayout(h4)
        v.addWidget(self.px_poe_row)
        form.addWidget(gb_px)

        # === 线条检测 ===
        gb_line = QGroupBox("线条检测 (Line Detect)")
        v = QVBoxLayout(gb_line)
        self.chk_line = QCheckBox("启用线条检测 (独立于深浅, 救胡须/细线)")
        self.chk_line.setChecked(False)
        v.addWidget(self.chk_line)
        h = QHBoxLayout()
        h.addWidget(QLabel("方法:"))
        self.line_method_combo = QComboBox()
        self.line_method_combo.addItems(["canny", "multiscale", "local", "hybrid", "frangi"])
        h.addWidget(self.line_method_combo)
        h.addWidget(QLabel("  阈值:"))
        self.line_thr_spin = QDoubleSpinBox()
        self.line_thr_spin.setRange(50, 99)
        self.line_thr_spin.setValue(88.0)
        h.addWidget(self.line_thr_spin)
        h.addStretch(1)
        v.addLayout(h)
        # 线条细节
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("min_pct:"))
        self.line_pct_spin = QDoubleSpinBox()
        self.line_pct_spin.setRange(0, 0.5)
        self.line_pct_spin.setSingleStep(0.01)
        self.line_pct_spin.setValue(0.05)
        h2.addWidget(self.line_pct_spin)
        h2.addWidget(QLabel("  min_count:"))
        self.line_count_spin = QSpinBox()
        self.line_count_spin.setRange(0, 20)
        self.line_count_spin.setValue(1)
        h2.addWidget(self.line_count_spin)
        h2.addStretch(1)
        v.addLayout(h2)
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("contrast:"))
        self.line_contrast_spin = QDoubleSpinBox()
        self.line_contrast_spin.setRange(0, 80)
        self.line_contrast_spin.setValue(30.0)
        h3.addWidget(self.line_contrast_spin)
        h3.addWidget(QLabel("  aspect_min:"))
        self.line_aspect_spin = QDoubleSpinBox()
        self.line_aspect_spin.setRange(1.0, 10.0)
        self.line_aspect_spin.setSingleStep(0.5)
        self.line_aspect_spin.setValue(2.5)
        h3.addWidget(self.line_aspect_spin)
        h3.addStretch(1)
        v.addLayout(h3)
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("min_region_size:"))
        self.line_region_spin = QSpinBox()
        self.line_region_spin.setRange(0, 50)
        self.line_region_spin.setValue(4)
        h4.addWidget(self.line_region_spin)
        h4.addStretch(1)
        v.addLayout(h4)
        form.addWidget(gb_line)

        # === 输出 ===
        gb_out = QGroupBox("输出")
        v = QVBoxLayout(gb_out)
        h = QHBoxLayout()
        h.addWidget(QLabel("目录:"))
        self.out_edit = QtWidgets.QLineEdit(str(ROOT / "outputs"))
        h.addWidget(self.out_edit)
        btn_browse_out = QPushButton("浏览...")
        btn_browse_out.clicked.connect(self._on_browse_out)
        h.addWidget(btn_browse_out)
        v.addLayout(h)
        form.addWidget(gb_out)

        form.addStretch(1)

        # 生成按钮（固定底部，主操作用主色填充）
        self.btn_gen = QPushButton("生成拼豆图纸")
        self.btn_gen.setMinimumHeight(40)
        self.btn_gen.setObjectName("primaryBtn")
        self.btn_gen.setStyleSheet(
            "#primaryBtn {"
            " background: #3b6ea5; color: #ffffff; border: none;"
            " border-radius: 6px; font-size: 14px; font-weight: 600;"
            " padding: 10px 16px;"
            "}"
            "#primaryBtn:hover { background: #4c7fb5; }"
            "#primaryBtn:pressed { background: #335f8f; }"
            "#primaryBtn:disabled { background: #b9c7d6; color: #ffffff; }"
        )
        form.addWidget(self.btn_gen)

        scroll.setWidget(inner)
        return scroll

    def _build_preview_panel(self) -> QWidget:
        """右侧预览面板：两个并排工作区（源图区 / 结果区）+ 可缩放滚动的统计区。

        布局（MATLAB 风格，可拖拽分隔条调整比例）：
            ┌───────────────────────────────────────────────┐
            │ ┌─源图区─┐ 拖拽  ┌─结果区─┐                    │
            │ │原图/卡通│<─────>│网格/图纸│                    │
            │ │/缓冲区 │ 分隔条 │        │                    │
            │ └────────┘       └────────┘                    │
            ├──────────────────── 拖拽分隔条 ────────────────┤
            │ 统计（可拉小，缩小后滚动浏览）                  │
            └───────────────────────────────────────────────┘
        """
        outer = QWidget()
        v = QVBoxLayout(outer)
        v.setContentsMargins(0, 0, 0, 0)

        # 外层垂直分隔器：图像区 / 统计区
        outer_splitter = QSplitter(Qt.Vertical)
        v.addWidget(outer_splitter, 1)
        self.preview_splitter = outer_splitter

        # 内层水平分隔器：源图区 | 结果区（并排）
        img_splitter = QSplitter(Qt.Horizontal)
        outer_splitter.addWidget(img_splitter)
        self.img_splitter = img_splitter

        splitter = img_splitter

        # ============ 工作区 1：源图区（原图 / 卡通图 / 转化缓冲区） ============
        src_tabs = QTabWidget()
        self.src_tabs = src_tabs

        # 源图区 - Tab: 原图
        tab_orig = QWidget()
        v1 = QVBoxLayout(tab_orig)
        v1.setContentsMargins(0, 0, 0, 0)
        self.lbl_orig = SelectableImageLabel()
        self.lbl_orig.setText("（先打开图片）")
        self.lbl_orig.setStyleSheet("background: #fbfcfe; border: 1px solid #e3e7ec; border-radius: 6px;")
        self.lbl_orig.setMinimumHeight(220)
        self.lbl_orig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_orig.setToolTip("拖拽鼠标框选区域，可只对选中区域像素化")
        self.lbl_orig.region_changed.connect(self._on_region_changed)
        v1.addWidget(self.lbl_orig, 1)
        # 框选提示 + 清除按钮
        h_orig = QHBoxLayout()
        self.lbl_region_tip = QLabel("拖拽框选区域，可只对选中区域像素化（留空=整图）")
        self.lbl_region_tip.setStyleSheet("color: #888; font-size: 12px;")
        h_orig.addWidget(self.lbl_region_tip)
        h_orig.addStretch(1)
        self.btn_clear_region = QPushButton("清除选区")
        self.btn_clear_region.setEnabled(False)
        self.btn_clear_region.clicked.connect(self._on_clear_region)
        h_orig.addWidget(self.btn_clear_region)
        v1.addLayout(h_orig)
        self.src_orig_idx = src_tabs.addTab(tab_orig, "原图")

        # 源图区 - Tab: 卡通图（仅卡通化后启用）
        tab_cartoon = QWidget()
        vc = QVBoxLayout(tab_cartoon)
        vc.setContentsMargins(0, 0, 0, 0)
        self.lbl_cartoon = SelectableImageLabel()
        self.lbl_cartoon.setText("（卡通化后显示）")
        self.lbl_cartoon.setStyleSheet("background: #fbfcfe; border: 1px solid #e3e7ec; border-radius: 6px;")
        self.lbl_cartoon.setMinimumHeight(220)
        self.lbl_cartoon.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_cartoon.setToolTip("拖拽鼠标框选区域，可只对选中区域像素化")
        self.lbl_cartoon.region_changed.connect(self._on_region_changed)
        vc.addWidget(self.lbl_cartoon, 1)
        self.tab_cartoon_idx = src_tabs.addTab(tab_cartoon, "卡通图")
        src_tabs.setTabEnabled(self.tab_cartoon_idx, False)

        # 源图区 - Tab: 转化缓冲区（框选后预览将被像素化的区域）
        tab_buffer = QWidget()
        vb = QVBoxLayout(tab_buffer)
        vb.setContentsMargins(0, 0, 0, 0)
        self.lbl_buffer = ZoomableImageLabel()
        self.lbl_buffer.setText("（框选区域后，这里显示将被像素化的裁剪图）")
        self.lbl_buffer.setStyleSheet("background: #f7f9fb; border: 1px solid #e3e7ec; border-radius: 6px;")
        self.lbl_buffer.setMinimumHeight(220)
        self.lbl_buffer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_buffer.setToolTip("转化缓冲区：框选后的裁剪预览，作为将被像素化的源图\nCtrl+滚轮缩放 · 拖拽平移")
        vb.addWidget(self.lbl_buffer, 1)
        self.lbl_buffer_tip = QLabel("未框选时等价于整图；框选后显示裁剪结果")
        self.lbl_buffer_tip.setStyleSheet("color: #888; font-size: 12px;")
        vb.addWidget(self.lbl_buffer_tip)
        self.tab_buffer_idx = src_tabs.addTab(tab_buffer, "转化缓冲区")

        splitter.addWidget(src_tabs)

        # ============ 工作区 2：结果区（网格预览 / 拼豆图纸） ============
        res_tabs = QTabWidget()
        self.res_tabs = res_tabs

        # 结果区 - Tab: 网格预览（放前面，方便对照修改）
        tab_grid = QWidget()
        v3 = QVBoxLayout(tab_grid)
        v3.setContentsMargins(0, 0, 0, 0)
        self.lbl_grid = GridEditWidget()
        self.lbl_grid.setText("（生成后显示，可在此编辑）")
        self.lbl_grid.setStyleSheet("background: #fbfcfe; border: 1px solid #e3e7ec; border-radius: 6px;")
        self.lbl_grid.setMinimumHeight(220)
        self.lbl_grid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_grid.setToolTip(
            "画笔：点击/拖动涂色\n框选：拖动选格子\n平移/Ctrl+滚轮缩放\n\n"
            "在左侧「编辑」页选择画笔色和工具"
        )
        v3.addWidget(self.lbl_grid, 1)
        # 编辑提示条
        self.lbl_grid_tip = QLabel("生成后可编辑 · 左侧「编辑」页选工具和颜色")
        self.lbl_grid_tip.setStyleSheet("color: #888; font-size: 12px;")
        v3.addWidget(self.lbl_grid_tip)
        self.res_grid_idx = res_tabs.addTab(tab_grid, "网格预览")

        # 结果区 - Tab: 拼豆图纸
        tab_chart = QWidget()
        v2 = QVBoxLayout(tab_chart)
        v2.setContentsMargins(0, 0, 0, 0)
        self.lbl_chart = ZoomableImageLabel()
        self.lbl_chart.setText("（生成后显示）")
        self.lbl_chart.setStyleSheet("background: #fbfcfe; border: 1px solid #e3e7ec; border-radius: 6px;")
        self.lbl_chart.setMinimumHeight(220)
        self.lbl_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_chart.setToolTip("Ctrl+滚轮缩放 · 拖拽平移 · 双击复位")
        v2.addWidget(self.lbl_chart, 1)
        self.res_chart_idx = res_tabs.addTab(tab_chart, "拼豆图纸")

        # 结果区 - Tab: 像素化预览（无网格线，模拟烫完后样子）
        tab_plain = QWidget()
        vp = QVBoxLayout(tab_plain)
        vp.setContentsMargins(0, 0, 0, 0)
        self.lbl_plain = ZoomableImageLabel()
        self.lbl_plain.setText("（生成后显示）")
        self.lbl_plain.setStyleSheet("background: #fbfcfe; border: 1px solid #e3e7ec; border-radius: 6px;")
        self.lbl_plain.setMinimumHeight(220)
        self.lbl_plain.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_plain.setToolTip(
            "无网格线的纯像素化预览，模拟拼豆烫好后融化的样子\n"
            "Ctrl+滚轮缩放 · 拖拽平移 · 双击复位"
        )
        vp.addWidget(self.lbl_plain, 1)
        self.lbl_plain_tip = QLabel("预览拼豆烫完后（无网格线）的样子")
        self.lbl_plain_tip.setStyleSheet("color: #888; font-size: 12px;")
        vp.addWidget(self.lbl_plain_tip)
        self.res_plain_idx = res_tabs.addTab(tab_plain, "像素化预览")

        splitter.addWidget(res_tabs)

        # ============ 统计区（可拉小，缩小后滚动浏览） ============
        self.stats_box = QGroupBox("统计")
        v_stats = QVBoxLayout(self.stats_box)
        v_stats.setContentsMargins(4, 4, 4, 4)
        self.lbl_stats = QLabel("— 生成后显示统计 —")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setStyleSheet("font-family: Consolas, monospace; color: #888;")
        # 统计内容放进滚动区，拉小后可滚动浏览全部数据
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setFrameShape(QFrame.NoFrame)
        stats_scroll.setWidget(self.lbl_stats)
        v_stats.addWidget(stats_scroll)
        outer_splitter.addWidget(self.stats_box)

        # 初始比例
        img_splitter.setStretchFactor(0, 5)
        img_splitter.setStretchFactor(1, 5)
        outer_splitter.setStretchFactor(0, 5)
        outer_splitter.setStretchFactor(1, 2)
        # 源图/结果区不可折叠，统计区可折叠
        img_splitter.setCollapsible(0, False)
        img_splitter.setCollapsible(1, False)
        outer_splitter.setCollapsible(0, False)
        outer_splitter.setCollapsible(1, True)
        # 统计区最小高度，允许拉到很小
        self.stats_box.setMinimumHeight(60)

        # 兼容旧代码引用：self.tabs 指向源图区 TabWidget（卡通图 Tab 索引在此）
        self.tabs = src_tabs

        return outer

    # -------- 信号/事件 --------

    def _connect_signals(self):
        self.act_open.triggered.connect(self._on_open)
        self.act_gen.triggered.connect(self._on_generate)
        self.act_save.triggered.connect(self._on_save)
        self.act_import.triggered.connect(self._on_import)
        self.act_settings.triggered.connect(self._on_settings)
        self.act_about.triggered.connect(self._on_about)
        self.btn_gen.clicked.connect(self._on_generate)
        self.chk_line.toggled.connect(self._update_param_state)
        self.px_combo.currentIndexChanged.connect(self._update_px_param_state)
        self._update_px_param_state()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # 左侧参数/编辑 Tab 切换：参数区时网格只读，编辑区时网格可编辑
        self.left_tabs.currentChanged.connect(self._on_left_tab_changed)

        # ==== 手动编辑相关信号 ====
        # 工具切换
        self.btn_tool_paint.clicked.connect(lambda: self._on_tool_changed("paint"))
        self.btn_tool_erase.clicked.connect(lambda: self._on_tool_changed("erase"))
        self.btn_tool_pick.clicked.connect(lambda: self._on_tool_changed("pick"))
        self.btn_tool_select.clicked.connect(lambda: self._on_tool_changed("select"))
        self.btn_tool_pan.clicked.connect(lambda: self._on_tool_changed("pan"))
        # 撤销 / 重做
        self.btn_undo.clicked.connect(self._on_undo)
        self.btn_redo.clicked.connect(self._on_redo)
        # 画笔色过滤
        self.chk_brush_used_only.toggled.connect(lambda _: self._rebuild_palette_swatch())
        # 选中
        self.btn_clear_sel.clicked.connect(self._on_clear_selection)
        self.btn_invert_sel.clicked.connect(self._on_invert_selection)
        self.btn_pick_on_canvas.toggled.connect(self._on_pick_on_canvas_toggled)
        # 批量改色
        self.btn_apply_replace.clicked.connect(self._on_apply_replace)
        # 颜色调整滑条（实时）
        self.slider_darken.valueChanged.connect(lambda v: self._on_adjust_slider("darken", v))
        self.slider_lighten.valueChanged.connect(lambda v: self._on_adjust_slider("lighten", v))
        self.slider_contrast.valueChanged.connect(lambda v: self._on_adjust_slider("contrast", v))
        self.slider_darken.sliderReleased.connect(self._on_adjust_slider_released)
        self.slider_lighten.sliderReleased.connect(self._on_adjust_slider_released)
        self.slider_contrast.sliderReleased.connect(self._on_adjust_slider_released)
        self.btn_reset_adjust.clicked.connect(self._on_reset_adjust)
        # 网格编辑回传
        self.lbl_grid.edit_started.connect(self._on_edit_started)
        self.lbl_grid.grid_changed.connect(self._on_grid_edited)
        self.lbl_grid.selection_changed.connect(self._on_selection_changed)
        self.lbl_grid.brush_picked.connect(self._on_brush_picked)

    def _on_tool_changed(self, tool: str):
        """切换网格编辑工具。"""
        self.lbl_grid.set_tool(tool)
        # 切到非 pick 工具时，退出「图上点选同色」模式
        if tool != "pick" and getattr(self, "_pick_on_canvas_active", False):
            self._pick_on_canvas_active = False
            self.btn_pick_on_canvas.setChecked(False)

    def _on_pick_on_canvas_toggled(self, checked: bool):
        """「图上点选同色」模式开关。"""
        if checked:
            self._pick_on_canvas_active = True
            self.lbl_grid.set_tool("pick")
            self.status.showMessage("点选模式：点击网格中某个格子，选中所有同色格子", 8000)
        else:
            self._pick_on_canvas_active = False

    def _on_selection_changed(self):
        """网格选中格子变化时更新计数显示。"""
        n = len(self.lbl_grid.selected_cells())
        self.lbl_sel_info.setText(f"已选中 {n} 格")

    def _on_clear_selection(self):
        self.lbl_grid.clear_selection()
        self.lbl_sel_info.setText("已选中 0 格")

    def _on_invert_selection(self):
        self.lbl_grid.invert_selection()
        self.lbl_sel_info.setText(f"已选中 {len(self.lbl_grid.selected_cells())} 格")

    def _on_select_all(self):
        """Ctrl+A 全选网格所有格子。"""
        if self.result is None:
            return
        # 仅当网格可编辑（在「编辑」页）时全选才生效
        if not self.lbl_grid.is_editable():
            return
        self.lbl_grid.select_all()
        self.lbl_sel_info.setText(f"已选中 {len(self.lbl_grid.selected_cells())} 格")
        self.status.showMessage("⬛ 已全选所有格子", 3000)

    def _on_brush_picked(self, code: str):
        """取色器/图上点选取到颜色。

        若处于「图上点选同色」模式，则切换（叠加/取消）同色格子的选中；
        否则设为画笔色（取色器）。
        """
        if getattr(self, "_pick_on_canvas_active", False):
            # 图上点选同色：若该色格子已全部选中则取消，否则叠加选中
            if self.lbl_grid.cells_for_codes_fully_selected({code}):
                self.lbl_grid.toggle_cells_by_codes({code}, add=False)
                self.status.showMessage(f"➖ 取消选中所有 {code} 色格子", 5000)
            else:
                self.lbl_grid.toggle_cells_by_codes({code}, add=True)
                self.status.showMessage(f"➕ 叠加选中所有 {code} 色格子", 5000)
            self.lbl_sel_info.setText(
                f"已选中 {len(self.lbl_grid.selected_cells())} 格"
            )
            # 保持点选模式，允许连续点选多种颜色
            return
        # 取色器：设为画笔色
        self._on_brush_pick(code)
        self.status.showMessage(f"取色: {code}", 3000)

    def _update_param_state(self):
        # 没开 line detect 时，line 相关控件灰显
        en = self.chk_line.isChecked()
        for w in [self.line_method_combo, self.line_thr_spin,
                  self.line_pct_spin, self.line_count_spin,
                  self.line_contrast_spin, self.line_aspect_spin, self.line_region_spin]:
            w.setEnabled(en)
        # 依赖缺失时禁用对应方法
        if not HAS_CV2:
            self.line_method_combo.model().item(0).setEnabled(False)  # canny
            self.line_method_combo.model().item(3).setEnabled(False)  # hybrid
            if self.line_method_combo.currentText() in ("canny", "hybrid"):
                self.line_method_combo.setCurrentText("multiscale")

    def _update_px_param_state(self):
        """像素化方法切换：只显示当前方法对应的参数，并控制超采样可用性。

        - SuperSample = 原生超采样 + 投票（唯一启用超采样参数）
        - area / dpid / pixeloe = 像素化方法，隐藏超采样参数
        """
        m = self.px_combo.currentText()
        is_ss = (m == "SuperSample")
        is_dpid = (m == "dpid")
        is_poe = (m == "pixeloe")
        # 参数行显示/隐藏
        self.px_dpid_row.setVisible(is_dpid)
        self.px_poe_row.setVisible(is_poe)
        # 超采样组：仅 SuperSample 可用
        self.gb_ss.setEnabled(is_ss)

    def _check_deps(self):
        """启动时检查依赖，提示用户。"""
        msgs = []
        if not HAS_CV2:
            msgs.append(
                "• opencv-python 未安装\n"
                "  线条检测的 'canny' 和 'hybrid' 方法不可用\n"
                "  安装: pip install opencv-python"
            )
        if msgs:
            QMessageBox.warning(
                self, "可选依赖缺失",
                "以下可选依赖未安装，对应功能不可用：\n\n" + "\n\n".join(msgs) +
                "\n\n基础量化 (vote + cleanup) 不受影响，可正常使用。"
            )
            self.status.showMessage("⚠ 部分可选依赖缺失", 0)
        else:
            self.status.showMessage(f"✓ 依赖完整 (opencv {CV2_VERSION})", 5000)

    # -------- 手动编辑：色板 / 选中 / 批量操作 --------

    def _rebuild_palette_swatch(self):
        """重建画笔色板（色块 + 色号代码标签，紧凑多列布局）。"""
        while self.palette_layout.count():
            item = self.palette_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        used_only = self.chk_brush_used_only.isChecked()
        if used_only and self.result is not None:
            used_codes = set(self.result.stats.keys())
            colors = [c for c in self.palette.colors if c.code in used_codes]
        else:
            colors = self.palette.colors

        cols = 8
        for i, c in enumerate(colors):
            # 每个色块：一个容器（色块按钮 + 下方色号）
            cell = QWidget()
            cell_v = QVBoxLayout(cell)
            cell_v.setContentsMargins(0, 0, 0, 0)
            cell_v.setSpacing(0)
            # 色块按钮：纯色 QPixmap 作为 icon（居中、四周留边），黑色弯角边框完整可见
            btn = SwatchButton(c.code)
            btn.setFixedSize(32, 20)
            btn.setIconSize(QtCore.QSize(26, 16))
            pm = QtGui.QPixmap(26, 16)
            pm.fill(QtGui.QColor(*c.rgb))
            btn.setIcon(QtGui.QIcon(pm))
            btn.setStyleSheet(
                "QPushButton { border: 1px solid #1a1a1a; border-radius: 4px; padding: 0px; }"
                "QPushButton:hover { border: 2px solid #1e78ff; }"
            )
            btn.setToolTip(f"{c.code}  {c.name}  #{c.hex}")
            btn.clicked.connect(lambda checked, cc=c.code: self._on_brush_pick(cc))
            btn.hovered.connect(self._on_swatch_hover)
            btn.left.connect(self._on_swatch_leave)
            cell_v.addWidget(btn, 0, Qt.AlignHCenter)
            # 色号标签
            lbl = QLabel(c.code)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 8px; color: #555;")
            cell_v.addWidget(lbl)
            self.palette_layout.addWidget(cell, i // cols, i % cols)
        # 撑满宽度，让色块均匀分布
        for col in range(cols):
            self.palette_layout.setColumnStretch(col, 1)

    def _on_swatch_hover(self, code: str):
        """色板色块悬浮：若有选中格子，临时预览该色。"""
        cells = self.lbl_grid.selected_cells()
        if cells:
            self.lbl_grid.set_preview(cells, code)

    def _on_swatch_leave(self):
        """色板色块移开：清除预览。"""
        self.lbl_grid.clear_preview()

    def _on_brush_pick(self, code: str):
        """点击色块：若有选中格子则确认批量改色，否则设为画笔色。"""
        c = self.palette.by_code(code)
        cells = self.lbl_grid.selected_cells()
        if cells and c is not None:
            # 有选中格子 → 确认批量改色（悬浮只是试色，点击确认）
            self.lbl_grid.clear_preview()
            self._push_undo()
            n = paint_cells(self.result, cells, c)
            self._after_edit(f"已将 {n} 格改为 {c.code} {c.name}")
            return
        # 无选中 → 设为画笔色
        self.lbl_grid.set_brush_code(code)
        if c:
            self.lbl_brush_cur.setText(f"当前画笔：{c.code} {c.name}")
            self.lbl_brush_cur.setStyleSheet("font-weight: bold; color: #000;")
            # 色块用 QPixmap 填充，确保颜色一定显示（不受 QLabel 样式影响）
            pm = QtGui.QPixmap(26, 26)
            pm.fill(QtGui.QColor(*c.rgb))
            self.lbl_brush_swatch.setPixmap(pm)
            self.lbl_brush_swatch.setStyleSheet(
                "border: 2px solid #444; border-radius: 3px;"
            )

    def _update_replace_target(self):
        """更新「改为」按钮的显示（当前目标色块 + 色号，不显示 Mard_ 名称）。"""
        code = self._replace_target_code
        if code == "":
            self.btn_replace_target.setText("空白（删除格子）")
            self.btn_replace_target.setIcon(QtGui.QIcon())
            return
        c = self.palette.by_code(code)
        if c is None:
            self.btn_replace_target.setText(code)
            self.btn_replace_target.setIcon(QtGui.QIcon())
            return
        pm = QtGui.QPixmap(16, 16)
        pm.fill(QtGui.QColor(*c.rgb))
        self.btn_replace_target.setIcon(QtGui.QIcon(pm))
        self.btn_replace_target.setText(code)

    def _show_replace_bubble(self):
        """点击「改为」按钮弹出非模态气泡：选择要改成的颜色（色块 + 色号，不含名称）。

        首项为「空白（删除格子）」，其余为全部色板色。
        """
        if getattr(self, "_replace_bubble", None) is not None:
            self._replace_bubble.close()
            self._replace_bubble = None

        bubble = QtWidgets.QDialog(self, Qt.Popup)
        bubble.setObjectName("colorBubble")
        self._replace_bubble = bubble
        layout = QVBoxLayout(bubble)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)

        title = QLabel("改为（点击选择，点击外部关闭）")
        title.setStyleSheet("font-weight: bold; color: #333; font-size: 12px;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(360)
        scroll.setMinimumWidth(200)
        inner = QWidget()
        inner_v = QVBoxLayout(inner)
        inner_v.setContentsMargins(0, 0, 0, 0)
        inner_v.setSpacing(2)

        def _make_item(code: str, label: str, rgb=None):
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setIconSize(QtCore.QSize(18, 18))
            if rgb is not None:
                pm = QtGui.QPixmap(18, 18)
                pm.fill(QtGui.QColor(*rgb))
                btn.setIcon(QtGui.QIcon(pm))
            btn.setStyleSheet(
                "QPushButton { border: 1px solid #bbb; border-radius: 3px; "
                "padding: 4px 6px; text-align: left; }"
                "QPushButton:hover { border: 2px solid #1e78ff; }"
                "QPushButton:checked { border: 2px solid #1e78ff; "
                "background-color: #dcecff; font-weight: bold; }"
            )
            btn.setText(label)
            btn.clicked.connect(lambda checked, cd=code: self._on_replace_color_click(cd))
            inner_v.addWidget(btn)
            return btn

        # 首项：空白（删除格子）
        blank_btn = _make_item("", "空白（删除格子）")
        blank_btn.setChecked(self._replace_target_code == "")
        # 全部色板色（色块 + 色号，不含 Mard_ 名称）
        for c in self.palette.colors:
            b = _make_item(c.code, c.code, c.rgb)
            b.setChecked(self._replace_target_code == c.code)
        inner_v.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        bubble.adjustSize()
        global_pos = self.btn_replace_target.mapToGlobal(
            QtCore.QPoint(0, self.btn_replace_target.height())
        )
        bubble.move(global_pos)
        bubble.show()

    def _on_replace_color_click(self, code: str):
        """气泡中点击目标色：更新「改为」按钮并关闭气泡。"""
        self._replace_target_code = code
        self._update_replace_target()
        if getattr(self, "_replace_bubble", None) is not None:
            self._replace_bubble.close()
            self._replace_bubble = None

    def _show_color_bubble(self):
        """点击「按颜色选中」弹出非模态气泡：按数量排序的色号（色块+代码）。

        气泡特点：
          - 非模态：选中一个颜色后气泡保留，可继续选其它颜色
          - 点击颜色：若该色格子已全部选中则取消，否则叠加选中
          - 已选中的颜色条目高亮（勾选标记 + 加粗边框）
          - 点击气泡外区域自动关闭（Qt.Popup 标志）
        """
        if self.result is None:
            QMessageBox.information(self, "尚未生成", "请先在「参数」页生成拼豆图纸")
            return
        # 若气泡已存在，先关闭旧的
        if getattr(self, "_color_bubble", None) is not None:
            self._color_bubble.close()
            self._color_bubble = None

        # 按用量降序的色号
        sorted_codes = sorted(self.result.stats.items(), key=lambda kv: -kv[1])

        # 非模态气泡（Qt.Popup：点击外部自动关闭）
        bubble = QtWidgets.QDialog(self, Qt.Popup)
        bubble.setObjectName("colorBubble")
        self._color_bubble = bubble
        self._bubble_buttons = {}  # code -> QPushButton
        layout = QVBoxLayout(bubble)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)

        title = QLabel("按颜色选中（可多选，点击外部关闭）")
        title.setStyleSheet("font-weight: bold; color: #333; font-size: 12px;")
        layout.addWidget(title)

        # 可滚动的颜色列表
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(360)
        scroll.setMinimumWidth(200)
        inner = QWidget()
        inner_v = QVBoxLayout(inner)
        inner_v.setContentsMargins(0, 0, 0, 0)
        inner_v.setSpacing(2)

        for code, cnt in sorted_codes:
            c = self.palette.by_code(code)
            if c is None:
                continue
            # 颜色项按钮：纯色 QPixmap icon（颜色必显示）+ 色号/名称/数量
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setIconSize(QtCore.QSize(18, 18))
            pm = QtGui.QPixmap(18, 18)
            pm.fill(QtGui.QColor(*c.rgb))
            btn.setIcon(QtGui.QIcon(pm))
            btn.setStyleSheet(
                "QPushButton { border: 1px solid #bbb; border-radius: 3px; "
                "padding: 4px 6px; text-align: left; }"
                "QPushButton:hover { border: 2px solid #1e78ff; }"
                "QPushButton:checked { border: 2px solid #1e78ff; "
                "background-color: #dcecff; font-weight: bold; }"
            )
            btn.setText(f"{code}  ({cnt})")
            btn.clicked.connect(lambda checked, cd=code: self._on_bubble_color_click(cd))
            # 根据当前选中状态初始化高亮
            if self.lbl_grid.cells_for_codes_fully_selected({code}):
                btn.setChecked(True)
            self._bubble_buttons[code] = btn
            inner_v.addWidget(btn)
        inner_v.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        # 定位在按钮旁
        bubble.adjustSize()
        global_pos = self.btn_select_by_color.mapToGlobal(
            QtCore.QPoint(0, self.btn_select_by_color.height())
        )
        bubble.move(global_pos)
        bubble.show()

    def _on_bubble_color_click(self, code: str):
        """气泡中点击某个颜色：切换（叠加/取消）该色格子的选中，气泡保留 + 高亮更新。"""
        if self.lbl_grid.cells_for_codes_fully_selected({code}):
            self.lbl_grid.toggle_cells_by_codes({code}, add=False)
            selected_now = False
        else:
            self.lbl_grid.toggle_cells_by_codes({code}, add=True)
            selected_now = True
        # 更新气泡按钮高亮状态
        btn = getattr(self, "_bubble_buttons", {}).get(code)
        if btn is not None:
            btn.setChecked(selected_now)
        self.lbl_sel_info.setText(
            f"已选中 {len(self.lbl_grid.selected_cells())} 格"
        )

    def _on_apply_replace(self):
        """把选中格子批量改为目标色（或删除为空白）。"""
        cells = self.lbl_grid.selected_cells()
        if not cells:
            QMessageBox.information(self, "无选中", "请先框选或按颜色选中格子")
            return
        target_code = self._replace_target_code
        if target_code is None:
            QMessageBox.information(self, "未选目标色", "请先选择要改成的颜色")
            return
        self._push_undo()
        if target_code == "":
            n = erase_cells(self.result, cells)
            self._after_edit(f"已删除 {n} 格为空白")
        else:
            target = self.palette.by_code(target_code)
            if target is None:
                return
            n = paint_cells(self.result, cells, target)
            self._after_edit(f"已将 {n} 格改为 {target.code} {target.name}")

    def _on_adjust_slider(self, mode: str, value: int):
        """滑条实时调整选中格子的颜色。

        采用「基准快照 + 实时重放」策略：滑条首次变化时记录选中格子的原始
        颜色作为基准，之后每次滑条变化都从基准重新计算，避免叠加漂移。
        """
        cells = self.lbl_grid.selected_cells()
        if not cells:
            return
        # 更新数值显示
        if mode == "contrast":
            self.lbl_contrast_val.setText(f"{value / 100.0:.2f}")
        elif mode == "darken":
            self.lbl_darken_val.setText(str(value))
        elif mode == "lighten":
            self.lbl_lighten_val.setText(str(value))

        # 记录基准快照（首次变化时）
        if getattr(self, "_adjust_base_grid", None) is None:
            self._adjust_base_grid = snapshot_grid(self.result)
            self._adjust_base_cells = set(cells)
        # 从基准恢复，再应用当前所有滑条
        restore_grid(self.result, self._adjust_base_grid)
        # 应用三个滑条的叠加效果
        darken_v = self.slider_darken.value()
        lighten_v = self.slider_lighten.value()
        contrast_v = self.slider_contrast.value() / 100.0
        # 依次应用（darken 和 lighten 互斥视觉上，取净效果）
        net_light = lighten_v - darken_v
        if net_light != 0:
            adjust_cells(self.result, cells, mode=("lighten" if net_light > 0 else "darken"),
                         amount=abs(net_light), palette=self.palette)
        if abs(contrast_v - 1.0) > 1e-4:
            adjust_cells(self.result, cells, mode="contrast",
                         amount=contrast_v, palette=self.palette)
        # 实时重渲染（不记录撤销）
        self._after_edit(silent=True)

    def _on_adjust_slider_released(self):
        """滑条释放：把本次调整作为一次撤销点（压入调整前的基准快照）。"""
        if getattr(self, "_adjust_base_grid", None) is not None:
            # 撤销点应该是「调整前」的状态，即基准快照
            self._undo_stack.append(self._adjust_base_grid)
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
            self._adjust_base_grid = None
            self._adjust_base_cells = None
            self._update_undo_buttons()

    def _on_reset_adjust(self):
        """重置三个滑条：恢复调整前的颜色（基准快照），滑条归零。"""
        base = getattr(self, "_adjust_base_grid", None)
        if base is not None:
            # 恢复到调整前的状态
            restore_grid(self.result, base)
            self._adjust_base_grid = None
            self._adjust_base_cells = None
        # 归零滑条（blockSignals 避免触发实时调整）
        for sl in (self.slider_darken, self.slider_lighten, self.slider_contrast):
            sl.blockSignals(True)
        self.slider_darken.setValue(0)
        self.slider_lighten.setValue(0)
        self.slider_contrast.setValue(100)
        for sl in (self.slider_darken, self.slider_lighten, self.slider_contrast):
            sl.blockSignals(False)
        self.lbl_darken_val.setText("0")
        self.lbl_lighten_val.setText("0")
        self.lbl_contrast_val.setText("1.00")
        self._after_edit(silent=True)

    def _on_edit_started(self):
        """画笔/橡皮擦按下时：记录撤销快照（此刻 grid 尚未被改）。"""
        self._push_undo()

    def _on_grid_edited(self):
        """网格被画笔/橡皮擦涂改后，重渲染（撤销快照已在按下时记录）。"""
        self._after_edit("已编辑")

    def _after_edit(self, msg: str = "", silent: bool = False):
        """编辑后统一处理：重算统计 + 重渲染网格/图纸 + 保存。"""
        if self.result is None:
            return
        rebuild_result_stats(self.result)
        self._rerender_outputs()
        if not silent and msg:
            self.status.showMessage(f"{msg}", 5000)

    # -------- 撤销 / 重做 --------
    def _push_undo(self):
        """把当前网格快照压入撤销栈，清空重做栈。"""
        if self.result is None:
            return
        self._undo_stack.append(snapshot_grid(self.result))
        if len(self._undo_stack) > 50:  # 限制栈深度
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_buttons()

    def _update_undo_buttons(self):
        self.btn_undo.setEnabled(len(self._undo_stack) > 0)
        self.btn_redo.setEnabled(len(self._redo_stack) > 0)

    def _on_undo(self):
        """撤销：恢复到上一个快照。"""
        if not self._undo_stack or self.result is None:
            return
        # 当前状态压入重做栈
        self._redo_stack.append(snapshot_grid(self.result))
        prev = self._undo_stack.pop()
        restore_grid(self.result, prev)
        # 更新网格控件引用
        self.lbl_grid._grid = self.result.grid
        self.lbl_grid.update()
        self._after_edit(silent=True)
        self._update_undo_buttons()
        self.status.showMessage("已撤销", 3000)

    def _on_redo(self):
        """重做：恢复撤销前状态。"""
        if not self._redo_stack or self.result is None:
            return
        self._undo_stack.append(snapshot_grid(self.result))
        nxt = self._redo_stack.pop()
        restore_grid(self.result, nxt)
        self.lbl_grid._grid = self.result.grid
        self.lbl_grid.update()
        self._after_edit(silent=True)
        self._update_undo_buttons()
        self.status.showMessage("已重做", 3000)

    def _rerender_outputs(self):
        """根据当前（可能已编辑的）result 重渲染网格预览与图纸。"""
        if self.result is None:
            return
        grid = self.grid_slider.value()
        cell_px = max(8, min(24, 1800 // grid))
        # 网格预览直接由控件绘制（result 已更新，直接 update）
        self.lbl_grid.update()
        # 图纸重渲染（色号全标注 + 放大图例）
        stem = Path(self.image_path).stem if self.image_path else "bead"
        suffix = "_region" if self.region is not None else ""
        out_dir = Path(self.out_edit.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        chart_path = out_dir / f"{stem}{suffix}_bead_chart_{grid}x{grid}.png"
        grid_path = out_dir / f"{stem}{suffix}_bead_grid_{grid}x{grid}.png"
        plain_path = out_dir / f"{stem}{suffix}_bead_plain_{grid}x{grid}.png"
        json_path = out_dir / f"{stem}{suffix}_bead_grid_{grid}x{grid}.json"

        # 图纸：色号全标注
        save_bead_chart(
            self.result, str(chart_path),
            cell_px=cell_px,
            label_every=1,
            label_all_cells=True,
            title=f"拼豆图纸 - {stem} · {grid}×{grid}",
            palette_name="MARD",
        )
        # 网格预览 PNG（纯色块，供保存）
        grid_img = render_grid_only(self.result, cell_px=cell_px, with_grid=True)
        grid_img.save(str(grid_path))
        # 像素化预览（无网格线，模拟烫完）
        plain_img = render_plain_image(self.result, cell_px=cell_px)
        plain_img.save(str(plain_path))
        # JSON
        grid_json = {
            "grid_w": self.result.grid_w,
            "grid_h": self.result.grid_h,
            "palette": "MARD",
            "cells": [[cell.code for cell in row] for row in self.result.grid],
        }
        json_path.write_text(
            json.dumps(grid_json, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

        # 更新路径 + 图纸显示
        self.last_chart_path = str(chart_path)
        self.last_grid_path = str(grid_path)
        self.last_json_path = str(json_path)
        self.lbl_chart.set_image(str(chart_path))
        self.lbl_plain.set_image(str(plain_path))
        # 刷新统计
        self._refresh_stats()
        # 刷新色板 swatch（若只看已用色）
        if self.chk_brush_used_only.isChecked():
            self._rebuild_palette_swatch()

    def _refresh_stats(self):
        """刷新统计标签。"""
        if self.result is None:
            return
        stats = self.result.stats
        top12 = sorted(stats.items(), key=lambda x: -x[1])[:12]
        lines = [
            f"用色数: {self.result.color_count}",
            f"总格数: {self.result.total_cells}",
            "",
            "色号用量 TOP 12:",
        ]
        for code, cnt in top12:
            c = self.palette.by_code(code)
            hex_ = c.hex if c else "?"
            lines.append(f"  {code:<6} {cnt:>5} 格  hex={hex_}")
        self.lbl_stats.setText("\n".join(lines))
        self.lbl_stats.setStyleSheet("font-family: Consolas, monospace;")
        self.stats_box.setVisible(True)

    # -------- 拖拽支持 --------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".webp")
            ):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self._load_image(urls[0].toLocalFile())

    # -------- 业务逻辑 --------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(ROOT / "samples"),
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str, reset_features: bool = True):
        if not Path(path).exists():
            QMessageBox.warning(self, "文件不存在", f"找不到文件：\n{path}")
            return
        # 验证图片
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.critical(
                self, "图片格式错误",
                f"无法加载图片，请检查格式：\n{path}\n\n"
                "支持: PNG / JPG / JPEG / BMP / WebP"
            )
            return
        w, h = pix.width(), pix.height()
        if w < 32 or h < 32:
            QMessageBox.warning(
                self, "图片太小",
                f"图片尺寸 {w}×{h} 太小，建议至少 200×200 像素。"
            )
        if w > 4096 or h > 4096:
            QMessageBox.warning(
                self, "图片很大",
                f"图片尺寸 {w}×{h} 很大，量化会比较慢。\n"
                "建议先缩小到 2048×2048 以内。"
            )
        self.image_path = path
        # 换图后清空旧的关键特征（避免错配到新图）
        # 卡通化完成时传 reset_features=False，保留刚算出的关键特征
        if reset_features:
            self.key_features = None
            self.analysis = None
            self.lbl_cartoon_status.setText("（未运行）")
            self.lbl_cartoon_status.setStyleSheet("color: #888;")
        # 清空选区
        self.region = None
        self.lbl_orig.clear_region()
        self.btn_clear_region.setEnabled(False)
        self.lbl_image.setText(f"✓ {Path(path).name}  ({w}×{h})")
        self.lbl_image.setStyleSheet("color: #2a7; font-weight: bold;")
        # 显示原图（可框选）
        self.lbl_orig.set_image(path)
        # 刷新转化缓冲区
        self._update_buffer()
        self.act_gen.setEnabled(True)
        self.btn_gen.setEnabled(True)
        if HAS_PIPELINE:
            self.btn_cartoon.setEnabled(True)
        self.status.showMessage(f"已加载: {Path(path).name}  ·  点「卡通化」或「生成」")

    def _on_browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    # -------- 框选区域 --------

    def _on_region_changed(self, region):
        """任一可框选控件框选后更新选区 + 刷新转化缓冲区。"""
        self.region = region
        if region is not None:
            x1, y1, x2, y2 = region
            pct = (x2 - x1) * 100, (y2 - y1) * 100
            self.lbl_region_tip.setText(
                f"✅ 已选区 ({pct[0]:.0f}% × {pct[1]:.0f}%)，将只对该区域像素化"
            )
            self.lbl_region_tip.setStyleSheet("color: #2a7; font-size: 12px; font-weight: bold;")
            self.btn_clear_region.setEnabled(True)
        else:
            self.lbl_region_tip.setText("拖拽框选区域，可只对选中区域像素化（留空=整图）")
            self.lbl_region_tip.setStyleSheet("color: #888; font-size: 12px;")
            self.btn_clear_region.setEnabled(False)
        # 刷新转化缓冲区（框选→裁剪预览；未框选→整图）
        self._update_buffer()

    def _update_buffer(self):
        """更新「转化缓冲区」Tab：框选后显示裁剪子图，未框选显示整图。"""
        # 判断当前源图（原图或卡通图）
        src_path = self._current_source_path()
        if not src_path:
            self.lbl_buffer.clear_image()
            self.lbl_buffer.setText("（请先打开图片）")
            self.lbl_buffer_tip.setText("未框选时等价于整图；框选后显示裁剪结果")
            return
        try:
            from PIL import Image as _PILImage
            im = _PILImage.open(src_path).convert("RGB")
            w, h = im.size
            if self.region is not None:
                x1, y1, x2, y2 = self.region
                box = (
                    max(0, int(x1 * w)), max(0, int(y1 * h)),
                    min(w, int(x2 * w)), min(h, int(y2 * h)),
                )
                if box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
                    im = im.crop(box)
                    self.lbl_buffer_tip.setText(
                        f"已裁剪选区 ({box[2]-box[0]}×{box[3]-box[1]} px) — 将被像素化"
                    )
                else:
                    self.lbl_buffer_tip.setText("选区太小，暂按整图")
            else:
                self.lbl_buffer_tip.setText("未框选 — 整图将被像素化")
            # 转成 QPixmap 显示（走临时文件，避免 image→QPixmap 转换繁琐）
            import tempfile, os
            tmp = Path(tempfile.gettempdir()) / "_bead_buffer_preview.png"
            im.save(str(tmp))
            self.lbl_buffer.set_image(str(tmp))
        except Exception as e:
            self.lbl_buffer.setText(f"（缓冲区预览失败: {e}）")

    def _current_source_path(self):
        """返回当前应作为量化/缓冲区源的图片路径（卡通图优先于原图）。

        依据 `_active_source`（记录最近激活的原图/卡通图 Tab），
        切到「转化缓冲区」Tab 时沿用该记录，不会漂移回原图。
        """
        if getattr(self, "_active_source", None) == "cartoon" and \
           getattr(self, "cartoon_path", None):
            return self.cartoon_path
        return self.image_path

    def _on_tab_changed(self, idx):
        """源图区切换 Tab 时处理选区。

        - 切到「转化缓冲区」Tab：不清选区（保留框，缓冲区预览保持框选结果）
        - 在原图 ↔ 卡通图之间切换：清选区（不同源图尺寸/内容不同，选区不通用）
        """
        # 记录最近激活的源图 Tab（原图 / 卡通图）
        if idx == getattr(self, "src_orig_idx", -1):
            self._active_source = "orig"
        elif idx == getattr(self, "tab_cartoon_idx", -1):
            self._active_source = "cartoon"
        # 切到缓冲区 Tab 时不动选区，直接返回
        if idx == getattr(self, "tab_buffer_idx", -1):
            return
        # 仅当切到原图/卡通图时才清空选区（避免误清）
        self.region = None
        self.lbl_orig.clear_region()
        if hasattr(self, "lbl_cartoon") and self.lbl_cartoon is not None:
            self.lbl_cartoon.clear_region()
        self._on_region_changed(None)

    def _on_clear_region(self):
        self.region = None
        self.lbl_orig.clear_region()
        if hasattr(self, "lbl_cartoon"):
            self.lbl_cartoon.clear_region()
        self._on_region_changed(None)

    def _on_left_tab_changed(self, idx):
        """左侧「参数 / 编辑」Tab 切换：参数区时网格只读，编辑区时可编辑。"""
        # idx 0 = 参数，idx 1 = 编辑
        editable = (idx == 1)
        self.lbl_grid.set_editable(editable)
        if editable:
            self.lbl_grid_tip.setText("已载入 · 左侧「编辑」页选工具和颜色开始编辑")
        else:
            self.lbl_grid_tip.setText("当前在「参数」页，网格为只读（切到「编辑」页可编辑）")

    # -------- 卡通化 (Stage 0 + 1) --------

    def _on_cartoonize(self):
        if not HAS_PIPELINE:
            QMessageBox.warning(
                self, "卡通化不可用",
                "卡通化功能依赖 openai 库，当前环境未安装。\n\n"
                "安装: pip install openai\n\n"
                "基础量化（生成拼豆图纸）不受影响。"
            )
            return
        if not self.image_path:
            QMessageBox.warning(self, "未选择图片", "请先选择或拖入一张图片")
            return
        # 检测 API Key 是否已配置
        if not self._check_api_keys():
            return
        if self.cartoon_worker and self.cartoon_worker.isRunning():
            QMessageBox.information(self, "正在卡通化", "上一次卡通化还在跑，请稍候")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "正在量化", "请等量化完成后再卡通化")
            return

        out_dir = self.out_edit.text()
        custom_prompt = self.prompt_edit.toPlainText().strip() or None
        self.cartoon_worker = CartoonizeWorker(self.image_path, out_dir, custom_prompt=custom_prompt)
        self.cartoon_worker.progress.connect(self._on_cartoon_progress)
        self.cartoon_worker.finished_ok.connect(self._on_cartoon_finished)
        self.cartoon_worker.failed.connect(self._on_cartoon_failed)

        self.btn_cartoon.setEnabled(False)
        self.btn_gen.setEnabled(False)
        self.act_gen.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_cartoon_status.setText("⏳ 卡通化中...")
        self.lbl_cartoon_status.setStyleSheet("color: #c90;")
        self.status.showMessage("卡通化中（LLM 理解 + SeedDream，约 1-2 分钟）...")
        self.cartoon_worker.start()

    def _on_cartoon_progress(self, msg: str):
        self.status.showMessage(f"{msg}")

    def _on_cartoon_finished(self, result: PipelineResult, cartoon_path: str):
        self.progress.setVisible(False)
        if HAS_PIPELINE:
            self.btn_cartoon.setEnabled(True)
        self.btn_gen.setEnabled(True)
        self.act_gen.setEnabled(True)

        self.key_features = result.key_features
        self.analysis = result.analysis
        # 记录卡通图路径，但不覆盖原图（原图保留在「原图」Tab）
        self.cartoon_path = cartoon_path

        # 在卡通图 Tab 显示结果，并启用该 Tab + 切过去
        self.lbl_cartoon.set_image(cartoon_path)
        self.tabs.setTabEnabled(self.tab_cartoon_idx, True)
        self.tabs.setCurrentIndex(self.tab_cartoon_idx)

        # 显示分析摘要
        if result.analysis:
            a = result.analysis
            n_feat = len(a.key_features)
            grid = a.recommended_grid
            gtxt = f"{grid.get('w', '?')}×{grid.get('h', '?')}" if grid else "?"
            self.lbl_cartoon_status.setText(
                f"✓ 主体: {a.subject}\n"
                f"  关键特征 {n_feat} 处 · 推荐网格 {gtxt}\n"
                f"  {Path(cartoon_path).name}"
            )
        else:
            self.lbl_cartoon_status.setText(f"✓ {Path(cartoon_path).name}")
        self.lbl_cartoon_status.setStyleSheet("color: #2a7;")
        self.status.showMessage(
            f"✓ 卡通化完成。现在可在「卡通图」Tab 查看，生成拼豆图纸时将使用卡通图", 8000
        )

    def _on_cartoon_failed(self, err: str):
        self.progress.setVisible(False)
        if HAS_PIPELINE:
            self.btn_cartoon.setEnabled(True)
        if self.image_path:
            self.btn_gen.setEnabled(True)
            self.act_gen.setEnabled(True)
        self.lbl_cartoon_status.setText("✗ 卡通化失败，详见弹窗")
        self.lbl_cartoon_status.setStyleSheet("color: #c33;")
        self.status.showMessage("✗ 卡通化失败", 5000)
        QMessageBox.critical(self, "卡通化失败", err)

    def _on_generate(self):
        if not self.image_path:
            QMessageBox.warning(self, "未选择图片", "请先选择或拖入一张图片")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "正在生成", "上一次任务还在跑，先等一下")
            return

        # 决定量化源图：若已有卡通图且最近激活源为卡通图，则用卡通图；否则用原图
        use_cartoon = bool(
            getattr(self, "cartoon_path", None) and
            getattr(self, "_active_source", None) == "cartoon"
        )
        quantize_image = self._current_source_path()
        if not quantize_image:
            QMessageBox.warning(self, "未选择图片", "请先选择或拖入一张图片")
            return

        # 收集参数
        params = {
            "image_path": quantize_image,
            "grid": self.grid_slider.value(),
            "supersample": self.ss_spin.value(),
            "ss_strategy": self.ss_combo.currentText(),
            "ss_min_dark_count": self.ss_dark_spin.value(),
            "ss_line_min_pixels": self.ss_line_spin.value(),
            "detail_mode": self.chk_detail.isChecked(),
            "outline_preserve": False,
            "min_island_size": self.island_spin.value(),
            "cleanup_min_contrast": self.deltae_spin.value(),
            "line_detect": self.chk_line.isChecked(),
            "line_method": self.line_method_combo.currentText(),
            "line_threshold_percentile": self.line_thr_spin.value(),
            "line_min_pct": self.line_pct_spin.value(),
            "min_line_count": self.line_count_spin.value(),
            "min_line_contrast": self.line_contrast_spin.value(),
            "line_aspect_min": self.line_aspect_spin.value(),
            "line_min_region_size": self.line_region_spin.value(),
            "out_dir": self.out_edit.text(),
        }
        # 像素化方法参数
        px_method = self.px_combo.currentText()
        px_params: dict = {}
        if px_method == "dpid":
            px_params["lam"] = self.px_dpid_lam.value()
        elif px_method == "pixeloe":
            px_params["mode"] = self.px_poe_mode.currentText()
            px_params["thickness"] = self.px_poe_thick.value()
            px_params["contrast"] = self.px_poe_contrast.value()
            px_params["saturation"] = self.px_poe_sat.value()
        # SuperSample -> 后端 pixelize_method 用 "none"（不走像素化，纯超采样+投票）
        params["pixelize_method"] = "none" if px_method == "SuperSample" else px_method
        params["pixelize_params"] = px_params
        params["chart_name"] = self.chart_name_edit.text().strip() or None

        self.worker = QuantizeWorker(**params)
        # 注入 Stage 0/1 得到的关键特征 bbox（仅当用卡通图时有效）
        self.worker.key_features = self.key_features if use_cartoon else None
        # 注入框选区域（若有）
        self.worker.region = self.region
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.btn_gen.setEnabled(False)
        self.act_gen.setEnabled(False)
        self.progress.setVisible(True)
        src_name = "卡通图" if use_cartoon else "原图"
        self.status.showMessage(f"生成中（量化源: {src_name}）...")
        self.worker.start()

    def _on_progress(self, msg: str):
        self.status.showMessage(f"⏳ {msg}")

    def _on_finished(self, result, chart_path, grid_path, json_path):
        self.progress.setVisible(False)
        self.btn_gen.setEnabled(True)
        self.act_gen.setEnabled(True)
        self.act_save.setEnabled(True)
        self.result = result
        self.last_chart_path = chart_path
        self.last_grid_path = grid_path
        self.last_json_path = json_path

        # 网格编辑控件载入结果
        self.lbl_grid.set_palette(self.palette)
        self.lbl_grid.set_result(result)
        self.lbl_grid_tip.setText("已载入 · 左侧「编辑」页选工具和颜色开始编辑")
        # 清空撤销/重做栈
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_undo_buttons()

        # 图纸：用色号全标注重新渲染（标题/文件名用用户命名的图纸名）
        grid = self.grid_slider.value()
        cell_px = max(8, min(24, 1800 // grid))
        stem = (self.chart_name_edit.text().strip()
                or (Path(self.image_path).stem if self.image_path else "bead"))
        suffix = "_region" if self.region is not None else ""
        out_dir = Path(self.out_edit.text())
        chart_path2 = out_dir / f"{stem}{suffix}_bead_chart_{grid}x{grid}.png"
        plain_path2 = out_dir / f"{stem}{suffix}_bead_plain_{grid}x{grid}.png"
        save_bead_chart(
            result, str(chart_path2),
            cell_px=cell_px,
            label_every=1,
            label_all_cells=True,
            title=stem,
            palette_name="MARD",
        )
        self.last_chart_path = str(chart_path2)

        # 显示图
        self.lbl_chart.set_image(str(chart_path2))
        # 像素化预览（无网格线）
        plain_img = render_plain_image(result, cell_px=cell_px)
        plain_img.save(str(plain_path2))
        self.last_plain_path = str(plain_path2)
        self.lbl_plain.set_image(str(plain_path2))

        # 初始化编辑面板
        self._init_edit_panel()

        # 统计
        stats = result.stats
        top12 = sorted(stats.items(), key=lambda x: -x[1])[:12]
        lines = [
            f"用色数: {result.color_count}",
            f"总格数: {result.total_cells}",
            f"清理小色块: {result.removed_islands}",
            "",
            "色号用量 TOP 12:",
        ]
        for code, cnt in top12:
            c = self.palette.by_code(code)
            hex_ = c.hex if c else "?"
            lines.append(f"  {code:<6} {cnt:>5} 格  hex={hex_}")
        lines.append("")
        lines.append(f"产物:")
        lines.append(f"  拼豆图: {Path(chart_path2).name}")
        lines.append(f"  网格:   {Path(grid_path).name}")
        lines.append(f"  JSON:   {Path(json_path).name}")
        self.lbl_stats.setText("\n".join(lines))
        self.lbl_stats.setStyleSheet("font-family: Consolas, monospace;")
        self.stats_box.setVisible(True)

        self.status.showMessage(f"✓ 完成: {Path(chart_path2).name} · 可在「编辑」页手动编辑", 8000)

    def _init_edit_panel(self):
        """生成图纸后初始化编辑面板。"""
        self.lbl_edit_status.setText(
            f"✓ 图纸已生成（{self.result.grid_w}×{self.result.grid_h}），"
            "可在网格预览上编辑"
        )
        self.lbl_edit_status.setStyleSheet("color: #2a7; font-size: 12px;")
        # 构建色板 swatch
        self._rebuild_palette_swatch()
        self._update_replace_target()
        # 重置调整滑条
        self.slider_darken.setValue(0)
        self.slider_lighten.setValue(0)
        self.slider_contrast.setValue(100)
        self._adjust_base_grid = None
        self._adjust_base_cells = None
        # 默认画笔选第一主色
        if self.result.stats:
            top_code = sorted(self.result.stats.items(), key=lambda x: -x[1])[0][0]
            self._on_brush_pick(top_code)

    def _on_failed(self, err: str):
        self.progress.setVisible(False)
        self.btn_gen.setEnabled(True)
        self.act_gen.setEnabled(True)
        self.status.showMessage("✗ 失败", 5000)
        QMessageBox.critical(self, "生成失败", err)

    def _on_save(self):
        """导出：选择导出哪些图 + 目标文件夹 + 文件名基名。

        基于当前 self.result 重新渲染（这样手动编辑后的结果也能正确导出）。
        """
        if self.result is None:
            return
        dlg = ExportDialog(self)
        # 默认文件名基名：优先用图纸命名，否则用图片文件名
        chart_name = self.chart_name_edit.text().strip()
        if chart_name:
            dlg.name_edit.setText(chart_name)
        elif self.image_path:
            dlg.name_edit.setText(Path(self.image_path).stem)
        if dlg.exec() != QDialog.Accepted:
            return

        out_dir = Path(dlg.dir_edit.text())
        stem = dlg.name_edit.text().strip() or "bead"
        out_dir.mkdir(parents=True, exist_ok=True)

        grid = self.result.grid_w
        cell_px = max(8, min(24, 1800 // grid))
        exported = []

        try:
            if dlg.chk_chart.isChecked():
                p = out_dir / f"{stem}_chart.png"
                save_bead_chart(
                    self.result, str(p), cell_px=cell_px,
                    label_every=max(5, grid // 12),
                    title=stem,
                    palette_name="MARD",
                )
                exported.append(p)
            if dlg.chk_grid.isChecked():
                p = out_dir / f"{stem}_grid.png"
                render_grid_only(self.result, cell_px=cell_px, with_grid=True).save(str(p))
                exported.append(p)
            if dlg.chk_plain.isChecked():
                p = out_dir / f"{stem}_plain.png"
                render_plain_image(self.result, cell_px=cell_px).save(str(p))
                exported.append(p)
            if dlg.chk_json.isChecked():
                p = out_dir / f"{stem}.json"
                grid_json = {
                    "grid_w": self.result.grid_w,
                    "grid_h": self.result.grid_h,
                    "palette": "MARD",
                    "cells": [[cell.code for cell in row] for row in self.result.grid],
                }
                p.write_text(
                    json.dumps(grid_json, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                exported.append(p)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return

        names = "、".join(p.name for p in exported)
        QMessageBox.information(self, "导出成功", f"已导出 {len(exported)} 个文件到:\n{out_dir}\n\n{names}")
        self.status.showMessage(f"✓ 已导出 {len(exported)} 个文件到 {out_dir}", 5000)

    def _on_import(self):
        """导入 JSON：解析色号矩阵，重建 result 并载入预览/编辑区。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入拼豆 JSON", self.out_edit.text(), "JSON 文件 (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            grid_w = int(data.get("grid_w", 0))
            grid_h = int(data.get("grid_h", 0))
            cells = data.get("cells")
            if not grid_w or not grid_h or not cells or len(cells) != grid_h:
                raise ValueError("JSON 结构无效：缺少 grid_w/grid_h/cells，或行数不匹配")
            # 重建 grid：code -> PaletteColor，空串 -> BLANK
            grid = []
            unknown = set()
            for row in cells:
                if len(row) != grid_w:
                    raise ValueError(f"JSON 列数不一致：某行 {len(row)} != {grid_w}")
                new_row = []
                for code in row:
                    code = str(code) if code is not None else ""
                    if code == "":
                        new_row.append(BLANK)
                    else:
                        c = self.palette.by_code(code)
                        if c is None:
                            unknown.add(code)
                            new_row.append(BLANK)
                        else:
                            new_row.append(c)
                grid.append(new_row)
            if unknown:
                QMessageBox.warning(
                    self, "部分色号未识别",
                    f"以下 {len(unknown)} 个色号不在 MARD 色板中，已置为空白:\n"
                    f"{', '.join(sorted(unknown))}",
                )
            # 构造 QuantizeResult
            result = QuantizeResult(
                grid=grid, stats={}, color_count=0,
                total_cells=grid_w * grid_h, grid_w=grid_w, grid_h=grid_h,
                removed_islands=0,
            )
            rebuild_result_stats(result)
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
            return

        # 载入：复用 _on_finished 的展示逻辑（但不依赖 worker）
        self.result = result
        self.act_save.setEnabled(True)
        self.lbl_grid.set_palette(self.palette)
        self.lbl_grid.set_result(result)
        self.lbl_grid_tip.setText("已导入 · 左侧「编辑」页可继续编辑")
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_undo_buttons()

        # 渲染到 outputs（沿用当前输出目录）
        grid = result.grid_w
        cell_px = max(8, min(24, 1800 // grid))
        stem = Path(path).stem
        out_dir = Path(self.out_edit.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        chart_path = out_dir / f"{stem}_import_chart.png"
        grid_path = out_dir / f"{stem}_import_grid.png"
        plain_path = out_dir / f"{stem}_import_plain.png"
        save_bead_chart(result, str(chart_path), cell_px=cell_px,
                        label_every=1, label_all_cells=True,
                        title=stem, palette_name="MARD")
        render_grid_only(result, cell_px=cell_px, with_grid=True).save(str(grid_path))
        render_plain_image(result, cell_px=cell_px).save(str(plain_path))

        self.last_chart_path = str(chart_path)
        self.last_grid_path = str(grid_path)
        self.last_plain_path = str(plain_path)
        self.last_json_path = path

        self.lbl_chart.set_image(str(chart_path))
        self.lbl_plain.set_image(str(plain_path))
        self._init_edit_panel()

        # 统计
        stats = result.stats
        top12 = sorted(stats.items(), key=lambda x: -x[1])[:12]
        lines = [
            f"用色数: {result.color_count}",
            f"总格数: {result.total_cells}",
            f"来源: {Path(path).name}",
            "",
            "色号用量 TOP 12:",
        ]
        for code, cnt in top12:
            c = self.palette.by_code(code)
            hex_ = c.hex if c else "?"
            lines.append(f"  {code:<6} {cnt:>5} 格  hex={hex_}")
        self.lbl_stats.setText("\n".join(lines))
        self.lbl_stats.setStyleSheet("font-family: Consolas, monospace;")
        self.stats_box.setVisible(True)

        self.status.showMessage(f"✓ 已导入 {Path(path).name}（{grid_w}×{grid_h}）", 8000)

    def _on_about(self):
        QMessageBox.about(
            self, "关于 拼豆图纸生成器",
            "<h3>拼豆图纸生成器 · Beadobe</h3>"
            "<p>把照片转成 MARD 色板的拼豆图纸。</p>"
            "<p><b>核心算法:</b> 5 阶段管线<br>"
            "1) LLM 视觉理解 (已集成)<br>"
            "2) SeedDream 卡通化 (已集成)<br>"
            "3) MARD 色板量化<br>"
            "4) 超采样 + 下采样<br>"
            "5) 小色块清理</p>"
            "<p>版本: Phase 3 (全流程版)</p>"
        )

    # -------- API 设置 --------

    def _on_settings(self):
        """打开 API 设置对话框。"""
        from core.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        dlg.exec()

    def _check_api_keys(self) -> bool:
        """检测卡通化所需的 API Key 是否已配置。

        缺 key 时弹提示框（带「去配置」按钮），返回 False；齐全返回 True。
        """
        try:
            from core.config_loader import has_llm_key, has_ark_key
            need_llm = not has_llm_key()
            need_ark = not has_ark_key()
        except Exception:
            # 配置模块异常时视为缺 key，引导去设置
            need_llm = True
            need_ark = True

        if not need_llm and not need_ark:
            return True

        # 组装缺失提示
        missing = []
        if need_llm:
            missing.append("• 对话型 LLM（视觉理解）的 API Key")
        if need_ark:
            missing.append("• 方舟生图（卡通化）的 API Key")

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("未配置 API Key")
        box.setText(
            "卡通化功能需要配置 API Key，当前缺少：\n\n"
            + "\n".join(missing)
            + "\n\n请点击「去配置」填写。"
        )
        go = box.addButton("去配置", QMessageBox.AcceptRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is go:
            self._on_settings()
        return False

    # -------- 退出时清理 --------

    def closeEvent(self, event: QtGui.QCloseEvent):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        if self.cartoon_worker and self.cartoon_worker.isRunning():
            self.cartoon_worker.cancel()
            self.cartoon_worker.wait(2000)
        event.accept()


# =============================================================================
# 入口
# =============================================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = BeadStudio()
    # 打包自检：--selftest 时构建窗口后立即退出，用于验证打包产物能正常加载
    # 色板与资源（不进入事件循环）。正常使用不带该参数。
    if "--selftest" in sys.argv:
        # 打包自检：除构建窗口 + 加载色板外，再实际跑一次核心链路
        # （图片 → 像素化 → 量化 → 图纸渲染），验证打包产物里核心功能可用。
        ok = True
        msgs = []
        if win.palette is None or len(win.palette.colors) == 0:
            ok = False
            msgs.append("palette empty")
        # 核心链路自检：用合成小图跑 pixeloe + 量化 + 渲染
        try:
            import numpy as np
            from PIL import Image
            import core.pixelize as pn
            import core.quantize as q
            import core.bead_chart as bc
            rng = np.random.default_rng(0)
            test_img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
            pix = pn.pixelize_pixeloe(test_img, 32, 32, mode="center",
                                      thickness=3, contrast=1.0, saturation=1.0)
            res = q.quantize(Image.fromarray(pix), grid_w=32, grid_h=32,
                             palette=win.palette, pixelize_method="none")
            n_used = len({c.code for row in res.grid for c in row})
            chart = bc.render_bead_chart(res, cell_px=20)
            msgs.append(f"pipeline OK used={n_used} colors")
        except Exception as e:
            ok = False
            msgs.append(f"pipeline FAIL: {e!r}")
        # API 设置模块自检：验证 settings_dialog / config_loader 可导入
        try:
            import core.settings_dialog as _sd
            import core.config_loader as _cl
            assert _cl.get_config_path().name == "config.yaml"
            assert _sd.SettingsDialog is not None
            msgs.append("settings OK")
        except Exception as e:
            ok = False
            msgs.append(f"settings FAIL: {e!r}")
        if ok:
            print("SELFTEST_OK " + " | ".join(msgs))
            sys.exit(0)
        print("SELFTEST_FAIL " + " | ".join(msgs))
        sys.exit(1)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
