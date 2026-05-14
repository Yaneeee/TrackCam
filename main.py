import sys
import time
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox, QFileDialog, QDoubleSpinBox,
    QSpinBox, QGroupBox, QFormLayout, QColorDialog, QScrollArea, QCheckBox
)


# ═══════════════════ 数据 ═══════════════════

@dataclass
class TrackPoint:
    time: str
    imageX: float
    imageY: float
    direction: float
    speed: float
    elapsedTimeFromStart: float
    lapNumber: int


def parse_track_points(path: str) -> List[TrackPoint]:
    try:
        tree = ET.parse(path)
        pts = []
        for s in tree.getroot().findall('.//Sample'):
            a = s.attrib
            pts.append(TrackPoint(
                a.get('time', ''), float(a.get('imageX', 0)),
                float(a.get('imageY', 0)), float(a.get('direction', 0)),
                float(a.get('speed', 0)), float(a.get('elapsedTimeFromStart', 0)),
                int(a.get('lapNumber', 0))))
        pts.sort(key=lambda p: p.elapsedTimeFromStart)
        return pts
    except Exception as e:
        print(f"XML 解析出错: {e}")
        return []


# ═══════════════════ 视频源 ═══════════════════

class VideoSource:
    def __init__(self):
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.duration = 0.0
        self.width = 0
        self.height = 0
        self._frame: Optional[np.ndarray] = None
        self._pos = 0.0

    @property
    def ok(self):
        return self.cap is not None and self.cap.isOpened()

    def load(self, path: str) -> bool:
        self.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            return False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, f = self.cap.read()
        if ok:
            self._frame = f
        return True

    def seek(self, sec: float):
        if not self.ok:
            return None
        sec = max(0.0, min(sec, self.duration))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, f = self.cap.read()
        if ok:
            self._frame = f
            self._pos = sec
        return self._frame

    def next(self):
        if not self.ok:
            return None
        ok, f = self.cap.read()
        if ok:
            self._frame = f
            self._pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return self._frame

    @property
    def frame(self):
        return self._frame

    @property
    def pos(self):
        return self._pos

    def release(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        self._frame = None
        self._pos = 0.0


# ═══════════════════ 帧缓冲池 ═══════════════════

class FrameBuffer:
    def __init__(self):
        self.output = None
        self._zeros = {}
        self._shape = (0, 0)

    def ensure(self, H, W):
        if self._shape != (H, W):
            self.output = np.empty((H, W, 3), np.uint8)
            self._shape = (H, W)
        return self.output

    def zeros(self, shape):
        if shape not in self._zeros:
            self._zeros[shape] = np.zeros(shape, np.uint8)
        return self._zeros[shape]

    def reset(self):
        self.output = None
        self._zeros.clear()
        self._shape = (0, 0)


# ═══════════════════ 蒙版与叠加 ═══════════════════

_mc = {}


def _mask(h, w):
    k = ('c', h, w)
    if k not in _mc:
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (w // 2, h // 2), min(w, h) // 2, 255, -1, cv2.LINE_AA)
        _mc[k] = m
    return _mc[k]


def _ring(h, w, t=3):
    k = ('cr', h, w, t)
    if k not in _mc:
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (w // 2, h // 2), min(w, h) // 2, 255, t, cv2.LINE_AA)
        _mc[k] = m
    return _mc[k]


# ═══════════════════ 圆形叠加（简化，patch 始终 = display_size）═══════════════════

def overlay_circle(canvas, patch, cx, cy, display_size,
                   border_color=(255, 255, 255), border_w=3):
    ds = display_size
    ch, cw = canvas.shape[:2]
    x1, y1 = max(0, cx), max(0, cy)
    x2, y2 = min(cw, cx + ds), min(ch, cy + ds)
    if x2 <= x1 or y2 <= y1:
        return
    px1, py1 = x1 - cx, y1 - cy
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    ph, pw = patch.shape[:2]
    if px2 > pw or py2 > ph:
        return
    ms = _mask(ds, ds)[py1:py2, px1:px2]
    rg = _ring(ds, ds, border_w)[py1:py2, px1:px2]
    roi = patch[py1:py2, px1:px2].copy()
    roi[rg > 0] = list(border_color)
    m3 = cv2.merge([ms, ms, ms])
    np.copyto(canvas[y1:y2, x1:x2], roi, where=m3.astype(bool))


def _rr_mask(h, w, r):
    k = ('rr', h, w, r)
    if k not in _mc:
        r = min(r, w // 2, h // 2)
        m = np.zeros((h, w), np.uint8)
        if r <= 0:
            m[:] = 255
        else:
            m[r:h - r, :] = 255
            m[:, r:w - r] = 255
            cv2.circle(m, (r, r), r, 255, -1, cv2.LINE_AA)
            cv2.circle(m, (w - 1 - r, r), r, 255, -1, cv2.LINE_AA)
            cv2.circle(m, (r, h - 1 - r), r, 255, -1, cv2.LINE_AA)
            cv2.circle(m, (w - 1 - r, h - 1 - r), r, 255, -1, cv2.LINE_AA)
        _mc[k] = m
    return _mc[k]


def _draw_rr_border(canvas, x, y, w, h, r, color, thickness):
    r = min(r, w // 2, h // 2)
    if r <= 0:
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), color, thickness)
        return
    cv2.line(canvas, (x + r, y), (x + w - 1 - r, y), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x + r, y + h - 1), (x + w - 1 - r, y + h - 1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x, y + r), (x, y + h - 1 - r), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x + w - 1, y + r), (x + w - 1, y + h - 1 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + w - 1 - r, y + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + r, y + h - 1 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + w - 1 - r, y + h - 1 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def overlay_rounded_rect(canvas, patch, cx, cy, radius,
                         border_color=(160, 160, 160), border_w=2):
    ph, pw = patch.shape[:2]
    ch, cw = canvas.shape[:2]
    x1, y1 = max(0, cx), max(0, cy)
    x2, y2 = min(cw, cx + pw), min(ch, cy + ph)
    if x2 <= x1 or y2 <= y1:
        return
    px1, py1 = x1 - cx, y1 - cy
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    mask_full = _rr_mask(ph, pw, radius)
    ms = mask_full[py1:py2, px1:px2]
    roi = patch[py1:py2, px1:px2].copy()
    m3 = cv2.merge([ms, ms, ms])
    np.copyto(canvas[y1:y2, x1:x2], roi, where=m3.astype(bool))
    _draw_rr_border(canvas, cx, cy, pw, ph, radius, border_color, border_w)


# ═══════════════════ 渲染工具 ═══════════════════

def resize_fit(frame, tw, th):
    if frame is None:
        return np.zeros((th, tw, 3), np.uint8)
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((th, tw, 3), np.uint8)
    s = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    c = np.zeros((th, tw, 3), np.uint8)
    yo, xo = (th - nh) // 2, (tw - nw) // 2
    if yo >= 0 and xo >= 0 and yo + nh <= th and xo + nw <= tw:
        c[yo:yo + nh, xo:xo + nw] = r
    return c


def fill_cover(frame, tw, th):
    """覆盖填充：等比缩放后裁剪中心，保证输出严格 tw×th 且不变形"""
    if frame is None:
        return np.zeros((th, tw, 3), np.uint8)
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((th, tw, 3), np.uint8)
    s = max(tw / w, th / h)
    # round + max 确保缩放后尺寸 ≥ 目标，消除浮点取整导致的 1px 欠缺
    nw = max(tw, int(round(w * s)))
    nh = max(th, int(round(h * s)))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    yo = max(0, (nh - th) // 2)
    xo = max(0, (nw - tw) // 2)
    return r[yo:yo + th, xo:xo + tw].copy()


# ═══════════════════ 渲染引擎 ═══════════════════
class RenderEngine:
    def __init__(self):
        self.map_img = None
        self._sc = self._sa = self._ss = None

    def load_map(self, path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            self.map_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            self.map_img = None
        return self.map_img is not None

    def reset(self):
        self._sc = self._sa = self._ss = None

    @staticmethod
    def interp(td, fi):
        i1 = int(fi)
        i2 = min(i1 + 1, len(td) - 1)
        f = fi - i1
        p1, p2 = td[i1], td[i2]
        ix = p1.imageX + (p2.imageX - p1.imageX) * f
        iy = p1.imageY + (p2.imageY - p1.imageY) * f
        d1, d2 = p1.direction, p2.direction
        if abs(d2 - d1) > 180:
            if d2 > d1:
                d1 += 360
            else:
                d2 += 360
        return ix, iy, (d1 + (d2 - d1) * f) % 360

    def render(self, ix, iy, idir, trail, laps, mode, size,
               ox, oy, ps, lw, lc, lerp, margin, scale_mult=1.0):
        """
        scale_mult: >1 放大细节(看到更少区域), <1 缩小(看到更多区域)
        """
        tw, th = size
        cx, cy = ix + ox, iy + oy
        if mode == "LapView" and len(laps) > 1:
            ap = np.array([[p.imageX + ox, p.imageY + oy] for p in laps], np.float32)
            mn, mx = np.min(ap, 0), np.max(ap, 0)
            tc = (mn + mx) / 2
            p0, p1 = laps[0], laps[-1]
            ta = np.degrees(np.arctan2(
                p1.imageY + oy - p0.imageY - oy,
                p1.imageX + ox - p0.imageX - ox)) + 90
            ts = min(tw / max(1, mx[0] - mn[0]),
                     th / max(1, mx[1] - mn[1])) * margin
        elif mode == "NorthUp":
            tc, ta, ts = np.array([cx, cy]), 0.0, 1.0
        else:
            tc, ta, ts = np.array([cx, cy]), idir, 1.0

        # 应用缩放：放大倍数 → ts 增大 → 地图放大 → 看到更少区域
        ts *= scale_mult

        if self._sc is None:
            self._sc = tc.copy() if isinstance(tc, np.ndarray) else np.array(tc)
            self._sa, self._ss = ta, ts
        else:
            self._sc += (tc - self._sc) * lerp
            self._ss += (ts - self._ss) * lerp
            d = ta - self._sa
            if d > 180:
                d -= 360
            if d < -180:
                d += 360
            self._sa += d * lerp
        M = cv2.getRotationMatrix2D(tuple(self._sc), self._sa, self._ss)
        M[0, 2] += tw / 2 - self._sc[0]
        M[1, 2] += th / 2 - self._sc[1]
        fr = cv2.warpAffine(self.map_img, M, (tw, th), flags=cv2.INTER_LINEAR) \
            if self.map_img is not None else np.zeros((th, tw, 3), np.uint8)
        if len(trail) > 1:
            d = np.array([[p.imageX + ox, p.imageY + oy] for p in trail], np.float32)
            t = cv2.transform(d.reshape(-1, 1, 2), M)
            cv2.polylines(fr, [t.astype(np.int32)], False, lc, lw, cv2.LINE_AA)
        tp = cv2.transform(np.array([[[cx, cy]]], np.float32), M)[0][0]
        cv2.circle(fr, (int(tp[0]), int(tp[1])), ps + 2, (255, 255, 255), -1)
        cv2.circle(fr, (int(tp[0]), int(tp[1])), ps, (0, 0, 255), -1)
        return fr


# ═══════════════════ 合成 ═══════════════════

SOURCES = ["Video", "HeadingUp", "NorthUp", "LapView"]


def compose(buf, bg, circle, rect,
            W, H, panel_r, circ_display, margin, border_w, darken,
            show_circle, show_rect, rr_radius):
    out = buf.ensure(H, W)
    if darken:
        z = buf.zeros((H, W, 3))
        cv2.addWeighted(bg, 0.72, z, 0.28, 0, dst=out)
    else:
        np.copyto(out, bg)
    if not (show_circle or show_rect):
        return out
    pw = max(60, int(W * panel_r))
    px = margin // 2
    py = margin // 2
    viewport_w = max(20, pw - margin * 2)
    y_cur = py + margin

    if show_circle and circle is not None:
        cx = px + (pw - circ_display) // 2
        cy = y_cur
        if cy + circ_display <= H and cx + circ_display <= W:
            overlay_circle(out, circle, cx, cy, circ_display)
        y_cur += circ_display + margin

    if show_rect and rect is not None:
        rh, rw = rect.shape[:2]
        rx = px + (pw - rw) // 2
        ry = y_cur
        if ry + rh <= H and rx + rw <= W:
            overlay_rounded_rect(out, rect, rx, ry, rr_radius,
                                 border_color=(160, 160, 160), border_w=border_w)
    return out



# ═══════════════════ 主窗口 ═══════════════════

WIN = "Track Viewer"


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("控制面板")
        self.e_bg = RenderEngine()
        self.e_circ = RenderEngine()
        self.e_rect = RenderEngine()
        self.engines = [self.e_bg, self.e_circ, self.e_rect]
        self.vs = VideoSource()
        self.buf = FrameBuffer()
        self.td: List[TrackPoint] = []
        self.fi = 0.0
        self.playing = False
        self.tcolor = QColor(0, 255, 0)
        self._rendering = False
        self._last_time = 0.0
        self._win_created = False
        self._render_w = 960
        self._render_h = 720

        self._ui()
        self._timer = QTimer()
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── cv2 窗口 ──

    def _ensure_window(self):
        if not self._win_created:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, self._render_w, self._render_h)
            self._win_created = True

    def _window_alive(self) -> bool:
        try:
            return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1
        except Exception:
            return False

    def _sync_window_size(self) -> bool:
        """检测窗口拖拽尺寸变化，自适应渲染分辨率。返回 True 表示尺寸变了"""
        if not self._win_created:
            return False
        try:
            x, y, w, h = cv2.getWindowImageRect(WIN)
            if w < 100 or h < 100:
                return False
            if w != self._render_w or h != self._render_h:
                self._render_w, self._render_h = w, h
                self.buf.reset()
                self.sp_dw.blockSignals(True)
                self.sp_dh.blockSignals(True)
                self.sp_dw.setValue(w)
                self.sp_dh.setValue(h)
                self.sp_dw.blockSignals(False)
                self.sp_dh.blockSignals(False)
                return True
        except Exception:
            pass
        return False

    # ── UI ──

    def _ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.setContentsMargins(0, 0, 0, 0)

        def grp(title, rows):
            g = QGroupBox(title)
            f = QFormLayout()
            f.setSpacing(4)
            for r in rows:
                if isinstance(r, tuple):
                    f.addRow(r[0], r[1])
                else:
                    f.addRow(r)
            g.setLayout(f)
            sl.addWidget(g)

        self.btn_map = QPushButton("加载地图")
        self.btn_xml = QPushButton("加载轨迹 XML")
        self.btn_vid = QPushButton("加载视频")
        grp("文件加载", [self.btn_map, self.btn_xml, self.btn_vid])

        self.ox = QDoubleSpinBox()
        self.ox.setRange(-99999, 99999)
        self.ox.setDecimals(1)
        self.oy = QDoubleSpinBox()
        self.oy.setRange(-99999, 99999)
        self.oy.setDecimals(1)
        grp("地图偏移", [("X:", self.ox), ("Y:", self.oy)])

        self.sp_fps = QSpinBox()
        self.sp_fps.setRange(10, 240)
        self.sp_fps.setValue(60)
        self.sp_fps.setSuffix(" FPS")
        self.cb_spd = QComboBox()
        self.cb_spd.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "8x", "16x"])
        self.cb_spd.setCurrentText("1x")
        grp("播放控制", [("帧率:", self.sp_fps), ("倍速:", self.cb_spd)])

        self.chk_bg = QCheckBox("显示")
        self.chk_bg.setChecked(True)
        self.cb_src_bg = QComboBox()
        self.cb_src_bg.addItems(SOURCES)
        self.cb_src_bg.setCurrentText("Video")
        self.chk_circ = QCheckBox("显示")
        self.chk_circ.setChecked(True)
        self.cb_src_circ = QComboBox()
        self.cb_src_circ.addItems(SOURCES)
        self.cb_src_circ.setCurrentText("HeadingUp")
        self.chk_rect = QCheckBox("显示")
        self.chk_rect.setChecked(True)
        self.cb_src_rect = QComboBox()
        self.cb_src_rect.addItems(SOURCES)
        self.cb_src_rect.setCurrentText("LapView")

        gv = QGroupBox("视窗源选择")
        fv = QFormLayout()
        fv.setSpacing(4)
        for label, chk, cb in [("背景:", self.chk_bg, self.cb_src_bg),
                                ("圆形:", self.chk_circ, self.cb_src_circ),
                                ("矩形:", self.chk_rect, self.cb_src_rect)]:
            row = QHBoxLayout()
            row.addWidget(chk)
            row.addWidget(cb)
            cont = QWidget()
            cont.setLayout(row)
            fv.addRow(label, cont)
        gv.setLayout(fv)
        sl.addWidget(gv)

        self.sp_lerp = QDoubleSpinBox()
        self.sp_lerp.setRange(0.01, 1.0)
        self.sp_lerp.setSingleStep(0.01)
        self.sp_lerp.setValue(0.15)
        self.sp_lerp.setDecimals(2)
        self.sp_margin = QDoubleSpinBox()
        self.sp_margin.setRange(0.1, 2.0)
        self.sp_margin.setSingleStep(0.05)
        self.sp_margin.setValue(0.60)
        self.sp_margin.setDecimals(2)
        grp("平滑参数", [("插值系数:", self.sp_lerp), ("LapView边距:", self.sp_margin)])

        self.sp_ttw = QSpinBox()
        self.sp_ttw.setRange(1, 600)
        self.sp_ttw.setValue(30)
        self.sp_ttw.setSuffix(" 秒")
        self.sp_lw = QSpinBox()
        self.sp_lw.setRange(1, 20)
        self.sp_lw.setValue(3)
        self.sp_lw.setSuffix(" px")
        self.sp_ps = QSpinBox()
        self.sp_ps.setRange(2, 30)
        self.sp_ps.setValue(8)
        self.sp_ps.setSuffix(" px")
        self.btn_col = QPushButton()
        self._upd_col()
        grp("轨迹显示", [
            ("回看:", self.sp_ttw), ("线宽:", self.sp_lw),
            ("点大小:", self.sp_ps), ("颜色:", self.btn_col),
        ])

        self.sp_panel = QDoubleSpinBox()
        self.sp_panel.setRange(0.1, 0.8)
        self.sp_panel.setSingleStep(0.05)
        self.sp_panel.setValue(0.30)
        self.sp_panel.setDecimals(2)
        self.sp_circ_zoom = QDoubleSpinBox()
        self.sp_circ_zoom.setRange(0.5, 4.0)
        self.sp_circ_zoom.setSingleStep(0.1)
        self.sp_circ_zoom.setValue(1.0)
        self.sp_circ_zoom.setDecimals(1)
        self.sp_circ_zoom.setToolTip(">1 看更多, <1 放大")
        self.sp_fm = QSpinBox()
        self.sp_fm.setRange(4, 40)
        self.sp_fm.setValue(12)
        self.sp_fm.setSuffix(" px")
        self.sp_rr = QSpinBox()
        self.sp_rr.setRange(0, 200)
        self.sp_rr.setValue(20)
        self.sp_rr.setSuffix(" px")
        self.sp_bd = QSpinBox()
        self.sp_bd.setRange(1, 8)
        self.sp_bd.setValue(2)
        self.sp_bd.setSuffix(" px")
        grp("浮窗布局", [
            ("面板宽度比:", self.sp_panel), ("圆形缩放:", self.sp_circ_zoom),
            ("内边距:", self.sp_fm), ("圆角半径:", self.sp_rr),
            ("矩形边框:", self.sp_bd),
        ])

        self.chk_darken = QCheckBox("背景暗化")
        self.chk_darken.setChecked(True)
        self.sp_voff = QDoubleSpinBox()
        self.sp_voff.setRange(-3600, 3600)
        self.sp_voff.setSingleStep(0.1)
        self.sp_voff.setValue(0.0)
        self.sp_voff.setDecimals(2)
        self.sp_voff.setSuffix(" 秒")
        self.lb_vi = QLabel("未加载")
        grp("视频同步", [
            self.chk_darken, ("时间偏移:", self.sp_voff), ("信息:", self.lb_vi),
        ])

        self.sp_dw = QSpinBox()
        self.sp_dw.setRange(320, 3840)
        self.sp_dw.setValue(960)
        self.sp_dw.setSingleStep(80)
        self.sp_dw.setSuffix(" px")
        self.sp_dh = QSpinBox()
        self.sp_dh.setRange(240, 2160)
        self.sp_dh.setValue(720)
        self.sp_dh.setSingleStep(60)
        self.sp_dh.setSuffix(" px")
        grp("窗口尺寸", [("宽:", self.sp_dw), ("高:", self.sp_dh)])

        sl.addStretch()
        scroll.setWidget(sw)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        self.sld = QSlider(Qt.Horizontal)
        self.sld.setEnabled(False)
        layout.addWidget(self.sld)

        hc = QHBoxLayout()
        self.btn_pp = QPushButton("播放 / 暂停")
        self.btn_rs = QPushButton("重置平滑")
        self.lb_t = QLabel("00:00:00 / 00:00:00")
        self.lb_i = QLabel("")
        hc.addWidget(self.btn_pp)
        hc.addWidget(self.btn_rs)
        hc.addWidget(self.lb_t)
        hc.addWidget(self.lb_i)
        layout.addLayout(hc)

        self.setCentralWidget(central)
        self.setFixedWidth(340)

        # ── 信号 ──
        self.btn_map.clicked.connect(self._open_map)
        self.btn_xml.clicked.connect(self._open_xml)
        self.btn_vid.clicked.connect(self._open_vid)
        self.btn_pp.clicked.connect(self._toggle_play)
        self.btn_rs.clicked.connect(self._reset)
        self.btn_col.clicked.connect(self._pick_col)
        self.sld.sliderMoved.connect(self._seek)
        self.cb_src_bg.currentTextChanged.connect(self._on_src_changed)
        self.cb_src_circ.currentTextChanged.connect(self._on_src_changed)
        self.cb_src_rect.currentTextChanged.connect(self._on_src_changed)

        # 窗口尺寸 spinbox → 同步 cv2 窗口大小
        self.sp_dw.valueChanged.connect(self._on_display_size)
        self.sp_dh.valueChanged.connect(self._on_display_size)

        redraw = [
            self.ox, self.oy, self.chk_bg, self.chk_circ, self.chk_rect,
            self.chk_darken, self.sp_lerp, self.sp_margin, self.sp_ttw,
            self.sp_lw, self.sp_ps, self.sp_panel, self.sp_circ_zoom,
            self.sp_fm, self.sp_rr, self.sp_bd, self.sp_voff,
            self.cb_spd, self.sp_fps,
        ]
        for w in redraw:
            if hasattr(w, 'valueChanged'):
                w.valueChanged.connect(self._on_param)
            if hasattr(w, 'stateChanged'):
                w.stateChanged.connect(self._on_param)
            if hasattr(w, 'currentTextChanged'):
                w.currentTextChanged.connect(self._on_param)

    # ── 辅助 ──

    def _upd_col(self):
        c = self.tcolor
        self.btn_col.setStyleSheet(
            f"background:{c.name()}; color:{'#000' if c.lightness() > 128 else '#fff'};"
            f"font-weight:bold; padding:4px; border:1px solid #888;")
        self.btn_col.setText(f"  {c.name()}  ")

    def _pick_col(self):
        c = QColorDialog.getColor(self.tcolor, self, "选择轨迹颜色")
        if c.isValid():
            self.tcolor = c
            self._upd_col()
            self._on_param()

    def _on_src_changed(self, _=None):
        self._reset()
        self._on_param()

    def _on_param(self):
        if not self.playing:
            self._render_and_show()

    def _on_display_size(self):
        """用户在 spinbox 改窗口尺寸 → 同步 cv2 窗口"""
        self._render_w = self.sp_dw.value()
        self._render_h = self.sp_dh.value()
        self.buf.reset()
        if self._win_created:
            try:
                cv2.resizeWindow(WIN, self._render_w, self._render_h)
            except Exception:
                pass
        if not self.playing:
            self._render_and_show()

    def _spd(self):
        return float(self.cb_spd.currentText().replace('x', ''))

    @staticmethod
    def _fmt(s):
        return f"{s // 3600:02}:{(s % 3600) // 60:02}:{s % 60:02}"

    # ── 核心定时器 ──

    def _tick(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            if self.playing:
                self._toggle_play()
            return
        if key == ord(' '):
            self._toggle_play()
            return

        if self._win_created and not self._window_alive():
            if self.playing:
                self._toggle_play()
            return

        # 检测窗口拖拽尺寸变化
        size_changed = self._sync_window_size()

        if self.playing and self.td:
            now = time.monotonic()
            elapsed = now - self._last_time
            self._last_time = now
            spd = max(0.01, self._spd())
            if len(self.td) > 1:
                dt = self.td[1].elapsedTimeFromStart - self.td[0].elapsedTimeFromStart
                rate = (1.0 / dt) if dt > 0 else 30.0
            else:
                rate = 30.0
            self.fi += elapsed * spd * rate
            if self.fi >= len(self.td) - 1:
                self.fi = float(len(self.td) - 1)
                self._toggle_play()
                self._render_and_show()
                self._upd_time()
                return
            self._render_and_show()
            self._upd_time()
        elif size_changed and self.td:
            # 窗口拖拽但未播放 → 仍需重绘以适应新尺寸
            self._render_and_show()

    # ── 渲染管线 ──

    def _get_video_frame(self):
        if not self.vs.ok:
            return None
        if not self.td:
            return self.vs.frame
        i1 = int(self.fi)
        i2 = min(i1 + 1, len(self.td) - 1)
        frac = self.fi - i1
        tt = (self.td[i1].elapsedTimeFromStart * (1 - frac) +
              self.td[i2].elapsedTimeFromStart * frac)
        vt = tt + self.sp_voff.value()
        diff = vt - self.vs.pos
        fd = 1.0 / self.vs.fps
        if 0 <= diff < fd * 3:
            f = self.vs.next()
            return f if f is not None else self.vs.frame
        return self.vs.seek(max(0, vt))

    # ═══════════════════ 视图渲染（加 zoom 处理视频）═══════════════════
    def _render_vp(self, engine, source, ix, iy, idir, trail, laps,
                   size, ox, oy, ps, lw, lc, lerp, margin, vf,
                   scale_mult=1.0):
        """统一渲染入口，map 源通过 scale_mult 缩放，Video 源通过裁剪/缩放"""
        tw, th = size
        if source == "Video":
            if vf is None:
                return np.zeros((th, tw, 3), np.uint8)
            return _zoom_video(vf, tw, th, scale_mult)
        return engine.render(ix, iy, idir, trail, laps,
                             source, size, ox, oy, ps, lw, lc, lerp, margin,
                             scale_mult=scale_mult)

    def _zoom_video(frame, tw, th, zoom):
        """视频帧缩放：zoom>1 放大中心, zoom<1 缩小看更多"""
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return np.zeros((th, tw, 3), np.uint8)

        if zoom > 1.0:
            # 放大：从原图中心裁剪 (1/zoom) 区域，再放大到目标尺寸
            cw = max(1, int(round(w / zoom)))
            ch = max(1, int(round(h / zoom)))
            x0 = max(0, (w - cw) // 2)
            y0 = max(0, (h - ch) // 2)
            cropped = frame[y0:y0 + ch, x0:x0 + cw]
            return cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_LINEAR)
        elif zoom < 1.0:
            # 缩小：缩小到 zoom 比例，居中放在画布上
            sw = max(1, int(round(tw * zoom)))
            sh = max(1, int(round(th * zoom)))
            scaled = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((th, tw, 3), np.uint8)
            yo = (th - sh) // 2
            xo = (tw - sw) // 2
            canvas[yo:yo + sh, xo:xo + sw] = scaled
            return canvas
        else:
            return resize_fit(frame, tw, th)

    def _render_and_show(self):
        if not self.td or self._rendering:
            return
        self._rendering = True
        try:
            self._ensure_window()

            W, H = self._render_w, self._render_h

            ix, iy, idir = RenderEngine.interp(self.td, self.fi)
            idx = int(self.fi)
            p = self.td[idx]
            tw = self.sp_ttw.value()
            trail = [t for t in self.td
                     if p.elapsedTimeFromStart - tw <= t.elapsedTimeFromStart <= p.elapsedTimeFromStart]
            laps = [t for t in self.td if t.lapNumber == p.lapNumber]

            c = self.tcolor
            lc = (c.blue(), c.green(), c.red())
            ox, oy = self.ox.value(), self.oy.value()
            lerp = self.sp_lerp.value()
            margin = self.sp_margin.value()
            ps = self.sp_ps.value()
            lw = self.sp_lw.value()
            panel_r = self.sp_panel.value()
            circ_zoom = self.sp_circ_zoom.value()
            fm = self.sp_fm.value()
            rr_radius = self.sp_rr.value()
            bd = self.sp_bd.value()

            src_bg = self.cb_src_bg.currentText()
            src_circ = self.cb_src_circ.currentText()
            src_rect = self.cb_src_rect.currentText()
            show_bg = self.chk_bg.isChecked()
            show_circ = self.chk_circ.isChecked()
            show_rect = self.chk_rect.isChecked()

            vf = self._get_video_frame()

            if show_bg:
                if src_bg == "Video":
                    bg = fill_cover(vf, W, H) if vf is not None \
                        else np.full((H, W, 3), (18, 18, 18), np.uint8)
                else:
                    bg = self.e_bg.render(ix, iy, idir, trail, laps,
                                          src_bg, (W, H), ox, oy, ps, lw, lc, lerp, margin)
            else:
                bg = np.full((H, W, 3), (18, 18, 18), np.uint8)

            # ── 布局 ──
            pw = max(60, int(W * panel_r))
            ph = max(60, H - fm * 2)
            viewport_w = max(20, pw - fm * 2)

            # 圆形：显示尺寸 = 渲染尺寸 = viewport_w，缩放由 scale_mult 控制
            circ_size = viewport_w

            # 矩形
            remaining = ph - fm
            if show_circ:
                remaining -= circ_size + fm
            rect_h = max(20, remaining) if show_rect else 0

            # ── 渲染（始终用显示尺寸，zoom 通过 scale_mult 传递）──
            circle_frame = None
            if show_circ:
                circle_frame = self._render_vp(
                    self.e_circ, src_circ, ix, iy, idir, trail, laps,
                    (circ_size, circ_size), ox, oy, ps, lw, lc, lerp, margin, vf,
                    scale_mult=circ_zoom)

            rect_frame = None
            if show_rect:
                rect_frame = self._render_vp(
                    self.e_rect, src_rect, ix, iy, idir, trail, laps,
                    (viewport_w, rect_h), ox, oy, ps, lw, lc, lerp, margin, vf,
                    scale_mult=1.0)

            # ── 合成 ──
            need_darken = self.chk_darken.isChecked() and src_bg == "Video" and show_bg
            out = compose(self.buf, bg, circle_frame, rect_frame,
                          W, H, panel_r, circ_size, fm, bd, need_darken,
                          show_circ, show_rect, rr_radius)

            cv2.imshow(WIN, out)
        finally:
            self._rendering = False

    def _upd_time(self):
        if not self.td:
            return
        idx = min(int(self.fi), len(self.td) - 1)
        ct = int(self.td[idx].elapsedTimeFromStart)
        tt = int(self.td[-1].elapsedTimeFromStart)
        self.lb_t.setText(f"{self._fmt(ct)} / {self._fmt(tt)}")
        parts = [f"帧 {idx}/{len(self.td) - 1}"]
        if self.vs.ok:
            parts.append(f"V {self.vs.pos:.1f}s/{self.vs.duration:.0f}s")
        self.lb_i.setText(" | ".join(parts))

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self._last_time = time.monotonic()

    def _reset(self):
        for e in self.engines:
            e.reset()

    def _seek(self, pos):
        self.fi = float(pos)
        self._render_and_show()
        self._upd_time()

    # ── 文件 ──

    def _open_map(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择底图", "", "Images (*.png *.jpg *.bmp)")
        if p:
            for e in self.engines:
                e.load_map(p)
            self._on_param()

    def _open_xml(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择轨迹 XML", "", "XML (*.xml)")
        if p:
            self.td = parse_track_points(p)
            if not self.td:
                return
            self.sld.setEnabled(True)
            self.sld.setRange(0, len(self.td) - 1)
            self.fi = 0.0
            self._reset()
            self._render_and_show()
            self._upd_time()

    def _open_vid(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "Videos (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;All (*)")
        if p and self.vs.load(p):
            v = self.vs
            self.lb_vi.setText(
                f"{v.width}×{v.height} | {v.fps:.1f}FPS | {v.duration:.1f}s")
            self._on_param()
        elif p:
            self.lb_vi.setText("加载失败!")

    def closeEvent(self, e):
        cv2.destroyAllWindows()
        self.vs.release()
        self.buf.reset()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoPlayer()
    win.show()
    sys.exit(app.exec())
