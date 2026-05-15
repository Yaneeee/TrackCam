import sys
import time
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
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
                float(a.get('speed', 0)),
                float(a.get('elapsedTimeFromStart', 0)),
                int(a.get('lapNumber', 0))))
        pts.sort(key=lambda p: p.elapsedTimeFromStart)
        return pts
    except Exception as e:
        print(f"XML 解析出错: {e}")
        return []


def get_lap_info(pts: List[TrackPoint]):
    if not pts:
        return 0, 0, 0
    laps = {p.lapNumber for p in pts}
    mn, mx = min(laps), max(laps)
    return mn, mx, mx - mn + 1


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


# ═══════════════════ 帧缓冲 ═══════════════════

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


# ═══════════════════ 蒙版 ═══════════════════

_mc = {}


def _mask(h, w):
    k = ('c', h, w)
    if k not in _mc:
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (w // 2, h // 2), min(w, h) // 2, 255, -1,
                   cv2.LINE_AA)
        _mc[k] = m
    return _mc[k]


def _ring(h, w, t=3):
    k = ('cr', h, w, t)
    if k not in _mc:
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (w // 2, h // 2), min(w, h) // 2, 255, t,
                   cv2.LINE_AA)
        _mc[k] = m
    return _mc[k]


def overlay_circle(canvas, patch, cx, cy, ds,
                   border_color=(255, 255, 255), bw=3):
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
    rg = _ring(ds, ds, bw)[py1:py2, px1:px2]
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
            for cx, cy in [(r, r), (w - 1 - r, r),
                           (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
                cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
        _mc[k] = m
    return _mc[k]


def _draw_rr_border(canvas, x, y, w, h, r, color, thickness):
    r = min(r, w // 2, h // 2)
    if r <= 0:
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1),
                      color, thickness)
        return
    cv2.line(canvas, (x + r, y), (x + w - 1 - r, y), color, thickness,
             cv2.LINE_AA)
    cv2.line(canvas, (x + r, y + h - 1), (x + w - 1 - r, y + h - 1),
             color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x, y + r), (x, y + h - 1 - r), color, thickness,
             cv2.LINE_AA)
    cv2.line(canvas, (x + w - 1, y + r), (x + w - 1, y + h - 1 - r),
             color, thickness, cv2.LINE_AA)
    for cx2, cy2, a0 in [
        (x + r, y + r, 180), (x + w - 1 - r, y + r, 270),
        (x + r, y + h - 1 - r, 90),
        (x + w - 1 - r, y + h - 1 - r, 0)
    ]:
        cv2.ellipse(canvas, (cx2, cy2), (r, r), a0, 0, 90, color,
                    thickness, cv2.LINE_AA)


def overlay_rounded_rect(canvas, patch, cx, cy, radius,
                         border_color=(160, 160, 160), bw=2):
    ph, pw = patch.shape[:2]
    ch, cw = canvas.shape[:2]
    x1, y1 = max(0, cx), max(0, cy)
    x2, y2 = min(cw, cx + pw), min(ch, cy + ph)
    if x2 <= x1 or y2 <= y1:
        return
    px1, py1 = x1 - cx, y1 - cy
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    ms = _rr_mask(ph, pw, radius)[py1:py2, px1:px2]
    roi = patch[py1:py2, px1:px2].copy()
    m3 = cv2.merge([ms, ms, ms])
    np.copyto(canvas[y1:y2, x1:x2], roi, where=m3.astype(bool))
    _draw_rr_border(canvas, cx, cy, pw, ph, radius, border_color, bw)


# ═══════════════════ 工具 ═══════════════════

def resize_fit(frame, tw, th):
    if frame is None:
        return np.zeros((th, tw, 3), np.uint8)
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((th, tw, 3), np.uint8)
    s = min(tw / w, th / h)
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    c = np.zeros((th, tw, 3), np.uint8)
    yo, xo = (th - nh) // 2, (tw - nw) // 2
    if yo >= 0 and xo >= 0 and yo + nh <= th and xo + nw <= tw:
        c[yo:yo + nh, xo:xo + nw] = r
    return c


def fill_cover(frame, tw, th):
    if frame is None:
        return np.zeros((th, tw, 3), np.uint8)
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((th, tw, 3), np.uint8)
    s = max(tw / w, th / h)
    nw = max(tw, int(round(w * s)))
    nh = max(th, int(round(h * s)))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    yo = max(0, (nh - th) // 2)
    xo = max(0, (nw - tw) // 2)
    return r[yo:yo + th, xo:xo + tw].copy()


def safe_imread(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(path)


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
        tw, th = size
        cx, cy = ix + ox, iy + oy
        if mode == "LapView" and len(laps) > 1:
            ap = np.array([[p.imageX + ox, p.imageY + oy] for p in laps],
                          np.float32)
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
        ts *= scale_mult
        if self._sc is None:
            self._sc = (tc.copy() if isinstance(tc, np.ndarray)
                        else np.array(tc))
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
        if self.map_img is not None:
            fr = cv2.warpAffine(self.map_img, M, (tw, th),
                                flags=cv2.INTER_LINEAR)
        else:
            fr = np.zeros((th, tw, 3), np.uint8)
        if len(trail) > 1:
            d = np.array([[p.imageX + ox, p.imageY + oy] for p in trail],
                         np.float32)
            t = cv2.transform(d.reshape(-1, 1, 2), M)
            cv2.polylines(fr, [t.astype(np.int32)], False, lc, lw,
                          cv2.LINE_AA)
        tp = cv2.transform(np.array([[[cx, cy]]], np.float32), M)[0][0]
        cv2.circle(fr, (int(tp[0]), int(tp[1])), ps + 2, (255, 255, 255), -1)
        cv2.circle(fr, (int(tp[0]), int(tp[1])), ps, (0, 0, 255), -1)
        return fr


# ═══════════════════ 合成 ═══════════════════

SOURCES = ["Video", "HeadingUp", "NorthUp", "LapView", "LapSegment"]


def compose(buf, bg, circle, rect, segment,
            W, H, panel_r, circ_display, margin, border_w,
            darken, show_circle, show_rect, show_segment, rr_radius):
    out = buf.ensure(H, W)
    if darken:
        z = buf.zeros((H, W, 3))
        cv2.addWeighted(bg, 0.72, z, 0.28, 0, dst=out)
    else:
        np.copyto(out, bg)
    if not (show_circle or show_rect or show_segment):
        return out
    pw = max(60, int(W * panel_r))
    px = margin // 2
    py = margin // 2
    y_cur = py + margin
    if show_circle and circle is not None:
        cx2 = px + (pw - circ_display) // 2
        if y_cur + circ_display <= H and cx2 + circ_display <= W:
            overlay_circle(out, circle, cx2, y_cur, circ_display)
        y_cur += circ_display + margin
    if show_rect and rect is not None:
        rh, rw = rect.shape[:2]
        rx = px + (pw - rw) // 2
        if y_cur + rh <= H and rx + rw <= W:
            overlay_rounded_rect(out, rect, rx, y_cur, rr_radius,
                                 (160, 160, 160), border_w)
        y_cur += rh + margin
    if show_segment and segment is not None:
        sh2, sw2 = segment.shape[:2]
        sx = px + (pw - sw2) // 2
        if y_cur + sh2 <= H and sx + sw2 <= W:
            overlay_rounded_rect(out, segment, sx, y_cur, rr_radius,
                                 (100, 200, 100), border_w)
    return out


# ═══════════════════ 分段 ═══════════════════

def compute_segments(img_h, roi, n):
    rx, ry, rw, rh = roi
    a = rh / (n + 1)
    segs = []
    for i in range(n):
        sy = int(round(ry + i * a))
        sh = int(round(2 * a))
        sh = min(sh, img_h - sy)
        if sh > 0:
            segs.append((i, rx, sy, rw, sh))
    return segs


def select_roi_on_image(img):
    h, w = img.shape[:2]
    scale = min(40, max(10, int(4000 / max(w, h) * 100)))
    nw = max(1, int(w * scale / 100))
    nh = max(1, int(h * scale / 100))
    scaled = cv2.resize(img, (nw, nh))
    roi = cv2.selectROI("拉选区域 → ENTER 确认", scaled,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("拉选区域 → ENTER 确认")
    if roi == (0, 0, 0, 0):
        return None
    f = 100.0 / scale
    rx = max(0, int(roi[0] * f))
    ry = max(0, int(roi[1] * f))
    rw = max(1, min(int(roi[2] * f), w - rx))
    rh = max(1, min(int(roi[3] * f), h - ry))
    return (rx, ry, rw, rh)


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
        self._min_lap = 0
        self._max_lap = 0
        self._num_laps = 0
        self._seg_images: Dict[int, np.ndarray] = {}

        # 导出状态
        self._exporting = False
        self._export_writer: Optional[cv2.VideoWriter] = None
        self._export_idx = 0
        self._export_total = 0

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

    def _window_alive(self):
        try:
            return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1
        except Exception:
            return False

    def _sync_window_size(self):
        if not self._win_created:
            return False
        try:
            _, _, w, h = cv2.getWindowImageRect(WIN)
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

        # 文件
        self.btn_map = QPushButton("加载地图")
        self.btn_xml = QPushButton("加载轨迹 XML")
        self.btn_vid = QPushButton("加载视频")
        grp("文件加载", [self.btn_map, self.btn_xml, self.btn_vid])

        # 偏移
        self.ox = QDoubleSpinBox()
        self.ox.setRange(-99999, 99999)
        self.ox.setDecimals(1)
        self.oy = QDoubleSpinBox()
        self.oy.setRange(-99999, 99999)
        self.oy.setDecimals(1)
        grp("地图偏移", [("X:", self.ox), ("Y:", self.oy)])

        # 播放
        self.sp_fps = QSpinBox()
        self.sp_fps.setRange(10, 240)
        self.sp_fps.setValue(60)
        self.sp_fps.setSuffix(" FPS")
        self.cb_spd = QComboBox()
        self.cb_spd.addItems(
            ["0.25x", "0.5x", "1x", "2x", "4x", "8x", "16x"])
        self.cb_spd.setCurrentText("1x")
        grp("播放控制", [("帧率:", self.sp_fps), ("倍速:", self.cb_spd)])

        # 四视窗
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
        self.chk_seg = QCheckBox("显示")
        self.chk_seg.setChecked(True)
        self.cb_src_seg = QComboBox()
        self.cb_src_seg.addItems(SOURCES)
        self.cb_src_seg.setCurrentText("LapSegment")

        gv = QGroupBox("视窗源选择")
        fv = QFormLayout()
        fv.setSpacing(4)
        for label, chk, cb in [
            ("背景:", self.chk_bg, self.cb_src_bg),
            ("圆形:", self.chk_circ, self.cb_src_circ),
            ("矩形:", self.chk_rect, self.cb_src_rect),
            ("分段:", self.chk_seg, self.cb_src_seg),
        ]:
            row = QHBoxLayout()
            row.addWidget(chk)
            row.addWidget(cb)
            cont = QWidget()
            cont.setLayout(row)
            fv.addRow(label, cont)
        gv.setLayout(fv)
        sl.addWidget(gv)

        # 平滑
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
        grp("平滑参数", [("插值系数:", self.sp_lerp),
                         ("LapView边距:", self.sp_margin)])

        # 轨迹
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
        grp("轨迹显示", [("回看:", self.sp_ttw), ("线宽:", self.sp_lw),
                         ("点大小:", self.sp_ps), ("颜色:", self.btn_col)])

        # 浮窗
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
            ("面板宽度比:", self.sp_panel),
            ("圆形缩放:", self.sp_circ_zoom),
            ("内边距:", self.sp_fm),
            ("圆角半径:", self.sp_rr),
            ("矩形边框:", self.sp_bd),
        ])

        # 视频
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

        # 显示
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

        # 分段
        self.btn_seg_img = QPushButton("加载图片 → 选区分段")
        self.btn_seg_frame = QPushButton("当前帧 → 选区分段")
        self.sp_seg_pad = QSpinBox()
        self.sp_seg_pad.setRange(0, 100)
        self.sp_seg_pad.setValue(4)
        self.sp_seg_pad.setSuffix(" px")
        self.lb_seg = QLabel("请先加载 XML 确定圈数")
        self.lb_seg_detail = QLabel("")
        ge = QGroupBox("分段截取 (自动匹配 XML 圈数)")
        fe = QFormLayout()
        fe.setSpacing(4)
        fe.addRow(self.btn_seg_img)
        fe.addRow(self.btn_seg_frame)
        fe.addRow("内边距:", self.sp_seg_pad)
        fe.addRow("状态:", self.lb_seg)
        fe.addRow("详情:", self.lb_seg_detail)
        ge.setLayout(fe)
        sl.addWidget(ge)

        # ── 导出视频 ──
        self.cb_export_fmt = QComboBox()
        self.cb_export_fmt.addItems([
            "AVI (无压缩)", "AVI (MJPG)", "MP4 (mp4v)", "MP4 (H264)"
        ])
        self.sp_export_fps = QSpinBox()
        self.sp_export_fps.setRange(10, 120)
        self.sp_export_fps.setValue(60)
        self.sp_export_fps.setSuffix(" FPS")
        self.sp_export_batch = QSpinBox()
        self.sp_export_batch.setRange(1, 30)
        self.sp_export_batch.setValue(5)
        self.sp_export_batch.setSuffix(" 帧/tick")
        self.chk_export_no_dark = QCheckBox("导出时关闭暗化")
        self.chk_export_no_dark.setChecked(True)  # ← 默认关闭暗化
        self.btn_export_start = QPushButton("导出视频")
        self.btn_export_cancel = QPushButton("取消导出")
        self.btn_export_cancel.setEnabled(False)
        self.lb_export = QLabel("")
        self.sld_export = QSlider(Qt.Horizontal)
        self.sld_export.setEnabled(False)

        ge2 = QGroupBox("导出视频")
        fe2 = QFormLayout()
        fe2.setSpacing(4)
        fe2.addRow("格式:", self.cb_export_fmt)
        fe2.addRow("导出帧率:", self.sp_export_fps)
        fe2.addRow("批量帧数:", self.sp_export_batch)
        fe2.addRow(self.chk_export_no_dark)  # ← 新增
        row_exp = QHBoxLayout()
        row_exp.addWidget(self.btn_export_start)
        row_exp.addWidget(self.btn_export_cancel)
        exp_cont = QWidget()
        exp_cont.setLayout(row_exp)
        fe2.addRow(exp_cont)
        fe2.addRow(self.sld_export)
        fe2.addRow("状态:", self.lb_export)
        ge2.setLayout(fe2)
        sl.addWidget(ge2)

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

        # 信号
        self.btn_map.clicked.connect(self._open_map)
        self.btn_xml.clicked.connect(self._open_xml)
        self.btn_vid.clicked.connect(self._open_vid)
        self.btn_pp.clicked.connect(self._toggle_play)
        self.btn_rs.clicked.connect(self._reset)
        self.btn_col.clicked.connect(self._pick_col)
        self.sld.sliderMoved.connect(self._seek)
        self.sp_dw.valueChanged.connect(self._on_display_size)
        self.sp_dh.valueChanged.connect(self._on_display_size)
        self.btn_seg_img.clicked.connect(self._seg_from_image)
        self.btn_seg_frame.clicked.connect(self._seg_from_frame)
        self.btn_export_start.clicked.connect(self._export_start)
        self.btn_export_cancel.clicked.connect(self._export_stop)

        for cb in [self.cb_src_bg, self.cb_src_circ, self.cb_src_rect,
                   self.cb_src_seg]:
            cb.currentTextChanged.connect(self._on_src_changed)

        redraw = [
            self.ox, self.oy, self.chk_bg, self.chk_circ, self.chk_rect,
            self.chk_seg, self.chk_darken, self.sp_lerp, self.sp_margin,
            self.sp_ttw, self.sp_lw, self.sp_ps, self.sp_panel,
            self.sp_circ_zoom, self.sp_fm, self.sp_rr, self.sp_bd,
            self.sp_voff, self.cb_spd, self.sp_fps, self.sp_seg_pad,
        ]
        for w in redraw:
            if hasattr(w, 'valueChanged'):
                w.valueChanged.connect(self._on_param)
            if hasattr(w, 'stateChanged'):
                w.stateChanged.connect(self._on_param)
            if hasattr(w, 'currentTextChanged'):
                w.currentTextChanged.connect(self._on_param)

    # ── 分段 ──

    def _seg_from_image(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "Images (*.png *.jpg *.bmp *.tiff);;All (*)")
        if not p:
            return
        img = safe_imread(p)
        if img is None:
            self.lb_seg.setText("读取失败")
            return
        self._do_segment(img)

    def _seg_from_frame(self):
        img = self._get_base_img()
        if img is None:
            self.lb_seg.setText("无可用画面")
            return
        self._do_segment(img)

    def _get_base_img(self):
        vf = self._get_video_frame() if self.vs.ok else None
        if vf is not None:
            return vf.copy()
        if self.e_bg.map_img is not None:
            return self.e_bg.map_img.copy()
        return None

    def _do_segment(self, img):
        if self._num_laps <= 0:
            self.lb_seg.setText("请先加载 XML！")
            return
        n = self._num_laps
        start_lap = self._min_lap
        self.lb_seg.setText(f"选区中... ({n}段, Lap {start_lap}~{self._max_lap})")
        QApplication.processEvents()
        roi = select_roi_on_image(img)
        if roi is None:
            self.lb_seg.setText("已取消")
            return
        h, w = img.shape[:2]
        segs = compute_segments(h, roi, n)
        self._seg_images.clear()
        preview = img.copy()
        colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (200, 0, 255),
                  (255, 255, 0), (0, 255, 255), (128, 255, 0), (255, 128, 0),
                  (128, 0, 255), (0, 128, 255)]
        rx, ry, rw, rh = roi
        a = rh / (n + 1)
        print(f"\n{'=' * 50}")
        print(f"ROI: ({rx},{ry},{rw},{rh})  圈数:{n}  a={a:.1f}px  段高={2 * a:.1f}px")
        print(f"{'=' * 50}")
        for idx, sx, sy, sw, sh in segs:
            lap = start_lap + idx
            seg_img = img[sy:sy + sh, sx:sx + sw].copy()
            self._seg_images[lap] = seg_img
            c = colors[idx % len(colors)]
            cv2.rectangle(preview, (sx, sy), (sx + sw, sy + sh), c, 2)
            cv2.putText(preview, f"L{lap}", (sx + 4, sy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
            print(f"  段{idx:2d} → Lap {lap:3d}  y={sy:5d} h={sh:5d}")
        ph2, pw2 = preview.shape[:2]
        ps2 = min(960 / pw2, 720 / ph2, 1.0)
        prev = cv2.resize(preview, (max(1, int(pw2 * ps2)), max(1, int(ph2 * ps2))))
        cv2.imshow("分段预览 (3秒)", prev)
        cv2.waitKey(3000)
        cv2.destroyWindow("分段预览 (3秒)")
        self.lb_seg.setText(f"完成: {n}段 → Lap {start_lap}~{self._max_lap}")
        self.lb_seg_detail.setText(f"a={a:.0f}px 段高={2 * a:.0f}px ROI=({rw}×{rh})")

    def _get_seg_for_lap(self, lap):
        return self._seg_images.get(lap)

    def _time_to_fi(self, time_sec: float) -> float:
        if len(self.td) < 2:
            return 0.0
        lo, hi = 0, len(self.td) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.td[mid].elapsedTimeFromStart <= time_sec:
                lo = mid
            else:
                hi = mid
        t1 = self.td[lo].elapsedTimeFromStart
        t2 = self.td[hi].elapsedTimeFromStart
        frac = (time_sec - t1) / (t2 - t1) if t2 > t1 else 0.0
        return lo + max(0.0, min(1.0, frac))

    # ── 修改：导出开始 ──

    def _export_start(self):
        if not self.td:
            self.lb_export.setText("无轨迹数据")
            return
        if self._exporting:
            self.lb_export.setText("正在导出中...")
            return

        fmt_text = self.cb_export_fmt.currentText()
        if "MP4" in fmt_text:
            ext = "MP4 Files (*.mp4)"
        else:
            ext = "AVI Files (*.avi)"
        path, _ = QFileDialog.getSaveFileName(self, "保存视频", "", ext)
        if not path:
            return

        W, H = self._render_w, self._render_h
        out_fps = self.sp_export_fps.value()

        # ★ 根据格式选择编码器
        if "无压缩" in fmt_text:
            # RAW：无损，色彩完全保真，文件较大
            fourcc = 0  # cv2.VideoWriter_fourcc 按平台自动
            if path.lower().endswith(('.mp4', '.avi')):
                if not path.lower().endswith('.avi'):
                    path = path.rsplit('.', 1)[0] + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'DIB ')  # 无压缩 RGB
        elif "H264" in fmt_text:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            if not path.lower().endswith('.mp4'):
                path = path.rsplit('.', 1)[0] + '.mp4'
        elif "mp4v" in fmt_text:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if not path.lower().endswith('.mp4'):
                path = path.rsplit('.', 1)[0] + '.mp4'
        else:  # MJPG
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            if not path.lower().endswith('.avi'):
                path = path.rsplit('.', 1)[0] + '.avi'

        self._export_writer = cv2.VideoWriter(path, fourcc, out_fps, (W, H))
        if not self._export_writer.isOpened():
            # H264 不可用时 fallback
            if "H264" in fmt_text:
                self._export_writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (W, H))
            if not self._export_writer or not self._export_writer.isOpened():
                self.lb_export.setText("创建失败!")
                self._export_writer = None
                return

        total_duration = self.td[-1].elapsedTimeFromStart
        self._export_total = max(1, int(total_duration * out_fps))
        self._export_out_fps = out_fps

        self._exporting = True
        self._export_idx = 0

        if self.playing:
            self._toggle_play()

        for e in self.engines:
            e.reset()

        self.btn_export_start.setEnabled(False)
        self.btn_export_cancel.setEnabled(True)
        self.sld_export.setEnabled(True)
        self.sld_export.setRange(0, self._export_total - 1)
        self.lb_export.setText(
            f"导出中 0/{self._export_total} "
            f"({total_duration:.1f}s)")

        self._export_timer = QTimer()
        self._export_timer.setInterval(1)
        self._export_timer.timeout.connect(self._export_tick)
        self._export_timer.start()

        print(f"[导出] {path}  {W}×{H}  {out_fps}FPS  "
              f"时长{total_duration:.1f}s  总帧{self._export_total}  "
              f"格式:{fmt_text}  暗化:{'关' if self.chk_export_no_dark.isChecked() else '开'}")

    # ── 修改：导出单步 ──

    def _export_tick(self):
        if not self._exporting or self._export_writer is None:
            self._export_stop()
            return

        batch = self.sp_export_batch.value()
        no_dark = self.chk_export_no_dark.isChecked()   # ★

        for _ in range(batch):
            if self._export_idx >= self._export_total:
                self._export_stop()
                return

            time_sec = self._export_idx / self._export_out_fps
            self.fi = self._time_to_fi(time_sec)

            frame = self._build_frame(no_dark=no_dark)  # ★
            if frame is not None:
                self._export_writer.write(frame)

            self._export_idx += 1

        pct = self._export_idx / self._export_total * 100
        elapsed_sec = self._export_idx / self._export_out_fps
        self.lb_export.setText(
            f"导出中 {self._export_idx}/{self._export_total} "
            f"({pct:.0f}%) {elapsed_sec:.1f}s")
        self.sld_export.setValue(self._export_idx)

        if self._export_idx % 5 == 0 and frame is not None:
            cv2.imshow(WIN, frame)
            cv2.waitKey(1)

    def _export_stop(self):
        if hasattr(self, '_export_timer') and self._export_timer.isActive():
            self._export_timer.stop()
        if self._export_writer is not None:
            self._export_writer.release()
            self._export_writer = None

        total = self._export_total
        done = self._export_idx
        self._exporting = False
        self._export_idx = 0
        self._export_total = 0

        self.btn_export_start.setEnabled(True)
        self.btn_export_cancel.setEnabled(False)
        self.sld_export.setEnabled(False)
        self.sld_export.setValue(0)

        if done >= total:
            duration = done / max(1, self.sp_export_fps.value())
            self.lb_export.setText(f"完成! {done}帧 ({duration:.1f}s)")
            print(f"[导出] 完成 {done}帧 ({duration:.1f}s)")
        else:
            self.lb_export.setText(f"已取消 ({done}/{total})")
            print(f"[导出] 取消 {done}/{total}")

    # ── 辅助 ──

    def _upd_col(self):
        c = self.tcolor
        self.btn_col.setStyleSheet(
            f"background:{c.name()}; "
            f"color:{'#000' if c.lightness() > 128 else '#fff'};"
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
        if not self.playing and not self._exporting:
            self._render_and_show()

    def _on_display_size(self):
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

    # ── 定时器 ──

    def _tick(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            if self._exporting:
                self._export_stop()
            elif self.playing:
                self._toggle_play()
            return
        if key == ord(' '):
            if not self._exporting:
                self._toggle_play()
            return
        if self._win_created and not self._window_alive():
            if self.playing:
                self._toggle_play()
            return

        # 导出期间跳过播放逻辑
        if self._exporting:
            return

        sz = self._sync_window_size()
        if self.playing and self.td:
            now = time.monotonic()
            elapsed = now - self._last_time
            self._last_time = now
            spd = max(0.01, self._spd())
            if len(self.td) > 1:
                dt = (self.td[1].elapsedTimeFromStart
                      - self.td[0].elapsedTimeFromStart)
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
        elif sz and self.td:
            self._render_and_show()

    # ── 核心渲染 ──

    def _get_video_frame(self):
        if not self.vs.ok:
            return None
        if not self.td:
            return self.vs.frame
        i1 = int(self.fi)
        i2 = min(i1 + 1, len(self.td) - 1)
        frac = self.fi - i1
        tt = (self.td[i1].elapsedTimeFromStart * (1 - frac)
              + self.td[i2].elapsedTimeFromStart * frac)
        vt = tt + self.sp_voff.value()
        diff = vt - self.vs.pos
        fd = 1.0 / self.vs.fps
        if 0 <= diff < fd * 3:
            f = self.vs.next()
            return f if f is not None else self.vs.frame
        return self.vs.seek(max(0, vt))

    def _render_vp(self, engine, source, ix, iy, idir, trail, laps,
                   size, ox, oy, ps, lw, lc, lerp, margin, vf,
                   scale_mult=1.0, current_lap=0):
        tw, th = size
        if source == "LapSegment":
            seg = self._get_seg_for_lap(current_lap)
            if seg is not None:
                return resize_fit(seg, tw, th)
            return np.zeros((th, tw, 3), np.uint8)
        if source == "Video":
            if vf is None:
                return np.zeros((th, tw, 3), np.uint8)
            if scale_mult > 1.0:
                fh, fw = vf.shape[:2]
                cw2 = max(1, int(round(fw / scale_mult)))
                ch2 = max(1, int(round(fh / scale_mult)))
                x0 = max(0, (fw - cw2) // 2)
                y0 = max(0, (fh - ch2) // 2)
                return cv2.resize(
                    vf[y0:y0 + ch2, x0:x0 + cw2],
                    (tw, th), interpolation=cv2.INTER_LINEAR)
            elif scale_mult < 1.0:
                sw2 = max(1, int(round(tw * scale_mult)))
                sh2 = max(1, int(round(th * scale_mult)))
                scaled = cv2.resize(vf, (sw2, sh2),
                                    interpolation=cv2.INTER_LINEAR)
                c = np.zeros((th, tw, 3), np.uint8)
                yo, xo = (th - sh2) // 2, (tw - sw2) // 2
                c[yo:yo + sh2, xo:xo + sw2] = scaled
                return c
            else:
                return resize_fit(vf, tw, th)
        return engine.render(ix, iy, idir, trail, laps, source, size,
                             ox, oy, ps, lw, lc, lerp, margin,
                             scale_mult=scale_mult)

    def _build_frame(self, no_dark=False):
        """渲染当前帧。no_dark=True 时强制关闭暗化（导出用）"""
        if not self.td:
            return None

        W, H = self._render_w, self._render_h
        ix, iy, idir = RenderEngine.interp(self.td, self.fi)
        idx = min(int(self.fi), len(self.td) - 1)
        p = self.td[idx]
        tw_s = self.sp_ttw.value()
        trail = [
            t for t in self.td
            if (p.elapsedTimeFromStart - tw_s
                <= t.elapsedTimeFromStart <= p.elapsedTimeFromStart)
        ]
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
        src_seg = self.cb_src_seg.currentText()
        show_bg = self.chk_bg.isChecked()
        show_circ = self.chk_circ.isChecked()
        show_rect = self.chk_rect.isChecked()
        show_seg = self.chk_seg.isChecked()

        current_lap = p.lapNumber
        vf = self._get_video_frame()

        # 背景
        if show_bg:
            if src_bg == "Video":
                bg = (fill_cover(vf, W, H) if vf is not None
                      else np.full((H, W, 3), (18, 18, 18), np.uint8))
            elif src_bg == "LapSegment":
                seg = self._get_seg_for_lap(current_lap)
                bg = (fill_cover(seg, W, H) if seg is not None
                      else np.full((H, W, 3), (18, 18, 18), np.uint8))
            else:
                bg = self.e_bg.render(
                    ix, iy, idir, trail, laps, src_bg,
                    (W, H), ox, oy, ps, lw, lc, lerp, margin)
        else:
            bg = np.full((H, W, 3), (18, 18, 18), np.uint8)

        # 布局
        pw = max(60, int(W * panel_r))
        ph = max(60, H - fm * 2)
        viewport_w = max(20, pw - fm * 2)
        circ_size = viewport_w
        remaining = ph - fm

        if show_circ:
            remaining -= circ_size + fm

        seg_h = 0
        if show_seg and self._seg_images:
            max_ratio = 0.0
            for seg_img in self._seg_images.values():
                si_h, si_w = seg_img.shape[:2]
                if si_w > 0:
                    max_ratio = max(max_ratio, si_h / si_w)
            if max_ratio > 0:
                seg_h = int(round(viewport_w * max_ratio))
            if seg_h <= 0:
                seg_h = max(20, viewport_w // 3)
            max_seg = remaining - (fm + 20 if show_rect else 0)
            seg_h = min(seg_h, max(20, max_seg))
            remaining -= seg_h + fm

        rect_h = max(20, remaining) if show_rect else 0

        circle_frame = None
        if show_circ:
            circle_frame = self._render_vp(
                self.e_circ, src_circ, ix, iy, idir, trail, laps,
                (circ_size, circ_size), ox, oy, ps, lw, lc, lerp,
                margin, vf, scale_mult=circ_zoom,
                current_lap=current_lap)

        rect_frame = None
        if show_rect:
            rect_frame = self._render_vp(
                self.e_rect, src_rect, ix, iy, idir, trail, laps,
                (viewport_w, rect_h), ox, oy, ps, lw, lc, lerp,
                margin, vf, scale_mult=1.0,
                current_lap=current_lap)

        seg_frame = None
        if show_seg:
            seg_pad = self.sp_seg_pad.value()
            content_w = max(10, viewport_w - seg_pad * 2)
            content_h = max(10, seg_h - seg_pad * 2)
            seg_raw = self._render_vp(
                self.e_rect, src_seg, ix, iy, idir, trail, laps,
                (content_w, content_h), ox, oy, ps, lw, lc, lerp,
                margin, vf, scale_mult=1.0,
                current_lap=current_lap)
            seg_frame = np.zeros((seg_h, viewport_w, 3), np.uint8)
            rh2, rw2 = seg_raw.shape[:2]
            yo = (seg_h - rh2) // 2
            xo = (viewport_w - rw2) // 2
            if (yo >= 0 and xo >= 0
                    and yo + rh2 <= seg_h and xo + rw2 <= viewport_w):
                seg_frame[yo:yo + rh2, xo:xo + rw2] = seg_raw

        # ★ 暗化：导出时可强制关闭
        if no_dark:
            need_darken = False
        else:
            need_darken = (self.chk_darken.isChecked()
                           and src_bg == "Video" and show_bg)

        out = compose(
            self.buf, bg, circle_frame, rect_frame, seg_frame,
            W, H, panel_r, circ_size, fm, bd, need_darken,
            show_circ, show_rect, show_seg, rr_radius)

        return out

    def _render_and_show(self):
        if not self.td or self._rendering:
            return
        self._rendering = True
        try:
            self._ensure_window()
            out = self._build_frame()
            if out is not None:
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
        lap = self.td[idx].lapNumber
        parts = [f"帧 {idx}/{len(self.td) - 1}", f"Lap {lap}"]
        if self._seg_images:
            parts.append(f"分段{'有' if lap in self._seg_images else '无'}")
        if self.vs.ok:
            parts.append(f"V {self.vs.pos:.1f}s")
        self.lb_i.setText(" | ".join(parts))

        # 进度条同步
        self.sld.blockSignals(True)
        self.sld.setValue(idx)
        self.sld.blockSignals(False)

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
        p, _ = QFileDialog.getOpenFileName(
            self, "选择底图", "", "Images (*.png *.jpg *.bmp)")
        if p:
            for e in self.engines:
                e.load_map(p)
            self._on_param()

    def _open_xml(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择轨迹 XML", "", "XML (*.xml)")
        if p:
            self.td = parse_track_points(p)
            if not self.td:
                return
            self.sld.setEnabled(True)
            self.sld.setRange(0, len(self.td) - 1)
            self.fi = 0.0
            self._reset()
            self._min_lap, self._max_lap, self._num_laps = get_lap_info(
                self.td)
            self.lb_seg.setText(
                f"XML: {self._num_laps} 圈 (Lap {self._min_lap}"
                f"~{self._max_lap})")
            self.lb_seg_detail.setText("加载图片后自动按圈数分段")
            self._render_and_show()
            self._upd_time()
            print(f"[XML] {len(self.td)} 点  {self._num_laps} 圈 "
                  f"(Lap {self._min_lap}~{self._max_lap})")

    def _open_vid(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "Videos (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;All (*)")
        if p and self.vs.load(p):
            v = self.vs
            self.lb_vi.setText(
                f"{v.width}×{v.height} | {v.fps:.1f}FPS | "
                f"{v.duration:.1f}s")
            self._on_param()
        elif p:
            self.lb_vi.setText("加载失败!")

    def closeEvent(self, e):
        if self._exporting:
            self._export_stop()
        cv2.destroyAllWindows()
        self.vs.release()
        self.buf.reset()
        super().closeEvent(e)


# ═══════════════════ 入口 ═══════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoPlayer()
    win.show()
    sys.exit(app.exec())
