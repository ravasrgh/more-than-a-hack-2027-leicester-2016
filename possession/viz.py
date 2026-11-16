"""Plot helpers for the explainer notebook: video frames with tracking + labels drawn on top.

NOTE: anything these functions render contains the competition video. Keep the outputs local --
do not commit a notebook with outputs to a public repository (RULES.md, "The data").
"""

import os
import sys

import numpy as np

from .data import ROOT

sys.path.insert(0, os.path.join(ROOT, "starter"))

COLORS = {"H": "#1f6fd1", "A": "#d62728", "D": "#8c8c8c", "X": "#ffffff"}
NAMES = {"H": "H  Barcelona pegang bola", "A": "A  Athletic pegang bola",
         "D": "D  bola mati", "X": "X  tidak dinilai"}
_cap = None
_hom = None


def video_frame(i):
    """RGB image of tactical video frame i."""
    import cv2
    global _cap
    if _cap is None:
        from games import asset_paths
        _cap = cv2.VideoCapture(asset_paths()["video"])
    _cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, img = _cap.read()
    if not ok:
        raise IOError("cannot read frame {}".format(i))
    return img[:, :, ::-1]


def homography():
    global _hom
    if _hom is None:
        import load_homography as lh
        _hom = lh.load()
    return _hom


def project(i, x, y):
    """Normalised pitch coords -> pixels for video frame i."""
    import load_homography as lh
    H, _, valid = homography()
    return lh.project(H[i], x, y)


def _pitch_lines():
    """Pitch markings as polylines in normalised coordinates (105 x 68 m pitch)."""
    L, W = 105.0, 68.0
    lines = [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], [(0.5, 0), (0.5, 1)]]
    for side in (0, 1):
        sx = lambda m: m / L if side == 0 else 1 - m / L  # noqa: E731
        box_y = (W / 2 - 20.16) / W, (W / 2 + 20.16) / W
        six_y = (W / 2 - 9.16) / W, (W / 2 + 9.16) / W
        lines.append([(sx(0), box_y[0]), (sx(16.5), box_y[0]), (sx(16.5), box_y[1]), (sx(0), box_y[1])])
        lines.append([(sx(0), six_y[0]), (sx(5.5), six_y[0]), (sx(5.5), six_y[1]), (sx(0), six_y[1])])
    t = np.linspace(0, 2 * np.pi, 60)
    lines.append(list(zip(0.5 + 9.15 / L * np.cos(t), 0.5 + 9.15 / W * np.sin(t))))
    return lines


def show_frame(ax, m, i, gt=None, pred=None, extra=None, lines=True, crop=None):
    """Video frame i with pitch lines, tracked players (team colour) and ball drawn on top."""
    img = video_frame(i)
    ax.imshow(img)
    if lines:
        for line in _pitch_lines():
            xs, ys = project(i, *np.array(line).T)
            ax.plot(xs, ys, color="white", lw=0.6, alpha=0.5)
    nx, ny = m.px[i] / m.length, m.py[i] / m.width
    for t, c in ((0, COLORS["H"]), (1, COLORS["A"])):
        sel = (m.team == t) & np.isfinite(nx)
        u, v = project(i, nx[sel], ny[sel])
        ax.scatter(u, v, s=55, facecolor="none", edgecolor=c, lw=2)
    if np.isfinite(m.bx[i]):
        u, v = project(i, m.bx[i] / m.length, m.by[i] / m.width)
        ax.scatter([u], [v], s=110, marker="o", facecolor="yellow", edgecolor="black", lw=1.5,
                   zorder=5)
        ball = "bola terlihat"
    else:
        ball = "BOLA TIDAK TERDETEKSI"
    title = "frame {}  (menit {:.1f})  |  {}".format(i, i / 25 / 60, ball)
    ax.set_title(title, fontsize=9)
    txt = []
    if gt is not None:
        txt.append(("label benar : " + NAMES[gt], COLORS[gt]))
    if pred is not None:
        txt.append(("prediksi    : " + NAMES[pred], COLORS[pred]))
    if extra:
        txt.append((extra, "black"))
    for k, (s, c) in enumerate(txt):
        ax.text(0.01, 0.98 - 0.075 * k, s, transform=ax.transAxes, fontsize=9, va="top",
                family="monospace", color="white",
                bbox=dict(facecolor=c if c != "#ffffff" else "black", alpha=0.85, pad=2))
    if crop is not None:
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[3], crop[2])
    ax.axis("off")


def label_strip(ax, labels, start, stop, y=0, h=0.8, name=""):
    """Coloured bar: one colour per label over frames [start, stop)."""
    lab = labels[start:stop]
    edges = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    b = np.concatenate(([0], edges, [len(lab)]))
    for a, e in zip(b[:-1], b[1:]):
        ax.barh(y, (e - a) / 25, left=(start + a - start) / 25, height=h,
                color=COLORS[lab[a]], edgecolor="none")
    ax.text(-1, y, name, ha="right", va="center", fontsize=9)


def draw_pitch(ax, length=105, width=68):
    for line in _pitch_lines():
        xs, ys = np.array(line).T
        ax.plot(xs * length, ys * width, color="#2b6e2b", lw=1)
    ax.set_xlim(-3, length + 3); ax.set_ylim(width + 3, -3); ax.set_aspect("equal")
    ax.set_facecolor("#e9f5e9"); ax.set_xticks([]); ax.set_yticks([])


def make_clip(m, start, stop, gt, pred, path, scale=0.5, fps_out=25, mark=None, panel=None,
              extra_inputs=None):
    """Write an H.264 mp4 of frames [start, stop) with tracking and both labels burned in."""
    import subprocess
    import cv2
    w, h = int(1920 * scale), int(1080 * scale)
    if panel is not None:
        h += panel.shape[0]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", "{}x{}".format(w, h), "-r", str(fps_out), "-i", "-"] + list(extra_inputs or []) +
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", path], stdin=subprocess.PIPE)
    h = int(1080 * scale)
    bgr = {k: tuple(int(c[j:j + 2], 16) for j in (5, 3, 1)) for k, c in COLORS.items()}
    import load_homography as lh
    Hm = homography()[0]
    global _cap
    video_frame(start)  # opens and seeks
    _cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    for i in range(start, stop):
        ok, img = _cap.read()
        if not ok:
            break
        for t, key in ((0, "H"), (1, "A")):
            sel = (m.team == t) & np.isfinite(m.px[i])
            u, v = lh.project(Hm[i], m.px[i][sel] / m.length, m.py[i][sel] / m.width)
            for a, b in zip(u, v):
                cv2.circle(img, (int(a), int(b)), 14, bgr[key], 3)
        if np.isfinite(m.bx[i]):
            u, v = lh.project(Hm[i], m.bx[i] / m.length, m.by[i] / m.width)
            cv2.circle(img, (int(u), int(v)), 10, (0, 255, 255), -1)
            cv2.circle(img, (int(u), int(v)), 10, (0, 0, 0), 2)
        img = cv2.resize(img, (w, h))
        rel = "" if mark is None else "  ({:+.2f}s dari pergantian label)".format((i - mark) / 25)
        rows = [("frame {}{}".format(i, rel), (0, 0, 0)),
                ("BENAR   : " + NAMES[gt[i]], bgr[gt[i]]),
                ("PREDIKSI: " + NAMES[pred[i]], bgr[pred[i]])]
        if not np.isfinite(m.bx[i]):
            rows.append(("bola tidak terdeteksi", (0, 0, 0)))
        for k, (s, c) in enumerate(rows):
            y = 22 + 26 * k
            cv2.rectangle(img, (5, y - 18), (5 + 11 * len(s), y + 6), c, -1)
            cv2.putText(img, s, (9, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
        if panel is not None:
            img = np.vstack([img, _panel_at(panel, i, start, stop, mark)])
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    return path


# ---------------------------------------------------------------------------------------------
# audio: spectrogram panel shown live under the video
# ---------------------------------------------------------------------------------------------

WHISTLE_BAND = (2500, 4500)
SPEC_MAX_HZ = 8000


def audio_segment(start, stop, sr=16000):
    """Mono float audio of video frames [start, stop) of the tactical video."""
    import subprocess
    from games import asset_paths
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(start / 25), "-t", str((stop - start) / 25),
         "-i", asset_paths()["video"], "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32), sr


def build_panel(start, stop, gt, pred, width, spec_h=200):
    """Static parts of the panel: label strips + spectrogram, as a BGR image + layout."""
    import cv2
    from scipy.signal import spectrogram
    a, sr = audio_segment(start, stop)
    f, t, S = spectrogram(a, sr, nperseg=1024, noverlap=896)
    keep = f <= SPEC_MAX_HZ
    db = 10 * np.log10(S[keep] + 1e-6)
    lo, hi = np.percentile(db, [5, 99.5])
    img = np.clip((db - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)[::-1]   # high freq on top
    spec = cv2.applyColorMap(cv2.resize(img, (width, spec_h)), cv2.COLORMAP_MAGMA)
    # whistle band guides
    for hz in WHISTLE_BAND:
        y = int(spec_h * (1 - hz / SPEC_MAX_HZ))
        for x in range(0, width, 12):
            cv2.line(spec, (x, y), (x + 6, y), (255, 255, 255), 1)
    cv2.putText(spec, "pita peluit 2.5-4.5 kHz", (width - 215, int(spec_h * (1 - 4500 / SPEC_MAX_HZ)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for hz in (2000, 4000, 6000):
        y = int(spec_h * (1 - hz / SPEC_MAX_HZ))
        cv2.putText(spec, "{}k".format(hz // 1000), (3, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)

    bgr = {k: tuple(int(c[j:j + 2], 16) for j in (5, 3, 1)) for k, c in COLORS.items()}
    strips = []
    for name, lab in (("BENAR", gt), ("PREDIKSI", pred)):
        s = np.zeros((18, width, 3), np.uint8)
        xs = np.linspace(start, stop, width, endpoint=False).astype(int)
        for x, i in enumerate(xs):
            s[:, x] = bgr[lab[i]]
        cv2.putText(s, name, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        strips.append(s)
    sep = np.full((3, width, 3), 30, np.uint8)
    return np.vstack([strips[0], sep, strips[1], sep, spec])


def _panel_at(panel, i, start, stop, mark):
    import cv2
    p = panel.copy()
    w = p.shape[1]
    if mark is not None:
        xm = int((mark - start) / (stop - start) * w)
        cv2.line(p, (xm, 0), (xm, p.shape[0]), (80, 255, 80), 1)
    x = int((i - start) / (stop - start) * w)
    cv2.line(p, (x, 0), (x, p.shape[0]), (255, 255, 0), 2)
    cv2.putText(p, "kursor = sekarang   garis hijau = label panitia berubah", (w - 430, p.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def make_clip_audio(m, start, stop, gt, pred, path, mark=None, gain_db=45, scale=0.5):
    """Like make_clip, plus a live spectrogram panel and the (amplified) match audio."""
    from games import asset_paths
    w = int(1920 * scale)
    panel = build_panel(start, stop, gt, pred, w)
    audio = ["-ss", str(start / 25), "-t", str((stop - start) / 25), "-i", asset_paths()["video"],
             "-map", "0:v", "-map", "1:a", "-af",
             "volume={}dB,alimiter=limit=0.9".format(gain_db), "-c:a", "aac", "-shortest"]
    return make_clip(m, start, stop, gt, pred, path, scale=scale, mark=mark, panel=panel,
                     extra_inputs=audio)
