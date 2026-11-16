"""Per-frame features, from one team's point of view, plus the degradation used to make clean
training games look like the competition's automatic tracking.

Everything is team-level (no player identities), so slot fragmentation and identity switches in
the ATD do not matter. Missing values stay NaN -- the gradient-boosting model handles them natively.
"""

import numpy as np
import pandas as pd

FPS = 25


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _shift(a, k):
    """a[t+k] with NaN padding (works on 1-D and 2-D along axis 0)."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _nanmean(a, mask=None):
    if mask is not None:
        a = np.where(mask, a, np.nan)
    with np.errstate(all="ignore"):
        cnt = np.sum(np.isfinite(a), 1)
        s = np.nansum(a, 1)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _nanstd(a, mask):
    a = np.where(mask, a, np.nan)
    mu = _nanmean(a)[:, None]
    return np.sqrt(_nanmean((a - mu) ** 2))


def _nanmin(a, mask):
    a = np.where(mask & np.isfinite(a), a, np.inf)
    out = a.min(1)
    return np.where(np.isinf(out), np.nan, out)


def _nanmax(a, mask):
    a = np.where(mask & np.isfinite(a), a, -np.inf)
    out = a.max(1)
    return np.where(np.isinf(out), np.nan, out)


def _roll(x, before, after, min_periods=1):
    """Mean of x over [t-before, t+after], NaN-aware."""
    s = pd.Series(x)
    w = before + after + 1
    r = s.rolling(w, min_periods=min_periods).mean().to_numpy()
    return _shift(r, after)


def _run_since(valid):
    """Frames since the last True (0 on True frames), capped; NaN before the first True."""
    idx = np.where(valid, np.arange(len(valid)), -1)
    last = np.maximum.accumulate(idx)
    out = (np.arange(len(valid)) - last).astype(np.float32)
    out[last < 0] = np.nan
    return out


def _ffill(x):
    return pd.Series(x).ffill().to_numpy()


def _bfill(x):
    return pd.Series(x).bfill().to_numpy()


def attack_signs(m):
    """sign[period][team] = +1 if that team attacks towards +x in that period."""
    signs = {}
    for p in (1, 2):
        sel = m.period == p
        if not sel.any():
            continue
        mean_x = [np.nanmean(m.px[sel][:, m.team == t]) for t in (0, 1)]
        home_left = mean_x[0] < mean_x[1]
        signs[p] = {0: 1.0 if home_left else -1.0, 1: -1.0 if home_left else 1.0}
    return signs


# ---------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------

def team_features(m, T):
    """DataFrame of features for every frame, seen from team T (0 home / 1 away)."""
    L, W = m.length, m.width
    signs = attack_signs(m)
    sign = np.ones(m.n, np.float32)
    for p, s in signs.items():
        sign[m.period == p] = s[T]

    O = 1 - T
    mt = (m.team == T)[None, :]
    mo = (m.team == O)[None, :]
    obs = np.isfinite(m.px)

    # oriented player coordinates: T attacks towards +x, 0 = halfway line
    xo = (m.px - L / 2) * sign[:, None]
    yc = m.py - W / 2
    k = 5
    vx = (_shift(m.px, k) - _shift(m.px, -k)) / (2 * k / FPS) * sign[:, None]
    vy = (_shift(m.py, k) - _shift(m.py, -k)) / (2 * k / FPS)
    spd = np.sqrt(vx ** 2 + vy ** 2)

    f = {}
    for tag, mask in (("T", mt), ("O", mo)):
        f["cx_" + tag] = _nanmean(xo, mask)
        f["sx_" + tag] = _nanstd(xo, mask)
        f["sy_" + tag] = _nanstd(yc, mask)
        f["vx_" + tag] = _nanmean(vx, mask)
        f["spd_" + tag] = _nanmean(spd, mask)
        f["xmax_" + tag] = _nanmax(xo, mask)
        f["xmin_" + tag] = _nanmin(xo, mask)
        f["n_" + tag] = np.sum(obs & mask, 1).astype(np.float32)
        f["fast_" + tag] = _nanmean((spd > 4).astype(np.float32) * np.where(np.isfinite(spd), 1, np.nan), mask)
    allm = np.ones_like(mt)
    f["spd_all"] = _nanmean(spd, allm)
    f["slow_all"] = _nanmean(np.where(np.isfinite(spd), (spd < 1.0).astype(np.float32), np.nan), allm)
    f["cx_gap"] = f["cx_T"] - f["cx_O"]
    f["vx_gap"] = f["vx_T"] - f["vx_O"]

    # ball
    bvis = np.isfinite(m.bx)
    bxo = (m.bx - L / 2) * sign
    byc = np.abs(m.by - W / 2)
    out_margin = np.maximum.reduce([-m.bx, m.bx - L, -m.by, m.by - W])
    bvx = (_shift(m.bx, 2) - _shift(m.bx, -2)) / (4 / FPS)
    bvy = (_shift(m.by, 2) - _shift(m.by, -2)) / (4 / FPS)
    bspd = np.sqrt(bvx ** 2 + bvy ** 2)
    dist = np.sqrt((m.px - m.bx[:, None]) ** 2 + (m.py - m.by[:, None]) ** 2)
    dT, dO = _nanmin(dist, mt), _nanmin(dist, mo)
    f["ball_vis"] = bvis.astype(np.float32)
    f["bxo"] = bxo
    f["byc"] = byc
    f["b_out"] = out_margin
    f["b_spd"] = bspd
    f["dT"], f["dO"] = dT, dO
    f["dgap"] = dT - dO
    f["nT5"] = np.where(bvis, np.sum((dist < 5) & mt, 1), np.nan)
    f["nO5"] = np.where(bvis, np.sum((dist < 5) & mo, 1), np.nan)
    f["b_vs_lineT"] = bxo - f["cx_T"]

    # what was seen around a gap in the ball track
    f["since_vis"] = np.minimum(_run_since(bvis), 3000)
    f["until_vis"] = np.minimum(_run_since(bvis[::-1])[::-1], 3000)
    for name in ("dgap", "bxo", "b_out"):
        f[name + "_ff"] = _ffill(f[name])
        f[name + "_bf"] = _bfill(f[name])

    # temporal context
    for name in ("dgap", "spd_all", "slow_all", "vx_T", "vx_O", "cx_gap", "ball_vis", "b_spd",
                 "sx_T", "sx_O", "b_out"):
        x = np.asarray(f[name], np.float64)
        f[name + "_c1"] = _roll(x, 25, 25)
        f[name + "_c4"] = _roll(x, 100, 100)
        f[name + "_p2"] = _roll(x, 50, 0)
        f[name + "_n2"] = _roll(x, 0, 50)
    for name in ("spd_all", "slow_all", "cx_gap", "vx_gap", "sx_T", "sx_O"):
        f[name + "_c10"] = _roll(np.asarray(f[name], np.float64), 250, 250)

    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


# ---------------------------------------------------------------------------------------------
# making clean training games look like automatic tracking
# ---------------------------------------------------------------------------------------------

def run_lengths(valid):
    """(lengths of True runs, lengths of False runs)."""
    v = np.asarray(valid, np.int8)
    edges = np.flatnonzero(np.diff(v)) + 1
    bounds = np.concatenate(([0], edges, [len(v)]))
    lens = np.diff(bounds)
    vals = v[bounds[:-1]]
    return lens[vals == 1], lens[vals == 0]


def degrade(m, vis_runs, miss_runs, rng, ball_sigma=0.5, player_sigma=0.2, player_drop=0.07):
    """Copy of a clean Match with ATD-like noise.

    * the clean ball is first interpolated through its own gaps (in the corrected sample data
      the ball is missing almost exactly when it is dead, which would be a label leak)
    * then masked with alternating visible / missing stretches resampled from the ATD's own
      run-length distribution, so the model learns to work through long gaps
    * Gaussian position noise; random player dropout segments
    """
    from copy import copy
    d = copy(m)
    bx, by = m.bx.astype(np.float64).copy(), m.by.astype(np.float64).copy()
    for p in np.unique(m.period[m.period > 0]):
        sel = np.flatnonzero(m.period == p)
        bx[sel] = pd.Series(bx[sel]).interpolate(limit_direction="both").to_numpy()
        by[sel] = pd.Series(by[sel]).interpolate(limit_direction="both").to_numpy()

    mask = np.zeros(m.n, bool)
    t, visible = 0, rng.random() < 0.6
    while t < m.n:
        n = int(rng.choice(vis_runs if visible else miss_runs))
        if visible:
            mask[t:t + n] = True
        t += n
        visible = not visible
    bx = bx + rng.normal(0, ball_sigma, m.n)
    by = by + rng.normal(0, ball_sigma, m.n)
    d.bx = np.where(mask, bx, np.nan).astype(np.float32)
    d.by = np.where(mask, by, np.nan).astype(np.float32)

    px = m.px + rng.normal(0, player_sigma, m.px.shape).astype(np.float32)
    py = m.py + rng.normal(0, player_sigma, m.py.shape).astype(np.float32)
    drop = np.zeros(m.px.shape, bool)
    seg = 250
    n_seg = int(player_drop * m.n * m.px.shape[1] / seg)
    starts = rng.integers(0, m.n, n_seg)
    cols = rng.integers(0, m.px.shape[1], n_seg)
    for s, c in zip(starts, cols):
        drop[s:s + seg, c] = True
    d.px = np.where(drop, np.nan, px).astype(np.float32)
    d.py = np.where(drop, np.nan, py).astype(np.float32)
    return d
