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

    if TOUCH:
        f.update(_touch_features(m, T, dist, mt, mo, bvis))

    if RESTART:
        f.update(_restart_features(m, f, xo, yc, spd, dist, mt, mo, bvis, bxo, byc, bspd))

    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


TOUCH = True       # ball-touch features (who last / next played the ball)
TOUCH_DV = 4.0     # m/s change of ball velocity over 0.24 s that counts as a touch
TOUCH_R = 2.5      # m: nearest player must be this close to own the touch


def touches(m, dist, mt, mo):
    """(frames, team 0/1) of detected ball touches:
    local maxima of the ball's velocity change with a player close enough to have caused it."""
    k = 3
    vx = (_shift(m.bx, 1) - _shift(m.bx, -1)) * (FPS / 2)
    vy = (_shift(m.by, 1) - _shift(m.by, -1)) * (FPS / 2)
    dv = np.hypot(_shift(vx, k) - _shift(vx, -k), _shift(vy, k) - _shift(vy, -k))
    dvf = np.where(np.isfinite(dv), dv, 0)
    w = 2 * k + 1
    peak = (dvf >= TOUCH_DV) & (dvf == pd.Series(dvf).rolling(w, center=True, min_periods=1).max().to_numpy())
    d0 = np.nanmin(np.where(m.team[None, :] == 0, dist, np.inf), 1)
    d1 = np.nanmin(np.where(m.team[None, :] == 1, dist, np.inf), 1)
    near = np.minimum(d0, d1) < TOUCH_R
    fr = np.flatnonzero(peak & near & (m.period > 0))
    return fr, (d1[fr] < d0[fr]).astype(np.int8)


def _touch_features(m, T, dist, mt, mo, bvis):
    fr, tm = touches(m, dist, mt, mo)
    f = {}
    n = m.n
    idx = np.arange(n)
    rel = np.where(tm == T, 1.0, -1.0)
    if len(fr):
        nxt = np.searchsorted(fr, idx, side="right")       # first touch > t
        prv = nxt - 1
        hp, hn = prv >= 0, nxt < len(fr)
        pi, ni = np.maximum(prv, 0), np.minimum(nxt, len(fr) - 1)
        f["tch_prev"] = np.where(hp, rel[pi], np.nan)
        f["tch_since"] = np.where(hp, np.minimum(idx - fr[pi], 750), np.nan)
        f["tch_next"] = np.where(hn, rel[ni], np.nan)
        f["tch_until"] = np.where(hn, np.minimum(fr[ni] - idx, 750), np.nan)
        p2 = np.maximum(prv - 1, 0)
        f["tch_prev2"] = np.where(prv >= 1, rel[p2], np.nan)
        n2 = np.minimum(nxt + 1, len(fr) - 1)
        f["tch_next2"] = np.where(nxt + 1 < len(fr), rel[n2], np.nan)
    else:
        for k in ("tch_prev", "tch_since", "tch_next", "tch_until", "tch_prev2", "tch_next2"):
            f[k] = np.full(n, np.nan)
    ev = np.zeros(n); ev[fr] = rel
    cnt = np.zeros(n); cnt[fr] = 1
    for name, (b, a) in (("1", (25, 25)), ("p1", (25, 0)), ("n1", (0, 25)), ("3", (75, 75))):
        s_ev = pd.Series(ev).rolling(a + b + 1, min_periods=1).sum().to_numpy()
        s_ct = pd.Series(cnt).rolling(a + b + 1, min_periods=1).sum().to_numpy()
        s_ev, s_ct = _shift(s_ev, a), _shift(s_ct, a)
        f["tch_bal" + name] = s_ev
        f["tch_cnt" + name] = s_ct
    # short-window control: share of visible frames where T is nearer to the ball
    dT = np.nanmin(np.where(mt, dist, np.inf), 1)
    dO = np.nanmin(np.where(mo, dist, np.inf), 1)
    nearer = np.where(bvis & np.isfinite(dT) & np.isfinite(dO), (dT < dO).astype(np.float64), np.nan)
    f["nearer_s"] = _roll(nearer, 6, 6)
    f["nearer_p"] = _roll(nearer, 12, 0)
    f["nearer_n"] = _roll(nearer, 0, 12)
    return f


RESTART = False    # restart-pattern features: kept, off -- LOGO +0.001 (noise level); --restart turns on


def _restart_features(m, f, xo, yc, spd, dist, mt, mo, bvis, bxo, byc, bspd):
    """Patterns of a set piece being prepared and taken, from tracking only.

    A restart is: players slow and re-shape while the ball sits still at a typical spot
    (touchline, corner, goal area, centre, free-kick wall at 9.15 m), then the ball is struck and
    everyone accelerates. Changes are written as explicit differences (after - before) because
    a tree cannot subtract two columns.
    """
    L, W = m.length, m.width
    g = {}
    spd_all = np.asarray(f["spd_all"], np.float64)
    for k, s in (("1", 25), ("2", 50), ("4", 100)):
        g["acc_all_" + k] = _roll(spd_all, 0, s) - _roll(spd_all, s, 0)
    for tag in ("T", "O"):
        x = np.asarray(f["spd_" + tag], np.float64)
        g["acc_{}_2".format(tag)] = _roll(x, 0, 50) - _roll(x, 50, 0)

    # ball standing still, then struck
    still = bvis & (bspd < 1.0)
    g["b_still_run"] = np.minimum(np.where(still, _run_since(~still | ~bvis), 0), 750)
    struck = bvis & (bspd > 6.0)
    g["until_struck"] = np.minimum(_run_since(struck[::-1])[::-1], 750)
    g["since_struck"] = np.minimum(_run_since(struck), 750)
    g["b_still_p2"] = _roll(still.astype(np.float64), 50, 0)
    g["b_kick"] = _roll(bspd, 0, 25) - _roll(np.where(bvis, bspd, np.nan), 25, 0)

    # typical restart spots (ball, or last/next seen ball position inside a gap)
    bx = np.where(bvis, bxo, np.asarray(f["bxo_ff"], np.float64))
    by_side = W / 2 - byc
    g["b_touch"] = by_side                                    # distance to touchline
    g["b_goalline"] = L / 2 - np.abs(bx)                      # distance to goal line
    g["b_corner"] = np.hypot(g["b_touch"], g["b_goalline"])
    g["b_centre"] = np.hypot(bxo, m.by - W / 2)
    g["b_six"] = ((L / 2 - np.abs(bxo) < 5.5) & (byc < 9.16)).astype(np.float32)
    g["b_six"][~bvis] = np.nan

    # people around the ball: taker alone, opponents pushed back to 9.15 m (free kicks)
    with np.errstate(invalid="ignore"):
        g["n_all2"] = np.where(bvis, np.sum((dist < 2) & np.isfinite(dist), 1), np.nan)
        g["nO9"] = np.where(bvis, np.sum((dist < 9.15) & mo, 1), np.nan)
        g["nT9"] = np.where(bvis, np.sum((dist < 9.15) & mt, 1), np.nan)
        g["n_all9_c1"] = _roll(g["nO9"] + g["nT9"], 25, 25)

    # set shapes: crowd in the box (corners / wide free kicks), each team in its own half (kick-off)
    with np.errstate(invalid="ignore"):
        inbox = (np.abs(xo) > L / 2 - 16.5) & (np.abs(yc) < 20.16)
        g["n_box"] = np.sum(inbox & np.isfinite(xo), 1).astype(np.float32)
        g["own_half_T"] = _nanmean(np.where(np.isfinite(xo), (xo < 0).astype(np.float32), np.nan), mt)
        g["own_half_O"] = _nanmean(np.where(np.isfinite(xo), (xo > 0).astype(np.float32), np.nan), mo)
        g["n_box_c1"] = _roll(g["n_box"], 25, 25)
    g["kickoff_shape"] = np.minimum(g["own_half_T"], g["own_half_O"])
    g["kickoff_shape_c1"] = _roll(g["kickoff_shape"], 25, 25)
    return g


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


NOISE_SMOOTH = 25   # frames; ATD positions are smooth (median 2nd difference 0.011 m ball,
                    # 0.001 m players), so position error is a slow drift, not per-frame jitter


def _smooth_noise(rng, sigma, shape):
    """Gaussian position error that drifts over ~1 s, rescaled to standard deviation sigma."""
    from scipy.ndimage import gaussian_filter1d
    z = gaussian_filter1d(rng.normal(0, 1, shape), NOISE_SMOOTH, axis=0)
    return z * (sigma / max(z.std(), 1e-9))


# How the ATD loses the ball, estimated without labels: mean player speed is 1.14 m/s while the
# ball is missing and 2.09 m/s while it is seen; in the sample games it is 1.04 m/s dead and
# 2.09 m/s alive. Solving the mixture gives P(missing | dead) ~ 0.95-1.0 and
# P(missing | alive) ~ 0.06 (dead share ~ 0.37). Edge jitter keeps a missing ball strong
# evidence of a stoppage, not proof.
BALL_SMOOTH = 6    # frames (Gaussian sigma): the ATD ball is low-pass filtered -- with 6 the
                   # ball's velocity-change quantiles over 1/3/6 frames match the ATD's (no labels)
MASK = "state"
P_MISS_DEAD = 1.0  # edge jitter brings the effective rate to ~0.85-0.9
P_MISS_ALIVE = 0.06
EDGE_JITTER = 25     # frames: where the ball vanishes / reappears around a stoppage


def _random_mask(n, vis_runs, miss_runs, rng):
    mask = np.zeros(n, bool)
    t, visible = 0, rng.random() < 0.6
    while t < n:
        k = int(rng.choice(vis_runs if visible else miss_runs))
        if visible:
            mask[t:t + k] = True
        t += k
        visible = not visible
    return mask


def _state_mask(labels, vis_runs, miss_runs, rng):
    """Ball visibility that depends on play state the way the ATD's does."""
    from .data import D
    # in-play gaps: ATD gap lengths, made rare enough that P(missing | alive) = P_MISS_ALIVE
    stretch = miss_runs.mean() * (1 - P_MISS_ALIVE) / P_MISS_ALIVE / vis_runs.mean()
    mask = _random_mask(len(labels), np.maximum(1, (vis_runs * stretch).astype(int)), miss_runs, rng)
    dead = (labels == D).astype(np.int8)
    e = np.flatnonzero(np.diff(np.concatenate(([0], dead, [0]))))
    for a, b in zip(e[::2], e[1::2]):
        if rng.random() < P_MISS_DEAD:
            a2 = max(0, a + int(rng.normal(0, EDGE_JITTER)))
            b2 = min(len(labels), b + int(rng.normal(0, EDGE_JITTER)))
            mask[a2:b2] = False
    return mask


def degrade(m, vis_runs, miss_runs, rng, ball_sigma=0.5, player_sigma=0.2, player_drop=0.07):
    """Copy of a clean Match with ATD-like noise.

    * the clean ball is first interpolated through its own gaps (in the corrected sample data
      the ball is missing almost exactly when it is dead, which would be a label leak)
    * then masked: MASK = "random" uses alternating visible / missing stretches resampled from the
      ATD's run lengths regardless of play state; MASK = "state" reproduces how ATD missingness
      depends on play state (see P_MISS_DEAD)
    * smooth (drifting) position noise -- the ATD has no frame-to-frame jitter; random player
      dropout segments
    """
    from copy import copy
    d = copy(m)
    bx, by = m.bx.astype(np.float64).copy(), m.by.astype(np.float64).copy()
    for p in np.unique(m.period[m.period > 0]):
        sel = np.flatnonzero(m.period == p)
        bx[sel] = pd.Series(bx[sel]).interpolate(limit_direction="both").to_numpy()
        by[sel] = pd.Series(by[sel]).interpolate(limit_direction="both").to_numpy()
        if BALL_SMOOTH:
            from scipy.ndimage import gaussian_filter1d
            bx[sel] = gaussian_filter1d(bx[sel], BALL_SMOOTH)
            by[sel] = gaussian_filter1d(by[sel], BALL_SMOOTH)

    if MASK == "state" and m.labels is not None:
        mask = _state_mask(m.labels, vis_runs, miss_runs, rng)
    else:
        mask = _random_mask(m.n, vis_runs, miss_runs, rng)
    bx = bx + _smooth_noise(rng, ball_sigma, m.n)
    by = by + _smooth_noise(rng, ball_sigma, m.n)
    d.bx = np.where(mask, bx, np.nan).astype(np.float32)
    d.by = np.where(mask, by, np.nan).astype(np.float32)

    px = m.px + _smooth_noise(rng, player_sigma, m.px.shape).astype(np.float32)
    py = m.py + _smooth_noise(rng, player_sigma, m.py.shape).astype(np.float32)
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
