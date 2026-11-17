"""How do label changes line up with what the ball and players do?

For every label change it measures, relative to the change frame (negative = before):
  * possession change X->Y : when a Y player first gets within 2 m of the ball (reception),
                              and when the last X player was within 2 m (last X touch)
  * alive -> D             : when the ball is first outside the pitch
  * D -> alive             : when the ball starts moving (speed > 4 m/s)

Run on the public BAR-ATH window (to understand the organisers' convention -- not to tune) and on
our event-derived labels for the sample games, then compare the conventions.

    python -m possession.definition_check
"""

import numpy as np
import pandas as pd

from .data import load_bar_ath, load_metrica_sample, ROOT
from .features import _shift

WIN = 100   # look +/- 4 s around each change
NEAR = 2.0  # metres


def _ball_kin(m):
    vx = (_shift(m.bx, 2) - _shift(m.bx, -2)) / (4 / 25)
    vy = (_shift(m.by, 2) - _shift(m.by, -2)) / (4 / 25)
    return np.sqrt(vx ** 2 + vy ** 2)


def _team_dist(m, t):
    d = np.sqrt((m.px - m.bx[:, None]) ** 2 + (m.py - m.by[:, None]) ** 2)
    d = np.where((m.team == t)[None, :] & np.isfinite(d), d, np.inf)
    out = d.min(1)
    return np.where(np.isinf(out), np.nan, out)


def _first(cond, lo, hi, reverse=False):
    idx = np.flatnonzero(cond[lo:hi])
    if not len(idx):
        return np.nan
    return lo + (idx[-1] if reverse else idx[0])


def analyse(m, labels, frames=None):
    """DataFrame with one row per label change inside graded frames."""
    lab = labels.astype(int)
    dist = {0: _team_dist(m, 0), 1: _team_dist(m, 1)}
    spd = _ball_kin(m)
    out_ = np.maximum.reduce([-m.bx, m.bx - m.length, -m.by, m.by - m.width]) > 0
    rows = []
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    if frames is not None:
        ch = ch[(ch > frames[0]) & (ch < frames[1])]
    for t in ch:
        a, b = lab[t - 1], lab[t]
        if a == 3 or b == 3:
            continue
        lo, hi = max(0, t - WIN), min(len(lab), t + WIN)
        vis = np.isfinite(m.bx[lo:hi]).mean()
        r = {"frame": t, "from": "HAD"[a], "to": "HAD"[b], "ball_vis": round(vis, 2)}
        if a < 2 and b < 2:
            r["kind"] = "possession"
            r["new_team_reaches_ball"] = _first(dist[b] < NEAR, lo, hi) - t
            r["old_team_last_near"] = _first(dist[a][:t] < NEAR, lo, t, reverse=True) - t \
                if t > lo else np.nan
            # nearest-player team flips from old to new: the switch closest to t
            near = np.where(np.isfinite(dist[a]) & np.isfinite(dist[b]),
                            (dist[b] < dist[a]).astype(float), np.nan)[lo:hi]
            nn = pd.Series(near).ffill().to_numpy()
            flips = np.flatnonzero((nn[1:] == 1) & (nn[:-1] == 0)) + 1 + lo
            r["nearest_flips_to_new"] = flips[np.argmin(np.abs(flips - t))] - t \
                if len(flips) else np.nan
        elif b == 2:
            r["kind"] = "alive->D"
            r["ball_leaves_pitch"] = _first(out_, lo, hi) - t
            r["ball_stops"] = _first(spd < 1.0, t - 25, hi) - t
        else:
            r["kind"] = "D->alive"
            r["ball_starts_moving"] = _first(spd > 4.0, lo, hi) - t
        rows.append(r)
    return pd.DataFrame(rows)


def summary(df, name):
    print("\n=== {} ===  ({} label changes)".format(name, len(df)))
    for kind, cols in (("possession", ["nearest_flips_to_new", "old_team_last_near"]),
                       ("alive->D", ["ball_leaves_pitch", "ball_stops"]),
                       ("D->alive", ["ball_starts_moving"])):
        sub = df[df.kind == kind]
        if not len(sub):
            continue
        parts = []
        for c in cols:
            v = sub[c].dropna().to_numpy() / 25
            if len(v):
                parts.append("{} median {:+.2f}s IQR [{:+.2f},{:+.2f}] (n={})".format(
                    c, np.median(v), *np.percentile(v, [25, 75]), len(v)))
        print("  {:<11} n={:<4} ".format(kind, len(sub)) + " | ".join(parts))


def main():
    m = load_bar_ath()
    gt = pd.read_csv(ROOT + "/evaluation/gt/BAR-ATH_possession_gt_public.csv").label
    gt = gt.map({"H": 0, "A": 1, "D": 2, "X": 3}).to_numpy()
    pub = analyse(m, gt, (11185, 18685))
    summary(pub, "BAR-ATH public window: organisers' labels")
    for g in (1, 2, 3):
        s = load_metrica_sample(g)
        summary(analyse(s, s.labels), "sample game {}: our event-derived labels".format(g))
    pub.to_csv(ROOT + "/data/cache/public_label_changes.csv", index=False)
    print("\nper-change table: data/cache/public_label_changes.csv")


if __name__ == "__main__":
    main()
