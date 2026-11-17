"""PFF FC FIFA World Cup 2022 open data: 64 matches, broadcast tracking (29.97 fps) + event data.
OTHER games. Free download after registration at PFF FC (data/external/<anything>/..., gitignored);
files are found by name anywhere under PFF_DIR, so the download can stay split in parts.

Labels are derived mechanically from PFF's event data, the same way as for the Metrica samples:
  * alive: owner = team of the most recent on-the-ball event (pass, carry, cross, shot, clearance,
    touch, initial touch); challenges and rebounds do not change possession
  * dead: from a ball-out event or a foul until the next set piece (throw-in, corner, goal kick,
    free kick, kick-off); before a set piece with no recorded stoppage, from the end of the
    previous event
Tracking is resampled to 25 fps by nearest timestamp.
"""

import bz2
import glob
import json
import os

import numpy as np

from .data import CACHE_DIR, ROOT, Match, H, A, D, X

PFF_DIR = os.environ.get("PFF_DIR") or os.path.join(ROOT, "data", "external")
OWNER_TYPES = {"PA", "BC", "CR", "SH", "CL", "TC", "IT"}
FPS_OUT = 25.0


def _find(sub, name):
    hits = sorted(glob.glob(os.path.join(PFF_DIR, "**", sub, name), recursive=True))
    if not hits:
        raise FileNotFoundError("PFF {}/{} not under {}".format(sub, name, PFF_DIR))
    return hits[0]


def all_matches():
    names = glob.glob(os.path.join(PFF_DIR, "**", "Tracking Data", "*.jsonl.bz2"), recursive=True)
    return tuple(sorted({os.path.basename(n).split(".")[0] for n in names}, key=int))


def _tracking(gid):
    t, per, hx, hy, ax, ay, bx, by = [], [], {}, {}, {}, {}, [], []
    with bz2.open(_find("Tracking Data", "{}.jsonl.bz2".format(gid)), "rt") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            p = r.get("period")
            if p not in (1, 2):
                continue
            k = len(t)
            t.append(r["videoTimeMs"] / 1000.0)
            per.append(p)
            for side, X_, Y_ in (("homePlayers", hx, hy), ("awayPlayers", ax, ay)):
                for pl in r.get(side) or ():
                    j = pl.get("jerseyNum")
                    if j is None or pl.get("x") is None:
                        continue
                    X_.setdefault(j, {})[k] = pl["x"]
                    Y_.setdefault(j, {})[k] = pl["y"]
            b = r.get("balls") or []
            b = b[0] if isinstance(b, list) and b else (b if isinstance(b, dict) else None)
            if b and b.get("x") is not None:
                bx.append(b["x"]); by.append(b["y"])
            else:
                bx.append(np.nan); by.append(np.nan)
    n = len(t)

    def mat(D_):
        keys = sorted(D_)
        a = np.full((n, len(keys)), np.nan, np.float32)
        for c, j in enumerate(keys):
            idx = np.fromiter(D_[j].keys(), int)
            a[idx, c] = np.fromiter(D_[j].values(), float)
        return a
    return (np.array(t), np.array(per, np.int8), mat(hx), mat(hy), mat(ax), mat(ay),
            np.array(bx, np.float32), np.array(by, np.float32))


def _events(gid):
    d = json.load(open(_find("Event Data", "{}.json".format(gid))))
    out = []
    for e in d:
        ge = e.get("gameEvents") or {}
        pe = e.get("possessionEvents") or {}
        fo = e.get("fouls") or {}
        out.append(dict(time=e.get("startTime"), end=e.get("endTime") or e.get("startTime"),
                        type=ge.get("gameEventType"), period=ge.get("period"),
                        home=ge.get("homeTeam"), sp=ge.get("setpieceType"),
                        ptype=pe.get("possessionEventType"), foul=fo.get("foulType") is not None))
    return [e for e in out if e["time"] is not None]


def load_pff(gid, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "pff_{}.npz".format(gid))
    meta = json.load(open(_find("Metadata", "{}.json".format(gid))))[0]
    pitch = meta["stadium"]["pitches"][0]
    L, W = float(pitch["length"]), float(pitch["width"])
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        return Match("pff_" + gid, z["px"], z["py"], z["team"], z["bx"], z["by"], z["period"],
                     L, W, z["labels"])

    t, per, hx, hy, ax, ay, bx, by = _tracking(gid)
    # 25 fps grid per period, nearest source frame
    sel_all, period = [], []
    for p in (1, 2):
        s = np.flatnonzero(per == p)
        if not len(s):
            continue
        grid = np.arange(t[s[0]], t[s[-1]], 1 / FPS_OUT)
        j = np.clip(np.searchsorted(t[s], grid), 1, len(s) - 1)
        j = np.where(np.abs(t[s][j - 1] - grid) < np.abs(t[s][j] - grid), j - 1, j)
        sel_all.append(s[j]); period.append(np.full(len(grid), p, np.int8))
    src = np.concatenate(sel_all)
    period = np.concatenate(period)
    tt = t[src]
    px = np.concatenate([hx, ax], 1)[src] + L / 2
    py = np.concatenate([hy, ay], 1)[src] + W / 2
    team = np.array([0] * hx.shape[1] + [1] * ax.shape[1], np.int8)
    bxx, byy = bx[src] + L / 2, by[src] + W / 2

    labels = np.full(len(src), X, np.int8)
    ev = sorted(_events(gid), key=lambda e: e["time"])
    for p in (1, 2):
        s = np.flatnonzero(period == p)
        if not len(s):
            continue
        tp = tt[s]
        evp = [e for e in ev if e["period"] == p]
        owner = np.full(len(s), -1, np.int8)
        for e in evp:
            if e["type"] in ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF") and e["home"] is not None and \
                    (e["ptype"] in OWNER_TYPES or e["sp"] not in (None, "O")):
                i = np.searchsorted(tp, e["time"])
                if i < len(s):
                    owner[i] = H if e["home"] else A
        o = owner.astype(float); o[o < 0] = np.nan
        import pandas as pd
        lab = pd.Series(o).ffill().bfill().to_numpy()
        lab = np.where(np.isnan(lab), H, lab).astype(np.int8)
        # dead spells before each restart
        stop = None
        prev_end = None
        for e in evp:
            if e["type"] == "OUT" or e["foul"]:
                stop = e["time"] if stop is None else stop
            if e["type"] == "OTB" and e["sp"] not in (None, "O"):
                a = stop if stop is not None else prev_end
                if a is not None and e["time"] > a:
                    i0, i1 = np.searchsorted(tp, a), np.searchsorted(tp, e["time"])
                    lab[i0:i1] = D
                stop = None
            if e["type"] in ("OTB",):
                prev_end = max(e["end"] or e["time"], e["time"])
        # play is over after the last event of the half
        last = max((e["end"] or e["time"]) for e in evp) if evp else tp[-1]
        first = min(e["time"] for e in evp) if evp else tp[0]
        lab[(tp > last) | (tp < first)] = X
        labels[s] = lab
    period = np.where(labels == X, 0, period).astype(np.int8)
    m = Match("pff_" + gid, px.astype(np.float32), py.astype(np.float32), team,
              bxx.astype(np.float32), byy.astype(np.float32), period, L, W, labels)
    np.savez(cache, px=m.px, py=m.py, team=team, bx=m.bx, by=m.by, period=period, labels=labels)
    return m


def restart_events(m):
    from .idsse import restart_events as r
    return r(m)


if __name__ == "__main__":
    import sys
    import time
    ids = sys.argv[1:] or all_matches()
    for gid in ids:
        t0 = time.time()
        m = load_pff(gid)
        g = m.period > 0
        lab = m.labels[g]
        runs = 0
        for p in (1, 2):
            x = m.labels[m.period == p]
            b = np.concatenate(([True], x[1:] != x[:-1]))
            runs += int(np.sum(b & (x < 2)))
        print("{} frames {} players {} H/A/D {} possessions {} ball vis {:.2f} ({:.0f}s)".format(
            gid, g.sum(), m.px.shape[1], np.round(np.bincount(lab, minlength=3)[:3] / len(lab), 2),
            runs, np.isfinite(m.bx[g]).mean(), time.time() - t0), flush=True)
