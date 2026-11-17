"""IDSSE open data (Bassek, Rein, Weber & Memmert, Scientific Data 2025): seven Bundesliga
matches, OTHER games, CC-BY 4.0, figshare DOI 10.6084/m9.figshare.28196177.

Download (about 2.6 GB) into data/external/idsse/ (gitignored):

    curl -s https://api.figshare.com/v2/articles/28196177 | python -c "import json,sys;
    [print(f['download_url'], f['name']) for f in json.load(sys.stdin)['files']]" |
    while read url name; do curl -sL -o "data/external/idsse/$name" "$url"; done

The DFL position feed carries, on every ball frame, the operator's BallStatus (1 alive / 0 dead)
and BallPossession (1 home / 2 guest). Those give H/A/D per frame directly; loaded into the same
Match structure as the Metrica samples (metres, origin at a corner, 25 fps).
"""

import glob
import os
import re

import numpy as np

from .data import CACHE_DIR, ROOT, Match, H, A, D, X

IDSSE_DIR = os.environ.get("IDSSE_DIR") or os.path.join(ROOT, "data", "external", "idsse")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SECTIONS = {"firstHalf": 1, "secondHalf": 2}

_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _file(kind, mid):
    hits = glob.glob(os.path.join(IDSSE_DIR, "DFL_{}_*{}.xml".format(kind, mid)))
    if not hits:
        raise FileNotFoundError("IDSSE {} file for {} not in {}".format(kind, mid, IDSSE_DIR))
    return hits[0]


def _home_team(mid):
    txt = open(_file("02_01_matchinformation", mid), encoding="utf-8").read()
    for tag in re.findall(r"<Team\b[^>]*>", txt):
        a = dict(_ATTR.findall(tag))
        if a.get("Role") == "home":
            return a["TeamId"]
    raise ValueError("no home team in match information " + mid)


def _pitch(mid):
    txt = open(_file("02_01_matchinformation", mid), encoding="utf-8").read()
    x, y = re.search(r'PitchX="([\d.]+)"', txt), re.search(r'PitchY="([\d.]+)"', txt)
    return float(x.group(1)) if x else 105.0, float(y.group(1)) if y else 68.0


def _read_positions(mid):
    """{(section, kind, person): (N array, X array, Y array, status, possession)}."""
    sets, cur, rows = {}, None, []
    frame_re = re.compile(rb'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"')
    ball_re = re.compile(rb'BallPossession="(\d)"|BallStatus="(\d)"')

    def close():
        if cur is not None and rows:
            arr = np.array(rows, np.float64)
            sets[cur] = arr
    with open(_file("04_03_positions_raw_observed", mid), "rb") as fh:
        for line in fh:
            if line.startswith(b"<Frame "):
                m = frame_re.match(line)
                if m is None:
                    continue
                if cur[1] == "ball":
                    st = po = np.nan
                    for p, s in ball_re.findall(line):
                        if p:
                            po = float(p)
                        if s:
                            st = float(s)
                    rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), st, po))
                else:
                    rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            elif line.startswith(b"<FrameSet "):
                close()
                a = dict(_ATTR.findall(line.decode()))
                team = a["TeamId"]
                kind = "ball" if team.lower() == "ball" else ("ref" if team == "referee" else team)
                cur, rows = (a["GameSection"], kind, a.get("PersonId", "")), []
        close()
    return sets


def load_idsse(mid, use_cache=True):
    """IDSSE match `mid` as a Match with frame labels from the DFL ball-status / possession feed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "idsse_{}.npz".format(mid))
    L, W = _pitch(mid)
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        return Match("idsse_" + mid, z["px"], z["py"], z["team"], z["bx"], z["by"], z["period"],
                     L, W, z["labels"])

    home = _home_team(mid)
    sets = _read_positions(mid)
    balls = {sec: arr for (sec, kind, _), arr in sets.items() if kind == "ball"}
    # one continuous timeline: halves back to back, frames indexed from each half's ball feed
    offs, n = {}, 0
    for sec in ("firstHalf", "secondHalf"):
        if sec in balls:
            N = balls[sec][:, 0]
            offs[sec] = (int(N.min()), n)
            n += int(N.max() - N.min()) + 1
    persons = sorted({(kind, pid) for (_, kind, pid) in sets if kind not in ("ball", "ref")})
    col = {p: i for i, p in enumerate(persons)}
    px = np.full((n, len(persons)), np.nan, np.float32)
    py = np.full_like(px, np.nan)
    team = np.array([0 if k == home else 1 for k, _ in persons], np.int8)
    bx = np.full(n, np.nan, np.float32)
    by = np.full_like(bx, np.nan)
    period = np.zeros(n, np.int8)
    labels = np.full(n, X, np.int8)
    for (sec, kind, pid), arr in sets.items():
        if sec not in offs or kind == "ref":
            continue
        n0, base = offs[sec]
        idx = arr[:, 0].astype(np.int64) - n0 + base
        ok = (idx >= 0) & (idx < n)
        idx, arr = idx[ok], arr[ok]
        x, y = arr[:, 1] + L / 2, arr[:, 2] + W / 2
        if kind == "ball":
            bx[idx], by[idx] = x, y
            period[idx] = SECTIONS[sec]
            st, po = arr[:, 3], arr[:, 4]
            labels[idx] = np.where(st == 0, D, np.where(po == 1, H, np.where(po == 2, A, X)))
        else:
            c = col[(kind, pid)]
            px[idx, c], py[idx, c] = x, y
    period[labels == X] = 0
    m = Match("idsse_" + mid, px, py, team, bx, by, period, L, W, labels)
    np.savez(cache, px=px, py=py, team=team, bx=bx, by=by, period=period, labels=labels)
    return m


def restart_events(m):
    """Metrica-style SET PIECE rows (1-based frames) at every dead -> alive change, so the
    training games can be given the same synthetic Smart-Tagging context as the samples.
    Restarts on a touchline are throw-ins (team unknown in Smart Tagging, like the real tags)."""
    import pandas as pd
    rows = []
    for p in (1, 2):
        sel = np.flatnonzero(m.period == p)
        if not len(sel):
            continue
        lab = m.labels[sel]
        t = sel[np.flatnonzero((lab[:-1] == D) & (lab[1:] < 2)) + 1]
        rows.append(dict(type="SET PIECE", subtype="KICK OFF", start=sel[0] + 1, end=sel[0] + 1,
                         team=int(m.labels[sel[0]]) if m.labels[sel[0]] < 2 else 0, period=p))
        for f in t:
            near_line = np.isfinite(m.by[f]) and min(m.by[f], m.width - m.by[f]) < 1.5
            rows.append(dict(type="SET PIECE", subtype="THROW IN" if near_line else "",
                             start=f + 1, end=f + 1, team=int(m.labels[f]), period=p))
        rows.append(dict(type="END", subtype="", start=sel[-1] + 1, end=sel[-1] + 1, team=0,
                         period=p))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time
    for mid in MATCHES:
        t0 = time.time()
        try:
            m = load_idsse(mid)
        except FileNotFoundError as e:
            print(e)
            continue
        g = m.period > 0
        lab = m.labels[g]
        runs = 0
        for p in (1, 2):
            x = m.labels[m.period == p]
            b = np.concatenate(([True], x[1:] != x[:-1]))
            runs += int(np.sum(b & (x < 2)))
        print("{} frames {} players {} H/A/D share {} possessions {} mean {:.1f}s  ({:.0f}s)".format(
            mid, m.n, m.px.shape[1], np.round(np.bincount(lab, minlength=3)[:3] / len(lab), 3),
            runs, (lab < 2).sum() / max(runs, 1) / 25, time.time() - t0))
