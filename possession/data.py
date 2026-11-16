"""Loaders that turn every tracking source we use into one common in-memory layout.

A `Match` holds, for T frames and S track slots:

    px, py   (T, S) float32   player positions in metres, NaN where the slot is not observed
    team     (S,)   int8      0 = home, 1 = away
    bx, by   (T,)   float32   ball position in metres, NaN where the ball is not observed
    period   (T,)   int8      1 / 2 for frames inside a half, 0 outside
    labels   (T,)   int8      0=H 1=A 2=D 3=X (only for training matches; None for BAR-ATH)
    length, width             pitch size in metres

Sources:
  * BAR-ATH            -- the competition match, Metrica automatic tracking (ATD, FIFA/EPTS)
  * Metrica sample 1-3 -- public games from github.com/metrica-sports/sample-data. Corrected
                          tracking plus human event data, used ONLY as training / validation data
                          from other games (allowed by RULES.md). Labels are derived from events.
"""

import io
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, asset_paths, half_video_blocks  # noqa: E402

H, A, D, X = 0, 1, 2, 3
CACHE_DIR = os.path.join(ROOT, "data", "cache")          # data/ is gitignored
SAMPLE_DIR = os.environ.get("METRICA_SAMPLE_DIR") or os.path.join(
    ROOT, "reference", "sample-data", "data")


@dataclass
class Match:
    name: str
    px: np.ndarray
    py: np.ndarray
    team: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    period: np.ndarray
    length: float
    width: float
    labels: Optional[np.ndarray] = None

    @property
    def n(self):
        return len(self.bx)


# ---------------------------------------------------------------------------------------------
# FIFA/EPTS raw text: "<frame>:<x,y[,s]>;<x,y[,s]>;...:<ball x,y[,s]>"
# ---------------------------------------------------------------------------------------------

def _parse_epts_block(lines, n_values):
    """Parse lines of one data-format block into a (n_lines, n_values) float array, fast."""
    text = "".join(lines).replace(":", ",").replace(";", ",")
    arr = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                      na_values=["NaN", "nan", ""], engine="c").to_numpy()
    if arr.shape[1] != n_values:
        raise ValueError("expected {} values per line, got {}".format(n_values, arr.shape[1]))
    return arr


def _epts_specs(meta_root):
    """[(start_frame, end_frame, [player ids in order], n_channels_per_player, ball_channels)]"""
    specs = []
    for spec in meta_root.iter("DataFormatSpecification"):
        players, per_player, ball = [], None, 0
        for split in spec.iter("SplitRegister"):
            refs = [r.get("playerChannelId") for r in split.findall("PlayerChannelRef")]
            if refs:
                pid = refs[0].split("_")[0]
                players.append(pid)
                per_player = len(refs)
            nb = len(split.findall("BallChannelRef"))
            if nb:
                ball = nb
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      players, per_player, ball))
    return specs


def _player_teams(meta_root):
    """{channel player prefix -> 0 home / 1 away}, first team in the file is home."""
    team_ids = [t.get("id") for t in meta_root.iter("Team")]
    out = {}
    pid_team = {p.get("id"): p.get("teamId") for p in meta_root.iter("Player")}
    for ch in meta_root.iter("PlayerChannel"):
        prefix = ch.get("id").split("_")[0]
        out[prefix] = 0 if pid_team[ch.get("playerId")] == team_ids[0] else 1
    return out


def read_epts(meta_path, raw_path):
    """Returns frame ids (T,), px, py (T,S) normalised 0-1, team (S,), bx, by (T,) normalised."""
    root = ET.parse(meta_path).getroot()
    specs = _epts_specs(root)
    teams = _player_teams(root)
    all_players = []
    for _, _, players, _, _ in specs:
        for p in players:
            if p not in all_players:
                all_players.append(p)
    col = {p: i for i, p in enumerate(all_players)}

    with open(raw_path) as handle:
        lines = handle.readlines()
    frame_ids = np.array([int(l[:l.index(":")]) for l in lines])
    T, S = len(lines), len(all_players)
    px = np.full((T, S), np.nan, np.float32)
    py = np.full((T, S), np.nan, np.float32)
    bx = np.full(T, np.nan, np.float32)
    by = np.full(T, np.nan, np.float32)

    for start, end, players, per_player, ball in specs:
        rows = np.flatnonzero((frame_ids >= start) & (frame_ids <= end))
        if not len(rows):
            continue
        n_values = 1 + per_player * len(players) + ball
        arr = _parse_epts_block([lines[i] for i in rows], n_values)
        for k, p in enumerate(players):
            px[rows, col[p]] = arr[:, 1 + per_player * k]
            py[rows, col[p]] = arr[:, 2 + per_player * k]
        bx[rows] = arr[:, 1 + per_player * len(players)]
        by[rows] = arr[:, 2 + per_player * len(players)]
    team = np.array([teams[p] for p in all_players], np.int8)
    return frame_ids, px, py, team, bx, by


# ---------------------------------------------------------------------------------------------
# BAR-ATH, the competition match
# ---------------------------------------------------------------------------------------------

def load_bar_ath(game="BAR-ATH", use_cache=True):
    """The competition match on the tactical-video timeline (row i = video frame i)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "{}_tactical_atd.npz".format(game))
    length, width = GAMES[game]["pitch"]
    n = GAMES[game]["n_video_frames"]
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        px, py, team, bx, by = z["px"], z["py"], z["team"], z["bx"], z["by"]
    else:
        paths = asset_paths(game)
        frame_ids, px, py, team, bx, by = read_epts(paths["atd_meta"], paths["atd_raw"])
        if len(frame_ids) != n or frame_ids[0] != 1:
            raise ValueError("unexpected ATD frame numbering")
        np.savez(cache, px=px, py=py, team=team, bx=bx, by=by)
    period = np.zeros(n, np.int8)
    for half, first, last in half_video_blocks(game):
        period[first:last + 1] = half
    return Match(game, px * length, py * width, team, bx * length, by * width,
                 period, length, width, None)


# ---------------------------------------------------------------------------------------------
# Metrica public sample games -- other games, used for training and validation only
# ---------------------------------------------------------------------------------------------

POSSESSION_EVENTS = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT"}


def _events_game12(g):
    e = pd.read_csv(os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(g),
                                 "Sample_Game_{}_RawEventsData.csv".format(g)))
    return pd.DataFrame({
        "team": np.where(e["Team"] == "Home", 0, 1),
        "type": e["Type"].str.upper(),
        "subtype": e["Subtype"].fillna("").str.upper(),
        "period": e["Period"],
        "start": e["Start Frame"], "end": e["End Frame"],
    })


def _events_game3():
    d = json.load(open(os.path.join(SAMPLE_DIR, "Sample_Game_3", "Sample_Game_3_events.json")))
    rows = []
    for ev in d["data"]:
        subs = ev["subtypes"]
        subs = subs if isinstance(subs, list) else ([subs] if subs else [])
        rows.append({
            "team": 0 if ev["team"]["id"] == "FIFATMA" else 1,
            "type": ev["type"]["name"].upper(),
            "subtype": "-".join(s["name"].upper() for s in subs),
            "period": ev["period"], "start": ev["start"]["frame"], "end": ev["end"]["frame"],
        })
    return pd.DataFrame(rows)


def labels_from_events(events, frame_ids):
    """Per-frame H/A/D/X derived mechanically from human event data of an OTHER game.

    * between kick-off and the last event of each period the ball is alive unless stopped
    * dead: from the end of the event preceding a SET PIECE (ball out, foul, goal ...) until the
      set piece is taken. Fouls (FAULT RECEIVED / CHALLENGE-FAULT) start the stoppage at once.
    * alive: owner is the team of the most recent on-ball event (pass, carry, shot, set piece,
      recovery, ball lost, ball out) -- loose balls stay with the last team in possession.
    """
    events = events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    idx_of = {f: i for i, f in enumerate(frame_ids)}
    T = len(frame_ids)
    first, last = frame_ids[0], frame_ids[-1]

    def to_idx(f):
        f = int(min(max(f, first), last))
        return idx_of.get(f, np.searchsorted(frame_ids, f))

    labels = np.full(T, X, np.int8)
    for period, ev in events.groupby("period"):
        ev = ev.reset_index(drop=True)
        p_start, p_end = to_idx(ev["start"].iloc[0]), to_idx(ev[["start", "end"]].max().max())
        owner = np.full(p_end - p_start + 1, -1, np.int8)
        # possession from on-ball events
        marks = ev[ev["type"].isin(POSSESSION_EVENTS)]
        for _, r in marks.iterrows():
            owner[to_idx(r["start"]) - p_start] = r["team"]
        owner = pd.Series(owner).replace(-1, np.nan).ffill().bfill().to_numpy()
        lab = owner.astype(np.int8)
        # dead-ball stretches before each restart
        for i in range(len(ev)):
            r = ev.iloc[i]
            if r["type"] != "SET PIECE" or i == 0:
                continue
            j = i - 1
            prev_end = max(int(ev.iloc[j]["end"]), int(ev.iloc[j]["start"]))
            # a foul stops play at the moment of the offence
            k = j
            while k >= 0 and int(ev.iloc[k]["start"]) >= prev_end - 50:
                if ev.iloc[k]["type"] == "FAULT RECEIVED" or "FAULT" in ev.iloc[k]["subtype"]:
                    prev_end = min(prev_end, int(ev.iloc[k]["start"]))
                k -= 1
            a, b = to_idx(prev_end) - p_start, to_idx(r["start"]) - p_start
            if b > a:
                lab[a:b] = D
        labels[p_start:p_end + 1] = lab
    return labels


def sample_events(g):
    """Event table of a sample game (frames are tracking frame ids, 1-based)."""
    return _events_game12(g) if g in (1, 2) else _events_game3()


def load_metrica_sample(g):
    """Sample game g (1, 2 or 3) as a Match with event-derived labels."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "metrica_sample_{}.npz".format(g))
    length, width = 105.0, 68.0
    if os.path.exists(cache):
        z = np.load(cache)
        return Match("sample{}".format(g), z["px"], z["py"], z["team"], z["bx"], z["by"],
                     z["period"], length, width, z["labels"])

    if g in (1, 2):
        base = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(g), "Sample_Game_{}_".format(g))
        frames = []
        for side, t in (("Home", 0), ("Away", 1)):
            df = pd.read_csv(base + "RawTrackingData_{}_Team.csv".format(side), skiprows=2)
            frames.append((df, t))
        df0 = frames[0][0]
        frame_ids = df0["Frame"].to_numpy()
        period_raw = df0["Period"].to_numpy()
        pxs, pys, teams = [], [], []
        for df, t in frames:
            cols = list(df.columns)
            for i, c in enumerate(cols):
                if c.startswith("Player"):
                    pxs.append(df.iloc[:, i].to_numpy(np.float32))
                    pys.append(df.iloc[:, i + 1].to_numpy(np.float32))
                    teams.append(t)
        bi = list(df0.columns).index("Ball")
        bx = df0.iloc[:, bi].to_numpy(np.float32)
        by = df0.iloc[:, bi + 1].to_numpy(np.float32)
        px, py = np.stack(pxs, 1), np.stack(pys, 1)
        team = np.array(teams, np.int8)
        events = _events_game12(g)
    else:
        base = os.path.join(SAMPLE_DIR, "Sample_Game_3", "Sample_Game_3_")
        frame_ids, px, py, team, bx, by = read_epts(base + "metadata.xml", base + "tracking.txt")
        events = _events_game3()
        period_raw = np.where(frame_ids <= 69661, 1, 2)

    labels = labels_from_events(events, frame_ids)
    period = np.where(labels == X, 0, period_raw).astype(np.int8)
    m = Match("sample{}".format(g), px * length, py * width, team, bx * length, by * width,
              period, length, width, labels)
    np.savez(cache, px=m.px, py=m.py, team=m.team, bx=m.bx, by=m.by, period=m.period,
             labels=m.labels)
    return m
