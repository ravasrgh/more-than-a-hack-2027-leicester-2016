"""Tactical-tag context: restart anchors and team phases.

For BAR-ATH they come from Metrica's automatic Smart Tagging. For the training games, which have
no Smart Tagging, an equivalent noisy context is synthesised from those games' own event data
(set pieces) and labels (phases), with error rates set to match what the real tagging looks like.
That lets the model learn how far to trust a tag instead of us guessing a weight.

Context = (anchors, phases[, whistles])
    anchors: list of (frame, team)   restart instants; team 0/1, or -1 when unknown (throw-ins)
    phases:  list of (start, stop, team)  team believed to be in possession over [start, stop)
    whistles: list of frames where a referee whistle was detected (BAR-ATH: possession.audio)
"""

import os
import sys

import numpy as np
import pandas as pd

from .data import ROOT, X, D

sys.path.insert(0, os.path.join(ROOT, "starter"))

# Metrica's set-piece tag opens ~3.5 s before the ball is played. Measured on the two kick-offs,
# whose true frames DATA.md / FAQ.md publish (tag start 11097 vs kick-off 11185, 113660 vs 113742),
# and consistent with player / ball acceleration profiles around every other set-piece tag.
SET_PIECE_LEAD = 88

POSSESSING = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
              "BALL IN FINAL THIRD", "BALL IN THE BOX", "LONG BALL", "PLAYERS IN THE BOX"}
DEFENDING = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


def smart_tagging_context(home="Barcelona", away="Atletic Club"):
    import load_smart_tagging as st
    team = {home.lower(): 0, away.lower(): 1}
    anchors, phases = [], []
    for e in st.load():
        code = (e["code"] or "").strip().upper()
        t = team.get((e["tags"].get("Team") or "").strip().lower(), -1)
        if code == "SET PIECES":
            anchors.append((e["start_frame"] + SET_PIECE_LEAD, t))
        elif t >= 0 and code in POSSESSING:
            phases.append((e["start_frame"], e["end_frame"], t))
        elif t >= 0 and code in DEFENDING:
            phases.append((e["start_frame"], e["end_frame"], 1 - t))
    return sorted(anchors), phases


def bar_ath_context():
    """Smart Tagging + detected referee whistles for the competition match."""
    from .audio import detect
    anchors, phases = smart_tagging_context()
    _, events = detect()
    return anchors, phases, sorted(int(e[0]) for e in events)


# Whistle detector behaviour on BAR-ATH, measured without labels (possession.audio):
# ~60% recall on moments that must carry a whistle, 79 events inside play.
WHISTLE_RECALL = 0.6
WHISTLES_PER_MATCH = 79


def synthetic_whistles(events, labels, rng, recall=WHISTLE_RECALL, total=WHISTLES_PER_MATCH):
    """Referee whistles a detector like ours would report on a training game.

    True whistles: fouls, offsides, goals, ends of halves (stop play) and kick-offs (restart).
    Each is detected with probability `recall` with ~0.3 s timing noise; false detections are
    spread uniformly over play so the total matches what the real detector reports per match.
    """
    ev = events.sort_values("start").reset_index(drop=True)
    true = []
    for _, r in ev.iterrows():
        sub = r["subtype"]
        if r["type"] == "FAULT RECEIVED" or (r["type"] == "CHALLENGE" and "FAULT" in sub):
            true.append(int(r["start"]) - 1 + 8)
        elif r["type"] == "BALL LOST" and "OFFSIDE" in sub:
            true.append(int(r["start"]) - 1 + 25)
        elif r["type"] == "SHOT" and "GOAL" in sub:
            true.append(int(r["end"]) - 1 + 25)
        elif r["type"] == "SET PIECE" and "KICK OFF" in sub:
            true.append(int(r["start"]) - 1 - 38)
    for _, g in ev.groupby("period"):
        true.append(int(g["end"].max()) - 1)
    true = sorted(set(true))
    out = [t + int(rng.normal(0, 8)) for t in true if rng.random() < recall]
    play = np.flatnonzero(labels != X)
    n_false = max(0, total - len(out))
    out += list(rng.choice(play, n_false, replace=False))
    return sorted(int(x) for x in out)
def synthetic_context(events, labels, rng, recall=0.6, jitter=12, phase_rate=0.8, flip=0.08):
    """Noisy Smart-Tagging-like context for a training game (frame index = event frame - 1)."""
    anchors = []
    sp = events[events["type"] == "SET PIECE"]
    for _, r in sp.iterrows():
        if rng.random() > recall:
            continue
        t = -1 if "THROW" in r["subtype"] else int(r["team"])
        anchors.append((int(r["start"]) - 1 + int(rng.normal(0, jitter)), t))

    phases = []
    lab = labels
    edges = np.flatnonzero(np.diff(lab)) + 1
    bounds = np.concatenate(([0], edges, [len(lab)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        t = int(lab[a])
        if t not in (0, 1) or b - a < 75 or rng.random() > phase_rate:
            continue
        s = a + int(rng.normal(0, 40))
        e = b + int(rng.normal(0, 50))
        if rng.random() < 0.5:
            e = max(e, s + int(rng.uniform(250, 700)))
        if rng.random() < flip:
            t = 1 - t
        phases.append((max(0, s), min(len(lab), e), t))
    return sorted(anchors), phases, synthetic_whistles(events, labels, rng)


def tag_features(n, context, T):
    anchors, phases = context[0], context[1]
    whistles = context[2] if len(context) > 2 else None
    f = {}
    idx = np.arange(n)
    if anchors:
        af = np.array([a for a, _ in anchors])
        at = np.array([t for _, t in anchors])
        rel = np.where(at < 0, 0.0, np.where(at == T, 1.0, -1.0))
        nxt = np.searchsorted(af, idx, side="left")      # first anchor >= t
        prv = nxt - 1
        has_n, has_p = nxt < len(af), prv >= 0
        f["until_sp"] = np.where(has_n, af[np.minimum(nxt, len(af) - 1)] - idx, np.nan)
        f["since_sp"] = np.where(has_p, idx - af[np.maximum(prv, 0)], np.nan)
        f["next_sp_team"] = np.where(has_n, rel[np.minimum(nxt, len(af) - 1)], np.nan)
        f["prev_sp_team"] = np.where(has_p, rel[np.maximum(prv, 0)], np.nan)
        for k in ("until_sp", "since_sp"):
            f[k] = np.minimum(f[k], 3000)
    else:
        for k in ("until_sp", "since_sp", "next_sp_team", "prev_sp_team"):
            f[k] = np.full(n, np.nan)
    score = np.zeros(n, np.float32)
    cover = np.zeros(n, np.float32)
    for s, e, t in phases:
        score[s:e] += 1.0 if t == T else -1.0
        cover[s:e] += 1.0
    f["phase"] = np.where(cover > 0, score / np.maximum(cover, 1), np.nan)
    f["phase_cover"] = cover
    if whistles is not None:
        w = np.array(sorted(whistles), int)
        nxt = np.searchsorted(w, idx, side="left")
        prv = np.searchsorted(w, idx, side="right") - 1
        f["since_wh"] = np.where(prv >= 0, np.minimum(idx - w[np.maximum(prv, 0)], 750), 750)
        f["until_wh"] = np.where(nxt < len(w), np.minimum(w[np.minimum(nxt, len(w) - 1)] - idx, 750), 750)
    ph = pd.Series(np.nan_to_num(f["phase"]))
    f["phase_c4"] = ph.rolling(201, center=True, min_periods=1).mean().to_numpy()
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
