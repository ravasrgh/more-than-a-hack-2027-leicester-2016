"""Leave-one-game-out validation on the Metrica public sample games (OTHER games only).

This is where every design choice and hyper-parameter is chosen. The competition match's public
five minutes are never used for that -- RULES.md forbids it, and EVALUATION.md shows why it would
pick the wrong answer anyway.

    python -m possession.validate
"""

import argparse
import itertools
import zlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

from .data import X, load_bar_ath, load_metrica_sample, sample_events  # noqa: E402
from .tags import synthetic_context  # noqa: E402
from .features import degrade, run_lengths  # noqa: E402
from .model import decode, fit, frame_table, predict  # noqa: E402

GAMES = (1, 2, 3)
PRE_STEP = 4      # external games are subsampled when their tables are built (memory); fit uses all kept frames
PFF_STEP = 8      # PFF: 64 games, keep every 8th frame
PRE_STEP_ALL = False   # also subsample IDSSE (needed when PFF is added)


def atd_ball_runs():
    m = load_bar_ath()
    g = m.period > 0
    return run_lengths(np.isfinite(m.bx[g]))


def _game(g):
    """(Match, Metrica-style events) for a Metrica sample (int) or an IDSSE match id (str)."""
    if isinstance(g, str) and g.startswith("pff:"):
        from .pff import load_pff, restart_events
        m = load_pff(g[4:])
        return m, restart_events(m)
    if isinstance(g, str):
        from .idsse import load_idsse, restart_events
        m = load_idsse(g)
        return m, restart_events(m)
    return load_metrica_sample(g), sample_events(g)


def build_tables(n_aug=2, seed=0, tags=True, whistles=True, games=GAMES, owner_only=()):
    """{game: [(Fh, Fa, labels, period), ...]} -- one per degraded copy."""
    vis, miss = atd_ball_runs()
    out = {}
    for g in games:
        # one stream per game so adding games never changes the copies of the others
        rng = np.random.default_rng([seed, zlib.crc32(g.encode())] if isinstance(g, str)
                                    else seed + 1000 * g)
        m, events = _game(g)
        out[g] = []
        for _ in range(n_aug):
            d = degrade(m, vis, miss, rng)
            ctx = synthetic_context(events, m.labels, rng) if tags else None
            if ctx is not None and not whistles:
                ctx = ctx[:2]
            Fh, Fa = frame_table(d, ctx)
            k = PFF_STEP if isinstance(g, str) and g.startswith("pff:") else \
                PRE_STEP if isinstance(g, str) and PRE_STEP_ALL else 1
            if k > 1:   # large external sets: keep every k-th frame now, fit uses them all
                out[g].append((Fh.iloc[::k].reset_index(drop=True), Fa.iloc[::k].reset_index(drop=True),
                               m.labels[::k], m.period[::k], g not in owner_only, 1))
            else:
                out[g].append((Fh, Fa, m.labels, m.period, g not in owner_only))
    return out


def score(labels, pred):
    gt = labels.astype(np.int64)
    pr = pred.astype(np.int64)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pr, runs)
    cm = pm.scored_confusion(pm.confusion(gt, pr))
    sp = pm.spell_scores(gt, filled, runs)
    return sp["mean_f1"], pm.macro_f1(cm), sp["n_gt"], sp["n_pred"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--no-whistles", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--mask", choices=("state", "random"), default="state")
    ap.add_argument("--touch", type=int, default=1, help="ball-touch features on/off")
    ap.add_argument("--ball-smooth", type=float, default=6, help="Gaussian sigma (frames) for the training ball, 0 = off")
    ap.add_argument("--pff", type=int, default=0, help="add the first N PFF World Cup 2022 matches")
    ap.add_argument("--pff-step", type=int, default=8)
    ap.add_argument("--pre-all", action="store_true", help="subsample IDSSE tables too (memory)")
    ap.add_argument("--save", help="write the held-out probabilities to this .npz (error analysis)")
    ap.add_argument("--idsse", choices=("all", "owner"), help="add the 7 IDSSE matches to training "
                    "(all = both models, owner = possession model only)")
    args = ap.parse_args()
    from . import features
    features.RESTART = args.restart
    features.MASK = args.mask
    features.TOUCH = bool(args.touch)
    features.BALL_SMOOTH = args.ball_smooth

    global PRE_STEP_ALL, PFF_STEP
    PFF_STEP = args.pff_step
    PRE_STEP_ALL = args.pre_all
    extra = ()
    if args.idsse:
        from .idsse import MATCHES
        extra = MATCHES
    if args.pff:
        from .pff import all_matches
        extra = tuple(extra) + tuple("pff:" + x for x in all_matches()[:args.pff])
    tables = build_tables(args.aug, seed=args.seed, tags=not args.no_tags, whistles=not args.no_whistles,
                          games=GAMES + tuple(extra),
                          owner_only=extra if args.idsse == "owner" else ())
    probs = {}
    for g in GAMES:                    # always tested on the Metrica samples (organisers' definition)
        train = [t for h in tables if h != g for t in tables[h]]
        models = fit(train)
        Fh, Fa, labels, period = tables[g][0][:4]
        probs[g] = predict(models, Fh, Fa) + (labels, period)
        pD, pH = probs[g][:2]
        alive = (labels < 2) & (period > 0)
        acc_owner = np.mean((pH[alive] > 0.5) == (labels[alive] == 0))
        acc_dead = np.mean((pD[period > 0] > 0.5) == (labels[period > 0] == 2))
        print("game {}: frame owner acc {:.3f}  dead acc {:.3f}".format(g, acc_owner, acc_dead))

    if args.save:
        np.savez(args.save, **{"{}_{}".format(k, g): v for g in GAMES
                               for k, v in zip(("pD", "pH", "labels", "period"), probs[g])})
    grid = [dict(switch=s, to_dead=d, from_dead=d) for s, d in
            itertools.product((6, 9, 12, 16, 20, 30), (6, 9, 12, 16, 20))] if args.grid else \
        [dict(switch=20, to_dead=16, from_dead=16)]   # = run.PARAMS
    best = None
    for params in grid:
        res = []
        for g in GAMES:
            pD, pH, labels, period = probs[g]
            res.append(score(labels, decode(pD, pH, period, **params)))
        res = np.array(res)
        spell, macro = res[:, 0].mean(), res[:, 1].mean()
        print("{}  spellF1 {:.4f}  macroF1 {:.4f}  poss pred/gt {}".format(
            params, spell, macro, [(int(r[3]), int(r[2])) for r in res]))
        if best is None or spell > best[0]:
            best = (spell, params)
    print("best", best)


if __name__ == "__main__":
    main()
