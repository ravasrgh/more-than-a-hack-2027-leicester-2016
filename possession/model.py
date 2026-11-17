"""Frame classifiers + a three-state Viterbi decoder that turns them into whole possessions.

Two gradient-boosting models, both trained only on OTHER games (Metrica public samples):
  * dead model  -- P(ball dead | frame)
  * owner model -- P(team T has the ball | ball alive, frame seen from T)

The owner model is applied from both teams' points of view and the two answers are combined, so
it is symmetric by construction. The decoder then finds the H/A/D sequence maximising

    sum_t log P(state_t | frame_t)  -  switch penalties

which is what makes the output a small number of long possessions instead of per-frame flicker.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import H, A, D, X
from .features import team_features
from .tags import tag_features

DEAD_EXCLUDE = ()       # all features usable for the dead model
SEED = 0


def frame_table(m, context=None):
    """(features from home view, features from away view), tag context optional."""
    out = []
    for T in (0, 1):
        F = team_features(m, T)
        if context is not None:
            F = F.join(tag_features(m.n, context, T))
        out.append(F)
    return tuple(out)


def _clf(seed):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, random_state=seed)


def fit(tables, step=4, seed=SEED):
    """tables: list of (F_home, F_away, labels, period[, use_for_dead]).
    Returns (dead_model, owner_model). A table with use_for_dead=False trains the owner model
    only (a game whose dead-ball definition differs from the target one)."""
    Xd, yd, Xo, yo = [], [], [], []
    for tab in tables:
        Fh, Fa, labels, period = tab[:4]
        use_dead = tab[4] if len(tab) > 4 else True
        sel = np.flatnonzero((period > 0) & (labels != X))[::step]
        lab = labels[sel]
        for F, T in ((Fh, 0), (Fa, 1)):
            if use_dead:
                Xd.append(F.iloc[sel]); yd.append(lab == D)
            alive = lab != D
            Xo.append(F.iloc[sel[alive]]); yo.append(lab[alive] == T)
    import pandas as pd
    dead = _clf(seed).fit(pd.concat(Xd), np.concatenate(yd))
    owner = _clf(seed).fit(pd.concat(Xo), np.concatenate(yo))
    return dead, owner


def predict(models, Fh, Fa):
    """Per-frame (pD, pH_given_alive)."""
    dead, owner = models
    pD = 0.5 * (dead.predict_proba(Fh)[:, 1] + dead.predict_proba(Fa)[:, 1])
    ph = owner.predict_proba(Fh)[:, 1]
    pa = owner.predict_proba(Fa)[:, 1]
    pH = ph * (1 - pa)
    pA = pa * (1 - ph)
    pH = (pH + 1e-6) / (pH + pA + 2e-6)
    return pD, pH


def viterbi(logp, switch, to_dead, from_dead):
    """Best state path for logp (T, 3) with per-transition costs (in nats)."""
    T = len(logp)
    cost = np.array([[0, switch, to_dead],
                     [switch, 0, to_dead],
                     [from_dead, from_dead, 0]], np.float64)
    back = np.zeros((T, 3), np.int8)
    score = logp[0].copy()
    for t in range(1, T):
        cand = score[:, None] - cost          # from i to j
        back[t] = np.argmax(cand, 0)
        score = cand[back[t], np.arange(3)] + logp[t]
    path = np.zeros(T, np.int8)
    path[-1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(pD, pH, period, switch=40.0, to_dead=15.0, from_dead=15.0, temp=1.0, eps=1e-4):
    """H/A/D codes for every frame in a half, X elsewhere."""
    pD = np.clip(pD, eps, 1 - eps)
    pH = np.clip(pH, eps, 1 - eps)
    logp = np.stack([np.log(1 - pD) + np.log(pH),
                     np.log(1 - pD) + np.log(1 - pH),
                     np.log(pD)], 1) / temp
    out = np.full(len(pD), X, np.int8)
    for p in np.unique(period[period > 0]):
        sel = np.flatnonzero(period == p)
        out[sel] = viterbi(logp[sel], switch, to_dead, from_dead)
    return out
