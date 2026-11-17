"""Produce the submission file for BAR-ATH.

    python -m possession.run --out evaluation/results/<team>.csv

Trains on other games only (Metrica public samples + IDSSE open data), applies to the competition ATD,
decodes with the parameters chosen by possession.validate, and writes frame,label.
"""

import argparse
import os

import numpy as np

from .data import load_bar_ath
from .model import decode, fit, frame_table, predict
from .tags import bar_ath_context, smart_tagging_context
from .idsse import MATCHES as IDSSE_MATCHES
from .validate import ENS_SEED_STEP, GAMES, ROOT, build_tables

LETTERS = np.array(["H", "A", "D", "X"])
WHISTLES = False   # whistle detector kept (possession.audio) but off: no measurable gain, precision unverified
PARAMS = dict(switch=20.0, to_dead=16.0, from_dead=16.0)   # centre of the flat LOGO optimum (possession.validate --grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "tracking_hmm.csv"))
    ap.add_argument("--aug", type=int, default=2)
    ap.add_argument("--pff", type=int, default=0, help="also train on the first N PFF World Cup 2022 "
                    "matches (needs ~16+ GB RAM; see possession/README.md)")
    ap.add_argument("--backend", choices=("hgb", "xgb"), default=None,
                    help="model library (default hgb); xgb with --device cuda trains on a GPU")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ens", type=int, default=1, help="average K models trained on differently "
                    "degraded training copies")
    args = ap.parse_args()
    from . import model as model_mod
    if args.backend:
        model_mod.BACKEND = args.backend
    if args.device:
        model_mod.DEVICE = args.device

    # Metrica samples + IDSSE Bundesliga matches (other games); LOGO: +0.010 spellF1 over 3 seeds
    games = GAMES + IDSSE_MATCHES
    if args.pff:
        from .pff import all_matches
        games = games + tuple("pff:" + x for x in all_matches()[:args.pff])
    m = load_bar_ath()
    Fh, Fa = frame_table(m, bar_ath_context() if WHISTLES else smart_tagging_context())
    pD = pH = 0
    for k in range(args.ens):   # member k: training copies degraded with seed 101 k (k = 0: as before)
        tables = build_tables(args.aug, seed=ENS_SEED_STEP * k, whistles=WHISTLES, games=games)
        models = fit([t for g in tables for t in tables[g]])
        del tables
        d, h = predict(models, Fh, Fa)
        pD, pH = pD + d / args.ens, pH + h / args.ens
    np.savez(os.path.join(ROOT, "data", "cache", "bar_ath_probs.npz"), pD=pD, pH=pH)
    codes = decode(pD, pH, m.period, **PARAMS)

    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(LETTERS[codes]))
    print("wrote", args.out, {k: int(np.sum(codes == i)) for i, k in enumerate("HADX")})


if __name__ == "__main__":
    main()
