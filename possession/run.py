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
from .validate import GAMES, ROOT, build_tables

LETTERS = np.array(["H", "A", "D", "X"])
WHISTLES = False   # whistle detector kept (possession.audio) but off: no measurable gain, precision unverified
PARAMS = dict(switch=20.0, to_dead=16.0, from_dead=16.0)   # centre of the flat LOGO optimum (possession.validate --grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "tracking_hmm.csv"))
    ap.add_argument("--aug", type=int, default=2)
    args = ap.parse_args()

    # Metrica samples + IDSSE Bundesliga matches (other games); LOGO: +0.010 spellF1 over 3 seeds
    tables = build_tables(args.aug, whistles=WHISTLES, games=GAMES + IDSSE_MATCHES)
    models = fit([t for g in tables for t in tables[g]])
    m = load_bar_ath()
    Fh, Fa = frame_table(m, bar_ath_context() if WHISTLES else smart_tagging_context())
    pD, pH = predict(models, Fh, Fa)
    np.savez(os.path.join(ROOT, "data", "cache", "bar_ath_probs.npz"), pD=pD, pH=pH)
    codes = decode(pD, pH, m.period, **PARAMS)

    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(LETTERS[codes]))
    print("wrote", args.out, {k: int(np.sum(codes == i)) for i, k in enumerate("HADX")})


if __name__ == "__main__":
    main()
