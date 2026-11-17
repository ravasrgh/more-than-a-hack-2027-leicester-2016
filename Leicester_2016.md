# Ball possession from automatic tracking — technical document

**Team:** Leicester 2016 · **Competition:** More than a Hack 2027, Phase 1 (Ball Possession)
**Submission:** `Leicester_2016.csv` · **Code:** `possession/` in this repository (`possession/README.md`)

## 1. Summary

We label every frame of BAR–ATH as H / A / D (X outside the halves) with a learned model that is
trained **only on other games**, then decode whole possessions with a three-state Viterbi. Inputs at
test time are the competition's automatic tracking (ATD) and Metrica's Smart Tagging; nothing else.

```
other games (Metrica samples 1–3, IDSSE 7 Bundesliga matches) — clean tracking + labels
   │  degrade to ATD quality (state-dependent ball loss, smooth noise, player dropout)
   │  + synthetic Smart-Tagging context
   ▼
team-level features per frame (seen from each team) ──► HGB "dead" model  P(D | frame)
                                                      ──► HGB "owner" model P(team has ball | alive)
   ▼
3-state Viterbi (H, A, D) with switch / to-dead / from-dead penalties  ──► frame,label
```

Leave-one-game-out validation (§5) on the Metrica games: **spellF1 0.676, macroF1 0.930**
(mean of 3 seeds).

## 2. Why this approach

* **The ball is missing on 41 % of graded ATD frames**, in ~150 long gaps (median 10.7 s). A
  rule such as "nearest player to the ball" has nothing to say on those frames. The owner has to be
  inferred from the team shape: who is pushing up, who is compact, and how fast everyone moves.
  That is a pattern-recognition problem, so we learn it from labelled games.
* **No labels of BAR–ATH may be used**, so labelled data must come from other games. Their
  tracking is far cleaner than the ATD. Making it *look like* the ATD (§4) mattered more than any
  feature or model change we tried.
* **spellF1 rewards whole possessions, not frames.** Per-frame argmax gives dozens of
  one-second flickers per minute. An HMM-style decoder with transition penalties is the simplest
  way to trade frame accuracy for clean possession boundaries.
* **Features are team-level** (centroids, spreads, speeds, nearest-player distances,
  counts near the ball). They do not depend on player identity, so the ATD's 46 anonymous,
  fragmenting track slots are not a problem.
* **Ball-touch features** catch short possessions that team shape reacts to too slowly.
  A touch is a peak in the ball's velocity change (> 4 m/s over 0.24 s) with a player within
  2.5 m. From each touch we derive the team of the previous / next touch (and the one before /
  after), the time since / until them, touch balance and counts per team in ±1 s and ±3 s, and
  which team is nearer the ball over ±0.25 s.

## 3. Training data

No data about BAR–ATH produced by a human is used anywhere. Labels are derived **mechanically**
from each game's own data.

| dataset | games | graded frames | labelled by | label source |
|---|---|---|---|---|
| Metrica Sports public sample data (github.com/metrica-sports/sample-data, commit `e706dd5`) | 3 | 424,886 (≈4.7 h) | Metrica's human event annotators | our rules on the event data (below) |
| IDSSE — Bassek, Rein, Weber & Memmert, *Scientific Data* 12 (2025), CC-BY 4.0, DOI 10.6084/m9.figshare.28196177 | 7 Bundesliga 2022/23 | 1,002,644 (≈11.1 h) | DFL / Sportec Solutions operators (official feed) | `BallStatus` (alive/dead) and `BallPossession` (home/guest) on every ball frame of the position feed |

Labels from Metrica events (`possession/data.py: labels_from_events`):

* The owner of the alive ball is the team of the most recent on-ball event (pass, carry, shot,
  set piece, recovery, ball lost, ball out). A loose ball stays with the last team in possession
  until the other team makes an on-ball event.
* The ball is **dead** from the end of the event before a SET PIECE until that set piece is taken.
  For a foul, the dead spell starts at the offence (FAULT RECEIVED / CHALLENGE-FAULT).
* Outside kick-off → last event of each half the label is X.

The two sources draw the line differently. IDSSE marks much more dead time (41–52 % of frames vs
25–33 %) and has more, shorter possessions (370–511 per game vs 357–404). We still add IDSSE to
both models because it helps on the Metrica validation games (+0.010 spellF1, §5). The validation
test games are always Metrica, since that is the organisers' definition.

## 4. Handling missing and noisy tracking

Clean training tracking is **degraded to match the ATD**. Every parameter was set from
**unlabelled** ATD statistics only (`possession/features.py: degrade`):

| ATD property (measured, no labels) | how we reproduce it |
|---|---|
| ball missing 41 % of frames, in long runs (median 10.7 s) | real ATD visible/missing run lengths are resampled |
| the ball goes missing mainly when play is dead: player speed is 1.14 m/s while the ball is missing vs 2.09 m/s while visible. A two-component mixture gives P(missing \| dead) ≈ 0.95–1.0 and P(missing \| alive) ≈ 0.06 | state-dependent mask: dead spells are hidden with ±1 s jitter at their edges; alive play loses the ball at 6 % |
| positions are smooth: median 2nd difference 0.011 m for the ball, 0.001 m for players | Gaussian noise smoothed over ~1 s (σ 0.5 m ball, 0.2 m players), not white jitter |
| the ATD ball is low-pass filtered: 90th / 99th percentile of the ball's velocity change over ±1, ±3, ±6 frames is 1.5/2.9, 4.3/8.7, 8.3/16.5 m/s (clean training ball: 0.4/16.9, 7.4/24.9, 12.9/27.9) | training ball smoothed with a Gaussian of σ = 6 frames, which gives 1.6/3.0, 4.6/8.8, 8.5/15.8 |
| track slots drop out | random 10 s dropout segments on 7 % of player-frames |

The clean ball is first interpolated through its own gaps. In the corrected sample data the ball is
missing almost exactly when it is dead, which would leak the label.

Two copies of each training game are degraded with different random streams. At test time the
model sees NaNs natively (gradient boosting handles missing values). The features also carry the
context around a gap: time since / until the ball was last seen, and ball position and
nearest-player gap carried forward and backward.

**Smart Tagging.** Set-piece tags give restart anchors, and phase tags (build-up, defending in
third …) give a noisy "team believed in possession". The training games have no Smart Tagging, so
an equivalent noisy context is synthesised from their events and labels: 60 % of set pieces kept
with ±0.5 s jitter, throw-ins with unknown team, 80 % of possessions tagged, 8 % with the wrong
team. The model therefore *learns* how far to trust a tag. The set-piece tag opens 88 frames
before the ball is played, a lead we measured on the two published kick-off frames.

## 5. Validation and model selection

Leave-one-game-out: train on two Metrica games (+ all IDSSE), test on the held-out Metrica game
(degraded the same way), and score with the official `possession_metrics`. **All choices were made
here.** The 5-minute public window of BAR–ATH was only used to check that the pipeline runs end to
end, never to select or tune.

| variant | spellF1 | macroF1 |
|---|---|---|
| random ball mask (missingness independent of play state) | 0.579 | 0.893 |
| state-dependent mask (final) | 0.650 | 0.924 |
| + IDSSE, owner model only | 0.666 (seed 0) | – |
| + IDSSE, both models, mean of seeds 0/1/2 | 0.664 (0.671 / 0.654 / 0.667) | 0.926 |
| same without IDSSE, seeds 0/1/2 | 0.654 (0.663 / 0.647 / 0.651) | 0.926 |
| + ATD-matched ball smoothing, mean of seeds 0/1/2 | 0.660 (0.666 / 0.651 / 0.664) | 0.926 |
| **+ ball-touch features (final)**, mean of seeds 0/1/2 | **0.676** (0.677 / 0.665 / 0.688) | 0.930 |

Rows that change the degradation (mask, smoothing) also change the degraded test games, so they
are not strictly comparable with the rows above them. The mask and the smoothing were chosen because
they match the ATD statistics (§4), not because of their score. Rows that change only the training
data or the features use the same test set and are directly comparable: IDSSE +0.010 and touch
features +0.016, both positive on every seed.

Decoder penalties (switch 20, to-dead 16, from-dead 16 nats) are the centre of a flat optimum of a
6 × 5 grid (all cells 0.649–0.656 without IDSSE; with IDSSE the chosen cell is 0.6710 vs best 0.6716), so they are not finely tuned.

**Label-free consistency check on BAR–ATH**, the only check of the output we allowed ourselves:

| | final output | first version | expected from ATD mixture |
|---|---|---|---|
| dead share | 0.382 | 0.435 | ≈ 0.37 |
| P(D \| ball missing) | 0.91 | 0.71 | ≈ 0.91 |
| P(D \| ball visible) | 0.02 | 0.24 | ≈ 0–0.03 |

Final output: H 59,493 · A 38,674 · D 60,690 · X 40,462 frames; 288 possessions.

## 6. Possession rules implemented

* **Loose balls, deflections, aerial duels:** stay with the team last in control until the other
  team makes a controlled on-ball action (inherited from the training labels). The switch penalty
  stops a single touch by the defending team from creating a possession.
* **Challenges / tackles:** possession changes only when the winning team keeps the ball. A foul
  makes the ball dead from the offence until the free kick is taken.
* **Restarts:** dead from the stoppage until the set piece is played. The restart belongs to the
  team taking it (kick-off, free kick, corner, goal kick, throw-in).
* **Outside the halves:** X, from the published half boundaries of the tactical video.

## 7. What we tried and dropped

| idea | result | decision |
|---|---|---|
| referee-whistle detector from broadcast audio (`possession/audio.py`), with synthetic whistles in training | ~60 % recall measured without labels; no LOGO gain | kept in code, switched off |
| restart-pattern features from tracking (ball still then struck, players gathering in the box, kick-off shape) | +0.001 LOGO (seed noise ≈ ±0.008) | kept in code, switched off (`features.RESTART`) |
| ball mask independent of play state | clearly worse BAR–ATH consistency (P(D \| visible) 0.24) | replaced by the state mask |
| IDSSE for the owner model only (its dead-ball definition differs) | +0.003 vs +0.008 for both models | IDSSE used for both |
| PFF FC FIFA World Cup 2022 open data (64 matches, broadcast tracking + events; free download after registration, no written licence; labels derived from PFF's events like the Metrica ones, loader `possession/pff.py`) | 16 matches added: 0.672 vs 0.674 without (XGBoost, seeds 0/1/2: +0.004 / −0.006 / −0.004), macroF1 lower on every seed. Its definition marks more dead time (median 45 %) | kept in code (`--pff N`), off |
| XGBoost instead of scikit-learn HGB (optionally on GPU, `--backend xgb --device cuda`) | 0.674 vs 0.676 (seeds 0/1/2), less memory | kept as an option, HGB used (deterministic on CPU) |
| tuning the decoder further: smoothing the frame probabilities before Viterbi, 48 penalty combinations | all 0.670–0.675 (seed noise) | not used |
| broadcast video (ball detection to fill gaps) | not synchronised to tactical; not feasible in time | not used |

## 8. Limitations

* **Too few possessions.** On validation we predict ~270 possessions per game against ~380 true
  ones. Half of the true possessions last under 5 s; only 13 % of those under 2 s are recovered
  (IoU ≥ 0.5), against 90 % of those over 10 s. Most missed short ones are given to the other team
  by the frame model, the rest are merged into a neighbour by the decoder. This is the main spellF1
  loss. (Breakdown measured before the touch features; they raise spellF1 but the predicted
  count stays ~270.)
* **Inferring the owner while the ball is invisible** relies on team shape. Long build-ups behind
  a missing ball are fine, but a turnover inside a ball gap is often placed late.
* **Only 3 games share the target definition.** The degradation is matched to ATD statistics, not
  learned. IDSSE uses a different line for dead ball and possession.
* **What we would do with more time:** more labelled games in Metrica's definition, a
  turnover-specific detector to recover short possessions, an ensemble over degradation seeds, and
  ball detection in the tactical video to shrink the gaps.

## 9. Reproduction

Full instructions are in `possession/README.md`. In short (Python 3.14.10):

```bash
pip install -r requirements-possession.txt
# competition bundle -> data/ ; Metrica samples -> reference/sample-data (commit e706dd5)
# IDSSE -> data/external/idsse (download command in possession/README.md)
python -m possession.run --out Leicester_2016.csv
python evaluation/validate_submission.py Leicester_2016.csv
```

All randomness is seeded: one random stream per training game, and fixed `random_state` for the
models. Re-running gives a byte-identical file (checked by regenerating and comparing).
Validation: `python -m possession.validate --idsse all [--seed k] [--grid]`.

## 10. Tools

Code was written with the help of an AI coding assistant (Claude Code). No model, API or service is
called by the pipeline itself. Everything runs locally from the repository and the open datasets
above.

## References

* Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and
  event data in elite soccer. *Scientific Data* 12. Data: doi.org/10.6084/m9.figshare.28196177
  (CC-BY 4.0).
* Metrica Sports sample data: github.com/metrica-sports/sample-data.
