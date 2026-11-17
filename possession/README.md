# possession — frame-level ball possession for BAR-ATH

Trains on **other games only** (Metrica public samples 1–3 + IDSSE Bundesliga open data), applies
to the competition ATD, and decodes H/A/D with a 3-state Viterbi. No label of the provided game is
used for training, tuning or validation.

## Reproduce the submission

```bash
# 1. environment (Python 3.14.10)
pip install -r requirements-possession.txt

# 2. competition data bundle -> data/ (as described in DATA.md)

# 3. Metrica sample data (other games, public)
git clone https://github.com/metrica-sports/sample-data.git reference/sample-data
git -C reference/sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996

# 4. IDSSE open data (7 Bundesliga matches, CC-BY 4.0, ~2.6 GB, DOI 10.6084/m9.figshare.28196177)
mkdir -p data/external/idsse
curl -s https://api.figshare.com/v2/articles/28196177 | python -c "import json,sys;
[print(f['download_url'], f['name']) for f in json.load(sys.stdin)['files']]" |
while read url name; do curl -sL -o "data/external/idsse/$name" "$url"; done

# 5. generate and check
python -m possession.run --out evaluation/results/<team>.csv
python evaluation/validate_submission.py evaluation/results/<team>.csv
```

Paths can be overridden with `METRICA_SAMPLE_DIR` and `IDSSE_DIR`. Intermediate caches go to
`data/cache/` (gitignored); deleting it only makes the first run slower.

## Validation (labels of other games only)

```bash
python -m possession.validate --idsse all           # leave-one-Metrica-game-out spellF1 / macroF1
python -m possession.validate --idsse all --grid    # decoder penalty grid
```

## Modules

| file | role |
|---|---|
| `data.py` | loaders: BAR-ATH ATD, Metrica samples (labels derived from events) |
| `idsse.py` | IDSSE loader (labels from DFL BallStatus / BallPossession) |
| `tags.py` | Smart-Tagging context (real for BAR-ATH, synthetic for training games) |
| `features.py` | per-frame features + degradation of clean tracking to ATD-like quality |
| `model.py` | gradient-boosting dead-ball and owner models, Viterbi decoder |
| `validate.py` | leave-one-game-out validation on the Metrica samples |
| `run.py` | trains on all other games and writes the submission |
| `audio.py` | whistle detector (kept, switched off) |
| `definition_check.py`, `viz.py` | pipeline sanity checks and visualisation (not used by `run.py`) |
