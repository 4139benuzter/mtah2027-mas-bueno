# Our solution: possession from control episodes

This fork adds `possession/`, a tracking-only pipeline that labels every frame of the tactical
video as `H` / `A` / `D`. It reads the provided ATD and Smart Tagging through `games.py`, uses no
video, no external data and no learned model beyond one logistic weight fitted to pseudo-labels
from the tracking itself. The full write-up is in
[docs/technical_document.md](docs/technical_document.md).

## Reproduce the submission

```bash
pip install -r requirements-possession.txt        # numpy 2.5.0, pandas 3.0.3 (Python >= 3.10)
# put the data bundle in ./data (see DATA.md) or set MTAH_DATA_DIR
python make_submission.py --team <team_name>
```

That writes `evaluation/results/<team_name>.csv`, runs the official validator on it and leaves
a cache of the parsed tracking in `cache/` (gitignored, like `data/`). The first run takes about
20 s to parse the tracking; later runs take about 10 s. The pipeline is deterministic: the same
data bundle produces a byte-identical file.

Score it against the public window like any other entry:

```bash
python evaluation/run_evaluation.py
python -m possession.diagnose evaluation/results/<team_name>.csv   # label-free health checks
```

## What is where

```
possession/
  atd.py        parse the FIFA/EPTS export into numpy arrays (metres), cached
  features.py   per-frame signals: ball visibility, distances, speeds, running share, team
                shape, out-of-bounds, restart hold
  tagging.py    Smart Tagging -> team prior, set pieces, goals
  episodes.py   contact episodes and their outcomes (the possession rules as evidence)
  model.py      emissions for a 3-state chain and the Viterbi decoder with fixed switch costs
  run.py        end to end -> frame,label CSV
  diagnose.py   label-free checks against Smart Tagging + the public-window segment table
  broadcast.py  broadcast-to-tactical alignment (evaluated, not used in the submission)
make_submission.py   the one command
```
