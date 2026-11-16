# Our solution: possession from control episodes

This fork adds `possession/`, a tracking-only pipeline that labels every frame of the tactical
video as `H` / `A` / `D`. It reads the provided ATD and Smart Tagging through `games.py`, uses no
video and no learned model beyond one logistic weight fitted to pseudo-labels from the tracking
itself. It is validated on ten other matches with public labels (Metrica's sample matches, the
DFL open dataset): `possession/validate.py`. Two existing settings (carry time, minimum contact)
were chosen on those matches by a rule fixed in advance (`calib/grid_rule.md`); nothing was
chosen on the competition match. The full write-up is in
[docs/technical_document.md](docs/technical_document.md).

## Reproduce the submission

```bash
pip install -r requirements-possession.txt        # numpy 2.5.0, pandas 3.0.3 (Python >= 3.12; tested on 3.13 and 3.14)
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
  keepers.py    goalkeeper team correction: a keeper in a penalty area -> the side defending it
  features.py   per-frame signals: ball visibility, distances, speeds, running share, team
                shape, out-of-bounds, restart hold
  tagging.py    Smart Tagging -> team prior, set pieces, goals
  episodes.py   contact episodes and their outcomes (the possession rules as evidence)
  model.py      emissions for a 3-state chain and the Viterbi decoder; a change of side is
                cheap only where the gaining side starts a control episode
  run.py        end to end -> frame,label CSV
  diagnose.py   label-free checks against Smart Tagging (reads no ground truth)
  broadcast.py  broadcast-to-tactical alignment (evaluated, not used in the submission)
  sample_game.py  Metrica's public sample matches 1-3 (other matches): tracking + event labels
  dfl.py          DFL/Sportec open matches (CC BY 4.0) -> arrays + labels (needs kloppy 3.19.1)
  validate.py     out-of-sample score of the current decoder on those matches
  grid.py         the settings grid on those matches; grid_select.py applies the fixed rule
  skillcorner.py  SkillCorner open matches (MIT) -> arrays + labels; sc_check.py, three_group.py
                  the independent check and its fixed follow-up rule
hpc/               the conversion and validation as PBS jobs (optional)
calib/             our validation results
make_submission.py   the one command
```
