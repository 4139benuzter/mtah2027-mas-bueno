# Adoption rule for the learned emissions (v35)

Written and committed on 2026-11-17, before any spellF1 of the learned model was computed (a smoke
test on one match checked the feature build, the team swap and that the numpy tree inference
equals LightGBM; it computed no score).

**Why.** v31's emissions are hand-built rules. Several other teams state in their public
repositories that gradient-boosted frame models trained on the open DFL and Metrica matches beat
rule-based decoders by about +0.15 spellF1 in their own held-out tests. We used none of their
code; the idea is common. Our attempts v32-v34 (calib/touch_*, domain_*) found no clear gain.

**Model** (`possession/learned.py`, `possession/learned_eval.py` at this commit). LightGBM,
multiclass H/A/D, the parameters in `LGB` and 120 rounds, on 67 per-frame features: tracking
features, v31's tag-free emissions and decoded path, contact-episode timing, and context over
+-0.5, 2 and 5 s. Training rows: every 10th graded frame of each degraded ball (seeds 0, 1, 2 of
`calib/domain_choice.json`), each also added with the two teams swapped. Decoding: the v31
Viterbi decoder with log-emissions T * log P; T, switch_team, switch_dead and the gate are chosen
from `GRID` (36 settings). On the competition match v31's Smart Tagging terms are added to the
log-emissions unchanged; they are zero on the public matches.

**Data.** The ten labelled other matches (Metrica 1-3, DFL x7), each degraded by
`calib/domain_choice.json` with seeds 0, 1, 2 (as in Amendment 1 of
`calib/domain_decoder_rule.md`). s(m) = spellF1 on match m, the mean over the three seeds. v31 on
the same degraded balls is the reference.

**Held-out gain (exact nesting).** For each match m: the decoding setting is the one with the
best mean s over the other nine matches, each of them predicted by a model trained without it
and without m (ties to the earlier `GRID` entry); m is predicted by a model trained on the other
nine; gain(m) = s(v35, m) - s(v31, m). `python -m possession.learned_eval select`.

**Adopt only if all hold:**
1. mean held-out gain over the ten matches >= +0.020;
2. gain(m) > 0 on at least 8 of the 10;
3. mean gain > 0 within the Metrica matches and within the DFL matches, separately;
4. the mean of the three group means (Metrica held-out, DFL held-out, SkillCorner with the final
   model, 19 matches as recorded, never trained on) is >= +0.005; SkillCorner alone may be
   negative;
5. the label-free checks of `calib/grid_rule.md` pass on the competition match, with the bound on
   possessions shorter than 2 s at 37.4%;
6. the numpy inference of the exported model equals LightGBM's (max abs difference < 1e-6).

**Final model.** Trained on all ten matches; the decoding setting chosen on all ten with
leave-one-match-out predictions (`python -m possession.learned_eval final`); exported to
`calib/learned_model.json`.

**If all hold:** the team says yes before any merge into main; `possession/run.py` uses the
learned emissions; the technical document describes v35 and the attempts v32-v34; the CSV is
regenerated from main and uploaded again. **Otherwise** v31 stays.

The public window is not scored for this decision.
