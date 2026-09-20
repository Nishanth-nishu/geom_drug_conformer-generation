# GEOM-Drugs training fixes (2026-09)

The model trained cleanly on QM9 but produced unusable, exploding coordinates
on GEOM-Drugs, and later plateaued at a training-invariant floor even after
the obvious bugs were fixed. This document records what was actually wrong
and what changed, file by file.

## 1. Mismatched diffusion-variable representation

GeoDiff's paper defines the standard scaled DDPM forward process,
`C_t = √ᾱ_t·C_0 + √(1-ᾱ_t)·ε`. Its official implementation (and this
project's `q_sample`) works instead with the rescaled variable
`C̃_t = C_t/√ᾱ_t = C_0 + √((1-ᾱ_t)/ᾱ_t)·ε` — the same forward process, just
divided through by `√ᾱ_t`. One reverse-sampling code path was recovering
`C_0` with the inversion formula valid for the *raw* scaled `C_t`
(`(1/√ᾱ_t)·C_t − √(1/ᾱ_t−1)·ε`) but applying it to `pos`, which was actually
already in the rescaled `C̃_t` form the rest of the codebase uses — where the
correct inversion is the simpler `C_0 = C̃_t − √((1-ᾱ_t)/ᾱ_t)·ε`, no extra
`1/√ᾱ_t` factor. At `t` near `T`, `1/√ᾱ_t` blows up (~300x at this project's
schedule), so mixing the two representations compounds into outright
coordinate explosion.

**Fix:** made every reverse-sampling code path consistently use the same
rescaled-`C̃_t` inversion formula that `q_sample` and GeoDiff's own code use,
in `models/dual_encoder_diffusion.py`.

## 2. Data pipeline standardization

`data/prepare_geom_drugs_standard.py` replaces an ad-hoc random 90/10 split
with:
- **Split membership** from GeoMol's published `smiles_splits/split0.npy`
  (SMILES-string → train/val/test) instead of a private random split.
  TorsionalDiffusion's own dataloader consumes GeoMol's index-based
  `splits/split0.npy`; the two representations were not independently checked
  to select identical molecules, so comparability with GeoMol/TorsionalDiffusion
  numbers is approximate. GeoDiff's own paper evaluates on the ConfGF
  (Shi et al., 2021) split with a 200-molecule test set, so its published
  numbers are not a same-test-set comparison.
- **Connectivity QC** (`clean_confs`, matching TorsionalDiffusion's
  `standardize_confs.py`): for each conformer, recompute its 2D bond graph
  from the 3D structure and discard the conformer if it doesn't match the
  molecule's intended graph.
- Two labeled, deliberate deviations from TorDiff's own defaults: a
  try/except around the per-conformer connectivity check (robustness against
  malformed records TorDiff's original doesn't guard against), and
  reproducing TorDiff's *opt-in* `--boltzmann top` sort policy rather than
  its actual default (`None`).
- `contextlib.ExitStack` for the three output file handles, a guard around
  `process_molecule()` so one malformed record can't crash a multi-hour run,
  and a duplicate-SMILES-aware hard cap on the fixed-size test subsample.

## 3. Training-loss weighting

`get_loss` was applying `w_global` as a multiplier on the training loss. In
GeoDiff's own reference implementation, `w_global` is exclusively an
inference-time force-mixing coefficient — it has no role in the training
objective. At the default `w_global=0.5` this had silently halved the
intended global/local training balance (effective global weight 1.0 instead
of the intended fixed 2.0). Fixed to `5.0·loss_local + 2.0·loss_global` with
no `w_global` factor, matching GeoDiff. A related diagnostic-only bug meant
the logged `local`/`global` loss breakdown printed the *combined* loss for
both fields instead of each component separately — cosmetic (never affected
backprop), now fixed.

## 4. Sampler interface and safety-clip wiring

`clip_local`/`clip_global` parameters existed on `ddim_sample` but were never
actually passed through to the underlying `eq_transform` clip logic — the
high-noise-explosion safety net was silently dead. `energy_guided_ddim_sample`
was missing the same `w_global`/`clip_pos`/`clip_local`/`clip_global`
parameters entirely, so a caller couldn't reproduce the same sampling
configuration through the energy-guided path. Both now take the same
parameters and wire them through identically.

## 5. Duplicate edges in the global encoder

`build_radius_graph` enumerates all atom pairs within the cutoff radius with
no exclusion of pairs already present in the local (bonded) graph. Since
bond lengths (~1-2 Å) are almost always well inside the default 10 Å cutoff,
essentially every bonded pair was being fed into the global encoder twice —
once under its real bond type, once again under radius-graph type 0 —
double-counting its contribution relative to GeoDiff's own coalesced,
duplicate-free edge construction. Fixed by dropping any radius-graph edge
whose `(row, col)` pair already exists in the local graph.

## 6. Checkpoint / config correctness

Checkpoints were saving the hardcoded module constants for `cutoff`,
`beta_start`, and `beta_end` instead of the actual CLI argument values the
model was built with. Harmless for the beta values (they're baked into
registered buffers that `load_state_dict` restores correctly regardless of
what's in the saved config dict), but a real bug for `cutoff`, a plain
Python attribute never touched by `load_state_dict` — a non-default
`--cutoff` run would have silently resumed with the wrong radius-graph
cutoff. Also: the experiment name (and therefore every checkpoint filename)
never encoded which data pipeline was used, so a legacy-split run and a
standardized-split run with otherwise-identical flags could silently
overwrite each other's checkpoints.

## 7. CLI / training-loop correctness

- Supplying only one of `--train-data`/`--val-data` used to silently fall
  back to the legacy random-split path — now a hard error, since a flag typo
  could otherwise burn a multi-day run on non-comparable data with no
  warning.
- The standardized-split path silently hardcoded `min_conformers=1`,
  ignoring `--min-confs`. Now honors the flag like the legacy path does.
- The validation loop's exception handler was a bare `except Exception: pass`
  — if every validation batch failed, the reported validation loss would
  read `0.0000`, indistinguishable from an excellent loss, with no trace of
  why. Now logs the exception.

## 8. The DDIM timestep-truncation bug (root cause of the training-invariant plateau)

This was the one that mattered most. Training loss declined normally for
125 epochs, but the evaluation metric (mean RMSD to reference conformers)
sat completely flat and coverage stayed at exactly 0%, regardless of how
much further the model trained — a sign that something structural, not
statistical, was capping quality.

`ddim_sample` and `energy_guided_ddim_sample` build a subsampled timestep
sequence for cheap evaluation:

```python
# before
seq = list(range(num_timesteps - num_steps, num_timesteps))
# num_timesteps=5000, num_steps=100 -> range(4900, 5000) -- only the LAST
# 100 integers: the highest-noise sliver of the schedule, never touching
# t=0..4899 at all.
```

The reverse-diffusion loop's final iteration hardcodes `ᾱ_next = 1.0`
(assumes zero noise remains — valid only if that step is genuinely `t=0`).
With the truncated sequence above, that assumption fired at `t=4900`
instead, where `ᾱ ≈ 0.008` (99.2% noise). Algebraically, forcing
`ᾱ_next=1` collapses the update to `mean = x0_pred` exactly — the entire
output became a single one-shot guess made from a nearly pure-noise input,
with none of the 4900 lower-noise refinement steps ever run. That guess's
quality has a hard ceiling with nothing to do with how well-trained the
network otherwise is — exactly matching the observed flat metric.

```python
# after
skip = max(num_timesteps // num_steps, 1)
seq = list(range(0, num_timesteps, skip))
# [0, 50, 100, ..., 4950] -- evenly spans the FULL schedule; the loop's
# final iteration now genuinely lands at t=0 (verified: alphas_cumprod at
# the final step is 0.999995, not 0.008).
```

This single fix, with zero additional training, dropped the mean-RMSD
metric roughly 27x and brought coverage off its 0% floor immediately upon
resuming from the existing checkpoint.

## 9. Multi-reference evaluation protocol

`run_geom_drugs_eval` was scoring against a single reference conformer per
molecule (whichever one the training-mode sampler happened to draw for that
batch), instead of GeoDiff's/TorDiff's own convention of scoring against
every quality-surviving ground-truth conformer per molecule. A single fixed
target is structurally harder to "cover" than any-of-N real references, so
this deflated the coverage metric independent of true generative quality.
Fixed to iterate the underlying dataset directly and pull every surviving
conformer as the reference set, with a real Boltzmann-weighted coverage
score in place of a placeholder that previously just copied the plain
coverage value.

## 10. Generation-count scaling

`eval_size_generalization.py` used one fixed `--n-gen` for every molecule
regardless of how many reference conformers it actually had. Fixed to scale
per molecule as `max(min_gen, gen_multiplier × reference_count)`, matching
GeoMol/TorDiff's "2x reference count" convention.

## 11. Evaluation RMSD was averaged over 3N coordinates

`kabsch_align` (`autoresearch/geodiff_eval.py`) and the duplicate
`kabsch_rmsd` implementations (`geom_drugs_eval.py`, `mol_prepare.py`, and two
visualization scripts) computed `sqrt(mean((P_rot - Q)**2))` on an `(N, 3)`
array, which divides the summed squared error by `3N` instead of `N`. Every
reported RMSD, including MAT-R and MAT-P, was therefore the true value divided
by `√3` (about 1.73x too small), and every coverage threshold was effectively
`√3` too generous (a "0.5 Å" threshold admitted true RMSD up to about 0.87 Å).

**Fix:** `sqrt(sum((P_rot - Q)**2) / N)`. Checked against SciPy's rotation
solver (max difference 7e-13 over 300 random cases) and a hand-computed case
(atoms at ±1 vs ±2 on one axis: RMSD exactly 1.0; the old code gave 0.577).

Ratios between two runs are unaffected. Absolute Å values from logs produced
before this fix must be multiplied by `√3`; coverage percentages cannot be
converted without the per-molecule RMSDs. The unaligned coordinate-match check
in `geom_drugs_eval/convert_to_mol_files.py` has the same form but is a
tolerance test, not a reported metric, and is unchanged.

## 12. Published reference numbers

Reference values printed by the evaluation and training scripts did not match
any source (for example GEOM-Drugs "GeoDiff MAT-R 0.528, TorDiff 0.481" and
QM9 "GeoDiff 71.0%, GeoMol 71.5%, TorDiff 73.2%"). They are replaced with values
read directly from Jing et al. (NeurIPS 2022), recall, mean, with GeoDiff
retrained by those authors on the GeoMol split:

| Method | GEOM-Drugs COV-R / MAT-R (Table 1, threshold 0.75 Å) | GEOM-QM9 COV-R / MAT-R (Table 7, threshold 0.5 Å) |
|---|---|---|
| RDKit ETKDG | 38.4% / 1.058 Å | 85.1% / 0.235 Å |
| GeoMol | 44.6% / 0.875 Å | 91.5% / 0.225 Å |
| GeoDiff (retrained) | 42.1% / 0.835 Å | 76.5% / 0.297 Å |
| Torsional Diffusion | 72.7% / 0.582 Å | 92.8% / 0.178 Å |

These are comparable to this repo's numbers only under the same test set and
RMSD variant. The published scripts use symmetry-aware RMSD, this repo uses
fixed atom indices, and the GEOM-Drugs coverage threshold there is 0.75 Å
whereas this repo reports coverage at 0.5 Å (only the MAT columns are
threshold-independent).

The QM9 experiments A-G used the single-DFT-geometry QM9 file (one reference
conformer per molecule), not the multi-conformer GEOM-QM9 that the published
numbers use, and their logged RMSDs predate fix 11; they are not comparable to
any published value. `visualization/expG_publication_plots.py`,
`visualization/expH_*.py`, `geom_drugs_eval/make_paper_figures.py`, and the
results tables in `full_flow.md` and `docs/README_exp_D.md` hard-code the older
values and result arrays; they are marked with a caution rather than
regenerated.

## Known, not-yet-fixed limitation

RMSD is computed via fixed atom-index correspondence (`kabsch_align` in
`geodiff_eval.py`), unlike TorsionalDiffusion's own default of searching over
topological automorphisms (`RDKit.AllChem.GetBestRMS`) for symmetric
substructures. This can overstate RMSD for molecules with swappable
substituents. Not fixed here because a real fix needs atom/bond information
threaded through a shared utility function used by multiple scripts — it
deserves its own change, not one bundled into this pass.
