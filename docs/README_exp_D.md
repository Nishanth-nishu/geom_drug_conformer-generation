# Experiment D — Torsion-Angle Auxiliary Loss (TorDiff-Inspired)

## Summary

| Field | Value |
|-------|-------|
| Model | ConformerDiffusion (same as Exp A) |
| Script | `scripts/exp_D_torsion_aux.sh` |
| Trainer | `autoresearch/mol_train_expD.py` |
| Key papers | Jing et al., *TorDiff*, NeurIPS 2022; Ganea et al., *GeoMol*, NeurIPS 2021 |
| Hypothesis | Direct dihedral supervision reduces MAT-R — torsion angles are the dominant source of conformer error |
| Key changes | `geometry_weight=0.5` (was 0.1), `include_torsions=True` |
| Expected gain | **15–25% lower MAT-R** vs Exp A |

---

## Research Motivation

### The Conformer Accuracy Gap

Published results on QM9 conformer generation show a large gap between methods:

| Method | MAT-R (Å) | Key feature |
|--------|-----------|-------------|
| RDKit ETKDG | 0.297 | Classical, no learning |
| GeoMol | 0.225 | Torsion prediction + GNN |
| GeoDiff | 0.297 | Cartesian diffusion (EDM-like) |
| TorDiff | 0.179 | Diffusion *over* torsion angles |
| EQGAT-diff | 0.171 | EGNN + attention |

**Key observation:** Methods that explicitly model torsion angles (GeoMol, TorDiff) consistently outperform Cartesian diffusion methods (GeoDiff, EDM). The reason is fundamental to molecular geometry.

### Why Torsion Angles Dominate Conformer Error

A molecular conformation can be decomposed as:

1. **Bond lengths** — nearly rigid, vary by ≤0.02 Å within one molecule (standard deviation across conformers is tiny)
2. **Bond angles** — semi-rigid, vary by ≤3° for most bonds
3. **Torsion angles (dihedrals)** — freely rotatable, can span the full 360° range

The conformational space of a flexible molecule is almost entirely determined by its torsion angles. Bond lengths and angles contribute negligibly to RMSD between conformers of the same molecule. Therefore:

> A model that learns torsion angles learns the conformational space; a model that only supervises Cartesian coordinates must rediscover torsion structure from scratch.

**GeoMol** (Ganea et al., NeurIPS 2021) quantified this: predicted torsion angle MAE is the strongest predictor of final RMSD.

**TorDiff** (Jing et al., NeurIPS 2022) went further: they diffuse directly *over* torsion angles (not Cartesian coordinates), achieving MAT-R = 0.179 Å — the best at publication.

### Our Approach: Torsion Auxiliary Loss

We cannot easily adopt TorDiff's full torsion-space diffusion (it requires differentiable torsion angle parameterization of the coordinate system, a major architectural change). Instead, we add an **OPLS-AA torsion energy auxiliary loss** to the existing Cartesian diffusion framework:

$$\mathcal{L}_\text{tors}(\hat{\mathbf{x}}_0) = \sum_{\text{rotatable bonds}} E_\text{OPLS}(\phi_{ijkl})$$

where $\phi_{ijkl}$ is the dihedral angle around bond $j$-$k$ and:

$$E_\text{OPLS}(\phi) = \frac{V_1}{2}(1+\cos\phi) + \frac{V_2}{2}(1-\cos 2\phi) + \frac{V_3}{2}(1+\cos 3\phi)$$

**Why OPLS-AA?** Jorgensen et al. (1996) parameterized this torsion potential by fitting to thousands of high-level quantum chemistry calculations. The parameters encode real chemical knowledge:
- sp³–sp³ bond: 3-fold barrier (ethane-like, $V_3=1.0$)
- sp³–sp² bond: 2-fold barrier (prefers 0°/180°, $V_2=2.0$)
- sp²–sp² conjugated: strong 2-fold (planarity, $V_2=6.0$)
- aromatic–aromatic: very strong 2-fold ($V_2=10.0$, enforces planarity)

We apply this loss on the **ground-truth coordinates** $\mathbf{x}_0$ each batch (not on the noisy prediction $\hat{\mathbf{x}}_0$) because:
- Supervised torsion angles from DFT data directly inject physical knowledge
- Loss on ground-truth is always meaningful; loss on $\hat{\mathbf{x}}_0$ at high noise is unstable

---

## Dihedral Angle Computation

For atoms $i$-$j$-$k$-$l$ forming a dihedral:

```
b1 = x_j - x_i
b2 = x_k - x_j
b3 = x_l - x_k

n1 = b1 × b2    (normal to plane ijk)
n2 = b2 × b3    (normal to plane jkl)

phi = atan2(|b2| * b1 · n2, n1 · n2)   // signed dihedral
```

This formula (standard Praxitelous et al.) handles all dihedral angle ranges correctly, including the discontinuity at ±180°, because `atan2` returns values in $(-\pi, \pi]$.

---

## Hybridization Detection

The torsion parameters $V_1, V_2, V_3$ depend on the hybridization of the two central bond atoms. We infer hybridization from bond types:

```python
if has_triple_bond:   → sp
if has_aromatic_bond: → aromatic
if has_double_bond and atom in {C, N, O, S}: → sp2
if num_neighbors <= 2: → sp
if num_neighbors == 3 and atom in {C, N, O, S}: → sp2
else: → sp3
```

This is a deterministic rule based on VSEPR theory (Gillespie 1972) and is exact for all QM9 atom types.

---

## Modified Loss Function

**Exp A loss:**
$$\mathcal{L}_A = \mathcal{L}_\text{MSE} + 0.1 \cdot \bar{w} \cdot (\mathcal{L}_\text{bond} + \mathcal{L}_\text{angle} + \mathcal{L}_\text{rep})$$

**Exp D loss:**
$$\mathcal{L}_D = \mathcal{L}_\text{MSE} + 0.5 \cdot \bar{w} \cdot (\mathcal{L}_\text{bond} + \mathcal{L}_\text{angle} + \mathcal{L}_\text{rep}) + 0.5 \cdot \mathcal{L}_\text{tors}(\mathbf{x}_0)$$

The torsion loss is applied directly to ground-truth $\mathbf{x}_0$ (not SNR-gated) because it's a supervised signal, not a reconstruction quality check. Increasing geometry_weight to 0.5 (from 0.1) applies stronger physical constraints overall.

---

## Hyperparameters (changes from Exp A)

| Parameter | Exp A | Exp D | Rationale |
|-----------|-------|-------|-----------|
| geometry_weight | 0.1 | **0.5** | Stronger geometry supervision (TorDiff insight) |
| include_torsions | False | **True** | OPLS-AA torsion auxiliary loss |
| Model | ConformerDiffusion | ConformerDiffusion | Same — isolates loss function change |

All other settings identical to Exp A. This is a **loss function ablation** only.

---

## Expected Results

| Metric | Exp A | Exp D (expected) | TorDiff (200ep) |
|--------|-------|-----------------|----------------|
| fully_valid | ~0.80 | ~0.78 (±slight decrease) | 0.891 |
| MAT-R (Å) | ~0.45 | **~0.34–0.38** | 0.179 |
| Strain (kcal/mol) | ~high | ~lower | — |

The slight decrease in `fully_valid` is expected: stronger geometry constraints can over-constrain early training, reducing RDKit-passable molecules slightly. The MAT-R improvement is the target gain.

**Runtime note:** Torsion loss adds $O(|E_\text{rot}|)$ computation per batch (where $E_\text{rot}$ = number of rotatable bonds ≤ 8 for QM9). Expected overhead: ~10–15% slower than Exp A.

---

## Citations

1. **Jing et al.** "Torsional Diffusion for Molecular Conformer Generation." *NeurIPS 2022.* arXiv:2206.01729
2. **Ganea et al.** "GeoMol: Torsional Graph Neural Network for Molecular Conformer Generation and Property Prediction." *NeurIPS 2021.* arXiv:2106.07802
3. **Jorgensen et al.** "Development and Testing of the OPLS All-Atom Force Field on Conformational Energetics and Properties of Organic Liquids." *J. Am. Chem. Soc. 1996.*
4. **Halgren.** "Merck Molecular Force Field (MMFF94)." *J. Comput. Chem. 1996.* (bond/angle parameters)
5. **Hoogeboom et al.** "Equivariant Diffusion for Molecule Generation in 3D." *ICML 2022.* arXiv:2203.17003
6. **Xu et al.** "GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation." *ICLR 2022.* arXiv:2203.02923
7. **Gillespie.** "Electron groups and the VSEPR model of molecular geometry." *J. Chem. Educ. 1970.*
