# NExT-Mol Gen — Full Pipeline: From Dataset to 3D Molecule Generation

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset: QM9](#2-dataset-qm9)
3. [Data Processing Pipeline](#3-data-processing-pipeline)
4. [Diffusion Framework](#4-diffusion-framework)
5. [Model Architecture](#5-model-architecture)
6. [Geometry Constraint System](#6-geometry-constraint-system)
7. [Training Loop](#7-training-loop)
8. [Inference: Molecule Generation](#8-inference-molecule-generation)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Architecture Justification](#10-architecture-justification)
11. [Experiment Overview](#11-experiment-overview)
12. [All Citations](#12-all-citations)

---

## 1. Problem Statement

**Goal:** Given a molecular graph (atom types + bond connectivity), generate the physically correct 3D arrangement of those atoms — the **conformer**.

**Why it matters:** A molecule's 3D shape determines its interaction with biological receptors. Two identical SMILES strings with different 3D conformations can have drastically different binding affinities, membrane permeabilities, and pharmacological effects. Computational conformer generation is the first step in drug discovery, materials design, and protein-ligand docking.

**Why it's hard:**

**(a) E(3) symmetry.** Molecules have no preferred orientation in space. Rotating or translating a molecule gives the same molecule — the model must be invariant to global rotation/translation of coordinates, and equivariant in its coordinate predictions.

**(b) Multi-modal distributions.** Flexible molecules have multiple low-energy conformers. The model must learn the full distribution, not just the mean.

**(c) Physical validity.** Generated molecules must respect chemistry: bond lengths within ~0.2 Å of ideal, no steric clashes (atom pair distance > sum of VDW radii × 0.7), correct hybridization angles.

---

## 2. Dataset: QM9

**Reference:** Ramakrishnan et al. *"Quantum chemistry structures and properties of 134 kilo molecules."* Scientific Data 2014.

QM9 is the standard benchmark for 3D molecular generation, used in EDM, GeoDiff, GeoMol, TorDiff, and EQGAT-diff.

| Property | Value |
|----------|-------|
| Size | 133,885 molecules |
| Atoms | H, C, N, O, F (atomic numbers 1, 6, 7, 8, 9) |
| Heavy atoms | ≤9 per molecule |
| Total atoms (with H) | ≤29 per molecule |
| Coordinate source | DFT optimization: B3LYP/6-31G(2df,p) |
| Coordinate quality | Electronic energy minima — physically stable |
| License | Public domain (Creative Commons) |

**Why DFT coordinates?** Density Functional Theory minimizes the electronic energy of the molecule. The resulting geometry is a true energy minimum — stable under small perturbations. This gives the model physically valid training signal that a force-field or heuristic dataset cannot provide.

**Why include explicit H?** Hydrogen atoms encode critical information: chirality (R/S configuration), ring puckering (axial vs equatorial H), and N-H/O-H bond angles that determine hydrogen bonding geometry. Removing H loses this information.

---

## 3. Data Processing Pipeline

### 3.1 Download and Parse

```
QM9 SDF file (deepchem S3, public)
    ↓  data/prepare_qm9.py
    ├── SDMolSupplier(removeHs=False, sanitize=False)
    ├── Per-molecule sanitization attempt (SanitizeMol)
    │   └── Skip if fails (radicals, bad valence — ~1,900 / 133,885 molecules)
    └── Write JSONL: {atom_types, coordinates, edge_index, bond_types, num_atoms}
```

**RDKit SanitizeMol** performs: aromaticity perception, valence checking, ring finding. Molecules that fail sanitization have non-physical bonding (radical species, transition states) that should not be in training data.

### 3.2 Graph Construction

Each molecule becomes a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

- **Nodes** $\mathcal{V}$: atoms, with feature = atomic number $z_i \in \{1,...,53\}$
- **Edges** $\mathcal{E}$: **all bonds in both directions** (undirected graph → directed edge pairs). Bond type $b_{ij} \in \{1,2,3,4\}$ = {single, double, triple, aromatic}.
- **3D coordinates** $\mathbf{x}_i \in \mathbb{R}^3$: from DFT geometry

Both directions are stored because the EGNN message passing is directed (from source to destination). An undirected bond $i$-$j$ creates edges $i \to j$ and $j \to i$.

### 3.3 Batching

PyTorch Geometric-style batching: concatenate all atom tensors, offset edge indices by cumulative atom counts, add `batch_idx` tensor mapping each atom to its molecule.

```
Batch of B=64 molecules:
  atom_types:  (N_total,)       N_total = sum of all atom counts
  coordinates: (N_total, 3)
  edge_index:  (2, E_total)     E_total = sum of all bond counts × 2
  bond_types:  (E_total,)
  batch_idx:   (N_total,)       batch_idx[i] = molecule index of atom i
```

### 3.4 Center-of-Mass Removal

**Reference:** Hoogeboom et al. EDM 2022.

Applied at the start of every training step:

$$\mathbf{x}_i \leftarrow \mathbf{x}_i - \frac{1}{N_m}\sum_{j \in \text{mol}_m} \mathbf{x}_j, \quad \forall \text{ molecule } m$$

This constrains all coordinates to the zero-CoM subspace $\mathcal{M}_0 = \{\mathbf{x} : \sum_i \mathbf{x}_i = \mathbf{0}\}$, removing the global translation degree of freedom. The model learns only the *shape* of the molecule.

---

## 4. Diffusion Framework

### 4.1 Forward Process

**Reference:** Ho et al. DDPM, NeurIPS 2020.

Given clean coordinates $\mathbf{x}_0$, the forward process adds Gaussian noise over $T=1000$ timesteps using the **cosine noise schedule** (Nichol & Dhariwal, ICML 2021):

$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon, \quad \boldsymbol\epsilon = \text{CoM-remove}(\mathcal{N}(\mathbf{0},\mathbf{I}))$$

$$\bar\alpha_t = \frac{\cos^2\!\!\left(\frac{t/T + 0.008}{1.008} \cdot \frac{\pi}{2}\right)}{\cos^2\!\!\left(\frac{0.008}{1.008} \cdot \frac{\pi}{2}\right)}$$

At $t=0$: $\bar\alpha_0 \approx 1$, so $\mathbf{x}_0 \approx \mathbf{x}_0$ (no noise).
At $t=T$: $\bar\alpha_T \approx 0$, so $\mathbf{x}_T \approx \boldsymbol\epsilon$ (pure noise).

The cosine schedule distributes learning signal more uniformly than the linear schedule, which wastes capacity near $t=0$ and $t=T$.

### 4.2 Reverse Process

The model learns to reverse the noise process. Using **x₀ parameterization** (predict clean coordinates directly):

$$\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_t, t, \mathcal{G}), \quad \hat{\mathbf{x}}_0 = \text{CoM-remove}(\hat{\mathbf{x}}_0)$$

Equivalent noise prediction:
$$\hat{\boldsymbol\epsilon} = \frac{\mathbf{x}_t - \sqrt{\bar\alpha_t}\hat{\mathbf{x}}_0}{\sqrt{1-\bar\alpha_t}}$$

**Why x₀ parameterization?** (Hoogeboom EDM 2022, Bao Analytic-DPM 2022)
- Allows applying geometry constraints (bond lengths, angles) directly on $\hat{\mathbf{x}}_0$ at every timestep
- More numerically stable for large molecules
- Soft-clamping $10\tanh(\hat{\mathbf{x}}_0/10)$ prevents outlier coordinates

### 4.3 Training Loss

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum_i w(t_{m(i)}) \cdot \|\hat{\boldsymbol\epsilon}_i - \boldsymbol\epsilon_i\|^2}_{\mathcal{L}_\text{MSE}} + \underbrace{\bar{w} \cdot \lambda_\text{geo} \cdot \mathcal{L}_\text{geo}}_{\text{geometry constraints}}$$

**Min-SNR weighting** (Hang et al., ICCV 2023):
$$w(t) = \min\!\left(1, \frac{\gamma}{\text{SNR}(t)}\right), \quad \text{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}, \quad \gamma=5$$

This prevents high-noise timesteps (where SNR→0 and any prediction is meaningless) from dominating the loss.

---

## 5. Model Architecture

### 5.1 Initial Feature Construction

```python
h_i = Embed(z_i, 54→256)              # atom type lookup, 54 = max atomic number
    + MLP(SinEmbed(t, 128)→256)[bi]   # time embedding, broadcast to per-atom
```

**Sinusoidal time embedding** (Vaswani et al., Transformer 2017):
$$\text{SinEmbed}(t)_k = \begin{cases}\sin(t \cdot 10000^{-2k/d}) & k \text{ even}\\\cos(t \cdot 10000^{-2k/d}) & k \text{ odd}\end{cases}$$

This gives the model a continuous, unique representation of each timestep. The sinusoidal form ensures distant timesteps have very different embeddings (important for the model to distinguish high-noise from low-noise predictions).

### 5.2 RBF Distance Features

Pairwise distance $d_{ij}$ encoded as 20 Gaussian RBF:
$$\phi_k(d_{ij}) = \exp\!\!\left(-\frac{(d_{ij} - \mu_k)^2}{2\sigma^2}\right), \quad \mu_k \in [0.5, 10.0]\text{ Å}$$

**Why RBF?** A raw distance scalar loses resolution at short distances (where most chemistry happens). RBF gives the model 20 parallel "sensors" tuned to different distance ranges, like a spectrogram for distances.

### 5.3 Equivariant Message Passing (EGNN, L=6 layers)

**Reference:** Satorras et al. ICML 2021.

Each layer computes 3 things:

**Edge messages** (chemistry context, invariant):
$$\mathbf{m}_{ij} = \text{MLP}_{e}\!\left([\mathbf{h}_i \| \mathbf{h}_j \| \phi(d_{ij}) \| \mathbf{e}_{ij}]\right) + \mathbf{e}_{ij}$$

where $\mathbf{e}_{ij} = \text{Embed}(b_{ij})$ is the bond-type embedding. The `+ e_ij` residual ensures bond type information persists through all layers.

**Coordinate update** (geometry, equivariant):
$$\mathbf{x}_i^{(l+1)} = \mathbf{x}_i^{(l)} + \frac{1}{\deg(i)+1}\sum_{j}\phi_x(\mathbf{m}_{ij}) \cdot \frac{\mathbf{x}_i^{(l)} - \mathbf{x}_j^{(l)}}{\|\mathbf{x}_i^{(l)} - \mathbf{x}_j^{(l)}\| + \epsilon}$$

This update is **equivariant** because: (1) unit displacement vectors $\hat{\mathbf{u}}_{ij}$ are equivariant (they rotate with the molecule), (2) scalar weights $\phi_x(\mathbf{m}_{ij})$ are invariant (distances are invariant to rotation). Equivariance guarantee: $f(R\mathbf{x}) = Rf(\mathbf{x})$ for any rotation $R \in O(3)$.

**Degree normalization:** Dividing by $\deg(i)+1$ prevents coordinate updates from growing with the number of neighbours — without it, dense aromatic rings would receive huge position updates while isolated atoms receive tiny ones.

**Node features** (chemistry, invariant):
$$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\!\!\left(\mathbf{h}_i^{(l)} + \text{MLP}_h\!\left([\mathbf{h}_i^{(l)} \| \textstyle\sum_j \mathbf{m}_{ij}]\right)\right)$$

LayerNorm prevents internal covariate shift across 6 layers. The residual connection enables gradient flow through all layers (analogous to ResNet).

### 5.4 Output Head

$$\hat{\mathbf{x}}_0 = \mathbf{x}^{(6)} + \text{MLP}_\text{out}(\mathbf{h}^{(6)}) \in \mathbb{R}^{N \times 3}$$

The $\mathbf{x}^{(6)}$ term is the coordinate stream after 6 equivariant refinements — already physically reasonable. The MLP adds a residual correction from the invariant features. This design is more stable than predicting $\hat{\mathbf{x}}_0$ from scratch.

---

## 6. Geometry Constraint System

**References:** Halgren MMFF94 1996; Jorgensen OPLS-AA 1996; Hoogeboom EDM 2022.

Since we predict $\hat{\mathbf{x}}_0$ directly, we can apply differentiable chemistry-aware constraints at every training step. All losses operate on $\hat{\mathbf{x}}_0$, not $\mathbf{x}_t$.

### 6.1 Bond Length Loss

MMFF94 lookup table (pre-built $54 \times 54 \times 5$ tensor, $O(1)$ lookup):

$$\mathcal{L}_\text{bond} = \frac{10.0}{|E|}\sum_{(i,j) \in E}\!\left(\|\hat{\mathbf{x}}_i - \hat{\mathbf{x}}_j\| - d^*_{z_i,z_j,b_{ij}}\right)^2$$

Example targets: C-C single=1.54 Å, C=C double=1.34 Å, C≡C triple=1.20 Å, C:C aromatic=1.40 Å.

### 6.2 Bond Angle Loss

Hybridization-inferred ideal angles:

$$\mathcal{L}_\text{angle} = \frac{3.0}{|A|}\sum_{(i,j,k) \in A}\!\left(\arccos\frac{\hat{\mathbf{v}}_{ji} \cdot \hat{\mathbf{v}}_{jk}}{\|\cdot\|} - \theta^*_j\right)^2$$

| Hybridization | Ideal angle | Detection rule |
|--------------|-------------|----------------|
| sp³ | 109.5° | 4 neighbours, no multiple bonds |
| sp² | 120.0° | 3 neighbours or double bond on C/N/O/S |
| sp | 180.0° | 2 neighbours or triple bond |
| aromatic | 120.0° | aromatic bonds present |

### 6.3 VDW Repulsion Loss

Prevents atomic clashes using Bondi VDW radii:

$$r_\text{clash}(i,j) = 0.70 \cdot (r_\text{VDW}(z_i) + r_\text{VDW}(z_j))$$

**Bug fix applied:** The original code checked `if N > 300: skip` where N was the batch size (64 mols × ~15 atoms = ~960 atoms → always skipped!). We now compute repulsion per-molecule:

$$\mathcal{L}_\text{rep} = \frac{5.0}{N_\text{clash}}\sum_{\substack{i<j,\; d_{ij}<r_\text{clash}\\\text{not 1-2 or 1-3}}}\!\!\!\left(r_\text{clash}(i,j) - d_{ij}\right)^2$$

1-2 pairs (directly bonded) and 1-3 pairs (bonded through one atom) are excluded — they are handled by the bond/angle losses.

### 6.4 SNR Gating

Geometry loss is gated by the batch-mean SNR weight:

$$\mathcal{L}_\text{geo,gated} = \bar{w} \cdot \mathcal{L}_\text{geo}, \quad \bar{w} = \text{mean}_i[w(t_{m(i)})]$$

At high noise ($t \approx T$, low SNR), $\hat{\mathbf{x}}_0$ is a poor estimate of the true structure. Applying geometry loss here would penalize meaningless predictions. SNR gating automatically suppresses geometry supervision when it would be uninformative.

---

## 7. Training Loop

```
for epoch in 1..50:
    lr = cosine_warmup_schedule(epoch, base_lr=1e-4, warmup=5)
    
    for batch in train_loader:
        x_0 = CoM_remove(batch.coordinates)
        
        # Sample random timestep per molecule
        t = randint(0, 999, size=(B,))
        
        # Forward process: add noise
        x_t, eps = q_sample(x_0, t, batch_idx)
        
        # Model forward: predict x_0
        x_0_hat = model.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
        x_0_hat = CoM_remove(x_0_hat)
        
        # Derive predicted noise
        eps_hat = (x_t - sqrt_alpha[t]*x_0_hat) / sqrt_one_minus[t]
        
        # Min-SNR weighted MSE
        snr_weight = min(1, gamma / SNR[t])  # per-molecule, broadcast to per-atom
        mse_loss = mean(snr_weight * ||eps_hat - eps||^2)
        
        # Geometry constraints on predicted x_0
        geo_loss = geometry.compute(x_0_hat, atom_types, edge_index, bond_types)
        snr_mean_weight = mean(snr_weight)
        
        loss = mse_loss + snr_mean_weight * geo_weight * geo_loss
        
        loss.backward()
        clip_grad_norm_(params, 1.0)
        optimizer.step()
    
    # Validation + checkpointing every epoch
    if val_loss < best_val_loss:
        save(checkpoint)
```

**Gradient clipping** to norm 1.0 prevents gradient explosions — observed in earlier experiments after epoch 150 when geometry losses occasionally produce large gradients for steric clashes.

---

## 8. Inference: Molecule Generation

### 8.1 DDIM Sampling (50 steps)

**Reference:** Song et al. DDIM, ICLR 2021.

```
Input: molecular graph (atom_types, edge_index, bond_types)
x_T ~ CoM-remove(N(0, I))   // start from Gaussian noise

timesteps = [1000, 980, 960, ..., 20, 0]   // 50 steps of size 20

for t_i, t_next in zip(timesteps[:-1], timesteps[1:]):
    // Predict clean structure
    x_0_hat = model(x_t, t_i, graph)
    x_0_hat = 10 * tanh(x_0_hat / 10)     // soft clamp to [-10, 10] Å
    x_0_hat = CoM_remove(x_0_hat)
    
    // Derive predicted noise direction
    alpha_t    = alphas_cumprod[t_i]
    alpha_next = alphas_cumprod[t_next]
    eps_hat = (x_t - sqrt(alpha_t) * x_0_hat) / sqrt(1 - alpha_t)
    
    // DDIM update (eta=0, deterministic)
    dir = sqrt(1 - alpha_next) * eps_hat
    x_t = sqrt(alpha_next) * x_0_hat + dir

return x_0_hat  // final generated 3D conformation
```

**Soft clamping** $10\tanh(\hat{\mathbf{x}}_0/10)$: limits coordinate values to $[-10, 10]$ Å (QM9 molecules are ≤5 Å radius), preventing runaway predictions at low SNR timesteps.

### 8.2 Post-Processing

```
Generated (atom_types, edge_index, bond_types, x_0_hat)
    ↓ RDKit: build Mol from atoms + bonds
    ↓ SanitizeMol()   → check valence, aromaticity, ring perception
    ├── PASS → valid molecule → export to SDF/PDB/MOL2
    └── FAIL → invalid (record as failure, contributes to fully_valid metric)
```

---

## 9. Evaluation Metrics

| Metric | Formula | Lower/Higher = Better |
|--------|---------|----------------------|
| **fully_valid** | % passing `SanitizeMol()` | Higher ↑ |
| **validity** | % with all bonds within ±0.20 Å of MMFF94 | Higher ↑ |
| **MAT-R** | $\text{mean}_{m} \min_{\hat{x} \in \hat{S}_m}\text{RMSD}(\hat{x}, x^*_m)$ | **Lower ↓** |
| **COV-R** | % of references within 0.5 Å RMSD of best generated | Higher ↑ |
| **strain** | MMFF94 force-field energy (kcal/mol) | Lower ↓ |
| **bond_error** | Mean absolute bond length deviation from ideal | Lower ↓ |

**MAT-R (Matching-Recall)** is the primary metric. It measures: for each reference conformation $x^*$, what is the closest generated sample? Averaging over the test set gives a measure of how well the model covers conformational space.

**GeoMol (NeurIPS 2021)** introduced MAT-R/COV-R as the standard evaluation protocol for conformer generation — universally adopted by all subsequent papers.

---

## 10. Architecture Justification

### Why E(3) Equivariance?

Without equivariance, the model must see a molecule in all possible orientations to generalize. With ~130K training molecules and ∞ possible orientations, this is impossible. Equivariance provides an **inductive bias** that says: rotating the input rotates the output by the same amount. This reduces the effective learning problem by a factor equal to the size of the rotation group (infinite for SO(3)).

**EGNN vs alternatives:**

| Architecture | Equivariant | Speed | Expressiveness |
|-------------|------------|-------|----------------|
| SchNet | ✗ (invariant only) | Fast | Low |
| DimeNet | ✗ (invariant) | Medium | Medium |
| SE(3)-Transformer | ✓ | Slow | High |
| NequIP | ✓ (tensor products) | Slow | Very High |
| **EGNN** | **✓** | **Fast** | **Medium-High** |
| EQGAT-diff | ✓ + attention | Medium | High |

EGNN is chosen as the backbone because it achieves equivariance without expensive tensor products (unlike NequIP), making it practical for 50-epoch training sweeps on a single RTX 3090.

### Why DDPM with x₀ Parameterization?

- **DDPM** provides a principled probabilistic framework with strong theoretical guarantees (can model the full distribution of conformers, not just the mean).
- **x₀ parameterization** allows applying geometry constraints directly — critical for physical validity.
- **Min-SNR weighting** stabilizes training across all noise levels.
- **DDIM sampling** provides 20× inference acceleration over full DDPM.

### Why Geometry Constraints?

DFT training coordinates are already valid. Why add explicit constraints?

1. The diffusion loss alone provides no guarantee the generated $\hat{\mathbf{x}}_0$ at intermediate timesteps is chemically reasonable.
2. Geometry constraints provide **every-step supervision** on the predicted clean structure, not just at $t=0$.
3. GCDM (Morehead 2023) and EQGAT-diff (Le 2024) both show that immediate geometry supervision from epoch 1 (not gradually ramped) gives faster convergence and better final validity.

---

## 11. Experiment Overview

| Exp | Model | Key Change | Hypothesis | Paper |
|-----|-------|-----------|------------|-------|
| A | ConformerDiffusion | Bug fixes only | Clean baseline | EDM 2022 |
| B | AttnConformerDiffusion | Multi-head attention on messages | Attention > sum-pooling | EQGAT-diff ICLR 2024 |
| C | FlowMatchingConformer | CFM objective + Euler ODE | Straight paths → fewer NFE | Lipman ICLR 2023 |
| D | ConformerDiffusion | Torsion auxiliary loss, geo_w=0.5 | Dihedral supervision → lower MAT-R | TorDiff NeurIPS 2022 |

See `docs/README_exp_A/B/C/D.md` for detailed per-experiment documentation.

---

## 12. All Citations

1. **Ho et al.** "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.* arXiv:2006.11239
2. **Satorras et al.** "E(n) Equivariant Graph Neural Networks." *ICML 2021.* arXiv:2102.09844
3. **Hoogeboom et al.** "Equivariant Diffusion for Molecule Generation in 3D (EDM)." *ICML 2022.* arXiv:2203.17003
4. **Song et al.** "Denoising Diffusion Implicit Models (DDIM)." *ICLR 2021.* arXiv:2010.02502
5. **Nichol & Dhariwal.** "Improved DDPM." *ICML 2021.* arXiv:2102.09672
6. **Hang et al.** "Efficient Diffusion Training via Min-SNR Weighting." *ICCV 2023.* arXiv:2303.09556
7. **Ganea et al.** "GeoMol: Torsional GNN for Molecular Conformer Generation." *NeurIPS 2021.* arXiv:2106.07802
8. **Xu et al.** "GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation." *ICLR 2022.* arXiv:2203.02923
9. **Le et al.** "EQGAT-diff: a novel equivariant graph attention model for molecular 3D generation." *ICLR 2024.* arXiv:2306.01916
10. **Jing et al.** "Torsional Diffusion for Molecular Conformer Generation (TorDiff)." *NeurIPS 2022.* arXiv:2206.01729
11. **Lipman et al.** "Flow Matching for Generative Modeling." *ICLR 2023.* arXiv:2210.02747
12. **Yim et al.** "SE(3) Diffusion Model with Application to Protein Backbone Generation (FrameDiff)." *ICML 2023.* arXiv:2302.02277
13. **Veličković et al.** "Graph Attention Networks." *ICLR 2018.* arXiv:1710.10903
14. **Ramakrishnan et al.** "Quantum chemistry structures and properties of 134 kilo molecules (QM9)." *Scientific Data 2014.*
15. **Halgren.** "Merck Molecular Force Field (MMFF94)." *J. Comput. Chem. 1996.*
16. **Jorgensen et al.** "Development and Testing of the OPLS All-Atom Force Field." *J. Am. Chem. Soc. 1996.*
17. **Vaswani et al.** "Attention is All You Need." *NeurIPS 2017.* arXiv:1706.03762
18. **Loshchilov & Hutter.** "Decoupled Weight Decay Regularization (AdamW)." *ICLR 2019.* arXiv:1711.05101
19. **Morehead & Cheng.** "Geometry-Complete Diffusion for 3D Molecule Generation (GCDM)." *MLSB Workshop NeurIPS 2023.* arXiv:2302.04313
20. **Bao et al.** "Analytic-DPM: Analytic Estimate of the Optimal Reverse Variance." *ICLR 2022.* arXiv:2201.06503
