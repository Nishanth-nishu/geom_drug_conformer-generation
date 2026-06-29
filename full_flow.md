# Full Mathematical Pipeline — mol_next_gen Conformer Generation

## References

| ID | Paper | Venue |
|----|-------|-------|
| [EDM] | Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D" | ICML 2022 |
| [GeoDiff] | Xu et al., "GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation" | ICML 2022 |
| [DDIM] | Song et al., "Denoising Diffusion Implicit Models" | ICLR 2021 |
| [DDPM] | Ho et al., "Denoising Diffusion Probabilistic Models" | NeurIPS 2020 |
| [EGNN] | Satorras et al., "E(n) Equivariant Graph Neural Networks" | ICML 2021 |
| [GeoMol] | Ganea et al., "GeoMol: Torsional Geometric Graph Neural Network for Molecular Conformer Generation" | NeurIPS 2021 |
| [GCDM] | Morehead & Cheng, "Geometry-Complete Diffusion for 3D Molecule Generation" | NeurIPS 2023 |
| [MinSNR] | Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy" | ICCV 2023 |
| [SchNet] | Schütt et al., "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions" | NeurIPS 2017 |
| [Kabsch] | Kabsch, "A solution for the best rotation to relate two sets of vectors" | Acta Cryst. 1976 |
| [Nichol] | Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" | ICML 2021 |
| [QM9] | Ramakrishnan et al., "Quantum chemistry structures and properties of 134 kilo molecules" | Sci. Data 2014 |
| [GEOM] | Axelrod & Gomez-Bombarelli, "GEOM: Energy-annotated molecular conformations" | Sci. Data 2022 |
| [TorDiff] | Jing et al., "Torsional Diffusion for Molecular Conformer Generation" | NeurIPS 2022 |

---

## 1. Dataset

### 1.1 QM9 [QM9]

QM9 contains **134,000 small organic molecules** with up to 9 heavy atoms (C, N, O, F) and one DFT-optimized 3D geometry per molecule at the B3LYP/6-31G(2df,p) level of theory.

**Preprocessing (heavy-atom stripping):**
- Remove all hydrogen atoms. Bonds are inferred from the molecular graph.
- Atom types: `{C=6, N=7, O=8, F=9, S=16, Cl=17}`
- Filter: keep only molecules with `N ≤ 9` heavy atoms and atomic numbers `z ∈ [1, 53]`
- Total after filtering: **131,967 molecules**

**Train/Val split (seed=42, 90/10):**

```
indices = torch.randperm(N, generator=Generator().manual_seed(42))
train_idx = indices[:n_train]   # 118,771 mols
val_idx   = indices[n_train:]   # 13,196 mols
```

### 1.2 GEOM-Drugs [GEOM]

**304,466 drug-like molecules** from the GEOM dataset, each with multiple DFT-optimized conformers (Boltzmann-weighted). Test set: `test_data_1k.pkl` — 1,000 pre-packed molecules, each with M reference conformers stacked as `pos_ref ∈ ℝ^{M·N×3}`.

### 1.3 Molecular Graph Representation

Each molecule is represented as an undirected graph **G = (V, E)**:

- **Node features:** atom type one-hot `z_i ∈ {6,7,8,9,16,17}`, embedded via `nn.Embedding(54, hidden_dim)`
- **Edge features:** bond type `b_{ij} ∈ {1=single, 2=double, 3=triple, 4=aromatic}`, embedded via `nn.Embedding(5, hidden_dim)`
- **3D coordinates:** `x_i ∈ ℝ^3` for each atom `i`
- **Center of mass (CoM) removal** applied at all stages [EDM §3.1]:

$$\mathbf{x} \leftarrow \mathbf{x} - \frac{1}{N}\sum_{i=1}^{N}\mathbf{x}_i$$

This enforces translation-invariance. The generative distribution is defined on the **CoM-free subspace** `ℝ^{3N}_{com=0}`.

---

## 2. Forward Diffusion Process

### 2.1 Cosine Noise Schedule [Nichol]

We use the cosine schedule rather than linear to prevent information collapse at the end of the chain:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

with `s = 0.008` (offset to prevent β_t being too small near t=0), `T = 1000`.

The variance schedule is derived as:

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \quad \beta_t \in (10^{-4},\ 0.9999)$$

Pre-computed buffers:
- `alphas_cumprod` = $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$
- `sqrt_alphas_cumprod` = $\sqrt{\bar{\alpha}_t}$
- `sqrt_one_minus_alphas_cumprod` = $\sqrt{1-\bar{\alpha}_t}$

### 2.2 Forward Process (q)

The forward process adds Gaussian noise to 3D coordinates [DDPM §2]:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\ (1-\bar{\alpha}_t)\mathbf{I}\right)$$

**Closed-form sampling** (reparameterization trick):

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

CoM is removed from both `x_0` and `ε` before applying the above, keeping the noised coordinates in CoM-free subspace [EDM Eq. 4].

### 2.3 Posterior (q reverse)

The true posterior for the reverse step is:

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\ \tilde{\boldsymbol{\mu}}_t,\ \tilde{\beta}_t \mathbf{I}\right)$$

$$\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t$$

$$\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})\,\beta_t}{1-\bar{\alpha}_t}$$

---

## 3. Neural Network Architecture (Denoiser)

### 3.1 Equivariant Graph Neural Network (E-GNN) [EGNN]

Each layer updates node embeddings `h_i` and coordinates `x_i` while preserving **E(3)-equivariance** — the outputs transform correctly under rotation, reflection, and translation of 3D space.

**Layer update equations:**

1. **Compute pairwise distances:**

$$d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|_2$$

2. **RBF edge features** [SchNet]:

$$\phi_k(d) = \exp\!\left(-\frac{(d - \mu_k)^2}{2\sigma^2}\right), \quad \mu_k = d_{\min} + k\cdot\frac{d_{\max}-d_{\min}}{K}$$

where `K = num_rbf = 32`, `d_min=0.5 Å`, `d_max=6.0 Å`. This gives **32-dimensional edge features** encoding pairwise geometry.

3. **Message computation:**

$$\mathbf{m}_{ij} = \phi_m\!\left([\mathbf{h}_i \,\|\, \mathbf{h}_j \,\|\, \phi(d_{ij}) \,\|\, e_{ij}]\right)$$

where `e_ij` is the bond type embedding and `φ_m` is an MLP.

4. **Coordinate update (equivariant):**

$$\mathbf{x}_i \leftarrow \mathbf{x}_i + \frac{1}{\deg(i)} \sum_{j \in \mathcal{N}(i)} (\mathbf{x}_i - \mathbf{x}_j) \cdot \phi_x(\mathbf{m}_{ij})$$

The `1/deg(i)` normalization corrects for bidirectional graph edge counting [FIX-1 in code]. The scalar `φ_x(m_ij)` preserves equivariance since `(x_i - x_j)` is equivariant.

5. **Node update (invariant):**

$$\mathbf{h}_i \leftarrow \mathbf{h}_i + \phi_h\!\left(\left[\mathbf{h}_i \,\bigg\|\, \sum_{j}\mathbf{m}_{ij}\right]\right)$$

### 3.2 Attention Gate

An additional self-attention gate on top of EGNN message aggregation:

$$\text{att}_{ij} = \text{softmax}_j\!\left(\frac{\mathbf{q}_i^\top \mathbf{k}_{ij}}{\sqrt{d_k}}\right), \quad \mathbf{m}_{ij}^{\text{att}} = \text{att}_{ij} \cdot \mathbf{v}_{ij}$$

This allows the model to selectively weight neighbors without breaking equivariance (attention weights are scalars).

### 3.3 Timestep Conditioning

Timestep `t ∈ {0,...,T-1}` is embedded via **sinusoidal positional encoding** [DDPM]:

$$\text{emb}(t)_{2k} = \sin\!\left(\frac{t}{10000^{2k/d}}\right), \quad \text{emb}(t)_{2k+1} = \cos\!\left(\frac{t}{10000^{2k/d}}\right)$$

This `time_dim=256`-dimensional embedding is projected and added to node features at each layer, injecting timestep information while preserving equivariance (it's invariant — same scalar for all atoms in the molecule).

### 3.4 Full Denoiser Architecture

```
x_0_pred = Denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)

Architecture:
  - Atom embedding:  nn.Embedding(54, 384)
  - Bond embedding:  nn.Embedding(5, 384)
  - Time projection: MLP(256 → 384)
  - 8× EquivariantLayer(hidden_dim=384, num_rbf=32)
  - Output head:     nn.Linear(384, 3)
  - CoM removal on output
```

Total parameters: **8.28M**

---

## 4. Training Objective

### 4.1 x₀ Parameterization [EDM, GeoDiff]

Rather than predicting noise `ε` (as in standard DDPM), the denoiser directly predicts the clean coordinates `x_0_pred`:

$$\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_t, t, \mathcal{G})$$

**Why x₀ parameterization?** At high noise levels (`ᾱ_t ≈ 0`), recovering `x_0` from a noise prediction requires:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\boldsymbol{\varepsilon}}}{\sqrt{\bar{\alpha}_t}}$$

The `1/√ᾱ_t` factor **amplifies errors catastrophically** when `ᾱ_t ≈ 0`. Predicting `x_0` directly avoids this instability [EDM Appendix B].

### 4.2 MSE Loss (per-molecule)

$$\mathcal{L}_{\text{mse}} = \frac{1}{N} \sum_{i=1}^{N} \|\hat{\mathbf{x}}_{0,i} - \mathbf{x}_{0,i}\|^2$$

Reduced per-molecule (not per-atom) to avoid large-molecule bias [FIX-6].

### 4.3 Min-SNR Weighting [MinSNR]

Raw MSE training over-weights high-noise timesteps (where `SNR_t = ᾱ_t/(1-ᾱ_t)` is small). Min-SNR clamps the effective weight:

$$w_t = \frac{\min(\text{SNR}_t,\ \gamma)}{\text{SNR}_t}, \quad \gamma = 5.0$$

$$\mathcal{L}_{\text{mse}}^{\text{weighted}} = \mathbb{E}_t\left[w_t \cdot \frac{1}{N}\sum_i \|\hat{\mathbf{x}}_{0,i} - \mathbf{x}_{0,i}\|^2\right]$$

This balances gradient magnitudes across timesteps.

### 4.4 Geometry Constraint Loss [GCDM]

Auxiliary supervision on predicted bond lengths, bond angles, and (optionally) torsion angles:

$$\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{bond}} + \mathcal{L}_{\text{angle}}$$

**Bond length loss:**

$$\mathcal{L}_{\text{bond}} = \frac{1}{|E|}\sum_{(i,j)\in E}\left(\|\hat{\mathbf{x}}_i - \hat{\mathbf{x}}_j\| - d^*_{b_{ij}}\right)^2$$

where `d*_{b}` is the ideal bond length for bond type `b` (e.g., C-C single = 1.54 Å, C=C double = 1.34 Å).

**Bond angle loss:**

$$\mathcal{L}_{\text{angle}} = \frac{1}{|\text{angles}|}\sum_{i-j-k}\left(\theta_{ijk} - \theta^*_{ijk}\right)^2, \quad \theta_{ijk} = \arccos\frac{(\mathbf{x}_i-\mathbf{x}_j)\cdot(\mathbf{x}_k-\mathbf{x}_j)}{\|\mathbf{x}_i-\mathbf{x}_j\|\|\mathbf{x}_k-\mathbf{x}_j\|}$$

### 4.5 Timestep Gating for Geometry Loss [GCDM §3.3]

At high noise levels `t > T·τ` (where `τ=0.3`), the predicted `x_0_pred` has RMSD ~1 Å from ground truth, making geometry gradients chaotic. We gate:

$$\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{geo}} \cdot \mathbf{1}[t < T \cdot \tau]$$

Only molecules with `t < 300` (out of 1000) receive geometry supervision.

### 4.6 Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mse}}^{\text{weighted}} + \lambda_{\text{geo}} \cdot \mathcal{L}_{\text{geo}}$$

where `λ_geo = 0.5` (geometry weight).

### 4.7 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 5×10⁻⁴ (cosine warmup) |
| Batch size | 128 (per GPU) |
| GPUs | 2× RTX 3090 (DDP) |
| Epochs | 500 |
| Gradient clip | 1.0 |
| Weight decay | 1×10⁻⁴ |
| `hidden_dim` | 384 |
| `num_layers` | 8 |
| `num_rbf` | 32 |
| `timesteps` | 1000 |

**Distributed Data Parallel (DDP):** Each GPU processes half the batch. Gradients are all-reduced via `torch.distributed.all_reduce` before the optimizer step. `DistributedSampler` ensures no sample overlap across GPUs.

---

## 5. Reverse Process — Generation (DDIM Sampling)

### 5.1 DDIM Formulation [DDIM]

Standard DDPM sampling requires T=1000 steps. DDIM achieves the same quality in 50-100 steps by deriving a **non-Markovian** reverse process.

**Initialization:**

$$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \mathbf{x}_T \leftarrow \text{RemoveCoM}(\mathbf{x}_T)$$

Starting from pure Gaussian noise in CoM-free subspace.

**DDIM update step** (from `x_t` to `x_{t-1}`, using x₀ parameterization):

1. Predict clean coordinates:
$$\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_t, t, \mathcal{G})$$

2. Derive noise prediction:
$$\hat{\boldsymbol{\varepsilon}} = \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\,\hat{\mathbf{x}}_0}{\sqrt{1-\bar{\alpha}_t}}$$

3. Compute sigma (stochasticity):
$$\sigma_t = \eta \cdot \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}\right)}$$

With `η=0` (deterministic DDIM), `σ_t = 0`.

4. Compute direction to `x_t`:
$$\text{dir} = \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \hat{\boldsymbol{\varepsilon}}$$

5. Update:
$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{\mathbf{x}}_0 + \text{dir} + \sigma_t\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$

**Numerical fix:** Clamp `(1 - ᾱ_t/ᾱ_{t-1})` to `≥ 0` before `sqrt` to prevent NaN [FIX-5 in code].

### 5.2 Sampling Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `num_steps` | 50–100 | More steps → better quality, slower |
| `eta` | 0.0 | Deterministic; η>0 adds stochasticity |
| Step size | `T // num_steps` = 10–20 | Evenly spaced timesteps |

**At final step** (`t=0`): directly output `x_0_pred` without DDIM update.

---

## 6. Evaluation Metrics

### 6.1 Kabsch RMSD [Kabsch]

Minimum RMSD between two point clouds `P, Q ∈ ℝ^{N×3}` after optimal rigid-body alignment:

1. Center both: `P ← P - mean(P)`, `Q ← Q - mean(Q)`
2. Compute covariance: `H = P^T Q`
3. SVD: `H = UΣV^T`
4. Correct for reflection: `D = diag(1,1, sign(det(V U^T)))`
5. Optimal rotation: `R = V D U^T`
6. RMSD: `√(mean‖R·P - Q‖²)`

### 6.2 COV-R (Coverage Recall) [GeoDiff]

Fraction of reference conformers covered by at least one generated conformer within threshold `δ`:

$$\text{COV-R}(\delta) = \frac{1}{M}\sum_{m=1}^{M} \mathbf{1}\!\left[\min_{g}\text{RMSD}(R_m, G_g) \leq \delta\right]$$

where `M` = number of reference conformers, `G = {G_g}` = generated set. **δ = 0.5 Å for QM9**, **δ = 1.25 Å for GEOM-Drugs**.

### 6.3 MAT-R (Matching Recall) [GeoDiff]

Mean of best-match RMSD for each reference:

$$\text{MAT-R} = \frac{1}{M}\sum_{m=1}^{M} \min_{g}\,\text{RMSD}(R_m, G_g)$$

Lower is better. This measures how accurately the model recovers reference geometries.

### 6.4 COV-P (Coverage Precision)

Fraction of **generated** conformers within δ of some reference (measures realism):

$$\text{COV-P}(\delta) = \frac{1}{|G|}\sum_{g} \mathbf{1}\!\left[\min_{m}\text{RMSD}(G_g, R_m) \leq \delta\right]$$

### 6.5 MAT-P (Matching Precision)

Mean of best-match RMSD for each generated conformer to its nearest reference:

$$\text{MAT-P} = \frac{1}{|G|}\sum_{g} \min_{m}\,\text{RMSD}(G_g, R_m)$$

### 6.6 Diversity

Mean pairwise Kabsch-RMSD among generated conformers for one molecule:

$$\text{Div} = \frac{1}{\binom{n}{2}} \sum_{i<j} \text{RMSD}(G_i, G_j)$$

**High diversity (> 0.1 Å)** indicates multi-modal sampling (no mode collapse).  
**Near-zero diversity** indicates the model always generates the same conformer.

### 6.7 GeoDiff Sampling Protocol [GeoDiff Table 2]

For a molecule with `M` reference conformers:
- Generate `2M` conformers (the "2× rule")
- Evaluate COV-R/MAT-R between reference set and generated set

---

## 7. Full End-to-End Pipeline Summary

```
Dataset (QM9/GEOM)
    ↓  Strip H, build graph G=(V,E), center CoM
    ↓  90/10 train/val split (seed=42)

Training Loop (500 epochs, DDP, 2× RTX 3090)
    For each batch:
    1. Sample timestep t ~ Uniform{0,...,999}
    2. Sample noise ε ~ N(0,I), remove CoM
    3. Forward diffuse: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    4. Predict: x̂_0 = Denoiser(x_t, t, G)  [8-layer E-GNN, 384 hidden]
    5. MSE loss: L_mse = w_t · mean_mol ||x̂_0 - x_0||²
    6. If t < 300: geo loss on bond lengths + angles
    7. Total: L = L_mse + 0.5 · L_geo
    8. Backprop + AdamW update

Inference (DDIM, 50 steps, η=0)
    1. x_T ~ N(0,I), remove CoM
    2. For t = T,...,0 (50 steps):
         x̂_0 = Denoiser(x_t, t, G)
         x_{t-1} = √ᾱ_{t-1}·x̂_0 + √(1-ᾱ_{t-1})·ε̂_derived
    3. Output x_0: generated 3D conformer

Evaluation (GeoDiff protocol)
    For each val molecule:
    1. Generate 10 conformers from Gaussian noise
    2. Compute Kabsch-RMSD between each gen↔ref pair
    3. Report COV-R, MAT-R, COV-P, MAT-P, Diversity
```

---

## 8. Results (exp_G, Epoch 500)

| Metric | Ours | GeoDiff [GeoDiff] | GeoMol [GeoMol] | TorDiff [TorDiff] |
|--------|------|-------------------|-----------------|-------------------|
| COV-R@0.5Å | **95.5%** | 71.0% | 71.5% | 73.2% |
| MAT-R (Å) | **0.229** | 0.297 | 0.225 | 0.219 |
| COV-P@0.5Å | **76.2%** | — | — | — |
| MAT-P (Å) | **0.343** | — | — | — |
| Diversity (Å) | **0.258** | — | — | — |

**Dataset:** QM9 heavy-atom val split, 200 molecules × 10 generated conformers.
