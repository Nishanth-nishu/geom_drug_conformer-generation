# NExT-Mol Gen — Base Model: ConformerDiffusion

## Overview

The base model is an **E(3)-equivariant denoising diffusion probabilistic model (DDPM)** for 3D molecular conformer generation. Given a molecular graph (atoms + bond topology from SMILES), the model learns to generate the physically correct 3D arrangement of those atoms.

---

## 1. Why This Problem Is Hard

A molecule's biological activity depends not only on which atoms it contains but on their **precise 3D arrangement**. Two molecules with identical SMILES (same connectivity) but different 3D conformations can have completely different receptor binding affinities.

The challenge: the output lives in 3D Euclidean space under the **E(3) symmetry group** — rotating or translating a molecule gives the same molecule. A naïve model that ignores this must memorize all possible orientations, which is intractable.

---

## 2. Dataset: QM9

**Reference:** Ramakrishnan et al. *"Quantum chemistry structures and properties of 134 kilo molecules."* Scientific Data 2014.

QM9 contains **133,885 small organic molecules** (C, H, N, O, F atoms, ≤9 heavy atoms) with **DFT-optimized 3D coordinates** at the B3LYP/6-31G(2df,p) level of theory. These are physically valid, electronically stable minimum-energy geometries — the gold-standard training signal for conformer generation.

**Dataset format** (our JSONL):
```json
{
  "atom_types":  [6, 1, 1, 1, 1],        // atomic numbers (H=1, C=6, N=7, O=8, F=9)
  "coordinates": [[-0.013, 1.086, 0.008], ...],  // DFT 3D coords in Ångströms
  "edge_index":  [[0,1,1,2,...],[1,0,2,1,...]],   // both-direction bond pairs
  "bond_types":  [1,1,1,1,...],           // 1=single, 2=double, 3=triple, 4=aromatic
  "num_atoms":   5
}
```

**Why explicit H?** QM9 retains all hydrogen atoms. This is important because H positions encode chirality, ring puckering, and torsional preferences that are invisible in heavy-atom-only representations.

**Split:** 90% train (118,773) / 10% val (13,197), fixed seed=42.

---

## 3. Mathematical Foundation: DDPM

**Reference:** Ho et al. *"Denoising Diffusion Probabilistic Models."* NeurIPS 2020.

### 3.1 Forward Process (Data → Noise)

Given clean coordinates $\mathbf{x}_0 \in \mathbb{R}^{N \times 3}$, DDPM defines a Markov chain that gradually adds Gaussian noise:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\; (1-\bar\alpha_t)\mathbf{I}\right)$$

where $\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$ and $\beta_s$ is the noise schedule. This allows sampling any noisy $\mathbf{x}_t$ in one step:

$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon, \quad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### 3.2 Cosine Noise Schedule

**Reference:** Nichol & Dhariwal. *"Improved DDPM."* ICML 2021.

The linear schedule wastes capacity at extreme timesteps. The cosine schedule distributes learning signal more uniformly:

$$\bar\alpha_t = \frac{\cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)}{\cos^2\!\left(\frac{s}{1+s} \cdot \frac{\pi}{2}\right)}, \quad s=0.008$$

### 3.3 Center-of-Mass (CoM) Removal

**Reference:** Hoogeboom et al. *"Equivariant Diffusion for Molecule Generation in 3D (EDM)."* ICML 2022.

Molecular conformations are translation-invariant. Diffusing in the full $\mathbb{R}^{3N}$ space forces the model to also learn global position — wasted capacity. EDM constrains all diffusion to the **zero-CoM subspace**:

$$\text{CoM-remove}(\mathbf{x}, \text{batch}) = \mathbf{x} - \frac{1}{N_m}\sum_{i \in m}\mathbf{x}_i \quad \forall \text{ molecule } m$$

Applied to: (1) training data, (2) noise $\boldsymbol\epsilon$, (3) predicted $\hat{\mathbf{x}}_0$ at every denoising step.

### 3.4 x₀ Parameterization

**Reference:** Hoogeboom et al. EDM 2022; Bao et al. *"Analytic-DPM."* ICLR 2022.

Instead of predicting the added noise $\boldsymbol\epsilon$ (standard DDPM), we predict the clean coordinates $\hat{\mathbf{x}}_0$ directly:

$$\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_t, t, \mathcal{G})$$

Noise is then derived: $\hat{\boldsymbol\epsilon} = (\mathbf{x}_t - \sqrt{\bar\alpha_t}\hat{\mathbf{x}}_0) / \sqrt{1-\bar\alpha_t}$

**Why x₀?** It allows applying geometry constraints (bond lengths, angles) directly on the predicted clean structure at every timestep, giving denser chemistry supervision.

### 3.5 Min-SNR Loss Weighting

**Reference:** Hang et al. *"Efficient Diffusion Training via Min-SNR Weighting."* ICCV 2023.

The signal-to-noise ratio $\text{SNR}(t) = \bar\alpha_t / (1-\bar\alpha_t)$ varies by orders of magnitude across timesteps. Without reweighting, high-noise timesteps dominate the loss and cause instability. Min-SNR clips the per-timestep weight:

$$w(t) = \min\!\left(1,\; \frac{\gamma}{\text{SNR}(t)}\right), \quad \gamma = 5$$

The training loss becomes:
$$\mathcal{L}_\text{MSE} = \frac{1}{N}\sum_{i=1}^N w(t_{m(i)}) \cdot \|\hat{\boldsymbol\epsilon}_i - \boldsymbol\epsilon_i\|^2$$

---

## 4. Model Architecture: ConformerDenoiser

### 4.1 Input Representation

Each molecule is a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with:

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\mathbf{z}$ | $(N,)$ | Atomic numbers (1–53) |
| $\mathbf{x}_t$ | $(N,3)$ | Noisy 3D coordinates at timestep $t$ |
| $\text{edge\_index}$ | $(2,E)$ | Both-direction bond pairs |
| $\mathbf{b}$ | $(E,)$ | Bond types: 1=single, 2=double, 3=triple, 4=aromatic |
| $\text{batch\_idx}$ | $(N,)$ | Molecule membership per atom |

**Atom embedding:** Learnable lookup $\mathbf{h}_i^{(0)} = \text{Embed}(\mathbf{z}_i) \in \mathbb{R}^{d_h}$, covering atoms H(1) through I(53).

**Time embedding:** Sinusoidal positional encoding (adapted from Transformer, Vaswani et al. 2017) mapped through an MLP:

$$\mathbf{t}_\text{emb} = \text{MLP}\!\left(\text{SinEmbed}(t, d_t)\right) \in \mathbb{R}^{d_h}$$

$$\text{SinEmbed}(t)_k = \begin{cases}\sin(t / 10000^{2k/d_t}) & k \text{ even} \\ \cos(t / 10000^{2k/d_t}) & k \text{ odd}\end{cases}$$

**Initial node features:** $\mathbf{h}_i = \text{Embed}(\mathbf{z}_i) + \mathbf{t}_\text{emb}[\text{batch}[i]]$ (additive time conditioning).

**RBF distance features:** Pairwise distances encoded with 20 radial basis functions:

$$\phi_k(d) = \exp\!\left(-\frac{(d - \mu_k)^2}{2\sigma^2}\right), \quad \mu_k \in [0.5, 10.0]\text{ Å}$$

RBF gives the model smooth, differentiable distance information rather than a raw scalar.

### 4.2 Equivariant Message Passing (EGNN)

**Reference:** Satorras et al. *"E(n) Equivariant Graph Neural Networks."* ICML 2021.

Each `EquivariantLayer` performs one round of E(3)-equivariant message passing. The key insight is that coordinate updates must be equivariant: rotating the input should rotate the output by the same amount.

**Step 1 — Edge messages** (invariant MLP):
$$\mathbf{m}_{ij} = \phi_e\!\left(\mathbf{h}_i \| \mathbf{h}_j \| \text{RBF}(d_{ij}) \| \mathbf{e}_{ij}\right) + \mathbf{e}_{ij}$$

where $d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|$ and $\mathbf{e}_{ij}$ is the bond-type embedding. The residual `+ e_ij` ensures bond type info is never washed out.

**Step 2 — Coordinate update** (equivariant):
$$\Delta\mathbf{x}_i = \frac{1}{\deg(i)+1}\sum_{j \in \mathcal{N}(i)} \phi_x(\mathbf{m}_{ij}) \cdot \hat{\mathbf{u}}_{ij}$$

where $\hat{\mathbf{u}}_{ij} = (\mathbf{x}_i - \mathbf{x}_j)/\|\mathbf{x}_i - \mathbf{x}_j\|$ is the unit displacement vector. This update is **equivariant** because: (i) unit vectors rotate with $R$, (ii) scalar weights $\phi_x(\mathbf{m}_{ij})$ are invariant (distances don't change under rotation). Degree normalization $1/(\deg(i)+1)$ prevents updates from growing with molecule size.

**Step 3 — Node feature update** (invariant MLP + residual):
$$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\!\left(\mathbf{h}_i^{(l)} + \phi_h\!\left(\mathbf{h}_i^{(l)} \| \textstyle\sum_j \mathbf{m}_{ij}\right)\right)$$

**Why EGNN?** It achieves E(3)-equivariance without expensive irreducible representation decompositions (unlike SE(3)-Transformers or NequIP), making it fast to train and simple to implement while maintaining the equivariance guarantee needed for molecules.

### 4.3 Output Head (x₀ Prediction)

After $L$ equivariant layers, a 3-layer MLP maps node features to coordinate residuals:

$$\hat{\mathbf{x}}_0 = \mathbf{x}^{(L)} + \text{MLP}(\mathbf{h}^{(L)})$$

The addition of $\mathbf{x}^{(L)}$ (which has already evolved through equivariant updates) as a residual helps the model output physically reasonable coordinates even early in training.

### 4.4 Default Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden dim $d_h$ | 256 | Balances expressivity and GPU memory |
| Layers $L$ | 6 | Covers molecular diameter of QM9 (≤9 heavy atoms) |
| RBF bins | 20 | Sufficient resolution for 0.5–10 Å range |
| Time dim $d_t$ | 128 | Matches time embedding literature |
| Diffusion steps $T$ | 1000 | Standard DDPM |
| Dropout | 0.1 | Prevents overfitting (train/val gap closes) |
| **Total params** | **~2.8M** | Fast, fits in 2GB VRAM |

---

## 5. Geometry Constraint System

**References:** Halgren (MMFF94, 1996); Jorgensen et al. (OPLS-AA, 1996); Hoogeboom EDM 2022.

Using $x_0$ parameterization, we can supervise the predicted clean structure with chemistry-aware losses at every training step:

$$\mathcal{L}_\text{geo} = \mathcal{L}_\text{bond} + \mathcal{L}_\text{angle} + \mathcal{L}_\text{repulsion} + \mathcal{L}_\text{planarity} + \mathcal{L}_\text{chirality} + \mathcal{L}_\text{ring}$$

**Bond length loss** (MMFF94 lookup table, vectorized):
$$\mathcal{L}_\text{bond} = \lambda_b \cdot \frac{1}{|E|}\sum_{(i,j)} \!\left(\|\hat{\mathbf{x}}_i - \hat{\mathbf{x}}_j\| - d^*_{z_i,z_j,b_{ij}}\right)^2$$

**Angle loss** (hybridization-aware: sp=180°, sp²=120°, sp³=109.5°, aromatic=120°):
$$\mathcal{L}_\text{angle} = \lambda_a \cdot \frac{1}{|A|}\sum_{(i,j,k)}\!\left(\angle(\hat{\mathbf{x}}_i, \hat{\mathbf{x}}_j, \hat{\mathbf{x}}_k) - \theta^*_j\right)^2$$

**Repulsion loss** (VDW-based, excludes 1-2 and 1-3 pairs, computed per-molecule to avoid batch-size bias):
$$\mathcal{L}_\text{rep} = \lambda_r \cdot \text{mean}_{d_{ij} < r_\text{clash}}\!\left(r_\text{clash}(i,j) - d_{ij}\right)^2$$

**SNR gating:** Geometry loss is weighted by the batch-mean SNR weight $\bar{w}$. At high noise (low SNR), the predicted $\hat{\mathbf{x}}_0$ is too inaccurate for geometry supervision to be meaningful:
$$\mathcal{L}_\text{total} = \mathcal{L}_\text{MSE} + \bar{w} \cdot \lambda_\text{geo} \cdot \mathcal{L}_\text{geo}$$

---

## 6. Inference: DDIM Sampling

**Reference:** Song et al. *"Denoising Diffusion Implicit Models."* ICLR 2021.

Full DDPM requires $T=1000$ denoiser forward passes at inference — too slow. DDIM reformulates the reverse process as a deterministic, non-Markovian ODE solvable with large steps:

Given subsequence $\tau_1 > \tau_2 > \cdots > \tau_S$ with $S=50$:

1. Predict clean coords: $\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_{\tau_i}, \tau_i, \mathcal{G})$, soft-clamped to $[-10,10]$ via $10\tanh(\hat{\mathbf{x}}_0/10)$
2. DDIM update ($\eta=0$, deterministic):
$$\mathbf{x}_{\tau_{i+1}} = \sqrt{\bar\alpha_{\tau_{i+1}}}\hat{\mathbf{x}}_0 + \sqrt{1-\bar\alpha_{\tau_{i+1}}}\cdot\hat{\boldsymbol\epsilon}_\theta$$

This gives a **20× speedup** (50 steps vs 1000) with negligible quality loss.

---

## 7. Evaluation Metrics

| Metric | Formula | Source |
|--------|---------|--------|
| **fully_valid** | Fraction passing RDKit `SanitizeMol()` | EDM 2022 |
| **validity** | Fraction with all bonds within ±0.20 Å of MMFF94 ideal | GeoDiff 2022 |
| **MAT-R** | $\min_{\hat{\mathbf{x}}\in\hat{S}}\text{RMSD}(\hat{\mathbf{x}}, \mathbf{x}^*)$ averaged over test set | GeoMol 2021 |
| **COV-R** | Fraction of references covered within RMSD threshold (0.5 Å) | GeoMol 2021 |
| **strain (kcal/mol)** | MMFF94 force-field energy of generated geometry | PhysNet 2019 |
| **clash-free** | Fraction with no atom pair closer than 1.4 Å | EDM 2022 |

**MAT-R** (Matching score, Recall) is the primary metric: it measures how close the best generated conformer is to the true DFT geometry. Lower is better. Published baselines: GeoMol = 0.225 Å, EDM = 0.44 Å, EQGAT-diff = 0.17 Å.

---

## 8. Training Recipe

| Setting | Value | Reference |
|---------|-------|-----------|
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.999$) | Loshchilov & Hutter 2019 |
| Learning rate | $1\times10^{-4}$ | EDM 2022 |
| Warmup | 5 epochs linear $0 \to \eta$ | SGDR (Loshchilov 2017) |
| Schedule | Cosine annealing $\eta \to 0$ over 50 ep | SGDR (Loshchilov 2017) |
| Grad clip | $\|\nabla\theta\|_2 \leq 1.0$ | Prevents explosion after ep 150 |
| Batch size | 64 molecules | Memory-compute tradeoff |
| Geometry weight | $\lambda_\text{geo} = 0.1$ | Ablated in Exp D |
| Min-SNR $\gamma$ | 5 | Hang et al. ICCV 2023 |

---

## 9. Full Citations

1. **Ho et al.** "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.* arXiv:2006.11239
2. **Satorras et al.** "E(n) Equivariant Graph Neural Networks." *ICML 2021.* arXiv:2102.09844
3. **Hoogeboom et al.** "Equivariant Diffusion for Molecule Generation in 3D." *ICML 2022.* arXiv:2203.17003
4. **Song et al.** "Denoising Diffusion Implicit Models." *ICLR 2021.* arXiv:2010.02502
5. **Nichol & Dhariwal.** "Improved DDPM." *ICML 2021.* arXiv:2102.09672
6. **Hang et al.** "Efficient Diffusion Training via Min-SNR Weighting." *ICCV 2023.* arXiv:2303.09556
7. **Ganea et al.** "GeoMol: Torsional GNN for Molecular Conformer Generation." *NeurIPS 2021.* arXiv:2106.07802
8. **Xu et al.** "GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation." *ICLR 2022.* arXiv:2203.02923
9. **Ramakrishnan et al.** "Quantum chemistry structures and properties of 134 kilo molecules." *Scientific Data 2014.*
10. **Halgren.** "Merck molecular force field (MMFF94)." *J. Comput. Chem. 1996.*
11. **Vaswani et al.** "Attention is All You Need." *NeurIPS 2017.* arXiv:1706.03762
