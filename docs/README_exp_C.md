# Experiment C — Conditional Flow Matching (CFM)

## Summary

| Field | Value |
|-------|-------|
| Model | FlowMatchingConformer (EGNN backbone + CFM training + Euler ODE) |
| Script | `scripts/exp_C_flow_matching.sh` |
| Trainer | `autoresearch/mol_train_expC.py` |
| Key paper | Lipman et al., *Flow Matching*, ICLR 2023 |
| Hypothesis | Straight ODE trajectories → same accuracy as DDPM at 4× fewer inference steps |
| Key advantage | 20 ODE steps vs 50 DDIM steps (2.5× faster inference) |

---

## Research Motivation

### The Problem with DDPM at Inference

DDPM's reverse process follows **curved stochastic paths** through noisy intermediate states. Even with DDIM acceleration (Song 2021), reaching high-quality samples requires ≥50 function evaluations (NFE) because:

1. The denoiser must be evaluated at many noise levels to navigate the curved path
2. The stochastic nature of DDPM means each trajectory is unique — no reuse between samples
3. Error accumulates at each step; fewer steps → larger discretization error

### Conditional Flow Matching: The Idea

**Reference:** Lipman et al. *"Flow Matching for Generative Modeling."* ICLR 2023. arXiv:2210.02747

CFM replaces the curved stochastic diffusion path with a **straight deterministic ODE path** between data and noise. The key insight is:

> If the path from data $\mathbf{x}_0$ to noise $\boldsymbol\epsilon$ is a straight line, any ODE solver can traverse it in far fewer steps.

**Forward process (linear interpolation):**
$$\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\boldsymbol\epsilon, \quad t \in [0,1], \quad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})$$

**Target velocity field** (the derivative of the path — constant!):
$$\mathbf{v}^* = \frac{d\mathbf{x}_t}{dt} = \boldsymbol\epsilon - \mathbf{x}_0$$

This is a **constant vector** — the velocity doesn't change along the path. The neural network just needs to learn to point from the noisy state toward the clean data.

**Training objective:**
$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t \sim U[0,1],\; \mathbf{x}_0,\; \boldsymbol\epsilon}\left[\|\mathbf{v}_\theta(\mathbf{x}_t, t) - (\boldsymbol\epsilon - \mathbf{x}_0)\|^2\right]$$

Compare to DDPM's loss: $\mathbb{E}\left[\|\hat{\boldsymbol\epsilon}_\theta - \boldsymbol\epsilon\|^2\right]$ — CFM predicts velocity while DDPM predicts noise. Both use the same GNN backbone.

### Why This Helps for Molecules

**FrameDiff** (Yim et al., ICML 2023) applied CFM-style straight paths to protein backbone generation and showed:
- Equal geometry accuracy to DDPM
- 10× fewer inference steps
- Cleaner loss landscape → faster convergence

For 3D molecular conformers, the straight-path structure means the geometry constraint loss on the predicted $\hat{\mathbf{x}}_0$ is **more accurate** at intermediate $t$:

$$\hat{\mathbf{x}}_0^\text{CFM} = \mathbf{x}_t - t \cdot \mathbf{v}_\theta$$

Since $\mathbf{x}_t$ is on a straight line between $\mathbf{x}_0$ and $\boldsymbol\epsilon$, this reconstruction is exact when $\mathbf{v}_\theta$ is perfect, and degrades gracefully otherwise. In contrast, DDPM's $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\hat{\boldsymbol\epsilon}) / \sqrt{\bar\alpha_t}$ can produce unstable estimates when $\bar\alpha_t$ is small.

---

## Architecture

The `FlowMatchingConformer` reuses the **same EGNN backbone** (`ConformerDenoiser`) as Exp A — only the training objective and sampling algorithm change.

```
Training:
  t ~ Uniform[0, 1]  (continuous, not discrete timesteps)
  eps = CoM-remove(randn(N, 3))
  x_t = (1-t) * x_0 + t * eps          // linear interpolation
  v_target = eps - x_0                  // constant velocity
  v_pred = velocity_net(x_t, t, graph)  // same EGNN as Exp A
  loss = MSE(v_pred, v_target)

Inference (Euler ODE, t: 1 → 0):
  x = CoM-remove(randn(N, 3))           // start from noise
  for t in linspace(1.0, 0.0+dt, 20):  // 20 steps
      v = velocity_net(x, t, graph)
      x = x + v * (-1/20)              // Euler step backward
      x = CoM-remove(x)
  return x  // generated conformer
```

### Key difference from DDPM sampling

DDPM (DDIM): navigates a *curved path* approximated by 50 steps  
CFM: navigates a *straight path* exactly traversable in 20 steps  
**Result:** 2.5× faster inference at equal or better accuracy.

### Bug Fixes Applied (vs. original `flow_matching.py`)

**Bug 1 — Wrong argument order** in `velocity_net` call:
```python
# ORIGINAL (broken):
self.velocity_net(x_t, atom_types, edge_index, bond_types, batch_idx, t_int[batch_idx])
# ConformerDenoiser.forward(x_noisy, t, atom_types, edge_index, bond_types, batch_idx)
# atom_types was being interpreted as the timestep!

# FIXED:
self.velocity_net(x_t, t_int_per_atom, atom_types, edge_index, bond_types, batch_idx)
```

**Bug 2 — Wrong x0_hat reconstruction** for geometry loss:
```python
# ORIGINAL (broken):
x0_hat = x_t - t * v_pred     # t is shape (B,) — wrong broadcast with (N,3)

# FIXED:
t_atom = t[batch_idx].unsqueeze(-1)   # (N, 1) — correct per-atom broadcast
x0_hat = x_t - t_atom * v_pred       # (N, 3) — correct
```

---

## Hyperparameters (changes from Exp A)

| Parameter | Exp A | Exp C | Rationale |
|-----------|-------|-------|-----------|
| Training objective | DDPM SNR-weighted MSE | CFM MSE on velocity | Lipman 2023 |
| Sampling | DDIM 50 steps | Euler ODE 20 steps | CFM straight paths |
| geometry_weight | 0.1 | 0.05 | CFM x0_hat less stable early — reduce geo pressure |
| t distribution | Discrete $t \in \{1,...,1000\}$ | Continuous $t \sim U[0,1]$ | CFM formulation |

---

## Connection to Optimal Transport

Lipman et al. (2023) extend CFM to **Optimal Transport CFM (OT-CFM)** where $\boldsymbol\epsilon$ is paired with $\mathbf{x}_0$ via a mini-batch optimal transport plan (instead of random pairing). OT-CFM further straightens paths and requires even fewer NFE. We use the basic CFM formulation for simplicity and extensibility.

**Future direction:** OT-CFM with molecule-specific couplings (e.g., pair $\boldsymbol\epsilon$ to the nearest-energy conformation from a MMFF94 ensemble) could further reduce the required ODE steps.

---

## Expected Results

| Metric | Exp A | Exp C (expected) |
|--------|-------|-----------------|
| fully_valid | ~0.80 | ≥0.80 (equal or better) |
| MAT-R (Å) | ~0.45 | ~0.40–0.45 (similar) |
| NFE at inference | 50 | **20** (2.5× faster) |
| Training convergence | 50 ep to plateau | ~40 ep (straighter loss landscape) |

---

## Citations

1. **Lipman et al.** "Flow Matching for Generative Modeling." *ICLR 2023.* arXiv:2210.02747
2. **Yim et al.** "SE(3) Diffusion Model with Application to Protein Backbone Generation (FrameDiff)." *ICML 2023.* arXiv:2302.02277
3. **Albergo & Vanden-Eijnden.** "Building Normalizing Flows with Stochastic Interpolants." *ICLR 2023.* arXiv:2209.15571 (parallel CFM derivation)
4. **Liu et al.** "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." *ICLR 2023.* arXiv:2209.03003 (rectified flow, related approach)
5. **Song et al.** "Score-Based Generative Modeling through SDEs." *ICLR 2021.* arXiv:2011.13456 (continuous-time SDE framework that unifies DDPM and flow matching)
6. **Satorras et al.** "E(n) Equivariant Graph Neural Networks." *ICML 2021.* arXiv:2102.09844
