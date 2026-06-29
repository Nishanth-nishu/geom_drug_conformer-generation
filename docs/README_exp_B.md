# Experiment B — EQGAT-diff Attention-Enhanced EGNN

## Summary

| Field | Value |
|-------|-------|
| Model | AttnConformerDiffusion (multi-head attention EGNN + DDPM) |
| Script | `scripts/exp_B_attention_egnn.sh` |
| Trainer | `autoresearch/mol_train_expB.py` |
| Key paper | Le et al., *EQGAT-diff*, ICLR 2024 |
| Hypothesis | Attention-weighted message aggregation outperforms sum-pooling for conformer accuracy |
| Expected gain | ~10% lower MAT-R vs Exp A |

---

## Research Motivation

Standard EGNN (Satorras 2021) aggregates incoming messages with **unweighted sum-pooling**:

$$\mathbf{h}_i \leftarrow \mathbf{h}_i + \text{MLP}\!\left(\mathbf{h}_i \| \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}\right)$$

This treats all neighbours equally. For molecules, this is chemically wrong:

- A **chiral centre** (sp³ carbon with 4 distinct substituents) must distinguish all 4 neighbours to predict the correct R/S configuration. Equal weighting loses this information.
- An **aromatic ring** needs to couple across the full π-system — certain long-range bond partners matter more than saturated-chain neighbours.
- An **amide bond** (sp²-sp³) has one highly relevant conjugated partner (the C=O) that should dominate the torsional representation.

**EQGAT-diff** (Le et al., ICLR 2024) showed that adding multi-head attention over EGNN messages reduces MAT-R by ~15% on QM9 compared to standard EGNN, specifically because attention correctly identifies which neighbours carry the most conformationally-relevant information.

### Why EQGAT-diff specifically?

EQGAT-diff was chosen over alternatives (e.g., SE(3)-Transformer, NequIP/Allegro) because:

1. **Same backbone:** It adds attention *on top of* the EGNN message-passing framework, not a complete architecture replacement. This isolates the attention contribution.
2. **Empirically validated:** EQGAT-diff achieves MAT-R = 0.17 Å on QM9 (vs EDM's 0.44 Å) — a 61% improvement from the same DDPM framework.
3. **Computational cost:** Attention over messages (per edge, not per atom-pair) scales as $O(E \cdot d_h)$, not $O(N^2)$. For QM9's small molecules this is fast.

---

## Architecture Change: Multi-Head Attention Aggregation

### Standard EGNN (Exp A)
```
m_agg_i = sum_{j in N(i)} m_ij           // all neighbours equally
h_i_new = LayerNorm(h_i + MLP([h_i || m_agg_i]))
```

### EQGAT-diff (Exp B)
```
// Step 1: compute attention logits per edge per head
attn_logit_ij = LeakyReLU(W_attn @ [h_i || h_j || RBF(d_ij)])   // (E, num_heads)

// Step 2: softmax over incoming edges per destination node (not global softmax)
a_ij = softmax_{j in N(i)}(attn_logit_ij)  // (E, num_heads)

// Step 3: attention-weighted aggregation
m_agg_i = sum_{j in N(i)} a_ij * m_ij     // heads broadcast over head_dim

// Step 4: node update (same as before)
h_i_new = LayerNorm(h_i + MLP([h_i || m_agg_i]))
```

### Mathematical formulation

**Attention gate** (EQGAT-diff eq. 5):
$$a_{ij}^{(h)} = \frac{\exp\!\left(\text{LeakyReLU}\!\left(\mathbf{w}_h^\top [\mathbf{h}_i \| \mathbf{h}_j \| \text{RBF}(d_{ij})]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\!\left(\text{LeakyReLU}\!\left(\mathbf{w}_h^\top [\mathbf{h}_i \| \mathbf{h}_k \| \text{RBF}(d_{ik})]\right)\right)}$$

Note: softmax is over **incoming edges per destination node**, not globally. This ensures each atom's attention weights sum to 1 over its neighbours.

**Attention-weighted aggregation:**
$$\mathbf{m}_\text{agg,i} = \sum_{j \in \mathcal{N}(i)} a_{ij} \cdot \mathbf{m}_{ij} \in \mathbb{R}^{d_h}$$

where $a_{ij} \in \mathbb{R}^{H}$ (H=4 heads) is broadcast over the $d_h/H$ head dimensions.

**Coordinate update:** **Unchanged** from EGNN — equivariance is preserved because the coordinate update still uses scalar weights on unit vectors.

### Why LeakyReLU not Tanh/Softplus?

Following GAT (Veličković 2018), LeakyReLU (negative slope=0.2) prevents dead attention units (neurons stuck at 0 gradient) while still allowing the gate to express strong preferences. The original GAT ablation showed LeakyReLU outperforms ELU and ReLU on graph tasks.

---

## Hyperparameters (changes from Exp A)

| Parameter | Exp A | Exp B | Rationale |
|-----------|-------|-------|-----------|
| Model class | ConformerDiffusion | AttnConformerDiffusion | EQGAT-diff style |
| num_heads | — | 4 | Le et al. use 4 heads; matches $d_h=256$, head_dim=64 |
| dropout | 0.1 | 0.1 | Applied to attention weights (prevents over-attention) |
| Parameters | ~2.8M | ~3.1M | +0.3M from attention projection weights |

All other hyperparameters (lr, batch_size, geometry_weight, optimizer) are **identical to Exp A** to ensure the architecture change is the only variable.

---

## What This Tests

This is a **controlled ablation** of message aggregation function:

- **Null hypothesis:** Attention does not improve over sum-pooling (MAT-R_B ≥ MAT-R_A)
- **Alternative hypothesis:** Attention improves conformer accuracy (MAT-R_B < MAT-R_A)

If Exp B improves MAT-R, the mechanism is clear: the model is learning to selectively attend to the most conformationally-relevant neighbours, supporting the hypothesis that sum-pooling is a bottleneck for molecular geometry learning.

---

## Expected Results

| Metric | Exp A (baseline) | Exp B (expected) | EQGAT-diff paper |
|--------|-----------------|------------------|-----------------|
| fully_valid | ~0.80 | ~0.82 | 0.917 (200 ep) |
| MAT-R (Å) | ~0.45 | ~0.40 | 0.17 (200 ep) |
| COV-R | ~0.30 | ~0.35 | 0.61 (200 ep) |

We expect partial improvement at 50 epochs (attention needs more epochs to learn complex attention patterns than simple sum-pooling).

---

## Citations

1. **Le et al.** "EQGAT-diff: a novel equivariant graph attention model for molecular 3D generation." *ICLR 2024.* arXiv:2306.01916
2. **Veličković et al.** "Graph Attention Networks." *ICLR 2018.* arXiv:1710.10903
3. **Satorras et al.** "E(n) Equivariant Graph Neural Networks." *ICML 2021.* arXiv:2102.09844
4. **Hoogeboom et al.** "Equivariant Diffusion for Molecule Generation in 3D." *ICML 2022.* arXiv:2203.17003
5. **Brody et al.** "How Attentive are Graph Attention Networks?" *ICLR 2022.* arXiv:2105.14491 (GATv2 — motivation for LeakyReLU gate design)
