"""
geom_drugs_eval.py — GEOM-Drugs Benchmark Evaluation (v2)

Implements the standard GEOM-Drugs conformer generation evaluation protocol
plus two novel energy-aware metrics:

Standard metrics:
  - COV-R  (Coverage Recall):  fraction of reference conformers covered by at
                                least one generated conformer within threshold Å RMSD
  - MAT-R  (Matching Recall):  mean minimum RMSD from each reference to its
                                closest generated conformer
  - COV-P  (Coverage Precision): fraction of generated conformers close to a reference
  - MAT-P  (Matching Precision): mean minimum RMSD from each generated to a reference

v2 Novel energy-aware metrics:
  - Bw-COV-R (Boltzmann-Weighted Coverage Recall):
      COV-R weighted by Boltzmann probability w_i = exp(-E_i/kT).
      Thermodynamically dominant conformers (low energy) matter more.
      Unlike standard COV-R which treats rare/common conformers equally.

  - MEE (Mean Energy Error):
      E[E_surrogate(x_generated)] - E_min_reference  (kcal/mol)
      Measures: are generated conformers thermodynamically stable?

Published GEOM-Drugs reference points (Jing et al., NeurIPS 2022, Table 1; threshold 0.75 A;
GeoMol test split; GeoDiff retrained on that split; mean):
  GeoDiff (retrained): COV-R=42.1%, MAT-R=0.835 A, COV-P=24.9%, MAT-P=1.136 A
  Torsional Diffusion: COV-R=72.7%, MAT-R=0.582 A, COV-P=55.2%, MAT-P=0.778 A
  RDKit ETKDG:         COV-R=38.4%, MAT-R=1.058 A, COV-P=40.9%, MAT-P=0.995 A
Coverage below is reported at 0.5 A, so only the MAT columns are directly comparable
(and only under the same test set and RMSD variant).
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Subset

BOLTZMANN_KT_KCAL = 0.5921  # kT at 298K in kcal/mol (= 1.987e-3 * 298)

# ──────────────────────────────────────────────────────────────────────────────
# Kabsch RMSD (explicit CoM centering + SVD alignment)
# ──────────────────────────────────────────────────────────────────────────────

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Kabsch-aligned RMSD between two conformers P and Q.
    Both P and Q are (N, 3) arrays. CoM centering applied.
    """
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return float(np.sqrt(np.sum((P @ R.T - Q) ** 2) / P.shape[0]))


# ──────────────────────────────────────────────────────────────────────────────
# Standard COV-MAT metric
# ──────────────────────────────────────────────────────────────────────────────

def cov_mat(refs: List[np.ndarray],
            gens: List[np.ndarray],
            threshold: float = 0.5) -> Tuple[float, float, float, float]:
    """
    Compute COV-R, MAT-R, COV-P, MAT-P for a single molecule.

    Args:
        refs:      List of reference conformer coordinate arrays (N, 3)
        gens:      List of generated conformer coordinate arrays (N, 3)
        threshold: RMSD threshold in A for 'coverage' (default 0.5 A for drugs)

    Returns:
        (cov_r, mat_r, cov_p, mat_p) — all scalars
    """
    if not refs or not gens:
        return 0.0, float('inf'), 0.0, float('inf')

    # Pairwise RMSD matrix: (n_refs, n_gens)
    rmsd_matrix = np.array([
        [kabsch_rmsd(r, g) for g in gens]
        for r in refs
    ])

    min_rmsd_per_ref = rmsd_matrix.min(axis=1)   # (n_refs,)
    cov_r = float(np.mean(min_rmsd_per_ref < threshold))
    mat_r = float(np.mean(min_rmsd_per_ref))

    min_rmsd_per_gen = rmsd_matrix.min(axis=0)   # (n_gens,)
    cov_p = float(np.mean(min_rmsd_per_gen < threshold))
    mat_p = float(np.mean(min_rmsd_per_gen))

    return cov_r, mat_r, cov_p, mat_p


# ──────────────────────────────────────────────────────────────────────────────
# v2: Boltzmann-weighted COV-R
# ──────────────────────────────────────────────────────────────────────────────

def bw_cov_r(
    refs: List[np.ndarray],
    gens: List[np.ndarray],
    ref_energies: List[float],         # GFN2-xTB energies in kcal/mol per atom
    threshold: float = 0.5,
    kT: float = BOLTZMANN_KT_KCAL,
) -> float:
    """
    Boltzmann-Weighted Coverage Recall (v2 novel metric).

    Standard COV-R treats all reference conformers equally:
        COV-R = (1/M) * sum_i 1[min_j RMSD(ref_i, gen_j) < thr]

    Bw-COV-R weights each reference conformer by its Boltzmann probability:
        w_i = exp(-E_i / kT)
        Bw-COV-R = sum_i w_i * 1[covered_i] / sum_i w_i

    Scientific motivation: the thermodynamically dominant conformer (lowest E)
    contributes the most weight. Nature does not treat all conformers equally —
    the Boltzmann distribution governs which conformers actually exist in solution.

    A model that only generates the lowest-energy conformer gets high Bw-COV-R
    even if it misses rare high-energy conformers. This is scientifically correct.

    Args:
        ref_energies: GFN2-xTB energies (kcal/mol) for each reference conformer.
                      Will be shifted to zero minimum before Boltzmann weighting.
        kT:           Thermal energy in kcal/mol (default: 0.592 = kT at 298K).
    """
    if not refs or not gens or len(refs) != len(ref_energies):
        return float('nan')

    # Shift energies so minimum is 0 (relative energies for Boltzmann weights)
    energies = np.array(ref_energies)
    delta_e   = energies - energies.min()   # DeltaE from lowest conformer
    weights   = np.exp(-delta_e / kT)
    weights   = weights / weights.sum()     # normalize to sum=1

    bw_cov = 0.0
    for i, (ref, w) in enumerate(zip(refs, weights)):
        # Is this reference conformer covered by any generated conformer?
        covered = any(kabsch_rmsd(ref, g) < threshold for g in gens)
        if covered:
            bw_cov += float(w)

    return bw_cov


# ──────────────────────────────────────────────────────────────────────────────
# v2: Mean Energy Error
# ──────────────────────────────────────────────────────────────────────────────

def mean_energy_error(
    gens_tensors: List[torch.Tensor],  # list of (N, 3) generated coordinate tensors
    atom_types: torch.Tensor,
    edge_index: torch.Tensor,
    bond_types: torch.Tensor,
    batch_idx: torch.Tensor,
    energy_surrogate,
    ref_min_energy: float,             # min reference energy (normalized)
    device,
) -> float:
    """
    Mean Energy Error (v2 novel metric).

    MEE = mean_g[ E_surrogate(x_g) ] - E_min_ref

    Measures: on average, how much higher in energy are generated conformers
    compared to the lowest-energy reference conformer?

    If MEE <= 0: model generates lower-energy structures than reference.
    If MEE > 0:  model generates higher-energy (less stable) structures.

    Lower MEE (close to 0 or negative) = model successfully generates
    thermodynamically stable structures.
    """
    if not gens_tensors or energy_surrogate is None:
        return float('nan')

    energy_surrogate.eval()
    energies = []
    with torch.no_grad():
        for x in gens_tensors:
            try:
                x_dev = x.to(device) if not x.is_cuda else x
                e = energy_surrogate(x_dev, atom_types, edge_index, bond_types, batch_idx)
                energies.append(e.mean().item())
            except Exception:
                pass

    if not energies:
        return float('nan')
    return float(np.mean(energies)) - ref_min_energy


# ──────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ──────────────────────────────────────────────────────────────────────────────

def generate_conformers(
        model,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        bond_types: torch.Tensor,
        batch_idx: torch.Tensor,
        n_gen: int = 2,
        num_steps: int = 50,
        energy_surrogate=None,
        guidance_scale: float = 0.0,
        guidance_power: float = 0.5,
        device=None,
) -> List[torch.Tensor]:
    """
    Generate n_gen conformers for a single molecule.

    Uses energy_guided_ddim_sample (v2 smooth schedule) if surrogate + guidance_scale > 0,
    otherwise falls back to standard ddim_sample.

    Returns list of (N, 3) tensors.
    """
    model.eval()
    generated = []
    with torch.no_grad():
        for _ in range(n_gen):
            try:
                if energy_surrogate is not None and guidance_scale > 0:
                    # v2: smooth power-law guidance gamma(t) = gamma_max * alpha_bar_t^p
                    x = model.energy_guided_ddim_sample(
                        atom_types, edge_index, bond_types, batch_idx,
                        energy_surrogate=energy_surrogate,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        guidance_power=guidance_power,
                    )
                else:
                    x = model.ddim_sample(
                        atom_types, edge_index, bond_types, batch_idx,
                        num_steps=num_steps,
                    )
                generated.append(x.cpu())
            except Exception:
                pass
    return generated


# ──────────────────────────────────────────────────────────────────────────────
# Main GEOM-Drugs Evaluation (v2)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_geom_drugs_eval(
        model,
        val_loader,
        device,
        energy_surrogate=None,
        guidance_scale: float = 0.0,
        guidance_power: float = 0.5,
        num_steps: int = 50,
        n_gen: int = 2,
        n_mols: int = 100,
        cov_threshold: float = 0.5,
        verbose: bool = True,
) -> Dict:
    """
    Run GEOM-Drugs COV-R/MAT-R/COV-P/MAT-P + v2 Bw-COV-R/MEE evaluation.

    For each molecule in the validation set (up to n_mols):
      1. Retrieve ALL ground-truth reference conformers for that molecule
         (multi-reference, matching GeoDiff/TorDiff's own COV-R/MAT-R
         protocol -- see FIX note below).
      2. Generate n_gen conformers using (optionally energy-guided) DDIM.
      3. Compute COV-R, MAT-R, COV-P, MAT-P per molecule.
      4. Compute Bw-COV-R (Boltzmann-weighted over the real multi-conformer
         reference set) and MEE (if surrogate available).

    Returns:
        dict with mean COV-R, MAT-R, COV-P, MAT-P, Bw-COV-R, MEE, and lists.

    FIX: this previously scored
    against a SINGLE reference conformer per molecule (whichever one the
    training-mode sampler happened to draw for that DataLoader batch),
    unlike GeoDiff's/TorDiff's own protocol of scoring against ALL surviving
    ground-truth conformers per molecule (see GeoDiff utils/evaluation/
    covmat.py and TorsionalDiffusion evaluate_confs.py). A single fixed
    target is structurally harder to "cover" than any-of-N real references,
    so this deflated COV-R (and affected MAT-R) independent of true
    generative quality, making the printed numbers non-comparable to the
    GeoDiff/TorDiff SOTA figures printed alongside them. Now iterates the
    underlying dataset directly by index (bypassing the DataLoader's
    single-conformer-per-item batching, which exists for training, not
    eval) and uses `get_reference_coords()`/`get_all_conformers()` to pull
    every QC-surviving conformer as the reference set. Falls back to the
    old single-reference behavior only if the dataset doesn't support
    multi-reference lookup (e.g. a legacy non-GeomDrugsDataset).
    """
    model.eval()
    if energy_surrogate is not None:
        energy_surrogate.eval()

    cov_r_list, mat_r_list = [], []
    cov_p_list, mat_p_list = [], []
    bw_cov_r_list = []
    mee_list = []
    rmsd_list = []
    n_done = 0
    t0 = time.time()

    # Resolve the underlying dataset + index mapping, unwrapping a Subset
    # (used by the legacy random-split path) if present.
    raw_dataset = val_loader.dataset
    if isinstance(raw_dataset, Subset):
        base_dataset = raw_dataset.dataset
        index_map = raw_dataset.indices
    else:
        base_dataset = raw_dataset
        index_map = range(len(raw_dataset))

    supports_multi_ref = hasattr(base_dataset, 'get_reference_coords')
    n_available = min(n_mols, len(index_map))

    for pos in range(n_available):
        real_idx = index_map[pos]

        item = base_dataset[real_idx]
        at_b = item['atom_types'].to(device)
        ei_b_local = item['edge_index'].to(device)
        bt_b = item['bond_types'].to(device)
        bi_b = torch.zeros(at_b.size(0), dtype=torch.long, device=device)

        if at_b.size(0) < 3:
            continue

        if supports_multi_ref:
            refs = [c.numpy() for c in base_dataset.get_reference_coords(real_idx)]
            ref_confs = base_dataset.get_all_conformers(real_idx)
            ref_energies_kcal = [
                float(c['energy_kcal'].item()) for c in ref_confs
                if 'energy_kcal' in c
            ]
        else:
            refs = [item['coordinates'].numpy()]
            ref_energies_kcal = (
                [float(item['energy_kcal'].item())] if 'energy_kcal' in item else []
            )

        if not refs:
            continue
        ref_np = refs[0]  # lowest-energy reference, for the scalar RMSD-mean stat

        # Reference energy for MEE: lowest (normalized) energy among references
        ref_e_norm = float(item['energy_norm'].item()) if 'energy_norm' in item else None

        # Generate n_gen conformers
        gens_tensors = generate_conformers(
            model, at_b, ei_b_local, bt_b, bi_b,
            n_gen=n_gen,
            num_steps=num_steps,
            energy_surrogate=energy_surrogate,
            guidance_scale=guidance_scale,
            guidance_power=guidance_power,
            device=device,
        )

        if not gens_tensors:
            n_done += 1
            continue

        gens = [g.numpy() for g in gens_tensors]

        # Standard COV-MAT (now genuinely multi-reference when supported)
        cr, mr, cp, mp = cov_mat(refs, gens, threshold=cov_threshold)
        cov_r_list.append(cr)
        mat_r_list.append(mr)
        cov_p_list.append(cp)
        mat_p_list.append(mp)
        rmsd_list.append(kabsch_rmsd(gens[0], ref_np))

        # Bw-COV-R: genuine Boltzmann-weighted coverage when energies for
        # every reference conformer are available; falls back to plain
        # COV-R (as before) when they aren't.
        if len(ref_energies_kcal) == len(refs) and len(refs) > 1:
            bw_cov_r_list.append(bw_cov_r(refs, gens, ref_energies_kcal, threshold=cov_threshold))
        else:
            bw_cov_r_list.append(cr)

        # MEE: mean energy error
        if energy_surrogate is not None and ref_e_norm is not None:
            mee = mean_energy_error(
                gens_tensors, at_b, ei_b_local, bt_b, bi_b,
                energy_surrogate, ref_e_norm, device
            )
            if not np.isnan(mee):
                mee_list.append(mee)

        n_done += 1

        if verbose and n_done % 20 == 0 and n_done > 0:
            elapsed = time.time() - t0
            mee_str = f"  MEE={np.mean(mee_list):.3f}" if mee_list else ""
            print(f"  GEOM Eval [{n_done}/{n_mols}] {elapsed:.0f}s  "
                  f"MAT-R={np.mean(mat_r_list):.4f}A  "
                  f"COV-R={np.mean(cov_r_list)*100:.1f}%{mee_str}", flush=True)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else float('nan')

    return {
        'cov_r_05':    safe_mean(cov_r_list),
        'mat_r_mean':  safe_mean(mat_r_list),
        'cov_p_05':    safe_mean(cov_p_list),
        'mat_p_mean':  safe_mean(mat_p_list),
        'bw_cov_r':    safe_mean(bw_cov_r_list),   # v2 novel metric
        'mee':         safe_mean(mee_list),          # v2 novel metric
        'rmsd_mean':   safe_mean(rmsd_list),
        'n_evaluated': n_done,
        'cov_r_list':  cov_r_list,
        'mat_r_list':  mat_r_list,
        'mee_list':    mee_list,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Results printer (v2 — includes novel metrics)
# ──────────────────────────────────────────────────────────────────────────────

def print_geom_results(results: Dict, tag: str = ""):
    hdr = f"── GEOM-Drugs Eval{' '+tag if tag else ''} " + "─" * 40
    print(f"\n{hdr}")
    print(f"  n_evaluated  : {results.get('n_evaluated', '?')}")
    print()
    print("  Standard Metrics (GeoDiff/TorDiff protocol):")
    print(f"  COV-R@0.5A   : {results.get('cov_r_05', float('nan'))*100:.1f}%"
          f"  [published @0.75A: GeoDiff 42.1%, TorDiff 72.7%]")
    print(f"  MAT-R        : {results.get('mat_r_mean', float('nan')):.4f} A"
          f"  [published: GeoDiff 0.835, TorDiff 0.582]")
    print(f"  COV-P@0.5A   : {results.get('cov_p_05', float('nan'))*100:.1f}%"
          f"  [published @0.75A: GeoDiff 24.9%, TorDiff 55.2%]")
    print(f"  MAT-P        : {results.get('mat_p_mean', float('nan')):.4f} A"
          f"  [published: GeoDiff 1.136, TorDiff 0.778]")
    print()
    print("  v2 Novel Energy-Aware Metrics:")
    bw = results.get('bw_cov_r', float('nan'))
    mee = results.get('mee', float('nan'))
    print(f"  Bw-COV-R     : {bw*100:.1f}%"
          f"  [Boltzmann-weighted; thermodynamically dominant conformers matter more]")
    mee_str = f"{mee:+.3f}" if not np.isnan(mee) else "N/A"
    print(f"  MEE          : {mee_str} (norm)"
          f"  [<0 = generated more stable than ref; >0 = less stable]")
    print(f"  RMSD mean    : {results.get('rmsd_mean', float('nan')):.4f} A")
    print()
