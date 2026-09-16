#!/usr/bin/env python3
"""
eval_size_generalization.py — MAT-R / COV-R as a function of molecule size.

WHY THIS SCRIPT EXISTS
-----------------------
Every eval script in this repo (geodiff_eval.py, geom_drugs_eval.py) reports a single
aggregate MAT-R/COV-R number over whatever molecule-size range its input dataset
happens to contain. That hides the one question this project's whole QM9 -> GEOM-Drugs
story turns on: does quality degrade gracefully with molecule size, or does it fall off
a cliff past whatever size the model was trained on?

This script buckets a GEOM-Drugs JSONL file (data/geom_drugs.jsonl, built by
data/prepare_geom_drugs.py) by heavy/total atom count, runs the SAME checkpoint's DDIM
sampler against every bucket, and reports MAT-R/COV-R per bucket. Point it at:
  - a QM9-only checkpoint (ConformerDiffusion / AttnConformerDiffusion, trained with
    max_atoms=9) to produce the "trained small, degrades how fast on bigger molecules"
    curve referenced in docs/GEOM_DRUGS_DIAGNOSIS_AND_PLAN.md
  - a GEOM-Drugs checkpoint (DualEncoderDiffusion, mol_train_v2.py) trained at some
    max_atoms cutoff (30/50/100 per the GEOM_SMALL/MEDIUM/FULL presets in
    data/geom_drugs_dataset.py) to see how well it extrapolates PAST that cutoff.

NOT YET RUN AGAINST REAL DATA. This was written by reading the existing eval code
(autoresearch/geom_drugs_eval.py, autoresearch/geodiff_eval.py, data/geom_drugs_dataset.py)
and mirroring their conventions, not by executing it — there was no GPU/dataset access
available at write time. Sanity-check it on a small --n-per-bin (e.g. 5) and --max-mols-scan
before trusting a full run. Please report back anything that breaks.

Usage:
  PYTHONPATH=. python3 autoresearch/eval_size_generalization.py \\
      --ckpt checkpoints/exp_G_heavy_atom_sota_ddp_best_matr.pt \\
      --model-type attn \\
      --data data/geom_drugs.jsonl \\
      --bins 9,20,30,50,70,100 \\
      --n-per-bin 30 --n-gen 10 --num-steps 50 \\
      --out-csv results_size_generalization.csv
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoresearch.geodiff_eval import kabsch_align, covmat_single_molecule


# ── Model construction from checkpoint config ─────────────────────────────────

def build_model(model_type: str, cfg: dict, device):
    """Reconstruct the right model class from a checkpoint's saved config dict."""
    if model_type == 'conformer':
        from models.conformer_diffusion import ConformerDiffusion
        model = ConformerDiffusion(
            num_timesteps=cfg.get('timesteps', cfg.get('num_timesteps', 1000)),
            hidden_dim=cfg.get('hidden_dim', 256),
            num_layers=cfg.get('num_layers', 6),
            num_rbf=cfg.get('num_rbf', 20),
            time_dim=cfg.get('time_dim', 128),
        )
    elif model_type == 'attn':
        from models.attn_conformer_diffusion import AttnConformerDiffusion
        model = AttnConformerDiffusion(
            num_timesteps=cfg.get('timesteps', cfg.get('num_timesteps', 1000)),
            hidden_dim=cfg.get('hidden_dim', 256),
            num_layers=cfg.get('num_layers', 6),
            num_rbf=cfg.get('num_rbf', 20),
            time_dim=cfg.get('time_dim', 128),
            num_heads=cfg.get('num_heads', 4),
        )
    elif model_type == 'dual':
        from models.dual_encoder_diffusion import DualEncoderDiffusion
        model = DualEncoderDiffusion(
            hidden_dim=cfg.get('hidden_dim', 256),
            num_convs_global=cfg.get('num_convs_global', 6),
            num_convs_local=cfg.get('num_convs_local', 4),
            num_gaussians=cfg.get('num_gaussians', 50),
            num_attn_heads=cfg.get('num_attn_heads', 8),
            num_bond_types=cfg.get('num_bond_types', 5),
            edge_order=cfg.get('edge_order', 3),
            cutoff=cfg.get('cutoff', 10.0),
            num_timesteps=cfg.get('num_timesteps', 2000),
            beta_start=cfg.get('beta_start', 1e-7),
            beta_end=cfg.get('beta_end', 2e-3),
        )
    else:
        raise ValueError(f"Unknown --model-type {model_type!r} (expected conformer|attn|dual)")
    return model.to(device).eval()


def load_checkpoint(ckpt_path: str, model_type: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = build_model(model_type, cfg, device)
    state_key = 'model' if 'model' in ckpt else 'model_state_dict'
    model.load_state_dict(ckpt[state_key])
    print(f"Loaded {ckpt_path}  (model_type={model_type})")
    print(f"  epoch={ckpt.get('epoch', '?')}  "
          f"saved_mat_r={ckpt.get('best_mat_r', ckpt.get('mat_r', 'N/A'))}")
    print(f"  config={cfg}")
    return model


# ── Minimal standalone JSONL loader (mirrors data/geom_drugs_dataset.py schema) ─

def stream_molecules_by_bin(data_path: str, bin_edges, n_per_bin: int,
                             max_conformers: int, max_mols_scan: int):
    """
    One pass over data/geom_drugs.jsonl, bucketing molecules by num_atoms into
    bins defined by bin_edges (upper-inclusive), stopping each bucket once it has
    n_per_bin molecules. Returns dict: bin_label -> list of molecule dicts, each
    with atom_types/edge_index/bond_types (torch tensors) + up to max_conformers
    reference conformer coordinate arrays.

    Standalone (doesn't import GeomDrugsDataset) so this script has no dependency
    on that class's private fields and can be run/tested independently.
    """
    edges = sorted(bin_edges)
    labels = []
    lo = 0
    for hi in edges:
        labels.append((lo + 1, hi))
        lo = hi

    buckets = {lab: [] for lab in labels}

    def bucket_for(n_atoms):
        for lab in labels:
            if lab[0] <= n_atoms <= lab[1]:
                return lab
        return None

    n_scanned = 0
    with open(data_path, 'r') as f:
        for line in f:
            if max_mols_scan > 0 and n_scanned >= max_mols_scan:
                break
            n_scanned += 1
            if all(len(v) >= n_per_bin for v in buckets.values()):
                break
            line = line.strip()
            if not line:
                continue
            try:
                mol = json.loads(line)
            except Exception:
                continue

            na = mol.get('num_atoms', 0)
            lab = bucket_for(na)
            if lab is None or len(buckets[lab]) >= n_per_bin:
                continue
            confs = mol.get('conformers', [])
            if len(confs) < 1:
                continue

            confs_sorted = sorted(confs, key=lambda c: c.get('boltzmann_weight', 0.0),
                                   reverse=True)[:max_conformers]
            ref_coords = [np.asarray(c['coordinates'], dtype=np.float32) for c in confs_sorted]

            buckets[lab].append({
                'num_atoms': na,
                'atom_types': torch.tensor(mol['atom_types'], dtype=torch.long),
                'edge_index': torch.tensor(mol['edge_index'], dtype=torch.long),
                'bond_types': torch.tensor(mol['bond_types'], dtype=torch.long),
                'ref_coords': ref_coords,
            })

    print(f"[stream_molecules_by_bin] scanned {n_scanned:,} lines")
    for lab, mols in buckets.items():
        print(f"  bin N in [{lab[0]:>3},{lab[1]:>3}] : {len(mols)} molecules found")
    return buckets


# ── Generation + metrics for one molecule ──────────────────────────────────────

@torch.no_grad()
def generate_n_conformers(model, atom_types, edge_index, bond_types, n_gen, num_steps, device):
    N = atom_types.size(0)
    batch_idx = torch.zeros(N, dtype=torch.long, device=device)
    gens = []
    for _ in range(n_gen):
        try:
            x = model.ddim_sample(atom_types, edge_index, bond_types, batch_idx,
                                   num_steps=num_steps)
            x_np = x.detach().cpu().numpy()
            gens.append(x_np - x_np.mean(0))
        except Exception as e:
            print(f"    [gen failed] {e}")
    return gens


def eval_bucket(model, mols, gen_multiplier, min_gen, num_steps, thresholds, device):
    """FIX: generation count now scales
    per molecule as max(min_gen, gen_multiplier * len(refs)), matching
    GeoMol/TorDiff's "2x reference count" convention, instead of a single
    fixed --n-gen applied uniformly regardless of how many reference
    conformers a given molecule actually has."""
    mat_rs, cov_r_arrs, n_atoms_list = [], [], []
    for m in mols:
        atom_types = m['atom_types'].to(device)
        edge_index = m['edge_index'].to(device)
        bond_types = m['bond_types'].to(device)

        refs = [c - c.mean(0) for c in m['ref_coords']]
        n_gen_this_mol = max(min_gen, gen_multiplier * len(refs))
        gens = generate_n_conformers(model, atom_types, edge_index, bond_types,
                                      n_gen_this_mol, num_steps, device)
        if not gens:
            continue

        cov_r, mat_r, cov_p, mat_p = covmat_single_molecule(refs, gens, thresholds)
        mat_rs.append(mat_r)
        cov_r_arrs.append(cov_r)
        n_atoms_list.append(m['num_atoms'])

    if not mat_rs:
        return None

    return {
        'n_mols': len(mat_rs),
        'mean_atoms': float(np.mean(n_atoms_list)),
        'mat_r_mean': float(np.mean(mat_rs)),
        'mat_r_median': float(np.median(mat_rs)),
        'mat_r_p90': float(np.percentile(mat_rs, 90)),
        'cov_r_by_threshold': np.mean(np.stack(cov_r_arrs), axis=0),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model-type', required=True, choices=['conformer', 'attn', 'dual'])
    p.add_argument('--data', required=True, help='Path to geom_drugs.jsonl')
    p.add_argument('--bins', default='9,20,30,50,70,100',
                    help='Comma-separated upper edges of size bins, e.g. 9,20,30,50,70,100 '
                         '-> bins [1-9],[10-20],[21-30],[31-50],[51-70],[71-100]')
    p.add_argument('--n-per-bin', type=int, default=30,
                    help='Max molecules to evaluate per size bin')
    p.add_argument('--max-conformers', type=int, default=5,
                    help='Max reference conformers per molecule to use for MAT-R/COV-R')
    p.add_argument('--max-mols-scan', type=int, default=-1,
                    help='Stop scanning the JSONL after this many lines (-1 = scan until '
                         'all bins are full or EOF; use a small value for a quick smoke test)')
    p.add_argument('--gen-multiplier', type=int, default=2,
                    help='Generate gen_multiplier x (number of reference conformers) per '
                         'molecule, matching GeoMol/TorDiff\'s "2x reference count" convention '
                         '(replaces the old fixed --n-gen)')
    p.add_argument('--min-gen', type=int, default=4,
                    help='Floor on generated conformers per molecule, for molecules with very '
                         'few reference conformers')
    p.add_argument('--num-steps', type=int, default=50, help='DDIM steps')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out-csv', default='results_size_generalization.csv')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    bin_edges = [int(x) for x in args.bins.split(',')]
    thresholds = np.array([0.5, 1.0, 1.25, 2.0])  # QM9 std=0.5A, GEOM-Drugs std=1.25A

    model = load_checkpoint(args.ckpt, args.model_type, device)

    print(f"\nScanning {args.data} into {len(bin_edges)} size bins "
          f"(n_per_bin={args.n_per_bin}) ...")
    buckets = stream_molecules_by_bin(args.data, bin_edges, args.n_per_bin,
                                       args.max_conformers, args.max_mols_scan)

    rows = []
    print(f"\n{'='*78}")
    print(f"  SIZE-GENERALIZATION SWEEP  |  ckpt={os.path.basename(args.ckpt)}  "
          f"model_type={args.model_type}")
    print(f"  {args.gen_multiplier}x reference count (min {args.min_gen}) generated conformers/mol, "
          f"{args.num_steps} DDIM steps")
    print(f"{'='*78}")
    print(f"{'bin (N atoms)':<16}{'n_mols':>8}{'mean_N':>9}{'MAT-R mean':>13}"
          f"{'MAT-R p90':>12}{'COV-R@0.5':>12}{'COV-R@1.25':>12}")

    for lab, mols in buckets.items():
        label_str = f"[{lab[0]},{lab[1]}]"
        if not mols:
            print(f"{label_str:<16}{'--- no molecules in this bin ---':>60}")
            continue
        result = eval_bucket(model, mols, args.gen_multiplier, args.min_gen, args.num_steps,
                              thresholds, device)
        if result is None:
            print(f"{label_str:<16}{'--- all generations failed ---':>60}")
            continue

        cov_05 = result['cov_r_by_threshold'][int(np.where(thresholds == 0.5)[0][0])]
        cov_125 = result['cov_r_by_threshold'][int(np.where(thresholds == 1.25)[0][0])]

        print(f"{label_str:<16}{result['n_mols']:>8}{result['mean_atoms']:>9.1f}"
              f"{result['mat_r_mean']:>13.4f}{result['mat_r_p90']:>12.4f}"
              f"{cov_05*100:>11.1f}%{cov_125*100:>11.1f}%")

        rows.append({
            'bin_lo': lab[0], 'bin_hi': lab[1],
            'n_mols': result['n_mols'], 'mean_atoms': result['mean_atoms'],
            'mat_r_mean': result['mat_r_mean'], 'mat_r_median': result['mat_r_median'],
            'mat_r_p90': result['mat_r_p90'],
            'cov_r_at_0.5A': cov_05, 'cov_r_at_1.25A': cov_125,
        })

    print(f"{'='*78}\n")

    if rows:
        import csv
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved -> {args.out_csv}")
        print("\nIf MAT-R climbs sharply / COV-R collapses once bin_lo exceeds the "
              "checkpoint's training max_atoms, that's the size-generalization cliff "
              "referenced in docs/GEOM_DRUGS_DIAGNOSIS_AND_PLAN.md section 5 step 4 — "
              "plot this CSV (bin midpoint vs MAT-R) for the paper figure.")
    else:
        print("[WARNING] No bins produced results — check --data path and --bins range "
              "against the actual num_atoms distribution in your GEOM-Drugs JSONL.")


if __name__ == '__main__':
    main()
