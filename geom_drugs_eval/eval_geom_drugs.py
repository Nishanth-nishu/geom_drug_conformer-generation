"""
eval_geom_drugs.py — Evaluate mol_next_gen on GEOM-Drugs dataset

GeoDiff Methodology (ICML 2022):
  - test_data_1k.pkl has already-packed molecules (1 item = 1 mol, pos_ref = all conformers stacked)
  - For each mol with M reference conformers → generate 2*M conformers
  - Metrics: COV-R, MAT-R, COV-P, MAT-P at δ=1.25 Å (standard GEOM-Drugs threshold)
  - Also reports: Diversity (mean pairwise RMSD), MAT-R binned by rotatable bonds

GEOM-Drugs SOTA reference (GeoDiff Table 2):
  GeoDiff (ICML 2022): COV-R=88.5%, MAT-R=0.88 Å
  GeoMol  (NeurIPS 2021): COV-R=84.7%, MAT-R=0.97 Å
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# Parent dir for model + eval utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoresearch.geodiff_eval import kabsch_align, covmat_single_molecule
from models.conformer_diffusion import ConformerDiffusion


# ─────────────────────────── helpers ────────────────────────────────────────

def get_field(obj, key):
    """Safe field access for both PyG Data objects (old/new) and plain dicts."""
    if isinstance(obj, dict):
        return obj[key]
    # Bypass broken __getattr__ in old PyG by going directly to __dict__
    raw = object.__getattribute__(obj, '__dict__')
    # New PyG stores fields inside a nested '_store' dict
    if '_store' in raw:
        store = raw['_store']
        if key in store:
            return store[key]
        raise KeyError(key)
    # Old PyG stores directly in __dict__
    if key in raw:
        return raw[key]
    raise KeyError(key)


def has_field(obj, key):
    """Check if a field exists without triggering PyG __getattr__."""
    try:
        get_field(obj, key)
        return True
    except (KeyError, AttributeError):
        return False


def count_rotatable_bonds(anums, esrc, edst, ebo):
    try:
        from rdkit import Chem
        em = Chem.RWMol()
        for z in anums:
            em.AddAtom(Chem.Atom(int(z)))
        BT = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC,
        }
        seen = set()
        for i, j, bo in zip(esrc, edst, ebo):
            k = (min(i, j), max(i, j))
            if k not in seen:
                seen.add(k)
                em.AddBond(i, j, BT.get(int(bo), Chem.rdchem.BondType.SINGLE))
        mol = em.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    except Exception:
        return 0


def pack_dataset(raw_list):
    """
    The GeoDiff test_data_1k.pkl stores one item per conformer.
    We pack conformers of the same molecule together, replicating
    PackedConformationDataset._pack_data_by_mol logic.
    """
    by_smiles = defaultdict(list)
    for item in raw_list:
        try:
            smiles = get_field(item, 'smiles')
        except (KeyError, AttributeError):
            continue
        by_smiles[smiles].append(item)

    packed = []
    for smiles, confs in by_smiles.items():
        first = confs[0]
        try:
            atom_type  = get_field(first, 'atom_type')
            edge_index = get_field(first, 'edge_index')
            edge_type  = get_field(first, 'edge_type')
        except (KeyError, AttributeError):
            continue

        all_pos = []
        for c in confs:
            try:
                all_pos.append(get_field(c, 'pos'))
            except (KeyError, AttributeError):
                continue
        if not all_pos:
            continue

        packed.append({
            'smiles':     smiles,
            'atom_type':  atom_type,
            'edge_index': edge_index,
            'edge_type':  edge_type,
            'pos_ref':    torch.cat(all_pos, dim=0),
            'num_confs':  len(all_pos),
        })
    return packed


# ─────────────────────────── main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEOM-Drugs COV-MAT Evaluation (GeoDiff methodology)")
    parser.add_argument('--ckpt',     type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--test-set', type=str, required=True, help='Path to test_data_1k.pkl')
    parser.add_argument('--device',   type=str, default='cuda')
    parser.add_argument('--n-mols',   type=int, default=200,  help='Number of molecules to evaluate')
    parser.add_argument('--n-steps',  type=int, default=100,  help='DDIM steps per conformer')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Load + pack dataset ───────────────────────────────────────────────
    print(f"\nLoading test set: {args.test_set}")
    with open(args.test_set, 'rb') as f:
        raw = pickle.load(f)
    print(f"  Raw items loaded: {len(raw)}")

    # Check if already packed (pos_ref exists on first item)
    if has_field(raw[0], 'pos_ref'):
        dataset = []
        for item in raw:
            try:
                nr = get_field(item, 'num_pos_ref')
                nc = int(nr.item()) if hasattr(nr, 'item') else int(nr)
                dataset.append({
                    'smiles':    get_field(item, 'smiles'),
                    'atom_type': get_field(item, 'atom_type'),
                    'edge_index':get_field(item, 'edge_index'),
                    'edge_type': get_field(item, 'edge_type'),
                    'pos_ref':   get_field(item, 'pos_ref'),
                    'num_confs': nc,
                })
            except Exception:
                continue
        print(f"  Already packed → {len(dataset)} molecules")
    else:
        print("  Packing conformers by molecule...")
        dataset = pack_dataset(raw)
        print(f"  Packed → {len(dataset)} unique molecules")

    dataset = dataset[:args.n_mols]
    print(f"  Evaluating on: {len(dataset)} molecules\n")

    # ── 2. Load model ────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    cfg  = ckpt.get('config', {})
    epoch = ckpt.get('epoch', '?')
    saved_mat = ckpt.get('mat_r', 'N/A')
    print(f"  Epoch={epoch}  |  Saved MAT-R (QM9 val)={saved_mat}")
    print(f"  Config: {cfg}")

    model = ConformerDiffusion(
        num_timesteps = cfg.get('timesteps',       1000),
        hidden_dim    = cfg.get('hidden_dim',       256),
        num_layers    = cfg.get('num_layers',         6),
        time_dim      = cfg.get('time_dim',          128),
        num_rbf       = cfg.get('num_rbf',            20),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params/1e6:.2f}M\n")

    # ── 3. Evaluation loop ───────────────────────────────────────────────────
    thresholds  = np.arange(0.05, 3.05, 0.05)
    # Standard δ for GEOM-Drugs = 1.25 Å  (GeoDiff Table 2)
    thr_idx     = int(round((1.25 - 0.05) / 0.05))  # index 24

    all_cov_r, all_cov_p = [], []
    all_mat_r, all_mat_p = [], []
    all_div, all_N, all_rot = [], [], []

    print(f"Running GEOM-Drugs eval [{len(dataset)} mols × 2× conformers, δ=1.25 Å] ...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset)):
            try:
                atom_types  = data['atom_type'].to(device)
                edge_index  = data['edge_index'].to(device)
                bond_types  = data['edge_type'].to(device)
                pos_ref_t   = data['pos_ref']         # CPU tensor
                N           = atom_types.shape[0]
                M           = data['num_confs']
                num_samples = max(2 * M, 4)           # at least 4 generated

                batch_idx = torch.zeros(N, dtype=torch.long, device=device)

                # Reference conformers (centered)
                pos_ref_np = pos_ref_t.numpy()
                ref_confs  = []
                for m in range(M):
                    p = pos_ref_np[m*N:(m+1)*N]
                    ref_confs.append(p - p.mean(0))

                # Generated conformers
                gen_confs = []
                for _ in range(num_samples):
                    try:
                        g = model.ddim_sample(
                            atom_types, edge_index, bond_types,
                            batch_idx, num_steps=args.n_steps
                        )
                        g_np = g.cpu().numpy()
                        gen_confs.append(g_np - g_np.mean(0))
                    except Exception:
                        continue

                if len(gen_confs) == 0:
                    continue

                # COV-MAT metrics
                cov_r, mat_r, cov_p, mat_p = covmat_single_molecule(
                    ref_confs, gen_confs, thresholds
                )

                # Diversity: mean pairwise RMSD between generated conformers
                pairs = [(gen_confs[a], gen_confs[b])
                         for a in range(len(gen_confs))
                         for b in range(a+1, len(gen_confs))]
                div = float(np.mean([kabsch_align(a, b) for a, b in pairs])) if pairs else 0.0

                # Rotatable bonds for granular analysis
                rot = count_rotatable_bonds(
                    atom_types.cpu().tolist(),
                    edge_index[0].cpu().tolist(),
                    edge_index[1].cpu().tolist(),
                    bond_types.cpu().tolist(),
                )

                all_cov_r.append(cov_r);  all_cov_p.append(cov_p)
                all_mat_r.append(mat_r);  all_mat_p.append(mat_p)
                all_div.append(div);      all_N.append(N);  all_rot.append(rot)

            except Exception as e:
                tqdm.write(f"[skip mol {i}] {e}")
                continue

    if not all_mat_r:
        print("\n[ERROR] 0 molecules were successfully evaluated.")
        return

    # ── 4. Results ───────────────────────────────────────────────────────────
    cov_r_arr = np.stack(all_cov_r)   # (n_mols, n_thresholds)
    cov_p_arr = np.stack(all_cov_p)
    mat_r_arr = np.array(all_mat_r)
    mat_p_arr = np.array(all_mat_p)

    print(f"\n{'─'*70}")
    print(f"  GEOM-Drugs Results  |  checkpoint: {os.path.basename(args.ckpt)}")
    print(f"  Threshold δ = 1.25 Å  (standard for GEOM-Drugs)")
    print(f"{'─'*70}")
    print(f"  Molecules evaluated : {len(all_mat_r)}")
    print(f"  COV-R               : {cov_r_arr[:, thr_idx].mean()*100:6.2f}%")
    print(f"  COV-P               : {cov_p_arr[:, thr_idx].mean()*100:6.2f}%")
    print(f"  MAT-R               : {mat_r_arr.mean():.4f} Å")
    print(f"  MAT-P               : {mat_p_arr.mean():.4f} Å")
    print(f"  Diversity           : {np.mean(all_div):.4f} Å  (mean pairwise RMSD)")
    print(f"{'─'*70}")

    print(f"\n  MAT-R by Rotatable Bonds:")
    bins = {"rot≤2": [], "rot 3-5": [], "rot≥6": []}
    for mat, r in zip(all_mat_r, all_rot):
        if   r <= 2: bins["rot≤2"].append(mat)
        elif r <= 5: bins["rot 3-5"].append(mat)
        else:        bins["rot≥6"].append(mat)
    for label, vals in bins.items():
        s = f"{np.mean(vals):.4f} Å  (n={len(vals)})" if vals else "no data"
        print(f"    {label:10s}: {s}")

    print(f"\n  MAT-R by Atom Count:")
    size_bins = {"N≤20": [], "N 21-40": [], "N≥41": []}
    for mat, n in zip(all_mat_r, all_N):
        if   n <= 20: size_bins["N≤20"].append(mat)
        elif n <= 40: size_bins["N 21-40"].append(mat)
        else:         size_bins["N≥41"].append(mat)
    for label, vals in size_bins.items():
        s = f"{np.mean(vals):.4f} Å  (n={len(vals)})" if vals else "no data"
        print(f"    {label:10s}: {s}")

    print(f"\n{'─'*70}")
    print(f"  SOTA Reference (GEOM-Drugs, 2x conformers, δ=1.25 Å):")
    print(f"    GeoDiff  (ICML 2022)  : COV-R=88.5%  MAT-R=0.88 Å")
    print(f"    GeoMol   (NeurIPS 21) : COV-R=84.7%  MAT-R=0.97 Å")
    print(f"    TorDiff  (NeurIPS 22) : COV-R=92.5%  MAT-R=0.59 Å")
    print(f"{'─'*70}\n")

    # ── 5. Save TSV ──────────────────────────────────────────────────────────
    out_tsv = os.path.join(os.path.dirname(args.ckpt), 'results_geom_drugs.tsv')
    with open(out_tsv, 'w') as f:
        f.write('\t'.join(['ckpt','n_mols','epoch','cov_r','cov_p','mat_r','mat_p','diversity']) + '\n')
        f.write('\t'.join([
            os.path.basename(args.ckpt),
            str(len(all_mat_r)),
            str(epoch),
            f"{cov_r_arr[:, thr_idx].mean()*100:.2f}",
            f"{cov_p_arr[:, thr_idx].mean()*100:.2f}",
            f"{mat_r_arr.mean():.4f}",
            f"{mat_p_arr.mean():.4f}",
            f"{np.mean(all_div):.4f}",
        ]) + '\n')
    print(f"  Results saved → {out_tsv}\n")


if __name__ == '__main__':
    main()
