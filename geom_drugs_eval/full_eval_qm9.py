"""
full_eval_qm9.py — Publication-grade QM9 evaluation with all metrics.

Metrics:
  - COV-R, MAT-R (Recall): coverage and matching from reference side
  - COV-P, MAT-P (Precision): coverage and matching from generated side
  - Diversity: mean pairwise RMSD among generated conformers per molecule
  - Binned MAT-R by atom count and rotatable bonds
  - Visual: best/median/worst case XYZ files for inspection

Usage:
  python geom_drugs_eval/full_eval_qm9.py \
      --ckpt checkpoints/exp_G_heavy_atom_sota_ddp_best_matr.pt \
      --data data/qm9_heavy.jsonl \
      --n-mols 200 --n-gen 10 --out-dir geom_drugs_eval/eval_outputs
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoresearch.geodiff_eval import kabsch_align, covmat_single_molecule
from models.conformer_diffusion import ConformerDiffusion

ELEMENT = {1:'H',6:'C',7:'N',8:'O',9:'F',16:'S',17:'Cl'}


def load_val_split(data_path, max_atoms=9):
    all_mols = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            if not item.get('coordinates'): continue
            if item.get('num_atoms', len(item['atom_types'])) > max_atoms: continue
            if any(z >= 54 or z <= 0 for z in item['atom_types']): continue
            all_mols.append(item)
    n_val = int(len(all_mols) * 0.1)
    n_train = len(all_mols) - n_val
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(all_mols), generator=gen).tolist()
    return [all_mols[i] for i in indices[n_train:]]


def count_rot_bonds(item):
    try:
        from rdkit import Chem
        n = len(item['atom_types'])
        em = Chem.RWMol()
        for z in item['atom_types']: em.AddAtom(Chem.Atom(int(z)))
        BT = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE,
              3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
        ei = item['edge_index']
        seen = set()
        for i, j, bo in zip(ei[0], ei[1], item['bond_types']):
            k = (min(i,j), max(i,j))
            if k not in seen:
                seen.add(k)
                em.AddBond(int(i), int(j), BT.get(int(bo), Chem.rdchem.BondType.SINGLE))
        mol = em.GetMol(); Chem.SanitizeMol(mol)
        return Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    except: return 0


def write_xyz(path, atom_types, coords, comment=""):
    """Write XYZ file for molecular visualization."""
    with open(path, 'w') as f:
        f.write(f"{len(atom_types)}\n{comment}\n")
        for z, (x, y, z_) in zip(atom_types, coords):
            sym = ELEMENT.get(z, f"X{z}")
            f.write(f"{sym:2s}  {x:10.4f}  {y:10.4f}  {z_:10.4f}\n")


def diversity_metric(conformers):
    """
    Sampling diversity = mean pairwise Kabsch-RMSD among generated conformers.
    
    Formally:
        Div = (1 / C(n,2)) * sum_{i<j} RMSD(C_i, C_j)
    where C_i are generated conformers and RMSD uses Kabsch alignment.
    
    High diversity (>0.2 A) means the model samples the conformer distribution.
    Low diversity (~0.0 A) means mode collapse.
    """
    n = len(conformers)
    if n < 2: return 0.0
    pairs = [(conformers[i], conformers[j]) for i in range(n) for j in range(i+1, n)]
    return float(np.mean([kabsch_align(a, b) for a, b in pairs]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',    required=True)
    parser.add_argument('--data',    required=True)
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--n-mols',  type=int, default=200)
    parser.add_argument('--n-gen',   type=int, default=10)
    parser.add_argument('--n-steps', type=int, default=50)
    parser.add_argument('--out-dir', default='geom_drugs_eval/eval_outputs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load val split ───────────────────────────────────────────────────────
    print(f"\nLoading val split from: {args.data}")
    val_mols = load_val_split(args.data)
    val_mols = val_mols[:args.n_mols]
    print(f"  Evaluating {len(val_mols)} molecules\n")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    cfg   = ckpt.get('config', {})
    epoch = ckpt.get('epoch', '?')
    print(f"  Epoch={epoch} | Best QM9 MAT-R={ckpt.get('mat_r','?'):.4f}")

    model = ConformerDiffusion(
        num_timesteps = cfg.get('timesteps', 1000),
        hidden_dim    = cfg.get('hidden_dim', 256),
        num_layers    = cfg.get('num_layers', 6),
        time_dim      = cfg.get('time_dim', 128),
        num_rbf       = cfg.get('num_rbf', 20),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")

    # ── Evaluation ───────────────────────────────────────────────────────────
    thresholds = np.arange(0.05, 3.05, 0.05)
    thr_idx    = int(round((0.5 - 0.05) / 0.05))   # δ=0.5 Å for QM9

    rows = []   # per-molecule results for analysis

    print(f"Running evaluation: {len(val_mols)} mols × {args.n_gen} conformers ...")
    with torch.no_grad():
        for item in tqdm(val_mols, desc="Eval"):
            try:
                atom_types = torch.tensor(item['atom_types'], dtype=torch.long, device=device)
                ei = np.array(item['edge_index'])
                if ei.ndim == 2 and ei.shape[1] == 2: ei = ei.T
                edge_index = torch.tensor(ei, dtype=torch.long, device=device)
                if edge_index.shape[0] != 2: edge_index = edge_index.T
                bond_types = torch.tensor(item['bond_types'], dtype=torch.long, device=device)
                N = atom_types.shape[0]
                batch_idx = torch.zeros(N, dtype=torch.long, device=device)

                ref_pos  = np.array(item['coordinates'], dtype=np.float32)
                ref_cent = ref_pos - ref_pos.mean(0)

                # Generate conformers from noise
                gen_confs = []
                for _ in range(args.n_gen):
                    try:
                        g    = model.ddim_sample(atom_types, edge_index, bond_types, batch_idx, num_steps=args.n_steps)
                        g_np = g.cpu().numpy()
                        gen_confs.append(g_np - g_np.mean(0))
                    except Exception: continue

                if not gen_confs: continue

                # COV-R, MAT-R, COV-P, MAT-P
                cov_r, mat_r, cov_p, mat_p = covmat_single_molecule([ref_cent], gen_confs, thresholds)

                # Diversity: mean pairwise RMSD across generated conformers
                div = diversity_metric(gen_confs)

                # Size / flexibility properties
                n_atoms   = N
                n_rot     = count_rot_bonds(item)

                # Best generated conformer (lowest RMSD to ref)
                rmsds_to_ref = [kabsch_align(ref_cent, g) for g in gen_confs]
                best_idx  = int(np.argmin(rmsds_to_ref))
                best_rmsd = rmsds_to_ref[best_idx]

                rows.append({
                    'atom_types':  item['atom_types'],
                    'ref_coords':  ref_cent,
                    'gen_confs':   gen_confs,
                    'best_gen':    gen_confs[best_idx],
                    'mat_r':       mat_r,
                    'mat_p':       mat_p,
                    'cov_r_05':    float(cov_r[thr_idx]),
                    'cov_p_05':    float(cov_p[thr_idx]),
                    'diversity':   div,
                    'n_atoms':     n_atoms,
                    'n_rot':       n_rot,
                    'best_rmsd':   best_rmsd,
                })

            except Exception as e:
                tqdm.write(f"[skip] {e}")
                continue

    if not rows:
        print("No molecules evaluated successfully.")
        return

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    mat_r_all  = np.array([r['mat_r']     for r in rows])
    mat_p_all  = np.array([r['mat_p']     for r in rows])
    cov_r_all  = np.array([r['cov_r_05']  for r in rows])
    cov_p_all  = np.array([r['cov_p_05']  for r in rows])
    div_all    = np.array([r['diversity']  for r in rows])
    n_atoms_all= np.array([r['n_atoms']   for r in rows])
    n_rot_all  = np.array([r['n_rot']     for r in rows])

    sep = "─" * 66
    print(f"\n{sep}")
    print(f"  PUBLICATION-GRADE EVALUATION RESULTS  |  {os.path.basename(args.ckpt)}")
    print(f"  Dataset: QM9 heavy-atom val split  |  Epoch {epoch}")
    print(f"{sep}")
    print(f"  Molecules evaluated : {len(rows)}")
    print(f"  Generated per mol   : {args.n_gen}  (δ = 0.5 Å for QM9)")
    print(f"{sep}")
    print(f"  RECALL METRICS  (reference → generated)")
    print(f"    COV-R@0.5Å    : {cov_r_all.mean()*100:6.2f}%   ± {cov_r_all.std()*100:.2f}")
    print(f"    MAT-R  mean   : {mat_r_all.mean():7.4f} Å  ± {mat_r_all.std():.4f}")
    print(f"    MAT-R  median : {np.median(mat_r_all):7.4f} Å")
    print(f"    MAT-R  p90    : {np.percentile(mat_r_all, 90):7.4f} Å  (worst 10%)")
    print(f"{sep}")
    print(f"  PRECISION METRICS  (generated → reference)")
    print(f"    COV-P@0.5Å    : {cov_p_all.mean()*100:6.2f}%   ± {cov_p_all.std()*100:.2f}")
    print(f"    MAT-P  mean   : {mat_p_all.mean():7.4f} Å  ± {mat_p_all.std():.4f}")
    print(f"    MAT-P  median : {np.median(mat_p_all):7.4f} Å")
    print(f"  (High COV-P means generated conformers are all realistic,")
    print(f"   not spread randomly to 'accidentally' cover reference)")
    print(f"{sep}")
    print(f"  DIVERSITY  (mean pairwise Kabsch-RMSD among generated)")
    print(f"  Formula: (1/C(n,2)) * sum_{{i<j}} RMSD(C_i, C_j)")
    print(f"    Diversity mean  : {div_all.mean():7.4f} Å")
    print(f"    Diversity median: {np.median(div_all):7.4f} Å")
    print(f"  (>0.10 Å = model samples diverse conformers; ~0.0 = mode collapse)")
    print(f"{sep}")

    # ── Binned analysis ───────────────────────────────────────────────────────
    print(f"\n  MAT-R by Atom Count (N):")
    for lo, hi, label in [(1,5,'N≤5'), (6,7,'N=6-7'), (8,8,'N=8'), (9,99,'N≥9')]:
        mask = (n_atoms_all >= lo) & (n_atoms_all <= hi)
        vals = mat_r_all[mask]
        s = f"{vals.mean():.4f} Å  ± {vals.std():.4f}  (n={len(vals)})" if len(vals) else "no data"
        print(f"    {label:8s}: {s}")

    print(f"\n  MAT-R by Rotatable Bonds:")
    for lo, hi, label in [(0,0,'rot=0'), (1,1,'rot=1'), (2,2,'rot=2'), (3,99,'rot≥3')]:
        mask = (n_rot_all >= lo) & (n_rot_all <= hi)
        vals = mat_r_all[mask]
        s = f"{vals.mean():.4f} Å  ± {vals.std():.4f}  (n={len(vals)})" if len(vals) else "no data"
        print(f"    {label:8s}: {s}")

    print(f"\n{sep}")
    print(f"  SOTA Reference (QM9 heavy-atom, δ=0.5 Å, 2× conformers):")
    print(f"    GeoDiff (ICML 2022) : COV-R=71.0%  MAT-R=0.297 Å")
    print(f"    GeoMol  (NeurIPS 21): COV-R=71.5%  MAT-R=0.225 Å")
    print(f"    TorDiff (NeurIPS 22): COV-R=73.2%  MAT-R=0.219 Å")
    print(f"{sep}\n")

    # ── Visual examples ───────────────────────────────────────────────────────
    sorted_rows = sorted(rows, key=lambda r: r['mat_r'])
    n = len(sorted_rows)
    cases = {
        'best':   sorted_rows[0],
        'median': sorted_rows[n // 2],
        'worst':  sorted_rows[-1],
    }
    print(f"  VISUAL EXAMPLES  (XYZ files → {args.out_dir}/)")
    print(f"  {'Case':8s}  {'MAT-R':>8s}  {'Diversity':>10s}  {'N atoms':>8s}  {'Rot bonds':>9s}")
    print(f"  {'-'*52}")
    for case, r in cases.items():
        print(f"  {case:8s}  {r['mat_r']:8.4f}  {r['diversity']:10.4f}  {r['n_atoms']:8d}  {r['n_rot']:9d}")
        prefix = os.path.join(args.out_dir, f"case_{case}")
        write_xyz(f"{prefix}_reference.xyz",  r['atom_types'], r['ref_coords'],
                  comment=f"Reference  MAT-R={r['mat_r']:.4f}")
        write_xyz(f"{prefix}_best_gen.xyz",   r['atom_types'], r['best_gen'],
                  comment=f"BestGen RMSD={r['best_rmsd']:.4f}")
        # Write all generated conformers
        for gi, gc in enumerate(r['gen_confs']):
            write_xyz(f"{prefix}_gen_{gi:02d}.xyz", r['atom_types'], gc,
                      comment=f"Gen{gi} RMSD={kabsch_align(r['ref_coords'], gc):.4f}")
    print(f"\n  Open XYZ files with VESTA, PyMol, or VMD to visualize.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style='whitegrid', font_scale=1.2)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'QM9 Evaluation — {os.path.basename(args.ckpt)}\n'
                     f'Epoch {epoch} | {len(rows)} molecules | {args.n_gen} generated each',
                     fontsize=13, fontweight='bold')

        # 1. MAT-R distribution
        ax = axes[0, 0]
        ax.hist(mat_r_all, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(mat_r_all.mean(), color='red',    lw=2, ls='--', label=f'Mean {mat_r_all.mean():.3f}Å')
        ax.axvline(np.median(mat_r_all), color='orange', lw=2, ls='--', label=f'Median {np.median(mat_r_all):.3f}Å')
        ax.axvline(0.297, color='gray', lw=1.5, ls=':', label='GeoDiff 0.297Å')
        ax.set_xlabel('MAT-R (Å)'); ax.set_ylabel('Count')
        ax.set_title('MAT-R Distribution'); ax.legend(fontsize=9)

        # 2. MAT-P distribution
        ax = axes[0, 1]
        ax.hist(mat_p_all, bins=30, color='coral', edgecolor='white', alpha=0.85)
        ax.axvline(mat_p_all.mean(), color='red', lw=2, ls='--', label=f'Mean {mat_p_all.mean():.3f}Å')
        ax.axvline(np.median(mat_p_all), color='orange', lw=2, ls='--', label=f'Median {np.median(mat_p_all):.3f}Å')
        ax.set_xlabel('MAT-P (Å)'); ax.set_ylabel('Count')
        ax.set_title('MAT-P Distribution (Precision)'); ax.legend(fontsize=9)

        # 3. Diversity distribution
        ax = axes[0, 2]
        ax.hist(div_all, bins=30, color='mediumseagreen', edgecolor='white', alpha=0.85)
        ax.axvline(div_all.mean(), color='red', lw=2, ls='--', label=f'Mean {div_all.mean():.3f}Å')
        ax.axvline(0.0, color='gray', lw=1.5, ls=':', label='Mode collapse = 0')
        ax.set_xlabel('Diversity (Å)'); ax.set_ylabel('Count')
        ax.set_title('Sampling Diversity\n(mean pairwise RMSD)'); ax.legend(fontsize=9)

        # 4. MAT-R vs atom count
        ax = axes[1, 0]
        for n_a in sorted(set(n_atoms_all)):
            mask = n_atoms_all == n_a
            ax.scatter([n_a]*mask.sum(), mat_r_all[mask], alpha=0.4, s=20, color='steelblue')
        # Bin means
        for lo, hi in [(1,5),(6,7),(8,8),(9,99)]:
            mask = (n_atoms_all >= lo) & (n_atoms_all <= hi)
            if mask.sum():
                mid = (lo + min(hi, 9)) / 2
                ax.plot(mid, mat_r_all[mask].mean(), 'r^', ms=10, zorder=5)
        ax.axhline(0.297, color='gray', lw=1.5, ls=':', label='GeoDiff')
        ax.set_xlabel('Heavy Atom Count (N)'); ax.set_ylabel('MAT-R (Å)')
        ax.set_title('MAT-R vs Molecule Size'); ax.legend(fontsize=9)

        # 5. MAT-R vs rotatable bonds
        ax = axes[1, 1]
        jitter = np.random.uniform(-0.1, 0.1, len(n_rot_all))
        ax.scatter(n_rot_all + jitter, mat_r_all, alpha=0.4, s=20, color='coral')
        for rb in sorted(set(n_rot_all)):
            mask = n_rot_all == rb
            if mask.sum():
                ax.plot(rb, mat_r_all[mask].mean(), 'r^', ms=10, zorder=5)
        ax.axhline(0.297, color='gray', lw=1.5, ls=':', label='GeoDiff')
        ax.set_xlabel('Rotatable Bonds'); ax.set_ylabel('MAT-R (Å)')
        ax.set_title('MAT-R vs Flexibility'); ax.legend(fontsize=9)

        # 6. COV-R vs COV-P scatter
        ax = axes[1, 2]
        ax.scatter(cov_r_all, cov_p_all, alpha=0.3, s=18, color='purple')
        ax.set_xlabel('COV-R (Recall)'); ax.set_ylabel('COV-P (Precision)')
        ax.set_title('Coverage: Recall vs Precision\n(per molecule)')
        ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='R=P line')
        ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
        ax.legend(fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, 'eval_summary.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Plots saved → {plot_path}")

    except Exception as e:
        print(f"\n  [Plotting skipped: {e}]")

    # ── Save TSV ──────────────────────────────────────────────────────────────
    tsv_path = os.path.join(args.out_dir, 'per_molecule_results.tsv')
    with open(tsv_path, 'w') as f:
        f.write('n_atoms\tn_rot\tmat_r\tmat_p\tcov_r_05\tcov_p_05\tdiversity\tbest_rmsd\n')
        for r in rows:
            f.write(f"{r['n_atoms']}\t{r['n_rot']}\t{r['mat_r']:.4f}\t{r['mat_p']:.4f}\t"
                    f"{r['cov_r_05']:.4f}\t{r['cov_p_05']:.4f}\t{r['diversity']:.4f}\t{r['best_rmsd']:.4f}\n")
    print(f"  Per-molecule TSV → {tsv_path}\n")


if __name__ == '__main__':
    main()
