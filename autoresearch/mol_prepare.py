"""
mol_prepare.py — Fixed constants, data utilities, and evaluation harness.
Mirrors autoresearch's prepare.py: this file is READ-ONLY for the agent.
The agent modifies mol_train.py, NOT this file.

Usage (imported by mol_train.py):
    from autoresearch.mol_prepare import (
        DATA_PATH, EVAL_MOLECULES, EPOCH_BUDGET,
        make_dataloaders, evaluate_all, print_report,
        metrics_to_tsv_row, TSV_HEADER
    )
"""

import os, sys, json, time, math, copy
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

DATA_PATH      = Path(os.environ.get("MOL_DATASET", os.path.join(PROJECT_ROOT, "data", "qm9_selfies.jsonl")))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

MAX_ATOMS      = int(os.environ.get("MOL_MAX_ATOMS", 29))
                       # Set MOL_MAX_ATOMS=9 for heavy-atom-only training (SOTA default)
                       # Default 29: explicit-H QM9 (all atoms including hydrogen)
VAL_SPLIT      = 0.1   # 10 % validation
RANDOM_SEED    = 42
EVAL_MOLECULES = 500   # fixed evaluation set size (comparable across all experiments)
EPOCH_BUDGET   = 200    # budget for SOTA training
FULL_BUDGET    = 200   # for final full training

# Evaluation thresholds — from EDM / GeoMol papers (DO NOT change)
BOND_TOLERANCE = 0.2   # Å
CLASH_DIST     = 1.4   # Å
COV_THRESHOLD  = 0.5   # Å

IDEAL_BONDS: Dict = {
    (6, 6, 1): 1.54, (6, 6, 2): 1.34, (6, 6, 3): 1.20, (6, 6, 4): 1.40,
    (6, 7, 1): 1.47, (6, 7, 2): 1.29, (6, 7, 4): 1.34,
    (6, 8, 1): 1.43, (6, 8, 2): 1.22,
    (6, 1, 1): 1.09, (7, 1, 1): 1.01, (8, 1, 1): 0.96,
    (6, 9, 1): 1.35, (6,17, 1): 1.77, (6,16, 1): 1.82,
    (6,35, 1): 1.94, (7, 7, 1): 1.45, (7, 8, 1): 1.36,
    (16,8, 2): 1.44, (15,8, 1): 1.63,
}

TSV_HEADER = (
    "commit\tfully_valid\tmat_r\trmsd_mean\tstrain_kcal\t"
    "cov_r\tvalidity\tbond_error\tstatus\tdescription"
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ConformerDataset(Dataset):
    def __init__(self, data_path: str = DATA_PATH, max_atoms: int = MAX_ATOMS):
        self.data = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('coordinates') is None:
                    continue
                if item['num_atoms'] > max_atoms:
                    continue
                ats = item.get('atom_types', [])
                if any(z >= 54 or z <= 0 for z in ats):
                    continue
                self.data.append(item)

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'atom_types':  torch.tensor(item['atom_types'],  dtype=torch.long),
            'coordinates': torch.tensor(item['coordinates'], dtype=torch.float32),
            'edge_index':  torch.tensor(item['edge_index'],  dtype=torch.long),
            'bond_types':  torch.tensor(item['bond_types'],  dtype=torch.long),
            'num_atoms':   item['num_atoms'],
        }


def _collate(batch):
    at, co, ei, bt, bi = [], [], [], [], []
    offset = 0
    for i, item in enumerate(batch):
        N = item['num_atoms']
        at.append(item['atom_types'])
        co.append(item['coordinates'])
        ei.append(item['edge_index'] + offset)
        bt.append(item['bond_types'])
        bi.append(torch.full((N,), i, dtype=torch.long))
        offset += N
    return {
        'atom_types': torch.cat(at), 'coordinates': torch.cat(co),
        'edge_index': torch.cat(ei, dim=1), 'bond_types': torch.cat(bt),
        'batch_idx':  torch.cat(bi), 'num_molecules': len(batch),
    }


def make_dataloaders(batch_size: int = 64, num_workers: int = 4):
    ds = ConformerDataset()
    n_val   = int(len(ds) * VAL_SPLIT)
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(RANDOM_SEED)
    tr, va = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)
    kw = dict(collate_fn=_collate, num_workers=num_workers, pin_memory=True)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(va, batch_size=batch_size, shuffle=False, **kw))

# ---------------------------------------------------------------------------
# Evaluation helpers (ground-truth — DO NOT CHANGE)
# ---------------------------------------------------------------------------

def _ideal_bond(a1, a2, order):
    return IDEAL_BONDS.get((min(a1,a2), max(a1,a2), order), 1.50)

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    P = P - P.mean(0); Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    D = np.eye(3); D[2,2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return float(np.sqrt(np.mean((P @ R.T - Q)**2)))

def _bond_valid(pos, anums, esrc, edst, ebo):
    errs = [abs(float(np.linalg.norm(pos[i]-pos[j])) - _ideal_bond(anums[i], anums[j], bo))
            for i,j,bo in zip(esrc, edst, ebo) if i < j]
    if not errs: return True, 0.0
    mae = float(np.mean(errs))
    return mae < BOND_TOLERANCE, mae

def _clash_free(pos, esrc, edst):
    bonded = {(min(i,j), max(i,j)) for i,j in zip(esrc, edst)}
    N = len(pos)
    for i in range(N):
        for j in range(i+1, N):
            if (i,j) not in bonded and float(np.linalg.norm(pos[i]-pos[j])) < CLASH_DIST:
                return False
    return True

def _rdkit_valid(anums, pos, esrc, edst, ebo):
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        em = Chem.RWMol()
        for z in anums: em.AddAtom(Chem.Atom(int(z)))
        BT = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE,
              3: Chem.rdchem.BondType.TRIPLE,  4: Chem.rdchem.BondType.AROMATIC}
        seen = set()
        for i,j,bo in zip(esrc, edst, ebo):
            k=(min(i,j),max(i,j))
            if k not in seen:
                seen.add(k); em.AddBond(i, j, BT.get(int(bo), Chem.rdchem.BondType.SINGLE))
        conf = Chem.Conformer(len(anums))
        for idx,(x,y,z) in enumerate(pos): conf.SetAtomPosition(idx, Point3D(float(x),float(y),float(z)))
        mol = em.GetMol(); mol.AddConformer(conf, assignId=True)
        try: Chem.SanitizeMol(mol); return mol, True
        except: return mol, False
    except: return None, False

def _mmff_strain(mol):
    try:
        from rdkit.Chem import AllChem
        mol2 = copy.deepcopy(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol2, AllChem.MMFFGetMoleculeProperties(mol2), confId=0)
        if ff is None: return None
        e0 = ff.CalcEnergy(); ff.Minimize(maxIts=500); e1 = ff.CalcEnergy()
        return float(e0 - e1)
    except: return None

def _cov_mat(refs, gens, thr=COV_THRESHOLD):
    if not refs or not gens: return 0.0, float('inf')
    mins = np.array([min(kabsch_rmsd(r,g) for g in gens) for r in refs])
    return float(np.mean(mins < thr)), float(np.mean(mins))

# ---------------------------------------------------------------------------
# FIXED EVALUATION HARNESS — ground truth metric (DO NOT CHANGE)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_all(model, val_loader, device, num_gen=EVAL_MOLECULES,
                 num_gen_per_ref=5, verbose=True) -> Dict:
    """Full 9-metric evaluation. Immutable across all experiments."""
    model.eval(); model.to(device)
    rdkit_v, bond_v, clash_f, bond_e, strain_l, rmsd_l = [], [], [], [], [], []
    cov_mat_data = []
    n_done = 0; t0 = time.time()

    for batch in val_loader:
        if n_done >= num_gen: break
        at  = batch['atom_types'].to(device)
        ct  = batch['coordinates'].to(device)
        ei  = batch['edge_index'].to(device)
        bt  = batch['bond_types'].to(device)
        bi  = batch['batch_idx'].to(device)

        # Centre coordinates
        B = int(bi.max().item()) + 1
        com = torch.zeros(B,3,device=device); cnt = torch.zeros(B,device=device)
        cnt.scatter_add_(0, bi, torch.ones(bi.size(0),device=device))
        com.scatter_add_(0, bi.unsqueeze(-1).expand(-1,3), ct)
        ct = ct - (com / cnt.unsqueeze(1).clamp(min=1))[bi]

        try:
            gen = model.ddim_sample(at, ei, bt, bi, num_steps=50)
        except Exception: continue

        n_mols = int(bi.max().item()) + 1
        for b in range(n_mols):
            if n_done >= num_gen: break
            mask  = (bi == b).cpu()
            emask = ((bi[ei[0]]==b) & (bi[ei[1]]==b)).cpu()
            if mask.sum() < 2: continue

            pos_t = ct[mask].cpu().numpy()
            pos_g = gen[mask].cpu().numpy()

            g2l = {g:l for l,g in enumerate(mask.nonzero(as_tuple=False).squeeze(-1).tolist())}
            le   = ei[:, emask].cpu()
            esrc = [g2l.get(int(i),0) for i in le[0].tolist()]
            edst = [g2l.get(int(i),0) for i in le[1].tolist()]
            ebo  = bt[emask].cpu().tolist()
            anums = at[mask].cpu().tolist()

            rmsd_l.append(kabsch_rmsd(pos_g, pos_t))
            bv, mae = _bond_valid(pos_g, anums, esrc, edst, ebo)
            bond_v.append(bv); bond_e.append(mae)
            clash_f.append(_clash_free(pos_g, esrc, edst))
            mol, valid = _rdkit_valid(anums, pos_g, esrc, edst, ebo)
            rdkit_v.append(valid)
            if mol is not None and valid:
                s = _mmff_strain(mol)
                if s is not None: strain_l.append(s)

            try:
                la = at[mask]; lei = ei[:, emask]
                off = mask.nonzero(as_tuple=False)[0].item()
                lei0 = lei - off; lbt = bt[emask]
                lbi  = torch.zeros(mask.sum(), dtype=torch.long, device=device)
                gens = [model.ddim_sample(la, lei0, lbt, lbi, num_steps=50).cpu().numpy()
                        for _ in range(num_gen_per_ref)]
                cov_mat_data.append(([pos_t], gens))
            except Exception: pass

            n_done += 1

        if verbose and n_done % 50 == 0 and n_done > 0:
            print(f"  Eval [{n_done}/{num_gen}] {time.time()-t0:.0f}s", flush=True)

    sm = lambda lst: float(np.mean(lst)) if lst else float('nan')
    all_cov, all_mat = [], []
    for refs, gens in cov_mat_data:
        c, m = _cov_mat(refs, gens); all_cov.append(c); all_mat.append(m)

    return {
        'validity':         sm([float(v) for v in rdkit_v]),
        'bond_valid_rate':  sm([float(v) for v in bond_v]),
        'clash_free_rate':  sm([float(v) for v in clash_f]),
        'fully_valid_rate': sm([float(rv and bv and cf)
                                 for rv,bv,cf in zip(rdkit_v,bond_v,clash_f)]),
        'mean_bond_error':  sm(bond_e),
        'cov_r':            sm(all_cov),
        'mat_r':            sm(all_mat),
        'rmsd_mean':        sm(rmsd_l),
        'rmsd_std':         float(np.std(rmsd_l)) if rmsd_l else float('nan'),
        'mean_strain_kcal': sm(strain_l),
        'n_evaluated':      n_done,
        'n_strain_mols':    len(strain_l),
    }


def print_report(metrics: Dict, tag: str = "", epoch: Optional[int] = None):
    hdr = f"── Eval{' ['+tag+']' if tag else ''}{' [Epoch '+str(epoch)+']' if epoch else ''} " + "─"*40
    print(f"\n{hdr}")
    print(f"  n_evaluated       : {metrics.get('n_evaluated','?')}")
    print(f"  RDKit valid       : {metrics['validity']*100:6.1f}%")
    print(f"  Bond valid (0.2Å) : {metrics['bond_valid_rate']*100:6.1f}%")
    print(f"  Clash-free        : {metrics['clash_free_rate']*100:6.1f}%")
    print(f"  Fully valid ←PRIMARY: {metrics['fully_valid_rate']*100:6.1f}%")
    print(f"  Mean bond error   : {metrics['mean_bond_error']:.4f} Å")
    print(f"  COV-R             : {metrics['cov_r']*100:6.1f}%")
    print(f"  MAT-R ←SECONDARY  : {metrics['mat_r']:.4f} Å")
    print(f"  Kabsch-RMSD       : {metrics['rmsd_mean']:.4f} ± {metrics['rmsd_std']:.4f} Å")
    print(f"  MMFF strain       : {metrics['mean_strain_kcal']:.2f} kcal/mol ({metrics.get('n_strain_mols',0)} mols)")
    print()


def metrics_to_tsv_row(commit: str, metrics: Dict, status: str, description: str) -> str:
    return (f"{commit}\t{metrics['fully_valid_rate']:.6f}\t{metrics['mat_r']:.6f}\t"
            f"{metrics['rmsd_mean']:.6f}\t{metrics['mean_strain_kcal']:.2f}\t"
            f"{metrics['cov_r']:.6f}\t{metrics['validity']:.6f}\t"
            f"{metrics['mean_bond_error']:.6f}\t{status}\t{description}")
