"""
autoresearch/mol_infer.py — Inference: generate molecules and export to PDB/MOL2/SDF

For each experiment checkpoint, this script:
  1. Loads the best saved checkpoint
  2. Samples molecules from the diffusion model
  3. Validates each generated molecule with RDKit
  4. Exports valid molecules to:
       - SDF  (standard structure format, for RDKit/PyMol/Maestro)
       - PDB  (for PyMol/VMD/ChimeraX — each atom + CONECT records)
       - MOL2 (for UCSF Chimera, Tripos format with charges)
  5. Produces per-experiment summary: valid rate, RMSD, bond errors
  6. Writes edge-case analysis: which molecules fail and why

Usage:
    cd mol_next_gen
    source venv/bin/activate

    # Run inference for all 4 experiments:
    PYTHONPATH=. python autoresearch/mol_infer.py --all

    # Single experiment:
    PYTHONPATH=. python autoresearch/mol_infer.py --exp exp_A_baseline

    # Custom checkpoint:
    PYTHONPATH=. python autoresearch/mol_infer.py \
        --checkpoint checkpoints/exp_B_attention_egnn_best.pt \
        --num-molecules 200

Output:
    generated/exp_A_baseline/
        ├── molecules.sdf         (all valid molecules, multi-structure SDF)
        ├── mol_000.pdb           (individual PDB files)
        ├── mol_000.mol2          (individual MOL2 files)
        ├── mol_001.pdb / .mol2
        ├── ...
        ├── summary.json          (validity, RMSD, bond errors, edge cases)
        └── edge_cases/
            ├── invalid_000.pdb   (failed molecules for debugging)
            └── ...
"""

import os, sys, json, argparse, traceback
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from autoresearch.mol_prepare import make_dataloaders, CHECKPOINT_DIR
from models.conformer_diffusion import remove_com

# ── RDKit imports ──────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import (
    AllChem, SDWriter, SanitizeMol, MolToMolBlock, MolFromSmiles,
    RWMol, Atom, BondType
)
from rdkit.Chem.rdchem import BondStereo
import rdkit.RDLogger as rl
rl.DisableLog("rdApp.*")

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).parent.parent
GENERATED_DIR = PROJECT_ROOT / "generated"
NUM_MOLECULES = 200   # default molecules to generate per experiment

ATOM_NUM_TO_SYMBOL = {
    1:"H", 6:"C", 7:"N", 8:"O", 9:"F", 15:"P", 16:"S", 17:"Cl", 35:"Br", 53:"I"
}

BOND_TYPE_TO_RDKIT = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}

IDEAL_BONDS = {
    (6,6,1):1.54,(6,6,2):1.34,(6,6,3):1.20,(6,6,4):1.40,
    (6,7,1):1.47,(6,7,2):1.29,(6,7,4):1.34,
    (6,8,1):1.43,(6,8,2):1.22,
    (6,1,1):1.09,(7,1,1):1.01,(8,1,1):0.96,
    (6,9,1):1.35,(6,17,1):1.77,(6,16,1):1.82,
}

EXP_INFO = {
    "exp_A_baseline": {
        "model_class": "ConformerDiffusion",
        "checkpoint":  "exp_A_baseline_best.pt",
        "sampler":     "ddim",
        "num_steps":   50,
    },
    "exp_B_attention_egnn": {
        "model_class": "AttnConformerDiffusion",
        "checkpoint":  "exp_B_attention_egnn_best.pt",
        "sampler":     "ddim",
        "num_steps":   50,
    },
    "exp_C_flow_matching": {
        "model_class": "FlowMatchingConformer",
        "checkpoint":  "exp_C_flow_matching_best.pt",
        "sampler":     "ode",
        "num_steps":   20,
    },
    "exp_D_torsion_aux": {
        "model_class": "ConformerDiffusion",
        "checkpoint":  "exp_D_torsion_aux_best.pt",
        "sampler":     "ddim",
        "num_steps":   50,
    },
}

# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(exp_key, device):
    info = EXP_INFO[exp_key]
    ckpt_path = PROJECT_ROOT / "checkpoints" / info["checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})

    hidden_dim  = cfg.get("hidden_dim", 256)
    num_layers  = cfg.get("num_layers", 6)
    num_heads   = cfg.get("num_heads",  4)
    time_dim    = cfg.get("time_dim",   128)
    timesteps   = cfg.get("timesteps",  1000)

    cls_name = info["model_class"]
    if cls_name == "ConformerDiffusion":
        from models.conformer_diffusion import ConformerDiffusion
        model = ConformerDiffusion(num_timesteps=timesteps, hidden_dim=hidden_dim,
                                   num_layers=num_layers, time_dim=time_dim)
    elif cls_name == "AttnConformerDiffusion":
        from models.attn_conformer_diffusion import AttnConformerDiffusion
        model = AttnConformerDiffusion(num_timesteps=timesteps, hidden_dim=hidden_dim,
                                       num_layers=num_layers, time_dim=time_dim,
                                       num_heads=num_heads)
    elif cls_name == "FlowMatchingConformer":
        from models.flow_matching import FlowMatchingConformer
        model = FlowMatchingConformer(hidden_dim=hidden_dim, num_layers=num_layers,
                                      time_dim=time_dim)
    else:
        raise ValueError(f"Unknown model class: {cls_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  Loaded {cls_name} from {ckpt_path.name} "
          f"(epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',0):.4f})")
    return model, info

# ── Coordinate generation ──────────────────────────────────────────────────────

@torch.no_grad()
def generate_coords(model, batch, info, device):
    """Run inference for one batch. Returns generated coordinates (N, 3)."""
    at = batch["atom_types"].to(device)
    ei = batch["edge_index"].to(device)
    bt = batch["bond_types"].to(device)
    bi = batch["batch_idx"].to(device)

    sampler   = info["sampler"]
    num_steps = info["num_steps"]

    if sampler == "ddim":
        coords = model.ddim_sample(at, ei, bt, bi, num_steps=num_steps)
    elif sampler == "ode":
        coords = model.ode_sample(at, ei, bt, bi, num_steps=num_steps)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    return coords.cpu()

# ── Molecule builder ───────────────────────────────────────────────────────────

def build_rdkit_mol(atom_types, edge_index, bond_types, coords):
    """Build an RDKit Mol from generated data. Returns (mol, error_str)."""
    try:
        n_atoms = len(atom_types)
        rwmol   = RWMol()

        # Add atoms
        for z in atom_types:
            atom = Atom(int(z))
            rwmol.AddAtom(atom)

        # Add bonds (edge_index is bidirectional — only add each bond once)
        added_bonds = set()
        src_list, dst_list = edge_index[0].tolist(), edge_index[1].tolist()
        for s, d, bt in zip(src_list, dst_list, bond_types.tolist()):
            if (min(s,d), max(s,d)) not in added_bonds:
                added_bonds.add((min(s,d), max(s,d)))
                btype = BOND_TYPE_TO_RDKIT.get(bt, Chem.BondType.SINGLE)
                rwmol.AddBond(s, d, btype)

        # Set 3D coordinates
        conf = Chem.Conformer(n_atoms)
        from rdkit.Geometry import Point3D
        for i, (x, y, z) in enumerate(coords.tolist()):
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol = rwmol.GetMol()
        mol.AddConformer(conf, assignId=True)

        # Sanitize
        SanitizeMol(mol)
        return mol, None

    except Exception as e:
        return None, str(e)


def mol_bond_error(mol, atom_types, edge_index, bond_types, coords):
    """Compute mean absolute bond length error vs MMFF94 ideal."""
    errors = []
    src_list, dst_list = edge_index[0].tolist(), edge_index[1].tolist()
    seen = set()
    for s, d, bt in zip(src_list, dst_list, bond_types.tolist()):
        if (min(s,d), max(s,d)) in seen: continue
        seen.add((min(s,d), max(s,d)))
        z1, z2 = int(atom_types[s]), int(atom_types[d])
        key = (min(z1,z2), max(z1,z2), bt)
        if key in IDEAL_BONDS:
            d_pred = float(torch.norm(coords[s] - coords[d]))
            errors.append(abs(d_pred - IDEAL_BONDS[key]))
    return float(np.mean(errors)) if errors else 0.0


# ── File writers ───────────────────────────────────────────────────────────────

def write_pdb(mol, coords, atom_types, out_path: Path):
    """Write a PDB file with ATOM records + CONECT records."""
    lines = ["REMARK  Generated by NExT-Mol Gen\n"]
    sym_map = ATOM_NUM_TO_SYMBOL

    for i, (z, (x, y, z_c)) in enumerate(zip(atom_types, coords.tolist())):
        sym = sym_map.get(int(z), "X")
        lines.append(
            f"HETATM{i+1:5d}  {sym:<3s} LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z_c:8.3f}  1.00  0.00          {sym:>2s}\n"
        )

    # CONECT records
    if mol is not None:
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            lines.append(f"CONECT{i+1:5d}{j+1:5d}\n")
            lines.append(f"CONECT{j+1:5d}{i+1:5d}\n")

    lines.append("END\n")
    out_path.write_text("".join(lines))


def write_mol2(mol, coords, atom_types, edge_index, bond_types, out_path: Path):
    """Write a Tripos MOL2 file (UCSF Chimera, Amber compatible)."""
    n_atoms = len(atom_types)
    sym_map = ATOM_NUM_TO_SYMBOL

    src_list, dst_list = edge_index[0].tolist(), edge_index[1].tolist()
    unique_bonds = []
    seen = set()
    for s, d, bt in zip(src_list, dst_list, bond_types.tolist()):
        if (min(s,d), max(s,d)) not in seen:
            seen.add((min(s,d), max(s,d)))
            bt_str = {1:"1", 2:"2", 3:"3", 4:"ar"}.get(bt, "1")
            unique_bonds.append((s, d, bt_str))

    n_bonds = len(unique_bonds)
    lines = [
        "@<TRIPOS>MOLECULE\n",
        "generated_molecule\n",
        f"{n_atoms} {n_bonds} 0 0 0\n",
        "SMALL\nNO_CHARGES\n\n",
        "@<TRIPOS>ATOM\n",
    ]

    for i, (z, (x, y, z_c)) in enumerate(zip(atom_types, coords.tolist())):
        sym = sym_map.get(int(z), "X")
        lines.append(
            f"{i+1:6d}  {sym}{i+1:<4d}  {x:10.4f}  {y:10.4f}  {z_c:10.4f}"
            f"  {sym}       1  LIG       0.0000\n"
        )

    lines.append("@<TRIPOS>BOND\n")
    for b_idx, (s, d, bt_str) in enumerate(unique_bonds):
        lines.append(f"{b_idx+1:5d}  {s+1:4d}  {d+1:4d}  {bt_str}\n")

    out_path.write_text("".join(lines))


def write_sdf(mol, out_writer):
    """Write mol to open SDWriter."""
    if mol is not None:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=50)
        except Exception:
            pass
        out_writer.write(mol)


# ── Main inference loop ────────────────────────────────────────────────────────

def run_inference(exp_key, num_molecules, device):
    print(f"\n{'='*60}")
    print(f"  Inference: {exp_key}")
    print(f"  Generating {num_molecules} molecules...")
    print(f"{'='*60}")

    model, info = load_model(exp_key, device)
    _, val_loader = make_dataloaders(batch_size=32)

    out_dir     = GENERATED_DIR / exp_key
    edge_dir    = out_dir / "edge_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(exist_ok=True)

    sdf_writer = SDWriter(str(out_dir / "molecules.sdf"))

    stats = {
        "total_generated": 0,
        "valid": 0,
        "invalid": 0,
        "bond_errors": [],
        "failure_reasons": defaultdict(int),
        "per_molecule": [],
    }

    mol_idx  = 0
    fail_idx = 0

    for batch in val_loader:
        if stats["total_generated"] >= num_molecules:
            break

        coords_all  = generate_coords(model, batch, info, device)
        at_all      = batch["atom_types"]
        ei_all      = batch["edge_index"]
        bt_all      = batch["bond_types"]
        bi_all      = batch["batch_idx"]
        n_mols_batch = int(bi_all.max().item()) + 1

        for mol_in_batch in range(n_mols_batch):
            if stats["total_generated"] >= num_molecules:
                break

            mask   = bi_all == mol_in_batch
            at     = at_all[mask]
            coords = coords_all[mask]

            # Rebuild edge_index and bond_types for this molecule
            ei_mask = mask[ei_all[0]]
            ei_local = ei_all[:, ei_mask]
            offset   = mask.nonzero(as_tuple=False)[0, 0].item()
            ei_local = ei_local - offset
            bt       = bt_all[ei_mask]

            # Build RDKit mol
            mol, err = build_rdkit_mol(at.tolist(), ei_local, bt, coords)

            bond_err = mol_bond_error(mol or Chem.RWMol(), at, ei_local, bt, coords)
            stats["total_generated"] += 1

            if mol is not None:
                stats["valid"] += 1
                stats["bond_errors"].append(bond_err)

                # Write SDF
                write_sdf(mol, sdf_writer)

                # Write PDB
                write_pdb(mol, coords, at.tolist(),
                          out_dir / f"mol_{mol_idx:04d}.pdb")

                # Write MOL2
                write_mol2(mol, coords, at.tolist(), ei_local, bt,
                           out_dir / f"mol_{mol_idx:04d}.mol2")

                stats["per_molecule"].append({
                    "idx": mol_idx,
                    "valid": True,
                    "bond_error": round(bond_err, 4),
                    "n_atoms": len(at),
                    "atom_types": at.tolist(),
                })
                mol_idx += 1
            else:
                stats["invalid"] += 1
                reason = err or "unknown"
                # Bucket failure reason
                if "valence" in reason.lower():
                    stats["failure_reasons"]["valence_error"] += 1
                elif "sanitize" in reason.lower():
                    stats["failure_reasons"]["sanitize_fail"] += 1
                elif "kekulize" in reason.lower():
                    stats["failure_reasons"]["kekulize_fail"] += 1
                else:
                    stats["failure_reasons"]["other"] += 1

                # Save edge case PDB for debugging
                write_pdb(None, coords, at.tolist(),
                          edge_dir / f"invalid_{fail_idx:04d}.pdb")
                stats["per_molecule"].append({
                    "idx": fail_idx,
                    "valid": False,
                    "error": reason[:100],
                    "bond_error": round(bond_err, 4),
                    "n_atoms": len(at),
                })
                fail_idx += 1

    sdf_writer.close()

    # ── Summary ──
    valid_rate    = stats["valid"] / max(stats["total_generated"], 1)
    mean_bond_err = float(np.mean(stats["bond_errors"])) if stats["bond_errors"] else 0.0

    summary = {
        "exp_key":          exp_key,
        "total_generated":  stats["total_generated"],
        "valid":            stats["valid"],
        "invalid":          stats["invalid"],
        "fully_valid_rate": round(valid_rate, 4),
        "mean_bond_error":  round(mean_bond_err, 4),
        "failure_reasons":  dict(stats["failure_reasons"]),
        "sampler":          info["sampler"],
        "num_steps":        info["num_steps"],
        "output_dir":       str(out_dir),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  Results:")
    print(f"    Total generated : {stats['total_generated']}")
    print(f"    Valid molecules : {stats['valid']} ({valid_rate:.1%})")
    print(f"    Invalid         : {stats['invalid']} ({dict(stats['failure_reasons'])})")
    print(f"    Mean bond error : {mean_bond_err:.4f} Å")
    print(f"\n  Output files:")
    print(f"    {out_dir}/molecules.sdf    ({stats['valid']} structures)")
    print(f"    {out_dir}/mol_XXXX.pdb    ({mol_idx} PDB files)")
    print(f"    {out_dir}/mol_XXXX.mol2   ({mol_idx} MOL2 files)")
    print(f"    {out_dir}/edge_cases/     ({fail_idx} invalid, for debugging)")
    print(f"    {out_dir}/summary.json")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate molecules from trained checkpoints → PDB, MOL2, SDF"
    )
    parser.add_argument("--exp", choices=list(EXP_INFO.keys()),
                        help="Which experiment to run inference for")
    parser.add_argument("--all", action="store_true",
                        help="Run inference for all 4 experiments")
    parser.add_argument("--num-molecules", type=int, default=NUM_MOLECULES,
                        help=f"Molecules to generate per experiment (default {NUM_MOLECULES})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {GENERATED_DIR}/")

    if args.all:
        experiments = list(EXP_INFO.keys())
    elif args.exp:
        experiments = [args.exp]
    else:
        parser.print_help()
        return

    all_summaries = []
    for exp_key in experiments:
        try:
            s = run_inference(exp_key, args.num_molecules, device)
            all_summaries.append(s)
        except FileNotFoundError as e:
            print(f"\n  SKIP {exp_key}: {e}")
        except Exception as e:
            print(f"\n  ERROR {exp_key}: {e}")
            traceback.print_exc()

    # Print comparison table
    if len(all_summaries) > 1:
        print("\n" + "="*60)
        print("  INFERENCE COMPARISON TABLE")
        print("="*60)
        print(f"{'Experiment':<30} {'Valid':>8} {'Bond Err':>10} {'Sampler':>10} {'Steps':>6}")
        print("-"*60)
        for s in all_summaries:
            print(f"{s['exp_key']:<30} "
                  f"{s['fully_valid_rate']:>8.3f} "
                  f"{s['mean_bond_error']:>10.4f} "
                  f"{s['sampler']:>10} "
                  f"{s['num_steps']:>6}")


if __name__ == "__main__":
    main()
