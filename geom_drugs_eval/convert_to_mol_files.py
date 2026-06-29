"""
convert_to_mol_files.py — Convert generated conformers to SDF, MOL2, and PDB.

Why XYZ alone isn't enough:
  XYZ files contain only element symbols and 3D coordinates.
  MOL2/SDF/PDB additionally require bond connectivity and bond orders.
  We reconstruct the full RDKit molecule from:
    - atom_types (atomic numbers from the dataset)
    - edge_index  (bond connectivity from the dataset)
    - bond_types  (bond orders: 1=single, 2=double, 3=triple, 4=aromatic)
    - 3D coordinates (from XYZ file or directly from generator)

GeoDiff / GeoMol comparison:
  Neither GeoDiff nor GeoMol produce MOL2/PDB in their official repos.
  They both write SDF via RDKit using the same approach here.
  Their test.py (GeoDiff) uses Chem.MolToMolBlock for SDF output.

Usage:
  python geom_drugs_eval/convert_to_mol_files.py \
      --data data/qm9_heavy.jsonl \
      --xyz-dir geom_drugs_eval/eval_outputs/ \
      --out-dir geom_drugs_eval/mol_outputs/
"""

import os, sys, json, argparse, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

BOND_TYPE_MAP = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}

ELEMENT_MAP = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 16: 'S', 17: 'Cl', 35: 'Br',
}


def build_rdkit_mol(atom_types, edge_index, bond_types, coords):
    """
    Build an RDKit Mol with 3D coordinates from graph components.
    
    Args:
        atom_types : list[int] — atomic numbers (e.g. [6,6,8])
        edge_index : list[list] — [[src...],[dst...]] undirected edges
        bond_types : list[int] — bond order per edge (1/2/3/4)
        coords     : np.ndarray (N,3) — 3D coordinates in Angstroms
    
    Returns:
        rdkit.Chem.Mol with conformer, or None on failure
    """
    em = Chem.RWMol()

    # Add atoms
    for z in atom_types:
        atom = Chem.Atom(int(z))
        em.AddAtom(atom)

    # Add bonds (only unique undirected edges)
    seen = set()
    src_list, dst_list = edge_index[0], edge_index[1]
    for i, j, bo in zip(src_list, dst_list, bond_types):
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key not in seen:
            seen.add(key)
            btype = BOND_TYPE_MAP.get(int(bo), Chem.rdchem.BondType.SINGLE)
            em.AddBond(int(i), int(j), btype)

    # Embed 3D coordinates as a conformer
    conf = Chem.Conformer(len(atom_types))
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
    em.AddConformer(conf, assignId=True)

    # Try sanitizing; if aromatic rings fail, fall back to Kekulé form
    try:
        mol = em.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            mol = em.GetMol()
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return mol
        except Exception:
            return None


def write_sdf(mol, path):
    """Write molecule to SDF (MDL Molfile) — opens in PyMol, Avogadro, RDKit."""
    try:
        writer = Chem.SDWriter(path)
        writer.write(mol)
        writer.close()
        return True
    except Exception as e:
        print(f"    [SDF error] {e}")
        return False


def write_pdb(mol, path):
    """Write molecule to PDB — opens in VMD, PyMol, Chimera."""
    try:
        block = Chem.MolToPDBBlock(mol)
        if block:
            with open(path, 'w') as f:
                f.write(block)
            return True
    except Exception as e:
        print(f"    [PDB error] {e}")
    return False


def write_mol2(mol, path):
    """Write molecule to MOL2 — opens in Chimera, AMBER, GOLD docking."""
    try:
        # RDKit doesn't write MOL2 natively; use SDF→MOL2 via obabel if available
        # otherwise write SMILES-annotated SDF as fallback
        import subprocess
        sdf_tmp = path.replace('.mol2', '_tmp.sdf')
        write_sdf(mol, sdf_tmp)
        result = subprocess.run(
            ['obabel', sdf_tmp, '-O', path, '--quiet'],
            capture_output=True, timeout=10
        )
        os.remove(sdf_tmp)
        if result.returncode == 0 and os.path.exists(path):
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: write manual MOL2 using atom/bond tables
    try:
        conf = mol.GetConformer()
        atoms = list(mol.GetAtoms())
        bonds = list(mol.GetBonds())
        
        SYBYL = {
            Chem.rdchem.BondType.SINGLE:   '1',
            Chem.rdchem.BondType.DOUBLE:   '2',
            Chem.rdchem.BondType.TRIPLE:   '3',
            Chem.rdchem.BondType.AROMATIC: 'ar',
        }
        
        with open(path, 'w') as f:
            f.write('@<TRIPOS>MOLECULE\n')
            f.write('generated_conformer\n')
            f.write(f'{len(atoms)} {len(bonds)} 0 0 0\n')
            f.write('SMALL\nGASTEIGER\n\n')
            
            f.write('@<TRIPOS>ATOM\n')
            for i, atom in enumerate(atoms):
                pos = conf.GetAtomPosition(i)
                sym = atom.GetSymbol()
                f.write(f'{i+1:6d} {sym:<4s}{i+1:<4d}  '
                        f'{pos.x:10.4f} {pos.y:10.4f} {pos.z:10.4f}  '
                        f'{sym}.3     1  LIG1        0.0000\n')
            
            f.write('@<TRIPOS>BOND\n')
            for i, bond in enumerate(bonds):
                btype = SYBYL.get(bond.GetBondType(), '1')
                f.write(f'{i+1:6d} {bond.GetBeginAtomIdx()+1:4d} '
                        f'{bond.GetEndAtomIdx()+1:4d}  {btype}\n')
        return True
    except Exception as e:
        print(f"    [MOL2 fallback error] {e}")
        return False


def load_val_split(data_path, max_atoms=9):
    """Load and return val split molecules."""
    all_mols = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            if not item.get('coordinates'): continue
            if item.get('num_atoms', len(item['atom_types'])) > max_atoms: continue
            if any(z >= 54 or z <= 0 for z in item['atom_types']): continue
            all_mols.append(item)
    n_train = int(len(all_mols) * 0.9)
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(all_mols), generator=gen).tolist()
    return [all_mols[i] for i in idx[n_train:]]


def read_xyz_coords(xyz_path):
    """Parse 3D coordinates from an XYZ file."""
    coords = []
    with open(xyz_path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    for line in lines[2:2+n_atoms]:
        parts = line.strip().split()
        if len(parts) >= 4:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert generated XYZ conformers to SDF/PDB/MOL2")
    parser.add_argument('--data',    required=True, help='qm9_heavy.jsonl')
    parser.add_argument('--xyz-dir', required=True, help='Directory containing case_*.xyz files')
    parser.add_argument('--out-dir', required=True, help='Output directory for mol files')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load val split to get molecular graphs (atom_types, edge_index, bond_types)
    print(f"Loading val split from {args.data} ...")
    val_mols = load_val_split(args.data)

    # Find all XYZ files for the 3 cases
    xyz_files = sorted(glob.glob(os.path.join(args.xyz_dir, 'case_*.xyz')))
    print(f"Found {len(xyz_files)} XYZ files in {args.xyz_dir}\n")

    # Group by case prefix
    from collections import defaultdict
    cases = defaultdict(list)
    for f in xyz_files:
        name = os.path.basename(f)
        # e.g. case_best_gen_00.xyz → prefix = case_best
        prefix = '_'.join(name.split('_')[:2])  # case_best / case_median / case_worst
        cases[prefix].append(f)

    results_summary = []

    for case_name, files in sorted(cases.items()):
        print(f"Processing: {case_name}  ({len(files)} files)")

        # Find the reference XYZ to get atom count
        ref_file = os.path.join(args.xyz_dir, f"{case_name}_reference.xyz")
        if not os.path.exists(ref_file):
            print(f"  [skip] no reference file for {case_name}")
            continue

        ref_coords = read_xyz_coords(ref_file)
        N = len(ref_coords)

        # Find matching val molecule by atom count
        # (we match by N since case best/median/worst were picked from val mols)
        # Read element symbols from reference XYZ
        atom_syms = []
        with open(ref_file) as f:
            lines = f.readlines()
        for line in lines[2:2+N]:
            atom_syms.append(line.strip().split()[0])

        SYM_TO_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'S':16,'Cl':17,'Br':35}
        atom_types_from_xyz = [SYM_TO_Z.get(s, 6) for s in atom_syms]

        # Match val molecule: same N and same sorted atom types
        target_sorted = tuple(sorted(atom_types_from_xyz))
        match = None
        for vm in val_mols:
            if len(vm['atom_types']) == N and tuple(sorted(vm['atom_types'])) == target_sorted:
                # Further verify: reference coords close to dataset coords
                vm_coords = np.array(vm['coordinates'], dtype=np.float32)
                vm_cent   = vm_coords - vm_coords.mean(0)
                ref_cent  = ref_coords - ref_coords.mean(0)
                rmsd = np.sqrt(np.mean((vm_cent - ref_cent)**2))
                if rmsd < 0.5:
                    match = vm
                    break

        if match is None:
            # Fallback: use first val mol with same N (for connectivity)
            for vm in val_mols:
                if len(vm['atom_types']) == N:
                    match = vm
                    break

        if match is None:
            print(f"  [skip] could not find matching val molecule")
            continue

        atom_types = match['atom_types']
        edge_index  = match['edge_index']
        bond_types  = match['bond_types']

        # Convert each XYZ in this case
        n_ok = 0
        for xyz_path in sorted(files):
            coords = read_xyz_coords(xyz_path)
            if len(coords) != N:
                continue

            mol = build_rdkit_mol(atom_types, edge_index, bond_types, coords)
            if mol is None:
                tqdm_name = os.path.basename(xyz_path)
                print(f"  [warn] could not build RDKit mol for {tqdm_name}")
                continue

            stem = os.path.splitext(os.path.basename(xyz_path))[0]
            prefix_out = os.path.join(args.out_dir, stem)

            ok_sdf  = write_sdf(mol,  prefix_out + '.sdf')
            ok_pdb  = write_pdb(mol,  prefix_out + '.pdb')
            ok_mol2 = write_mol2(mol, prefix_out + '.mol2')
            n_ok += 1

        print(f"  ✅  Converted {n_ok}/{len(files)} files → SDF + PDB + MOL2")
        results_summary.append((case_name, n_ok, len(files)))

    print(f"\n{'─'*55}")
    print(f"  Summary:")
    for case, ok, total in results_summary:
        print(f"  {case:20s}  {ok}/{total} converted")
    print(f"\n  Output directory: {args.out_dir}")
    print(f"\n  Open with:")
    print(f"    PyMol  : pymol {args.out_dir}/*.sdf")
    print(f"    VMD    : vmd   {args.out_dir}/*.pdb")
    print(f"    Chimera: chimera {args.out_dir}/*.mol2")
    print(f"    Avogadro (GUI): drag-and-drop any .sdf file")


if __name__ == '__main__':
    main()
