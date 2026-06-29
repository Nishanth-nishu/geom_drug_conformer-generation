"""
prepare_qm9.py — Download QM9 dataset and convert to qm9_selfies.jsonl

Downloads the QM9 SDF file from the public AWS S3 repository (deepchem),
parses 3D coordinates + molecular graph topology using RDKit, and writes
the JSONL format expected by mol_prepare.py.

Output format (one JSON per line):
{
  "atom_types":  [6, 6, 8, 1, ...],      # atomic numbers (int)
  "coordinates": [[x,y,z], ...],          # 3D coords in Angstroms (float, DFT)
  "edge_index":  [[0,1,1,2,...],[1,0,2,1,...]], # undirected bond graph (both dirs)
  "bond_types":  [1, 1, 2, ...],          # 1=single,2=double,3=triple,4=aromatic
  "num_atoms":   9                        # total atoms (including H)
}

Filters applied:
  - num_atoms <= MAX_ATOMS (15, covering ~95% of QM9)
  - atom types in [1, 53] (H to I)
  - must have valid 3D coordinates (all non-zero or non-trivial)

Usage:
  cd /scratch/nishanth.r/nextmol_experiment/mol_next_gen
  source venv/bin/activate
  python data/prepare_qm9.py --output data/qm9_selfies.jsonl

  # Smoke test (first 1000 molecules):
  python data/prepare_qm9.py --output data/qm9_selfies.jsonl --max-mols 1000

References:
  Ramakrishnan et al. "Quantum chemistry structures and properties of 134
  kilo molecules" Scientific Data 2014. (QM9 dataset)

  RDKit: Open-source cheminformatics. rdkit.org
"""

import os
import sys
import json
import argparse
import tarfile
import urllib.request
import tempfile
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
QM9_SDF = "gdb9.sdf"
MAX_ATOMS = 29      # QM9 with explicit H: max 29 atoms (9 heavy + ~20 H)
                    # BUG-FIX: was 15, captured only 19.7% of QM9.
                    # Actual distribution: ≤15=19.7%, ≤22=92.9%, ≤29=100%
BOND_TYPE_MAP = {   # RDKit BondType → integer
    "SINGLE":   1,
    "DOUBLE":   2,
    "TRIPLE":   3,
    "AROMATIC": 4,
}

# ── Download ──────────────────────────────────────────────────────────────────

def download_qm9(cache_dir: Path) -> Path:
    """Download and extract QM9 SDF. Returns path to .sdf file."""
    sdf_path = cache_dir / QM9_SDF
    if sdf_path.exists():
        print(f"[prepare_qm9] SDF already exists: {sdf_path}")
        return sdf_path

    tar_path = cache_dir / "gdb9.tar.gz"
    print(f"[prepare_qm9] Downloading QM9 from {QM9_URL} ...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _progress(block, block_size, total):
        downloaded = block * block_size
        if total > 0:
            pct = 100 * downloaded / total
            mb = downloaded / 1e6
            print(f"\r  {pct:.1f}% ({mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(QM9_URL, tar_path, reporthook=_progress)
    print()

    print(f"[prepare_qm9] Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(cache_dir)

    if not sdf_path.exists():
        # Find the SDF inside extracted contents
        sdfs = list(cache_dir.glob("*.sdf"))
        if sdfs:
            sdf_path = sdfs[0]
        else:
            raise FileNotFoundError(f"Could not find .sdf in {cache_dir}")

    print(f"[prepare_qm9] SDF ready: {sdf_path}")
    return sdf_path

# ── Molecule parser ───────────────────────────────────────────────────────────

def mol_to_record(mol) -> dict | None:
    """Convert RDKit Mol with conformer to JSONL record. Returns None if invalid."""
    try:
        from rdkit.Chem import AllChem

        if mol is None:
            return None
        if not mol.GetNumConformers():
            return None

        conf = mol.GetConformer(0)
        n_atoms = mol.GetNumAtoms()

        if n_atoms > MAX_ATOMS or n_atoms < 2:
            return None

        # Atom types (atomic numbers)
        atom_types = []
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z <= 0 or z >= 54:
                return None   # filter out-of-range atoms
            atom_types.append(z)

        # 3D coordinates
        coords = []
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            coords.append([round(pos.x, 6), round(pos.y, 6), round(pos.z, 6)])

        # Sanity check: at least one non-zero coordinate
        if all(c == [0.0, 0.0, 0.0] for c in coords):
            return None

        # Bond graph (undirected → both directions stored)
        src_list, dst_list, bond_type_list = [], [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bt_str = bond.GetBondTypeAsDouble()
            # Map RDKit bond type string to integer
            btype_name = bond.GetBondType().name   # e.g. "SINGLE"
            btype = BOND_TYPE_MAP.get(btype_name, 1)

            # Both directions (undirected graph → directed edges)
            src_list += [i, j]
            dst_list += [j, i]
            bond_type_list += [btype, btype]

        if not src_list:
            return None   # no bonds → skip

        return {
            "atom_types":  atom_types,
            "coordinates": coords,
            "edge_index":  [src_list, dst_list],
            "bond_types":  bond_type_list,
            "num_atoms":   n_atoms,
        }
    except Exception:
        return None

# ── Main ──────────────────────────────────────────────────────────────────────

def prepare(sdf_path: Path, output_path: Path, max_mols: int = -1):
    """Parse SDF and write JSONL. Suppresses RDKit valence errors (expected for
    a handful of QM9 radical species — they are correctly skipped)."""
    from rdkit.Chem import SDMolSupplier
    import rdkit.RDLogger as rl
    rl.DisableLog('rdApp.*')   # suppress "Explicit valence" noise to stderr

    print(f"[prepare_qm9] Parsing {sdf_path} ...")
    # sanitize=False: parse all, then attempt per-molecule sanitization
    # This recovers aromatic molecules that SDMolSupplier would reject with sanitize=True
    supplier = SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_sanitize_fail = 0
    with open(output_path, "w") as fout:
        for i, mol in enumerate(supplier):
            if max_mols > 0 and n_written >= max_mols:
                break
            if mol is None:
                n_skipped += 1
                continue

            # Attempt sanitization — skip molecules that fail (radicals, bad valence)
            try:
                from rdkit.Chem import SanitizeMol
                SanitizeMol(mol)
            except Exception:
                n_sanitize_fail += 1
                n_skipped += 1
                continue

            record = mol_to_record(mol)
            if record is None:
                n_skipped += 1
                continue
            fout.write(json.dumps(record) + "\n")
            n_written += 1
            if n_written % 10000 == 0:
                print(f"  {n_written} molecules written ({i+1} processed, "
                      f"{n_skipped} skipped, {n_sanitize_fail} sanitize-fail)",
                      flush=True)

    print(f"\n[prepare_qm9] Done!")
    print(f"  Written         : {n_written} molecules")
    print(f"  Skipped (total) : {n_skipped} (too large / invalid / no 3D)")
    print(f"  Sanitize fails  : {n_sanitize_fail} (radicals/bad valence — expected)")
    print(f"  Output          : {output_path}")

    # Sanity check on first record
    with open(output_path) as f:
        first = json.loads(f.readline())
    print(f"\n  Sample record:")
    print(f"    num_atoms  : {first['num_atoms']}")
    print(f"    atom_types : {first['atom_types']}")
    print(f"    coords[0]  : {first['coordinates'][0]}")
    print(f"    bond_types : {first['bond_types'][:6]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare QM9 → qm9_selfies.jsonl")
    parser.add_argument("--output", default="data/qm9_selfies.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--cache-dir", default="data/qm9_raw",
                        help="Where to download/cache the raw QM9 SDF")
    parser.add_argument("--max-mols", type=int, default=-1,
                        help="Max molecules to write (-1 = all)")
    parser.add_argument("--sdf", default=None,
                        help="Path to existing gdb9.sdf (skip download)")
    args = parser.parse_args()

    proj_root = Path(__file__).parent.parent
    output_path = proj_root / args.output
    cache_dir = proj_root / args.cache_dir

    if args.sdf:
        sdf_path = Path(args.sdf)
        print(f"[prepare_qm9] Using provided SDF: {sdf_path}")
    else:
        sdf_path = download_qm9(cache_dir)

    if not sdf_path.exists():
        print(f"ERROR: SDF not found at {sdf_path}", file=sys.stderr)
        sys.exit(1)

    prepare(sdf_path, output_path, max_mols=args.max_mols)
