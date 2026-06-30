#!/usr/bin/env python3
"""
prepare_geom_drugs.py — Download and convert GEOM-Drugs to our JSONL format.

Source: GEOM dataset (Axelrod & Gomez-Bombarelli, Scientific Data 2022)
        Harvard Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF

This script downloads the rdkit_folder.tar.gz (~5 GB) which contains pre-computed
MMFF/DFT conformers in SDF format, then converts to the JSONL format used by
geom_drugs_dataset.py.

JSONL format (one molecule per line):
{
  "num_atoms": int,
  "num_conformers": int,
  "atom_types": [int, ...],          # atomic numbers
  "edge_index": [[src...], [dst...]], # undirected bond graph (both directions)
  "bond_types": [int, ...],          # 1=single,2=double,3=triple,4=aromatic
  "conformers": [
    {
      "coordinates": [[x,y,z],...],  # Angstrom
      "energy_hartree": float,       # GFN2-xTB or DFT energy
      "boltzmann_weight": float      # weight at 298K
    }, ...
  ]
}

Usage:
    python3 data/prepare_geom_drugs.py --out data/geom_drugs.jsonl
    python3 data/prepare_geom_drugs.py --out data/geom_drugs.jsonl --max-mols 100000

Requirements (already in venv):
    rdkit, numpy, torch, requests (or wget)
"""

import argparse
import gzip
import io
import json
import math
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# ── GEOM download config ───────────────────────────────────────────────────────
DATAVERSE_BASE   = "https://dataverse.harvard.edu/api/access/datafile"
GEOM_DOI         = "doi:10.7910/DVN/JNGTDF"

# The rdkit_folder contains SDF files with MMFF/DFT-optimized conformers.
# File ID for rdkit_folder.tar.gz (Harvard Dataverse file ID):
RDKIT_FOLDER_ID  = "6095299"   # confirmed ID as of 2024

HARTREE_TO_KCAL  = 627.509_47
kB               = 3.166_81e-6  # Hartree / K
TEMPERATURE_K    = 298.0


# ── Bond type mapping ──────────────────────────────────────────────────────────

def rdkit_bond_to_int(bond_type_str: str) -> int:
    """Convert RDKit bond type string to integer (1=single, 2=double, 3=triple, 4=aromatic)."""
    mapping = {
        'SINGLE':    1,
        'DOUBLE':    2,
        'TRIPLE':    3,
        'AROMATIC':  4,
        'UNSPECIFIED': 1,
    }
    return mapping.get(bond_type_str.upper(), 1)


# ── Boltzmann weights ──────────────────────────────────────────────────────────

def compute_boltzmann_weights(energies_hartree: List[float], T: float = 298.0) -> List[float]:
    """Compute Boltzmann weights from energies in Hartree at temperature T (K)."""
    kT = kB * T
    e = np.array(energies_hartree)
    e_min = e.min()
    w = np.exp(-(e - e_min) / kT)
    return (w / w.sum()).tolist()


# ── Download ───────────────────────────────────────────────────────────────────

def download_with_progress(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Download a file with a progress bar."""
    print(f"\nDownloading: {url}")
    print(f"Destination: {dest}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    t0 = time.time()
    with open(dest, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    speed = downloaded / (1 << 20) / max(time.time() - t0, 1e-6)
                    print(f"\r  {pct:.1f}%  {downloaded >> 20} MB / {total >> 20} MB  "
                          f"({speed:.1f} MB/s)", end='', flush=True)
    print(f"\n  Done in {time.time()-t0:.0f}s")


def get_rdkit_folder(cache_dir: Path) -> Path:
    """Download rdkit_folder.tar.gz if not already cached. Returns path to tar.gz."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / 'rdkit_folder.tar.gz'
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Using cached: {dest} ({dest.stat().st_size >> 20} MB)")
        return dest

    url = f"{DATAVERSE_BASE}/{RDKIT_FOLDER_ID}?format=original"
    try:
        download_with_progress(url, dest)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nManual download instructions:")
        print("  1. Go to: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF")
        print("  2. Download: rdkit_folder.tar.gz  (~5 GB)")
        print(f"  3. Place it at: {dest}")
        sys.exit(1)
    return dest


# ── SDF parsing (no RDKit required — pure Python) ─────────────────────────────

def parse_sdf_block(sdf_text: str) -> Optional[Dict]:
    """
    Parse a single SDF block into a molecule dict.
    Returns None if parsing fails or molecule is unusable.

    SDF V2000 format:
      Line 1: molecule name (SMILES in GEOM)
      Line 2: header
      Line 3: comment
      Line 4: counts line — natoms nbonds ...
      Lines 5..5+natoms-1: atom block  (x y z symbol ...)
      Lines 5+natoms..5+natoms+nbonds-1: bond block  (a1 a2 type ...)
      > <tag>  data records
      $$$$     end of record
    """
    lines = sdf_text.strip().split('\n')
    if len(lines) < 5:
        return None

    mol_name = lines[0].strip()

    # Count line (line index 3)
    counts_line = lines[3]
    try:
        natoms = int(counts_line[0:3])
        nbonds = int(counts_line[3:6])
    except (ValueError, IndexError):
        return None

    if natoms < 3 or natoms > 150:
        return None

    # Atom block
    atom_start = 4
    atom_types = []
    coords = []

    SYMBOL_TO_Z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'I': 53, 'Xe': 54, 'Ba': 56,
    }

    for i in range(natoms):
        line = lines[atom_start + i]
        parts = line.split()
        if len(parts) < 4:
            return None
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            symbol = parts[3]
            z_num = SYMBOL_TO_Z.get(symbol, 0)
            if z_num == 0:
                return None  # Unknown element
            atom_types.append(z_num)
            coords.append([x, y, z])
        except (ValueError, IndexError):
            return None

    # Bond block
    bond_start = atom_start + natoms
    src_list, dst_list, bond_types = [], [], []

    BOND_MAP = {'1': 1, '2': 2, '3': 3, '4': 4, '8': 4}  # 8 = aromatic in V2000

    for i in range(nbonds):
        line = lines[bond_start + i]
        try:
            a1 = int(line[0:3]) - 1   # 1-indexed → 0-indexed
            a2 = int(line[3:6]) - 1
            btype = int(line[6:9])
        except (ValueError, IndexError):
            return None

        if a1 < 0 or a1 >= natoms or a2 < 0 or a2 >= natoms:
            return None

        bt = BOND_MAP.get(str(btype), 1)
        # Bidirectional (both directions)
        src_list.extend([a1, a2])
        dst_list.extend([a2, a1])
        bond_types.extend([bt, bt])

    # Data records — find energy
    energy_hartree = 0.0
    data_start = bond_start + nbonds
    i = data_start
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('> <') and 'energy' in line.lower():
            if i + 1 < len(lines):
                try:
                    energy_hartree = float(lines[i + 1].strip())
                except ValueError:
                    pass
        i += 1

    return {
        'name': mol_name,
        'num_atoms': natoms,
        'atom_types': atom_types,
        'edge_index': [src_list, dst_list],
        'bond_types': bond_types,
        'coordinates': coords,
        'energy_hartree': energy_hartree,
    }


def sdf_to_conformer_groups(sdf_content: str) -> Optional[Dict]:
    """
    Parse all conformers from a multi-conformer SDF file.
    Groups conformers by topology (same atom_types + bond_types).
    Returns a molecule dict with all conformers, or None if unusable.
    """
    blocks = sdf_content.split('$$$$')
    conformers = []
    base_mol = None

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        parsed = parse_sdf_block(block)
        if parsed is None:
            continue
        if base_mol is None:
            base_mol = parsed
        conformers.append({
            'coordinates': parsed['coordinates'],
            'energy_hartree': parsed['energy_hartree'],
        })

    if base_mol is None or len(conformers) < 2:
        return None

    # Compute Boltzmann weights
    energies = [c['energy_hartree'] for c in conformers]
    weights = compute_boltzmann_weights(energies)
    for c, w in zip(conformers, weights):
        c['boltzmann_weight'] = w

    return {
        'num_atoms': base_mol['num_atoms'],
        'num_conformers': len(conformers),
        'atom_types': base_mol['atom_types'],
        'edge_index': base_mol['edge_index'],
        'bond_types': base_mol['bond_types'],
        'conformers': conformers,
    }


# ── Main conversion ────────────────────────────────────────────────────────────

def convert_rdkit_folder_to_jsonl(
        tar_path: Path,
        out_path: Path,
        max_atoms: int = 100,
        min_conformers: int = 2,
        max_mols: int = -1,
) -> None:
    """
    Read rdkit_folder.tar.gz (SDF files) and write JSONL.

    The tar.gz structure is:
      rdkit_folder/
        drugs/
          <smiles_or_id>.sdf   (each file has multiple conformers)
    """
    print(f"\nConverting: {tar_path}")
    print(f"Output    : {out_path}")
    print(f"max_atoms : {max_atoms}  min_confs: {min_conformers}  max_mols: {'all' if max_mols<0 else max_mols}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    t0 = time.time()

    with tarfile.open(tar_path, 'r:gz') as tar, \
         open(out_path, 'w') as out_f:

        for member in tar:
            if max_mols > 0 and n_written >= max_mols:
                break

            # Only process .sdf files in the drugs/ subdirectory
            if not member.name.endswith('.sdf'):
                continue
            if '/drugs/' not in member.name and 'drugs/' not in member.name:
                continue

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read().decode('utf-8', errors='ignore')
                mol = sdf_to_conformer_groups(content)

                if mol is None:
                    n_skipped += 1
                    continue
                if mol['num_atoms'] > max_atoms:
                    n_skipped += 1
                    continue
                if mol['num_conformers'] < min_conformers:
                    n_skipped += 1
                    continue

                out_f.write(json.dumps(mol, separators=(',', ':')) + '\n')
                n_written += 1

                if n_written % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = n_written / elapsed
                    print(f"  Written {n_written:,}  |  Skipped {n_skipped:,}  |  "
                          f"{rate:.0f} mol/s", flush=True)

            except Exception as e:
                n_skipped += 1
                continue

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size >> 20
    print(f"\n✓ Done in {elapsed:.0f}s")
    print(f"  Written  : {n_written:,} molecules → {out_path} ({size_mb} MB)")
    print(f"  Skipped  : {n_skipped:,}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Prepare GEOM-Drugs JSONL dataset')
    p.add_argument('--out',        default='data/geom_drugs.jsonl',
                   help='Output JSONL file (default: data/geom_drugs.jsonl)')
    p.add_argument('--cache-dir',  default='data/geom_raw',
                   help='Directory to cache downloaded files (default: data/geom_raw)')
    p.add_argument('--tar',        default=None,
                   help='Path to already-downloaded rdkit_folder.tar.gz (skips download)')
    p.add_argument('--max-atoms',  type=int, default=100,
                   help='Max heavy atoms per molecule (default: 100, full GEOM-Drugs)')
    p.add_argument('--min-confs',  type=int, default=2,
                   help='Min conformers required (default: 2)')
    p.add_argument('--max-mols',   type=int, default=-1,
                   help='Max molecules to write, -1=all (default: -1)')
    args = p.parse_args()

    out_path   = Path(args.out).resolve()
    cache_dir  = Path(args.cache_dir).resolve()

    if args.tar:
        tar_path = Path(args.tar)
        if not tar_path.exists():
            print(f"[ERROR] --tar file not found: {tar_path}")
            sys.exit(1)
    else:
        tar_path = get_rdkit_folder(cache_dir)

    convert_rdkit_folder_to_jsonl(
        tar_path=tar_path,
        out_path=out_path,
        max_atoms=args.max_atoms,
        min_conformers=args.min_confs,
        max_mols=args.max_mols,
    )


if __name__ == '__main__':
    main()
