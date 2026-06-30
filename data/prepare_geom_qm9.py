#!/usr/bin/env python3
"""
prepare_geom_qm9.py — Download and parse GEOM-QM9 dataset.

Unlike the 50GB GEOM-Drugs dataset, the QM9 dataset is much smaller (~1 GB)
and is distributed efficiently as a MessagePack tarball. 
Because the msgpack file maps SMILES strings to 3D conformers, we can use 
RDKit to extract the exact 2D topology from the SMILES string and then pair 
it with the DFT-optimized 3D coordinates.

Usage:
    python3 data/prepare_geom_qm9.py
"""

import os
import io
import time
import json
import tarfile
import argparse
import requests
from pathlib import Path

import msgpack
import numpy as np
from rdkit import Chem

# Dataverse file ID for qm9_crude.msgpack.tar.gz
QM9_FILE_ID = "4327190"
DATAVERSE_URL = f"https://dataverse.harvard.edu/api/access/datafile/{QM9_FILE_ID}"

kB = 3.166_81e-6  # Hartree / K
TEMPERATURE_K = 298.0

def compute_boltzmann_weights(energies_hartree, T=298.0):
    kT = kB * T
    e = np.array(energies_hartree)
    e_min = e.min()
    w = np.exp(-(e - e_min) / kT)
    return (w / w.sum()).tolist()

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"File {dest} already exists. Skipping download.")
        return
    
    print(f"Downloading GEOM-QM9 from {url} (~1 GB)")
    print(f"To: {dest}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    t0 = time.time()
    
    with open(dest, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            if not chunk: continue
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed = downloaded / elapsed / 1e6
            if total > 0:
                pct = downloaded / total * 100
                print(f"\rDownloading: {pct:.1f}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB) at {speed:.1f} MB/s", end="")
            else:
                print(f"\rDownloaded {downloaded/1e6:.1f} MB at {speed:.1f} MB/s", end="")
    print("\nDownload complete!")

def build_topology_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
    
    mol = Chem.AddHs(mol)
    
    num_atoms = mol.GetNumAtoms()
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    
    src_list, dst_list, bond_types = [], [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        b_type = int(bond.GetBondTypeAsDouble())
        if bond.GetIsAromatic():
            b_type = 4
        
        # Bidirectional graph
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_types.extend([b_type, b_type])
        
    return {
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'edge_index': [src_list, dst_list],
        'bond_types': bond_types
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/geom_qm9.jsonl')
    p.add_argument('--cache', default='data/geom_raw/qm9_crude.msgpack.tar.gz')
    p.add_argument('--min-confs', type=int, default=2)
    args = p.parse_args()

    tar_path = Path(args.cache).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(DATAVERSE_URL, tar_path)

    print(f"Parsing {tar_path} and converting to {out_path} ...")
    n_written = 0
    n_skipped = 0
    t0 = time.time()
    
    with tarfile.open(tar_path, 'r:gz') as tar, open(out_path, 'w') as out_f:
        for member in tar:
            if not member.name.endswith('.msgpack'):
                continue
            
            f = tar.extractfile(member)
            if f is None:
                continue
            
            unpacker = msgpack.Unpacker(f, raw=False)
            for mol_dict in unpacker:
                for smiles, mol_data in mol_dict.items():
                    if 'conformers' not in mol_data:
                        n_skipped += 1
                        continue
                    
                    confs = mol_data['conformers']
                    if len(confs) < args.min_confs:
                        n_skipped += 1
                        continue
                        
                    topo = build_topology_from_smiles(smiles)
                    if topo is None:
                        n_skipped += 1
                        continue
                        
                    conformers_list = []
                    energies = []
                    for c in confs:
                        geom = c.get('geom')
                        e = c.get('totalenergy')
                        if geom is None or e is None:
                            continue
                        
                        # msgpack geom is flat list [x1, y1, z1, x2, y2, z2...]
                        coords = []
                        for i in range(0, len(geom), 3):
                            coords.append([geom[i], geom[i+1], geom[i+2]])
                            
                        if len(coords) != topo['num_atoms']:
                            continue
                            
                        conformers_list.append({
                            'coordinates': coords,
                            'energy_hartree': e
                        })
                        energies.append(e)
                    
                    if len(conformers_list) < args.min_confs:
                        n_skipped += 1
                        continue
                        
                    weights = compute_boltzmann_weights(energies)
                    for i in range(len(conformers_list)):
                        conformers_list[i]['boltzmann_weight'] = weights[i]
                        
                    topo['num_conformers'] = len(conformers_list)
                    topo['conformers'] = conformers_list
                    
                    out_f.write(json.dumps(topo, separators=(',', ':')) + '\n')
                    n_written += 1
                    
                    if n_written % 5000 == 0:
                        elapsed = time.time() - t0
                        rate = n_written / elapsed
                        print(f"  Written {n_written:,} | Skipped {n_skipped:,} | Rate: {rate:.0f} mol/s")

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n✓ Done in {elapsed:.0f}s")
    print(f"  Written: {n_written:,} molecules -> {out_path} ({size_mb:.1f} MB)")
    print(f"  Skipped: {n_skipped:,}")

if __name__ == '__main__':
    main()
