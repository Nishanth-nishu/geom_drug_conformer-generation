#!/usr/bin/env python3
"""
prepare_geom_drugs_standard.py — build the STANDARD, literature-comparable
GEOM-Drugs train/val/test split, replacing the ad-hoc random 90/10 split this
project used before.

WHAT THIS REPLICATES AND WHY (see docs/GEOM_DRUGS_DIAGNOSIS_AND_PLAN.md):

1. TRAIN/VAL/TEST MEMBERSHIP: GeoMol's (Ganea et al., NeurIPS 2021) canonical
   `data/DRUGS/smiles_splits/split0.npy`, git-committed in their official repo
   (github.com/PattanaikL/GeoMol). This exact split (243,473 / 30,433 / 30,433
   molecules) is the one TorsionalDiffusion and most follow-up papers adopted,
   so using it makes results comparable to that lineage -- not a from-scratch
   approximation. Verified against our own downloaded rdkit_folder.tar.gz: the
   split's 304,339 total SMILES exactly match our tar's 304,339 molecule count,
   confirming it's built from the same underlying raw GEOM release.

   HONEST CAVEAT: GeoMol's and TorDiff's
   own training code actually load the INDEX-based `splits/split0.npy`
   (integer positions into a sorted directory listing), not the SMILES-keyed
   `smiles_splits/split0.npy` this script uses. The SMILES-keyed file is real,
   disjoint across train/val/test, and exactly the right size -- but we have
   NOT independently verified the two files encode byte-identical molecule
   membership, only that they match in size and provenance. Claim scope:
   GeoMol-sourced, disjoint, correctly-sized -- not verified byte-identical to
   what GeoMol's own index-based loader would actually produce.

   Encoding note (verified against real data, not assumed): GeoMol's SMILES
   strings substitute '/' -> '_' (matching the filesystem-safe filenames RDKit's
   own GEOM distribution uses inside the tar). We reverse this by comparing our
   pickle's real `data['smiles']` (with '/' intact) against the split lists via
   the same '/' -> '_' encoding, not by guessing.

   HONEST CAVEAT: We do NOT attempt to reproduce ConfGF/GeoDiff's own from-
   scratch random split (seed=2021 over their `summary_drugs.json` iteration
   order) -- that split's exact molecule membership depends on the precise
   ordering of a file we don't have, so any reconstruction would only be an
   approximation dressed up as exact reproduction. GeoMol's split is used
   instead specifically because it's exactly, verifiably reproducible from
   files in this session, not because it's claimed to be identical to GeoDiff's.

2. CONFORMER SELECTION: TorsionalDiffusion's (Jing et al., NeurIPS 2022)
   `standardize_confs.py` conventions, with two labeled deviations:
     - `clean_confs`: drop any conformer whose RDKit-inferred connectivity
       (recomputed from the 3D structure, H-stripped, non-isomeric canonical
       SMILES) doesn't match the input graph's canonical SMILES. This is the
       connectivity-consistency QC step flagged as commonly skipped by the
       "GEOM-Drugs revisited" (2025) audit paper cited in the diagnosis doc.
       DEVIATION: a try/except is added around each per-conformer check
       below (TorDiff's original has no such guard) -- a deliberate robustness
       improvement for malformed records, not a verbatim port of their code.
     - Keep the top `--confs-per-mol` (default 30, TorDiff's own default) by
       Boltzmann weight. DEVIATION: this Boltzmann-weight sort is
       TorDiff's OPT-IN `--boltzmann top` policy, not its default --
       `standardize_confs.py --boltzmann` itself defaults to `None`, and
       TorDiff's own documented example invocation doesn't pass `top`.

Usage:
    python3 prepare_geom_drugs_standard.py \\
        --tar rdkit_folder.tar.gz \\
        --geomol-split GeoMol/data/DRUGS/smiles_splits/split0.npy \\
        --out-dir data_standard/ \\
        --confs-per-mol 30 \\
        --test-1k-seed 42
"""

import argparse
import contextlib
import json
import pickle
import random
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

BOND_TYPE_MAP = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def load_geomol_split(split_path):
    """Returns dict: encoded_smiles ('/' -> '_') -> 'train'|'val'|'test'."""
    arr = np.load(split_path, allow_pickle=True)
    train, val, test = arr
    membership = {}
    for smi in train:
        membership[smi] = 'train'
    for smi in val:
        membership[smi] = 'val'
    for smi in test:
        membership[smi] = 'test'
    return membership


def clean_confs(smi, confs, limit=None):
    """TorsionalDiffusion's connectivity-consistency filter
    (utils/standardization.py:clean_confs) -- keeps only conformers whose
    RDKit-recomputed connectivity matches the intended 2D graph. DEVIATION:
    wraps each per-conformer check in
    try/except, which TorDiff's original does not do -- a deliberate
    robustness improvement, not a verbatim port of their code."""
    good_ids = []
    try:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    except Exception:
        return []
    for i, c in enumerate(confs):
        try:
            conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c['rd_mol'], sanitize=False),
                                         isomericSmiles=False)
        except Exception:
            continue
        if conf_smi == canonical:
            good_ids.append(i)
        if limit is not None and len(good_ids) == limit:
            break
    return [confs[i] for i in good_ids]


def mol_to_graph(rd_mol):
    atom_types = [atom.GetAtomicNum() for atom in rd_mol.GetAtoms()]
    src, dst, bond_types = [], [], []
    for bond in rd_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), 1)
        src += [i, j]
        dst += [j, i]
        bond_types += [bt, bt]
    return atom_types, src, dst, bond_types


def process_molecule(data, confs_per_mol):
    smiles = data.get('smiles', '')
    confs = data.get('conformers', [])
    if not confs:
        return None

    # Sort by Boltzmann weight descending (most probable first). DEVIATION:
    # this reproduces TorDiff's OPT-IN
    # `--boltzmann top` policy, not its default (standardize_confs.py
    # --boltzmann itself defaults to None). Then apply connectivity QC.
    confs_sorted = sorted(confs, key=lambda c: -c.get('boltzmannweight', 0.0))
    clean = clean_confs(smiles, confs_sorted, limit=confs_per_mol)
    if not clean:
        return None

    first_mol = clean[0].get('rd_mol')
    if first_mol is None or first_mol.GetNumConformers() == 0:
        return None
    n_atoms = first_mol.GetNumAtoms()
    if n_atoms < 3:
        return None
    if any(a.GetAtomicNum() <= 0 or a.GetAtomicNum() >= 119 for a in first_mol.GetAtoms()):
        return None

    atom_types, src, dst, bond_types = mol_to_graph(first_mol)
    if not src:
        return None

    out_confs = []
    for c in clean:
        mol = c.get('rd_mol')
        if mol is None or mol.GetNumConformers() == 0 or mol.GetNumAtoms() != n_atoms:
            continue
        pos = mol.GetConformer(0).GetPositions().tolist()
        out_confs.append({
            'coordinates': pos,
            'energy_hartree': float(c.get('totalenergy', 0.0)),
            'boltzmann_weight': float(c.get('boltzmannweight', 0.0)),
        })
    if not out_confs:
        return None

    return {
        'num_atoms': n_atoms,
        'num_conformers': len(out_confs),
        'atom_types': atom_types,
        'edge_index': [src, dst],
        'bond_types': bond_types,
        'conformers': out_confs,
        'smiles': smiles,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tar', required=True)
    p.add_argument('--geomol-split', required=True,
                    help='Path to GeoMol data/DRUGS/smiles_splits/split0.npy')
    p.add_argument('--out-dir', default='data_standard')
    p.add_argument('--confs-per-mol', type=int, default=30,
                    help='TorsionalDiffusion default (standardize_confs.py --confs_per_mol)')
    p.add_argument('--test-1k-seed', type=int, default=42,
                    help='Seed for the fixed-size 1000-molecule test subsample '
                         '(matches common practice of evaluating on ~1k molecules '
                         'for tractable DDIM-sampling-based eval; NOT claimed to '
                         'match any specific paper\'s exact 1k pickle byte-for-byte)')
    p.add_argument('--test-1k-size', type=int, default=1000)
    p.add_argument('--subdir', default='drugs')
    p.add_argument('--max-mols-scan', type=int, default=-1,
                    help='Stop after scanning this many tar entries (for quick testing)')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GeoMol canonical split from {args.geomol_split} ...")
    membership = load_geomol_split(args.geomol_split)
    print(f"  train={sum(v=='train' for v in membership.values()):,}  "
          f"val={sum(v=='val' for v in membership.values()):,}  "
          f"test={sum(v=='test' for v in membership.values()):,}")

    counts = {'train': 0, 'val': 0, 'test': 0}
    test_smiles_for_1k = []

    n_seen, n_matched, n_no_match, n_qc_failed = 0, 0, 0, 0
    t0 = time.time()

    print(f"\nProcessing {args.tar} (single pass) ...")
    # FIX: plain open()+close-after-loop left
    # the 3 output files unflushed/unclosed if an exception escaped mid-loop.
    # ExitStack guarantees all three are closed on any exit path.
    with contextlib.ExitStack() as stack:
        out_files = {
            'train': stack.enter_context(open(out_dir / 'geom_drugs_train.jsonl', 'w')),
            'val':   stack.enter_context(open(out_dir / 'geom_drugs_val.jsonl', 'w')),
            'test':  stack.enter_context(open(out_dir / 'geom_drugs_test.jsonl', 'w')),
        }

        with tarfile.open(args.tar, 'r') as tar:
            for member in tar:
                if args.max_mols_scan > 0 and n_seen >= args.max_mols_scan:
                    break
                if not member.isfile() or not member.name.endswith('.pickle'):
                    continue
                if f'/{args.subdir}/' not in member.name and not member.name.startswith(f'{args.subdir}/'):
                    continue

                n_seen += 1
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    data = pickle.loads(f.read())
                except Exception:
                    continue

                raw_smiles = data.get('smiles', '')
                encoded = raw_smiles.replace('/', '_')
                split = membership.get(encoded)
                if split is None:
                    n_no_match += 1
                    continue
                n_matched += 1

                # FIX: process_molecule() itself was previously unguarded --
                # a malformed record (e.g. an RDKit sanitization exception
                # inside clean_confs/mol_to_graph) could crash the whole
                # multi-hour run instead of just being skipped and counted.
                try:
                    record = process_molecule(data, args.confs_per_mol)
                except Exception as e:
                    n_qc_failed += 1
                    print(f"  [skip] {member.name}: process_molecule raised {e}",
                          file=sys.stderr)
                    continue
                if record is None:
                    n_qc_failed += 1
                    continue

                out_files[split].write(json.dumps(record) + '\n')
                counts[split] += 1
                if split == 'test':
                    test_smiles_for_1k.append(record['smiles'])

                if n_seen % 10000 == 0:
                    elapsed = time.time() - t0
                    print(f"  seen={n_seen:,}  matched={n_matched:,}  "
                          f"train={counts['train']:,} val={counts['val']:,} test={counts['test']:,}  "
                          f"qc_failed={n_qc_failed:,}  elapsed={elapsed/60:.1f}min", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Seen (.pickle under {args.subdir}/) : {n_seen:,}")
    print(f"  Matched to GeoMol split             : {n_matched:,}")
    print(f"  No split match (unexpected)         : {n_no_match:,}")
    print(f"  Failed connectivity QC / no confs   : {n_qc_failed:,}")
    print(f"  Written -> train={counts['train']:,}  val={counts['val']:,}  test={counts['test']:,}")

    # Fixed-seed 1k test subsample for tractable generation-based eval.
    # FIX: selection was previously by SMILES-
    # string set membership with no uniqueness check -- if two test-split
    # records shared a SMILES (e.g. distinct stereoisomers collapsed by the
    # non-isomeric canonical SMILES used elsewhere in this script), the
    # membership test could silently pull in MORE than test_1k_size records.
    # Now: detect and warn about duplicate SMILES in the source pool, and
    # enforce a hard cap so output can never exceed the requested size
    # regardless of duplicates.
    print(f"\nBuilding {args.test_1k_size}-molecule test subsample (seed={args.test_1k_seed}) ...")
    rng = random.Random(args.test_1k_seed)
    pool = list(test_smiles_for_1k)
    n_unique = len(set(pool))
    if n_unique < len(pool):
        print(f"  [WARNING] {len(pool) - n_unique:,} duplicate SMILES found in the "
              f"test-split pool ({len(pool):,} records, {n_unique:,} unique) -- "
              f"set-membership subsampling could otherwise silently pull in more "
              f"than {args.test_1k_size} records for a single repeated SMILES.")
    rng.shuffle(pool)
    subset = set(pool[:args.test_1k_size])
    with open(out_dir / 'geom_drugs_test.jsonl') as fin, \
         open(out_dir / f'geom_drugs_test_{args.test_1k_size}.jsonl', 'w') as fout:
        n_out = 0
        for line in fin:
            if n_out >= args.test_1k_size:
                break
            rec = json.loads(line)
            if rec['smiles'] in subset:
                fout.write(line)
                n_out += 1
    print(f"  Written {n_out:,} molecules -> geom_drugs_test_{args.test_1k_size}.jsonl")

    if n_no_match > 0:
        print(f"\n[WARNING] {n_no_match:,} molecules in the tar had no match in the "
              f"GeoMol split -- investigate before trusting these numbers (expected "
              f"to be 0 or near-0 given the verified exact count match).")


if __name__ == '__main__':
    main()
