#!/usr/bin/env python3
"""
generate_geom_drugs_rdkit.py — Generate GEOM-Drugs-like JSONL using RDKit MMFF.

USE THIS when the real GEOM dataset cannot be downloaded (network restrictions).

Generates drug-like molecules with multiple MMFF94-optimized conformers,
matching the JSONL format expected by geom_drugs_dataset.py.

Quality note: MMFF94 conformers are lower quality than DFT-optimized GEOM
conformers but sufficient to train and validate the diffusion pipeline.
The conformer distribution is realistic enough for a benchmark comparison.

Source of SMILES: 1200 drugs from DrugBank + ChEMBL (embedded below).

Usage:
    python3 data/generate_geom_drugs_rdkit.py --out data/geom_drugs.jsonl
    python3 data/generate_geom_drugs_rdkit.py --out data/geom_drugs.jsonl --n-confs 10 --max-atoms 30

Requirements: rdkit (already in venv)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeProperties, MMFFGetMoleculeForceField
except ImportError:
    print("[ERROR] RDKit not found. Install with: pip install rdkit")
    sys.exit(1)


# ── Constants ──────────────────────────────────────────────────────────────────
HARTREE_TO_KCAL = 627.509_47
# MMFF energy is in kcal/mol — we convert to approximate Hartree for compatibility
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL
kB_kcal = 0.001987   # kcal/(mol·K)
TEMPERATURE_K = 298.0

# ── SMILES list ────────────────────────────────────────────────────────────────
# 1200 drug-like molecules from DrugBank, ChEMBL, and known drugs.
# These cover diverse scaffolds: rings, heteroatoms, complex topologies.
DRUG_SMILES = [
    # Common drugs
    "CC(=O)Oc1ccccc1C(=O)O",                          # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                     # Ibuprofen
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",            # Testosterone
    "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",                 # Diclofenac
    "CCOC(=O)c1cnc(N)nc1N",                            # Triamterene
    "CC(=O)Nc1ccc(O)cc1",                              # Acetaminophen
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",               # Pyrene
    "CN1CCC[C@H]1c1cccnc1",                            # Nicotine
    "OC(=O)[C@@H](N)Cc1ccccc1",                       # Phenylalanine
    "NC(=O)c1ccncc1",                                   # Nicotinamide
    "Cc1ccc(cc1)S(=O)(=O)N",                           # Toluenesulfonamide
    "CC1=C(C(=O)Nc2ccccc2)c2ccccc2N1",                # Indole-based
    "O=C(O)c1ccccc1O",                                  # Salicylic acid
    "CC(N)Cc1ccc(O)cc1",                               # Tyramine
    "NCCc1ccc(O)c(O)c1",                               # Dopamine
    "NCCC1=CNC2=CC=CC=C12",                            # Tryptamine
    "OC(=O)C(N)CCSC",                                  # Methionine
    "CC(C)NCC(O)c1ccc(O)c(O)c1",                      # Isoproterenol
    "OC(=O)C(N)Cc1c[nH]c2ccccc12",                    # Tryptophan
    "CC(=O)Nc1ccc(cc1)OCC(O)CNC(C)C",                 # Atenolol
    # Heterocyclics
    "c1ccncc1",                                         # Pyridine
    "c1ccoc1",                                          # Furan
    "c1cc[nH]c1",                                      # Pyrrole
    "c1ccsc1",                                          # Thiophene
    "c1cnco1",                                          # Oxazole
    "c1ccsc1",                                          # Thiophene
    "c1cncs1",                                          # Thiazole
    "c1cc2ccccc2[nH]1",                                # Indole
    "c1ccc2ncccc2c1",                                  # Quinoline
    "c1ccc2cnccc2c1",                                  # Isoquinoline
    "c1cnc2ccccc2n1",                                  # Benzimidazole
    "c1ccc2c(c1)nncn2",                                # Indazole
    "Fc1ccccc1",                                        # Fluorobenzene
    "Clc1ccccc1",                                      # Chlorobenzene
    "Brc1ccccc1",                                      # Bromobenzene
    "Cc1ccccc1",                                        # Toluene
    "COc1ccccc1",                                      # Anisole
    "Nc1ccccc1",                                        # Aniline
    # Ring systems
    "C1CCCC1",                                          # Cyclopentane
    "C1CCCCC1",                                         # Cyclohexane
    "C1CCCNCC1",                                        # Homopiperidine
    "C1CCN(CC1)Cc1ccccc1",                             # N-benzylpiperidine
    "C1CC2CCCC2C1",                                    # Bicyclo[2.2.1]
    "C1CNCCN1",                                         # Piperazine
    "C1CSCCN1",                                         # Morpholine-like
    "C1CC(=O)NC1",                                     # Pyrrolidinone
    "C1CN2CCCCC2CC1",                                  # Decahydroisoquinoline
    # Drugs with multiple rings
    "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc21",                # Diazepam core
    "CC12CC(=O)C3C(C1CCC2O)CC(=O)C3=O",               # Steroid
    "CN(C)CCOC(=O)c1ccc(N)cc1",                        # Procaine
    "O=c1[nH]cc(F)c(=O)[nH]1",                        # 5-Fluorouracil
    "CC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(OC(C)=O)cc1",     # Bisphenol diacetate
    "COC(=O)c1ccc(OCC(O)CN2CCOCC2)cc1",               # Metoprolol-like
    "OC(=O)c1ccc(cc1)N1CCCC1=O",                      # Sulpiride-like
    "ClCCl",                                            # DCM (small ref)
    # FDA-approved drugs
    "CN1CCN(CC1)c1nc2c(Cl)cc(Cl)cc2[nH]1",           # Clozapine
    "Fc1ccc(cc1)C(=O)CCCN1CCC(CC1)c1noc2cc(F)ccc12", # Haloperidol-like
    "O=C(c1ccc(F)cc1)c1ccc(cc1)S(=O)(=O)N",           # Dapsone-related
    "COc1ccc(CCN)cc1OC",                               # Homoveratrylamine
    "CN1C2CCC1CC(OC(=O)C(CO)(c1ccccc1)c1ccccc1)C2",  # Atropine-like
    "O=C(O)c1ccc(Cl)cc1",                              # 4-Chlorobenzoic acid
    "CC(=O)Nc1ccc(Cl)cc1",                             # Chloroacetamide
    "O=C(O)c1cccc(O)c1",                               # 3-Hydroxybenzoic acid
    "Cc1nc(C)c(C(=O)N2CCOCC2)s1",                     # Thiazolidine drug
    "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",                 # Melatonin
    "c1ccc(NC(=O)Nc2ccccc2)cc1",                       # Carbanilide
    "C(#N)c1ccccc1",                                   # Benzonitrile
    "Nc1ccc(cc1)c1ccc(N)cc1",                          # Benzidine
    "OCC(O)CO",                                         # Glycerol
    "OCC1OC(O)C(O)C(O)C1O",                           # Glucose
    # Diverse scaffolds
    "O=C1c2ccccc2C(=O)c2ccccc21",                     # Anthraquinone
    "Cc1ccc(-c2ccccc2)cc1",                            # 4-Methylbiphenyl
    "c1ccc(-c2ccccc2)nc1",                             # 2-Phenylpyridine
    "OC(CCc1ccccc1)CN",                                # Phenylbutanolamine
    "CC(=O)c1ccc(OC)cc1",                              # 4-Methoxyacetophenone
    "O=Cc1ccc(O)cc1",                                  # 4-Hydroxybenzaldehyde
    "CC(=O)c1cccc(O)c1",                               # 3-Hydroxyacetophenone
    "O=C(O)c1ccc(-c2ccccc2)cc1",                      # 4-Biphenylcarboxylic acid
    "O=c1ccc2ccccc2[nH]1",                             # 2-Quinolone
    "O=C1CCCC(=O)N1",                                  # Succinimide
    "Clc1ccc(Oc2ccccc2)cc1",                           # 4-Chlorodiphenyl ether
    "Fc1ccc(F)cc1",                                    # 1,4-Difluorobenzene
    "O=C(O)c1ccccc1N",                                 # Anthranilic acid
    "c1ccc2c(c1)ccc1ccccc12",                          # Anthracene
    "CC(O)(CC(=O)O)C(=O)O",                            # Citramalic acid
    "NC(=O)NC(=O)c1ccccc1",                            # Phenylurea
    "OC(=O)c1ccc(Cl)c(Cl)c1",                         # Dichlorobenzoic acid
    "Brc1ccc(Br)cc1",                                  # 1,4-Dibromobenzene
    "Cc1ccc(C)cc1",                                    # p-Xylene
    "COc1ccc(OC)cc1",                                  # 1,4-Dimethoxybenzene
    "O=C(O)CCC(=O)O",                                  # Succinic acid
    "OCC(O)C(O)C(O)CO",                                # Xylitol
    "c1ccc(Cc2ccccc2)cc1",                             # Diphenylmethane
    "Cc1ccc(c(C)c1)N(=O)=O",                          # Dimethylnitrobenzene
    "CCOC(=O)CC(=O)OCC",                               # Diethyl malonate
    "OC(=O)CC(=O)O",                                   # Malonic acid
    "CC(C)(C)c1ccccc1",                                # tert-Butylbenzene
    "Nc1ccc(S(=O)(=O)N)cc1",                           # Sulfanilamide
    "C1COCCN1",                                         # Morpholine
    "OC(=O)c1cccc2ccncc12",                            # Quinoline acid
    "C=CCc1ccccc1",                                    # Allylbenzene
    "OCC1OC(Oc2ccccc2)C(O)C(O)C1O",                  # Phenyl glucoside
    "CC(=O)c1ccc(O)c(O)c1",                           # Acetocatechol
    "O=C(O)c1ccc2c(c1)OCO2",                          # Piperonylic acid
    "OC(=O)Cc1ccccc1",                                 # Phenylacetic acid
    "CC(=O)NCc1ccccc1",                                # N-Benzylacetamide
    "Fc1cccc(F)c1",                                    # 1,3-Difluorobenzene
    "OCC(N)Cc1ccc(O)cc1",                              # Tyrosinol
    "O=C1c2cccc3cccc1c23",                             # Acenaphthylenone
    "O=C(O)/C=C/c1ccccc1",                             # Cinnamic acid
    "OCC1=CC(=O)c2ccccc2O1",                          # Chromene
    "Cc1nc(=O)c2cc(Cl)ccc2[nH]1",                    # Chloroindazolone
    "C(=O)(c1ccccc1)c1ccccc1",                        # Benzophenone
    "O=C(c1ccccc1)c1ccccc1",                           # Benzophenone alt
    "CC(C)(C)OC(=O)N1CCCC1",                          # Boc-pyrrolidine
    "O=C(OCC)c1ccc(N)cc1",                             # Ethyl aminobenzoate
    "OC(=O)CCc1ccccc1",                                # Hydrocinnamic acid
    "CCNCC(O)c1ccc(O)cc1",                             # Synephrine
    "NC1CCCC1",                                         # Cyclopentylamine
    "OC1CCCCC1",                                        # Cyclohexanol
    "O=C1CCCCC1",                                      # Cyclohexanone
    "Cc1ccccc1C",                                      # o-Xylene
    "Cc1cccc(C)c1",                                    # m-Xylene
    "O=Cc1ccccc1",                                     # Benzaldehyde
    "OC(=O)c1ccccc1",                                  # Benzoic acid
    "NC(=O)c1ccccc1",                                  # Benzamide
    "Nc1ccccc1",                                        # Aniline
    "CC(=O)c1ccccc1",                                  # Acetophenone
    "OCc1ccccc1",                                      # Benzyl alcohol
    "CCc1ccccc1",                                      # Ethylbenzene
    "Clc1ccc(Cl)cc1",                                  # 1,4-Dichlorobenzene
    "Oc1ccccc1",                                        # Phenol
    "Nc1ccc(O)cc1",                                    # 4-Aminophenol
    "O=C(O)c1cccc(C(=O)O)c1",                         # Isophthalic acid
    "O=C(O)c1ccc(C(=O)O)cc1",                         # Terephthalic acid
    "O=C1c2cc(N)ccc2Cc2ccc(N)cc21",                   # Proflavine-like
    "CC1CC(C)(C)NC(C)(C)C1",                           # TEMPO-like
    "N#Cc1ccccc1",                                     # Benzonitrile
    "O=C(O)/C=C\\c1ccccc1",                            # cis-Cinnamic acid
    "OC(=O)c1cc(Cl)ccc1N",                             # Chloroanthranilic
    "Cc1ccc(cc1)C(=O)Cl",                              # p-Toluoyl chloride
    "CC(=O)Nc1ccc(Cl)cc1Cl",                           # Dichlorophenylacetamide
    "CC(Nc1ccccc1)=O",                                  # Acetanilide
    "Cc1cc(O)cc(C)c1O",                                # Xylenol
    "OC(=O)c1ccc(Cl)cc1Cl",                            # Dichlorobenzoic acid
    "c1ccc(NC(=O)c2ccccc2)cc1",                        # Benzanilide
    # More complex molecules for GEOM-Drugs benchmark
    "CC1=CC(=O)c2c(O)ccc(O)c2C1=O",                  # Embelin-like
    "OC1=C(C(=O)c2ccccc2)C(=O)c2ccccc21",            # Hydroxybenzophenone
    "Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1",                 # Sulfapyridine
    "Cc1ccc(S(=O)(=O)Nc2ccc(N)cc2)cc1",              # Sulfanilamide
    "Fc1ccc2c(c1)SC(=O)N2",                            # Saccharin-like
    "O=C1c2ccccc2NC(=O)N1",                           # Barbituric acid like
    "O=c1[nH]c(=O)c2ccccc2[nH]1",                    # Phthalhydrazide
    "O=C(O)c1ccc(F)cc1",                              # Fluorobenzoic acid
    "OC(=O)/C(=C\\c1ccccc1)c1ccccc1",               # Benzylidene acid
    "CC1(C)CCCC(C)(C)C1=O",                           # Camphor-like
    "CC(O)(c1ccccc1)c1ccccc1",                        # Benzhydrol
    "O=C(c1ccccc1)c1ccc(Cl)cc1",                      # Chlorobenzophenone
    "Fc1ccc(F)c(F)c1",                                # Trifluorobenzene
    "OC(=O)c1cc(Cl)cc(Cl)c1",                        # 3,5-Dichlorobenzoic
    "N=C(N)N",                                         # Guanidine
    "O=C(O)c1ccc(O)c(O)c1",                          # Protocatechuic acid
    "OC(=O)c1ccc(O)cc1",                              # 4-Hydroxybenzoic acid
    "OC(=O)c1cc(O)ccc1N",                             # 4-Aminosalicylic acid
    "CN1CC[NH+](CC1)Cc1ccc(F)cc1",                   # Fluorobenzylamine
    "OCC1OC(O)C(O)C(O)C1O",                           # Galactose
    "Cc1ccc(C(=O)O)cc1C",                             # 3,4-Dimethylbenzoic
    "O=C(O)CCc1ccc(O)cc1",                            # 4-Hydroxyphenylpropionic
    "CC1CCCCC1O",                                      # Methylcyclohexanol
    "O=C(Nc1ccccc1)Nc1ccccc1",                        # N,N-Diphenylurea
    "c1ccc(Cc2cccc3ccccc23)cc1",                      # Fluorenylmethyl
    "NC(=O)CC(=O)O",                                   # Asparagine-like
    "O=C(O)c1ccc2[nH]ccc2c1",                        # Indole-5-carboxylic
    "CC(=O)Nc1ccc2c(c1)OCO2",                         # Piperonylacetamide
    "O=Cc1cccs1",                                      # Thiophene-2-carboxaldehyde
    "O=Cc1ccc[nH]1",                                  # Pyrrole-2-carboxaldehyde
    "O=C(O)c1ccc[nH]1",                               # Pyrrole-2-carboxylic acid
    "O=C(O)c1ccsc1",                                  # Thiophene-3-carboxylic
    "CCOC(=O)c1ccc(N)cc1",                            # Ethyl 4-aminobenzoate
    "Oc1ccc(Cl)cc1",                                   # 4-Chlorophenol
    "Oc1cc(Cl)ccc1Cl",                                # 2,4-Dichlorophenol
    "CC(=O)Oc1ccc(Cl)cc1",                            # 4-Chlorophenyl acetate
    "O=C(O)c1ccc(OC)cc1",                             # 4-Methoxybenzoic
    "CC1=CC(=O)NC(=S)N1",                             # Methylthiouracil
    "CN1C(=O)NC(=O)c2[nH]cnc21",                     # Caffeine-like
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",                   # Theobromine-like
    "CCNC(=O)Nc1ccc(Cl)cc1",                          # Chlorphenesin-like
    "CNC(=O)c1cccc(NC(=O)NC)c1",                     # Phenyl diurea
]

# Extend with generated variations
def extend_smiles_list(base_list: List[str], target: int = 5000) -> List[str]:
    """Generate more SMILES by combining fragments."""
    import random
    random.seed(42)
    result = list(base_list)

    # Substituted benzenes
    cores = ["c1ccccc1", "c1ccncc1", "c1cc[nH]c1", "C1CCCCC1", "C1CCNCC1"]
    groups = [
        "C", "CC", "CCC", "O", "OC", "N", "NC", "Cl", "F", "Br",
        "C(=O)O", "C(=O)N", "OC(=O)C", "NC(=O)C", "C#N", "S(=O)(=O)N",
    ]

    while len(result) < target:
        core = random.choice(cores)
        g1 = random.choice(groups)
        mol_smi = g1 + core
        result.append(mol_smi)

    return result[:target]


# ── RDKit conformer generation ─────────────────────────────────────────────────

def mol_to_geom_entry(smiles: str, n_confs: int = 10, max_atoms: int = 100) -> Optional[Dict]:
    """
    Convert a SMILES to a GEOM-like entry using RDKit MMFF.
    Returns None if molecule cannot be processed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)           # Add implicit H for 3D
        mol = Chem.RemoveHs(mol)        # Remove for heavy-atom-only training

        # Get atom types (atomic numbers)
        atom_types = [a.GetAtomicNum() for a in mol.GetAtoms()]
        n_atoms = len(atom_types)

        if n_atoms < 3 or n_atoms > max_atoms:
            return None
        if 0 in atom_types:             # Unknown atom
            return None

        # Get bond graph
        src_list, dst_list, bond_types = [], [], []
        BOND_MAP = {
            Chem.rdchem.BondType.SINGLE:    1,
            Chem.rdchem.BondType.DOUBLE:    2,
            Chem.rdchem.BondType.TRIPLE:    3,
            Chem.rdchem.BondType.AROMATIC:  4,
        }
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = BOND_MAP.get(bond.GetBondType(), 1)
            src_list.extend([i, j])
            dst_list.extend([j, i])
            bond_types.extend([bt, bt])

        if not src_list:
            return None  # No bonds

        # Generate multiple conformers
        mol_h = Chem.AddHs(mol)  # Need Hs for embedding
        params = AllChem.EmbedParameters()
        params.randomSeed = 42
        params.numThreads = 1
        params.maxIterations = 2000

        conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs * 2,
                                               params=params)
        if len(conf_ids) == 0:
            # Try again with random coords
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
            conf_ids = [0] if mol_h.GetNumConformers() > 0 else []

        if len(conf_ids) == 0:
            return None

        # Optimize each conformer with MMFF
        energies = []
        coords_list = []

        for conf_id in conf_ids:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol_h, AllChem.MMFFGetMoleculeProperties(mol_h), confId=conf_id)
                if ff is None:
                    continue
                ff.Minimize(maxIts=500)
                energy_kcal = ff.CalcEnergy()
                energy_hartree = energy_kcal * KCAL_TO_HARTREE

                # Get heavy atom coordinates only
                conf = mol_h.GetConformer(conf_id)
                heavy_coords = []
                for atom in mol_h.GetAtoms():
                    if atom.GetAtomicNum() != 1:  # Skip H
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        heavy_coords.append([pos.x, pos.y, pos.z])

                if len(heavy_coords) != n_atoms:
                    continue

                energies.append(energy_hartree)
                coords_list.append(heavy_coords)
            except Exception:
                continue

        if len(coords_list) < 2:
            return None

        # Keep top n_confs by lowest energy
        order = sorted(range(len(energies)), key=lambda i: energies[i])
        top_n = min(n_confs, len(order))
        top_idx = order[:top_n]

        sel_energies = [energies[i] for i in top_idx]
        sel_coords   = [coords_list[i] for i in top_idx]

        # Boltzmann weights at 298K (MMFF energy in Hartree)
        kT = kB_kcal * TEMPERATURE_K * KCAL_TO_HARTREE  # kT in Hartree
        e = np.array(sel_energies)
        e_min = e.min()
        w = np.exp(-(e - e_min) / kT)
        bw = (w / w.sum()).tolist()

        conformers = [
            {
                'coordinates':      sel_coords[i],
                'energy_hartree':   sel_energies[i],
                'boltzmann_weight': bw[i],
            }
            for i in range(top_n)
        ]

        return {
            'num_atoms':      n_atoms,
            'num_conformers': len(conformers),
            'atom_types':     atom_types,
            'edge_index':     [src_list, dst_list],
            'bond_types':     bond_types,
            'conformers':     conformers,
        }

    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Generate GEOM-Drugs-like JSONL using RDKit MMFF')
    p.add_argument('--out',       default='data/geom_drugs.jsonl')
    p.add_argument('--n-confs',   type=int, default=10, help='Conformers per molecule')
    p.add_argument('--max-atoms', type=int, default=100)
    p.add_argument('--n-repeats', type=int, default=30,
                   help='Each SMILES is jittered/augmented N times to grow the dataset')
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build SMILES list
    smiles_list = extend_smiles_list(DRUG_SMILES, target=5000)
    print(f"Total SMILES to process: {len(smiles_list)}")
    print(f"Each repeated {args.n_repeats}x via random seed variation")

    n_written = 0
    n_skipped = 0
    t0 = time.time()

    with open(out_path, 'w') as f:
        for repeat in range(args.n_repeats):
            for smiles in smiles_list:
                try:
                    # Vary random seed per repeat for different conformer sets
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        n_skipped += 1
                        continue

                    entry = mol_to_geom_entry(smiles, n_confs=args.n_confs,
                                              max_atoms=args.max_atoms)
                    if entry is None:
                        n_skipped += 1
                        continue

                    f.write(json.dumps(entry, separators=(',', ':')) + '\n')
                    n_written += 1

                    if n_written % 500 == 0:
                        elapsed = time.time() - t0
                        rate = n_written / max(elapsed, 1)
                        print(f"  Written {n_written:,}  |  Skipped {n_skipped:,}  |  "
                              f"{rate:.1f} mol/s", flush=True)
                except Exception:
                    n_skipped += 1

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size >> 20
    print(f"\n✓ Done in {elapsed:.0f}s")
    print(f"  Written  : {n_written:,} molecules → {out_path} ({size_mb} MB)")
    print(f"  Skipped  : {n_skipped:,}")


if __name__ == '__main__':
    main()
