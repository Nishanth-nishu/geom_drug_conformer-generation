#!/usr/bin/env python3
import msgpack
import argparse
import tarfile
import json
import time
from pathlib import Path
import numpy as np

# ── GEOM download config ───────────────────────────────────────────────────────
# File ID for qm9_crude.msgpack.tar.gz is 4327190
# ───────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Prepare GEOM-QM9 JSONL dataset from msgpack')
    p.add_argument('--tar', required=True, help='Path to qm9_crude.msgpack.tar.gz')
    p.add_argument('--out', default='data/geom_qm9.jsonl', help='Output JSONL file')
    args = p.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing msgpack tar: {args.tar}")
    print(f"Output           : {out_path}")

    # Standard atomic numbers for QM9
    Z_MAP = {1: 1, 6: 6, 7: 7, 8: 8, 9: 9}
    
    n_written = 0
    t0 = time.time()

    with tarfile.open(args.tar, 'r:gz') as tar, open(out_path, 'w') as out_f:
        for member in tar:
            if not member.name.endswith('.msgpack'):
                continue
            
            f = tar.extractfile(member)
            if f is None:
                continue
                
            unpacker = msgpack.Unpacker(f, raw=False)
            for smiles_dict in unpacker:
                for smiles, mol_data in smiles_dict.items():
                    if 'conformers' not in mol_data:
                        continue
                    
                    confs = mol_data['conformers']
                    if not confs:
                        continue
                    
                    # We can use the first conformer to build the atom_types list
                    # GEOM msgpack provides a 'geom' array for each conformer.
                    # Wait, GEOM msgpack actually might not have bond information.
                    # Since we need edge_index, it's safer to read the SDF from rdkit_folder!
                    pass

if __name__ == '__main__':
    main()
