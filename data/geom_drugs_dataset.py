"""
geom_drugs_dataset.py — PyTorch Dataset for GEOM-Drugs multi-conformer training.

Memory-efficient lazy loading: stores only byte offsets at init time,
reads one molecule from disk per __getitem__ call.

Supports:
  - Top-K conformers by Boltzmann weight (filters out rare high-energy conformers)
  - Uniform or Boltzmann-weighted conformer sampling during training
  - Molecule count limit (max_mols) for progressive dataset scaling
  - GEOM-Small/Medium/Full atom-count presets
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

HARTREE_TO_KCAL = 627.509    # 1 Hartree = 627.509 kcal/mol
MIN_ATOMS = 3
MAX_ATOMS = 100

# Preset atom-count cutoffs for progressive experiments
GEOM_SMALL  = 30    # Start here: overlaps with QM9 size regime
GEOM_MEDIUM = 50
GEOM_FULL   = 100


def _boltzmann_weights(energies_hartree: List[float],
                        temp_K: float = 298.0) -> List[float]:
    """Compute Boltzmann weights from energies in Hartree at temperature temp_K."""
    kT_hartree = 3.1668e-6 * temp_K   # kT in Hartree at temp_K
    e = [x / kT_hartree for x in energies_hartree]
    e_min = min(e)
    w = [math.exp(-(ei - e_min)) for ei in e]
    total = sum(w)
    return [wi / total for wi in w]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset — lazy (offset-indexed) to avoid OOM
# ──────────────────────────────────────────────────────────────────────────────

class GeomDrugsDataset(Dataset):
    """
    Memory-efficient GEOM-Drugs dataset with Boltzmann-weighted conformer sampling.

    Memory model:
      __init__    : scans JSONL once, stores (byte_offset, byte_length) per mol
                    Memory use: ~1.5 MB for 96K molecules. Zero OOM risk.
      __getitem__ : opens file, seeks to offset, reads exactly one line.
                    Only one molecule is in RAM at a time per worker.

    Top-K conformer filtering:
      Conformers are sorted by Boltzmann weight (highest first = lowest energy).
      Only the top max_conformers are kept. This removes rarely-populated
      high-energy conformers that add noise without physical relevance.

    Sampling during training:
      uniform_sampling=True  → pick one of top-K uniformly at random (recommended
                               for pure GeoDiff-style: the top-K selection already
                               gives the low-energy bias; no double-weighting)
      uniform_sampling=False → Boltzmann-weighted pick (more aggressive low-E bias)

    Args:
        data_path:        Path to geom_drugs.jsonl
        max_atoms:        Maximum heavy atoms per molecule (default 30 = GEOM-Small)
        min_conformers:   Minimum conformers required (default 2)
        max_conformers:   Keep only top-K conformers by Boltzmann weight (default 10)
        max_mols:         Hard cap on total molecules (default -1 = no cap).
                          Use 20_000, 50_000, 100_000 for progressive scaling.
        training_mode:    If True, sample one conformer per __getitem__ call.
        uniform_sampling: If True, sample uniformly from top-K (recommended).
                          If False, sample ∝ Boltzmann weight.
        boltzmann_temp:   Temperature for recomputing stored weights (None = use JSONL weights).
        return_energy:    If True, return GFN2-xTB energy fields.
        normalize_energy: If True, normalize energy to zero-mean unit-variance.
        seed:             Random seed.
    """

    def __init__(
        self,
        data_path: str,
        max_atoms: int = GEOM_SMALL,
        min_conformers: int = 2,
        max_conformers: int = 10,
        max_mols: int = -1,
        training_mode: bool = True,
        uniform_sampling: bool = True,
        boltzmann_temp: Optional[float] = None,
        return_energy: bool = False,
        normalize_energy: bool = False,
        seed: int = 42,
    ):
        self.data_path       = str(Path(data_path).resolve())  # Always absolute — prevents cluster CWD bugs
        self.training_mode   = training_mode
        self.uniform_sampling = uniform_sampling
        self.return_energy   = return_energy
        self.normalize_energy = normalize_energy
        self.rng             = random.Random(seed)

        # ── Phase 1: index scan — byte offsets only ────────────────────────────
        print(f"[GeomDrugsDataset] Indexing {data_path} ...", flush=True)
        print(f"  max_atoms={max_atoms}  min_confs={min_conformers}  "
              f"max_confs={max_conformers}  max_mols={'all' if max_mols<0 else max_mols}",
              flush=True)

        self._offsets: List[Tuple[int, int]] = []
        self._num_atoms:  List[int]          = []
        self._bw_list:    List[List[float]]  = []   # top-K weights (for sampling)
        self._energies_list: List[List[float]] = [] # top-K energies (for stats)
        # Store number of conformers kept per mol (used by get_all_conformers)
        self._n_confs: List[int] = []

        n_filtered = 0

        with open(self.data_path, 'rb') as f:
            while True:
                if max_mols > 0 and len(self._offsets) >= max_mols:
                    break

                offset = f.tell()
                raw = f.readline()
                if not raw:
                    break
                raw = raw.rstrip(b'\n')
                if not raw:
                    continue

                try:
                    mol = json.loads(raw)
                except Exception:
                    n_filtered += 1
                    continue

                na = mol.get('num_atoms', 0)
                nc = mol.get('num_conformers', 0)
                at = mol.get('atom_types', [])

                if na < MIN_ATOMS or na > max_atoms:
                    n_filtered += 1
                    continue
                if nc < min_conformers:
                    n_filtered += 1
                    continue
                if any(z <= 0 or z >= 119 for z in at):
                    n_filtered += 1
                    continue
                if not mol.get('conformers'):
                    n_filtered += 1
                    continue

                confs = mol['conformers']

                # Recompute Boltzmann weights at custom temperature if needed
                if boltzmann_temp is not None:
                    engs = [c['energy_hartree'] for c in confs]
                    bws  = _boltzmann_weights(engs, boltzmann_temp)
                    for c, w in zip(confs, bws):
                        c = dict(c)
                        c['boltzmann_weight'] = w
                else:
                    bws = [c.get('boltzmann_weight', 1.0 / len(confs)) for c in confs]

                # Sort by Boltzmann weight descending (= lowest energy first)
                order = sorted(range(len(confs)), key=lambda i: bws[i], reverse=True)
                top_k = order[:max_conformers]

                top_bws  = [bws[i]                                      for i in top_k]
                top_engs = [confs[i].get('energy_hartree', 0.0)         for i in top_k]

                # Re-normalize top-K weights so they sum to 1
                total_w = sum(top_bws) or 1.0
                top_bws = [w / total_w for w in top_bws]

                self._offsets.append((offset, len(raw) + 1))
                self._num_atoms.append(na)
                self._bw_list.append(top_bws)
                self._energies_list.append(top_engs)
                self._n_confs.append(len(top_k))

        print(f"[GeomDrugsDataset] Indexed {len(self._offsets):,} molecules "
              f"({n_filtered:,} filtered)", flush=True)

        if len(self._offsets) == 0:
            raise RuntimeError(
                f"No molecules passed the filters! "
                f"Check max_atoms={max_atoms}, min_conformers={min_conformers}. "
                f"The dataset may need to be regenerated with a different filter."
            )

        # ── Phase 2: energy stats (streaming Welford) ──────────────────────────
        if normalize_energy and return_energy:
            self._energy_mean, self._energy_std = \
                self._compute_energy_stats_streaming()
        else:
            self._energy_mean, self._energy_std = 0.0, 1.0

    # ── Core I/O ──────────────────────────────────────────────────────────────

    def _read_mol(self, idx: int) -> dict:
        """Read one molecule from disk by byte offset. O(1) RAM."""
        offset, length = self._offsets[idx]
        with open(self.data_path, 'rb') as f:
            f.seek(offset)
            raw = f.read(length)
        return json.loads(raw)

    # ── Energy stats ──────────────────────────────────────────────────────────

    def _compute_energy_stats_streaming(self) -> Tuple[float, float]:
        """Welford's online algorithm — O(1) memory."""
        n = 0; mean = 0.0; M2 = 0.0
        for na, engs in zip(self._num_atoms, self._energies_list):
            for e_h in engs:
                e_pa  = (e_h * HARTREE_TO_KCAL) / max(na, 1)
                n    += 1
                delta = e_pa - mean
                mean += delta / n
                M2   += delta * (e_pa - mean)
        std = math.sqrt(M2 / max(n - 1, 1)) + 1e-8
        print(f"[GeomDrugsDataset] Energy stats (per atom): "
              f"mean={mean:.2f} kcal/mol, std={std:.2f} kcal/mol")
        return mean, std

    def normalize_energy_val(self, energy_hartree: float, n_atoms: int) -> float:
        e_pa = (energy_hartree * HARTREE_TO_KCAL) / max(n_atoms, 1)
        return (e_pa - self._energy_mean) / self._energy_std

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict:
        mol  = self._read_mol(idx)
        bws  = self._bw_list[idx]
        nc   = self._n_confs[idx]

        # Keep only top-K conformers (same order as stored weights)
        confs = mol['conformers']
        # Re-sort by Boltzmann weight descending, take top nc
        sorted_confs = sorted(confs,
                              key=lambda c: c.get('boltzmann_weight', 0.0),
                              reverse=True)[:nc]

        if self.training_mode:
            if self.uniform_sampling:
                conf = self.rng.choice(sorted_confs)
            else:
                conf = self.rng.choices(sorted_confs, weights=bws, k=1)[0]
        else:
            conf = sorted_confs[0]   # lowest energy (highest Boltzmann weight)

        return self._make_item(mol, conf)

    def _make_item(self, mol: dict, conf: dict) -> dict:
        item = {
            'atom_types':       torch.tensor(mol['atom_types'],   dtype=torch.long),
            'coordinates':      torch.tensor(conf['coordinates'], dtype=torch.float32),
            'edge_index':       torch.tensor(mol['edge_index'],   dtype=torch.long),
            'bond_types':       torch.tensor(mol['bond_types'],   dtype=torch.long),
            'num_atoms':        mol['num_atoms'],
            'boltzmann_weight': torch.tensor(
                conf.get('boltzmann_weight', 1.0), dtype=torch.float32),
        }
        if self.return_energy:
            e_h   = conf.get('energy_hartree', 0.0)
            norm_e = self.normalize_energy_val(e_h, mol['num_atoms'])
            item['energy_hartree'] = torch.tensor(e_h,                  dtype=torch.float32)
            item['energy_kcal']    = torch.tensor(e_h * HARTREE_TO_KCAL, dtype=torch.float32)
            item['energy_norm']    = torch.tensor(norm_e,                dtype=torch.float32)
        return item

    def get_all_conformers(self, idx: int) -> List[dict]:
        """All top-K conformers for molecule idx (for COV-R/MAT-R eval)."""
        mol = self._read_mol(idx)
        nc  = self._n_confs[idx]
        sorted_confs = sorted(mol['conformers'],
                              key=lambda c: c.get('boltzmann_weight', 0.0),
                              reverse=True)[:nc]
        return [self._make_item(mol, c) for c in sorted_confs]

    def get_reference_coords(self, idx: int) -> List[torch.Tensor]:
        """All top-K conformer coords as (N,3) tensors."""
        mol = self._read_mol(idx)
        nc  = self._n_confs[idx]
        sorted_confs = sorted(mol['conformers'],
                              key=lambda c: c.get('boltzmann_weight', 0.0),
                              reverse=True)[:nc]
        return [torch.tensor(c['coordinates'], dtype=torch.float32) for c in sorted_confs]

    @property
    def energy_stats(self) -> Tuple[float, float]:
        return self._energy_mean, self._energy_std


# ──────────────────────────────────────────────────────────────────────────────
# Collation
# ──────────────────────────────────────────────────────────────────────────────

def collate_geom(batch: List[dict]) -> dict:
    at, co, ei, bt, bi = [], [], [], [], []
    bw, en_raw, en_norm = [], [], []
    offset = 0

    for i, item in enumerate(batch):
        N = item['num_atoms']
        at.append(item['atom_types'])
        co.append(item['coordinates'])
        ei.append(item['edge_index'] + offset)
        bt.append(item['bond_types'])
        bi.append(torch.full((N,), i, dtype=torch.long))
        bw.append(item['boltzmann_weight'].unsqueeze(0))
        offset += N
        if 'energy_norm' in item:
            en_norm.append(item['energy_norm'].unsqueeze(0))
        if 'energy_kcal' in item:
            en_raw.append(item['energy_kcal'].unsqueeze(0))

    out = {
        'atom_types':        torch.cat(at),
        'coordinates':       torch.cat(co),
        'edge_index':        torch.cat(ei, dim=1),
        'bond_types':        torch.cat(bt),
        'batch_idx':         torch.cat(bi),
        'boltzmann_weights': torch.cat(bw),
        'num_molecules':     len(batch),
    }
    if en_norm:
        out['energy_norm'] = torch.cat(en_norm)
    if en_raw:
        out['energy_kcal'] = torch.cat(en_raw)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def make_geom_dataloaders(
    data_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.1,
    max_atoms: int = GEOM_SMALL,
    min_conformers: int = 2,
    max_conformers: int = 10,
    max_mols: int = -1,
    return_energy: bool = False,
    uniform_sampling: bool = True,
    seed: int = 42,
):
    """
    Create train/val DataLoaders for GEOM-Drugs.
    Single index shared between train and val (memory efficient).
    Val subset always returns lowest-energy conformer (eval mode).
    """
    # Full dataset index
    full_ds = GeomDrugsDataset(
        data_path=data_path,
        max_atoms=max_atoms,
        min_conformers=min_conformers,
        max_conformers=max_conformers,
        max_mols=max_mols,
        training_mode=True,
        uniform_sampling=uniform_sampling,
        return_energy=return_energy,
        seed=seed,
    )

    n = len(full_ds)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    # Split indices
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    # Val subset uses eval mode (always returns best conformer)
    val_ds = GeomDrugsDataset(
        data_path=data_path,
        max_atoms=max_atoms,
        min_conformers=min_conformers,
        max_conformers=max_conformers,
        max_mols=max_mols,
        training_mode=False,
        uniform_sampling=False,
        return_energy=return_energy,
        seed=seed,
    )

    from torch.utils.data import Subset
    train_subset = Subset(full_ds, train_idx)
    val_subset   = Subset(val_ds,  val_idx)

    print(f"[GeomDataLoaders] train={n_train:,}  val={n_val:,}  "
          f"batch={batch_size}  workers={num_workers}", flush=True)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_geom,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_geom,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader
