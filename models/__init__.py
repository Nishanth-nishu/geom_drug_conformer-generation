"""
models/__init__.py — Model registry for NExT-Mol Gen

Available models:
  - ConformerDiffusion:       DDPM-based E(3)-equivariant diffusion (experiments A, D, G, H)
  - FlowMatchingConformer:    Conditional Flow Matching (experiment C)
  - AttnConformerDiffusion:   EQGAT-diff attention EGNN (experiment B)
  - EnergySurrogate:          GFN2-xTB energy predictor for GEOM-Drugs guidance (experiment H)
"""

from models.conformer_diffusion import ConformerDiffusion, ConformerDenoiser, remove_com
from models.flow_matching import FlowMatchingConformer
from models.energy_surrogate import EnergySurrogate

__all__ = [
    "ConformerDiffusion",
    "ConformerDenoiser",
    "FlowMatchingConformer",
    "EnergySurrogate",
    "remove_com",
]
