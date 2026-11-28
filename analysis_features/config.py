"""Fixed constants and default paths for the BBBP cleaning deliverable.

The circuit constants mirror the published 10+2 feature map. Keeping them in
one place means the cleaning code, the embedding demo and the tests cannot
drift apart.
"""

from __future__ import annotations

from pathlib import Path

# --- 10+2 circuit ---------------------------------------------------------
N_BACKBONE = 10          # individually resolved atom qubits (q0..q9)
N_TAIL = 2               # summary qubits for every remaining heavy atom
N_QUBITS = N_BACKBONE + N_TAIL
ALPHA = 0.1              # theta_atom = ALPHA * electronegativity**2
TAIL_MARGIN = 0.05       # global tail angles live in [margin, pi - margin]
TPSA_CAP = 35.0          # Angstrom^2, fixed scale for the optional TPSA angle

# --- Paths ----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "BBBP.csv"
DEFAULT_PREVIOUS = REPO_ROOT / "data" / "raw" / "BBBP_previous_250.csv"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "processed"
