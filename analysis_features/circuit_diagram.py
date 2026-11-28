"""Draw the actual 10+2 feature-map circuit with Qiskit's matplotlib drawer.

    python -m analysis_features circuit-diagram

The circuit is not a cartoon: every angle is taken from
:func:`analysis_features.embedding.build_statevector` for the example molecule
(aspirin). It has the same 12 qubits, the same ten ``Ry`` rotations
(``theta_i = 0.1 * chi_i**2``), the same two tail rotations and the same
``Rxx`` bond gates as the NumPy state-vector simulator used for the results.

Outputs: ``results/quantum_circuit_qiskit.png`` and ``results/quantum_circuit.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import __version__ as QISKIT_VERSION

from .benchmark import RESULTS_DIR
from .embedding import build_statevector
from .standardize import parse

EXAMPLE_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
EXAMPLE_NAME = "aspirin"


def build_circuit(smiles: str = EXAMPLE_SMILES) -> tuple[QuantumCircuit, dict]:
    """Return the Qiskit circuit and a JSON-serialisable gate manifest."""
    mol = parse(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")

    state, qubits, tail_angles, bond_gates = build_statevector(
        mol, property="pauling", tail_mode="paper"
    )

    circuit = QuantumCircuit(12, name="10+2 feature map")
    for info in qubits:
        circuit.ry(float(info.angle), info.qubit)
    circuit.barrier(label="tail")
    circuit.ry(float(tail_angles[0]), 10)
    circuit.ry(float(tail_angles[1]), 11)
    circuit.barrier(label="bonds")
    for left, right, theta in bond_gates:
        circuit.rxx(float(theta), left, right)

    manifest = {
        "molecule": EXAMPLE_NAME,
        "smiles": smiles,
        "qiskit_version": QISKIT_VERSION,
        "n_qubits": circuit.num_qubits,
        "backbone": [
            {"qubit": q.qubit, "atom_index": q.atom_index,
             "element": q.element, "ry_angle": round(q.angle, 6)}
            for q in qubits
        ],
        "tail_ry_angles": [round(float(a), 6) for a in tail_angles],
        "bond_rxx_gates": [
            {"qubits": [q1, q2], "angle": round(float(theta), 6)}
            for q1, q2, theta in bond_gates
        ],
        "readout": "12 <Z> + 66 <ZZ> moments -> RBF SVM (C=10)",
        "note": "angles are identical to the NumPy simulator used for the results",
    }
    return circuit, manifest


def draw(outdir: Path = RESULTS_DIR) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    circuit, manifest = build_circuit()

    # fold=-1 keeps all 12 qubits on one canvas instead of splitting the circuit.
    figure = circuit.draw(
        output="mpl", fold=-1, scale=0.62, style="iqp", plot_barriers=True
    )
    figure.suptitle(
        f"10+2 quantum feature map for {EXAMPLE_NAME} "
        f"(Qiskit {QISKIT_VERSION}): Ry = 0.1*chi^2, Rxx = bond_order/2",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.015,
        "q0-q9 = backbone atoms | q10-q11 = alternating electronegativity tail | "
        "78 Pauli moments -> RBF SVM",
        ha="center",
        fontsize=9,
        transform=figure.transFigure,
    )
    path = outdir / "quantum_circuit_qiskit.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    (outdir / "quantum_circuit.json").write_text(json.dumps(manifest, indent=2))
    return path


if __name__ == "__main__":
    print(draw())
