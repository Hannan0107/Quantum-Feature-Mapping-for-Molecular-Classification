"""Full-BBBP cleaning and the 10+2 molecular embedding.

Public API::

    from analysis_features import clean_dataset, embed

    result = clean_dataset(raw_frame, previous_smiles)
    result.clean      # modelling table
    result.audit      # provenance table (never use source_num as a feature)
    result.report     # counts + methods metadata

    circuit = embed("[Cl].CC(C)NCC(O)COc1cccc2ccccc12")
    circuit.qubits    # what sits on q0..q9 after salt stripping
"""

from .cleaning import CleanResult, CleaningReport, clean_dataset
from .config import ALPHA, N_BACKBONE, N_QUBITS, N_TAIL
from .elements import HANNAN_EN, PAULING_EN, audit_elements, electronegativity
from .embedding import Embedding, embed
from .standardize import canonical_parent, canonical_smiles, choose_parent, parse

__version__ = "1.0.0"

__all__ = [
    "ALPHA",
    "CleanResult",
    "CleaningReport",
    "Embedding",
    "HANNAN_EN",
    "N_BACKBONE",
    "N_QUBITS",
    "N_TAIL",
    "PAULING_EN",
    "audit_elements",
    "canonical_parent",
    "canonical_smiles",
    "choose_parent",
    "clean_dataset",
    "electronegativity",
    "embed",
    "parse",
]
