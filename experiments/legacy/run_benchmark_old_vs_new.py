"""
run_benchmark_old_vs_new.py
---------------------------
Compare Machine Learning performance (Accuracy, AUC-ROC, F1) on BBBP between:
  1. Classical Baseline: Random Forest on Morgan Fingerprints (ECFP4, 256-bit)
  2. Original Quantum Feature Map: qmse_hero_normal (SMILES order) + RBF SVM
  3. New Quantum Feature Map: molecular_encoding (Scaffold + Centrality) + RBF SVM
  4. Control Negative: molecular_encoding (Random atom ordering) + RBF SVM

Uses Qiskit exact statevector simulation to generate the 4,096-dim probability features.
Evaluated across 10-split Stratified ShuffleSplit cross-validation.
"""

import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from molecular_encoding import encode_molecule
from molecular_encoding.chemistry.atom_selection import Strategy, assign_qubits
from molecular_encoding.quantum.circuit_builder import build_circuit
from molecular_encoding.chemistry.preprocessing import preprocess_smiles

# --- 1. ORIGINAL QMSE FEATURE MAP (from QAMP 2026 paper) ---
EN_MAP = {'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
          'N': 3.04, 'O': 3.44, 'F': 3.98, 'S': 2.58, 'Cl': 3.16}
ALPHA = 0.1
N_QUBITS = 12

def qmse_old(smiles):
    mol = Chem.MolFromSmiles(smiles)
    qc = QuantumCircuit(N_QUBITS)
    if not mol:
        return qc
    atoms = list(mol.GetAtoms())
    backbone = atoms[:10]
    tail = atoms[10:]
    mol_to_q = {a.GetIdx(): i for i, a in enumerate(backbone)}
    for atom in backbone:
        en = EN_MAP.get(atom.GetSymbol(), 1.0)
        qc.ry(ALPHA * (en**2), mol_to_q[atom.GetIdx()])
    if tail:
        tA, tB = tail[0::2], tail[1::2]
        if tA: qc.ry((ALPHA/2)*sum(EN_MAP.get(a.GetSymbol(),1.0) for a in tA), 10)
        if tB: qc.ry((ALPHA/2)*sum(EN_MAP.get(a.GetSymbol(),1.0) for a in tB), 11)
    for bond in mol.GetBonds():
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        q1, q2 = mol_to_q.get(i1), mol_to_q.get(i2)
        phi = bond.GetBondTypeAsDouble() / 2
        if q1 is not None and q2 is not None:
            qc.rxx(phi, q1, q2)
        elif q1 is not None:
            qc.rxx(phi, q1, 10); qc.rxx(phi, q1, 11)
        elif q2 is not None:
            qc.rxx(phi, q2, 10); qc.rxx(phi, q2, 11)
    return qc

def extract_statevector_probs(qc):
    # remove measurements if present
    qc_clean = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_clean)
    return sv.probabilities()

def main():
    print("="*75)
    print("BENCHMARK: OLD vs. NEW QUANTUM ENCODING ON BBBP DATASET")
    print("="*75)

    # 1. Load dataset
    print("\n[1/4] Loading and filtering BBBP.csv...")
    df = pd.read_csv("BBBP.csv").dropna(subset=['p_np', 'smiles'])
    # Filter length < 70 and valid mol, exactly as paper
    df = df[df['smiles'].apply(lambda s: Chem.MolFromSmiles(s) is not None and len(s) < 70)]

    # Take a balanced subset (e.g. 50 permeable, 50 non-permeable = 100 molecules)
    # to evaluate cleanly with fast execution
    n_per_class = 40  # 80 molecules total (ideal for fast CV & clear statistical comparison)
    df_pos = df[df['p_np'] == 1].sample(n_per_class, random_state=42)
    df_neg = df[df['p_np'] == 0].sample(n_per_class, random_state=42)
    df_sub = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    y = df_sub['p_np'].values
    smiles_list = df_sub['smiles'].values
    n_total = len(y)
    print(f"Dataset prepared: {n_total} balanced molecules ({n_per_class} per class).")

    # 2. Extract Classical Features (Morgan Fingerprints, 256 bits)
    print("\n[2/4] Extracting Classical Morgan Fingerprints (ECFP4)...")
    X_class = np.array([
        np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=256))
        for s in smiles_list
    ])

    # 3. Simulate Quantum Features (4096-dimensional probability vectors)
    print("\n[3/4] Simulating Quantum Feature Vectors (Statevector, 4096 dimensions)...")
    
    print("  -> Method A: Original QMSE (SMILES order)...")
    t0 = time.time()
    X_quantum_old = []
    for s in smiles_list:
        qc = qmse_old(s)
        X_quantum_old.append(extract_statevector_probs(qc))
    X_quantum_old = np.array(X_quantum_old)
    print(f"     Done in {time.time()-t0:.2f}s.")

    print("  -> Method B: New Scaffold + Centrality Encoding...")
    t0 = time.time()
    X_quantum_new = []
    for s in smiles_list:
        clean = preprocess_smiles(s) or s
        a = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY)
        qc = build_circuit(a)
        X_quantum_new.append(extract_statevector_probs(qc))
    X_quantum_new = np.array(X_quantum_new)
    print(f"     Done in {time.time()-t0:.2f}s.")

    print("  -> Method C: Control Negative (Random Atom Ordering)...")
    t0 = time.time()
    X_quantum_rand = []
    for s in smiles_list:
        clean = preprocess_smiles(s) or s
        a = assign_qubits(clean, strategy=Strategy.RANDOM, random_seed=42)
        qc = build_circuit(a)
        X_quantum_rand.append(extract_statevector_probs(qc))
    X_quantum_rand = np.array(X_quantum_rand)
    print(f"     Done in {time.time()-t0:.2f}s.")

    # 4. Cross-Validation (10 splits StratifiedShuffleSplit)
    print("\n[4/4] Running 10-Split Stratified Cross-Validation...")
    n_splits = 10
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=42)

    results = {
        'Classical (RF)': [],
        'Quantum Original (SMILES order)': [],
        'Quantum New (Scaffold+Centrality)': [],
        'Quantum Control (Random order)': []
    }

    feature_sets = [
        ('Classical (RF)', X_class, 'rf'),
        ('Quantum Original (SMILES order)', X_quantum_old, 'svm'),
        ('Quantum New (Scaffold+Centrality)', X_quantum_new, 'svm'),
        ('Quantum Control (Random order)', X_quantum_rand, 'svm'),
    ]

    for train_idx, test_idx in sss.split(y, y):
        y_train, y_test = y[train_idx], y[test_idx]

        for name, X, model_type in feature_sets:
            X_tr, X_te = X[train_idx], X[test_idx]

            if model_type == 'rf':
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_tr, y_train)
                preds = clf.predict(X_te)
                probs = clf.predict_proba(X_te)[:, 1]
            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                clf = SVC(kernel='rbf', C=10, probability=True, random_state=42)
                clf.fit(X_tr_s, y_train)
                preds = clf.predict(X_te_s)
                probs = clf.predict_proba(X_te_s)[:, 1]

            results[name].append({
                'Accuracy': accuracy_score(y_test, preds),
                'Precision': precision_score(y_test, preds, zero_division=0),
                'Recall': recall_score(y_test, preds, zero_division=0),
                'F1': f1_score(y_test, preds, zero_division=0),
                'AUC-ROC': roc_auc_score(y_test, probs)
            })

    # Summary table
    print("\n" + "="*85)
    print(f"{'Method':<36} | {'Accuracy':^13} | {'AUC-ROC':^13} | {'F1-Score':^13}")
    print("-"*85)
    for name, metric_list in results.items():
        acc = [m['Accuracy'] for m in metric_list]
        auc = [m['AUC-ROC'] for m in metric_list]
        f1  = [m['F1'] for m in metric_list]
        print(f"{name:<36} | {np.mean(acc):.3f} ± {np.std(acc):.3f} | {np.mean(auc):.3f} ± {np.std(auc):.3f} | {np.mean(f1):.3f} ± {np.std(f1):.3f}")
    print("="*85)

if __name__ == '__main__':
    main()
