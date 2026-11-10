"""Separate consensus-close IMPPAT molecules from the far set.

Each anti-target is its own comparison. Descriptors are tried alone, in pairs,
and in triples. The chosen combination is then compared with a label permutation.
"""

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, rdMolDescriptors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
IMPPAT = ROOT / "input"
OUT = ROOT / "results" / "paper"
BASIC_N = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(N-a)]")
NAMES = (
    "mw", "logp", "mr", "tpsa", "hbd", "hba", "rotatable", "rings", "aromatic_rings",
    "aliphatic_rings", "heteroatoms", "heavy_atoms", "fraction_csp3", "bertz",
    "chi0v", "kappa1", "labute_asa", "basic_n", "formal_charge", "nhoh", "amide",
    "bridgehead", "spiro",
)


def parent_mol(mol):
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return None
    parent = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
    return parent if parent.GetNumAtoms() else None


def descriptor_row(mol):
    parent = parent_mol(mol)
    if parent is None:
        return None
    try:
        key = Chem.MolToInchiKey(parent)
        balaban = GraphDescriptors.BalabanJ(parent) if parent.GetNumBonds() else 0.0
        values = [
            Descriptors.MolWt(parent),
            Crippen.MolLogP(parent),
            Crippen.MolMR(parent),
            rdMolDescriptors.CalcTPSA(parent),
            Lipinski.NumHDonors(parent),
            Lipinski.NumHAcceptors(parent),
            Lipinski.NumRotatableBonds(parent),
            Lipinski.RingCount(parent),
            Lipinski.NumAromaticRings(parent),
            Lipinski.NumAliphaticRings(parent),
            Lipinski.NumHeteroatoms(parent),
            Lipinski.HeavyAtomCount(parent),
            rdMolDescriptors.CalcFractionCSP3(parent),
            GraphDescriptors.BertzCT(parent),
            GraphDescriptors.Chi0v(parent),
            GraphDescriptors.Kappa1(parent),
            rdMolDescriptors.CalcLabuteASA(parent),
            float(len(parent.GetSubstructMatches(BASIC_N))),
            float(Chem.GetFormalCharge(parent)),
            float(Lipinski.NHOHCount(parent)),
            float(rdMolDescriptors.CalcNumAmideBonds(parent)),
            float(rdMolDescriptors.CalcNumBridgeheadAtoms(parent)),
            float(rdMolDescriptors.CalcNumSpiroAtoms(parent)),
        ]
    except Exception:
        return None
    if not np.isfinite(values).all() or not np.isfinite(balaban):
        return None
    return key, values + [balaban]


def load_matrix():
    labels = pd.read_csv(OUT / "imppat_similarity.csv", usecols=["inchikey", "consensus_hERG", "consensus_CYP", "consensus_PXR"])
    rows = {}
    for path in sorted((IMPPAT / "sdf_files").glob("*.sdf")):
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
        mol = next((item for item in supplier if item is not None), None)
        packed = descriptor_row(mol)
        if packed is None:
            continue
        key, values = packed
        rows.setdefault(key, values)
    desc_names = NAMES + ("balaban",)
    matrix = pd.DataFrame.from_dict(rows, orient="index", columns=desc_names)
    matrix.index.name = "inchikey"
    merged = labels.join(matrix, on="inchikey", how="inner")
    keep = [name for name in desc_names if merged[name].std() > 0]
    return merged, keep


def cv_auc(x, y, seed):
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros(len(y))
    for train, test in fold.split(x, y):
        scaler = StandardScaler()
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=400, random_state=seed)
        model.fit(scaler.fit_transform(x[train]), y[train])
        scores[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
    return float(roc_auc_score(y, scores))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("descriptors", flush=True)
    frame, names = load_matrix()
    print(f"molecules {len(frame)} descriptors {len(names)}", flush=True)
    values = frame[list(names)].to_numpy(dtype=float)
    name_index = {name: i for i, name in enumerate(names)}
    results = []
    rng = np.random.default_rng(0)
    for target, column in (("hERG", "consensus_hERG"), ("CYP", "consensus_CYP"), ("PXR", "consensus_PXR")):
        y = (frame[column] >= 0.80).to_numpy(dtype=int)
        print(f"{target} close {int(y.sum())}", flush=True)
        singles = []
        for name in names:
            score = cv_auc(values[:, [name_index[name]]], y, 0)
            singles.append((score, (name,)))
        singles.sort(reverse=True)
        pool = [item[1][0] for item in singles[:8]]
        candidates = list(singles[:8])
        for width in (2, 3):
            for combo in itertools.combinations(pool, width):
                cols = [name_index[name] for name in combo]
                candidates.append((cv_auc(values[:, cols], y, 0), combo))
        full = cv_auc(values, y, 0)
        candidates.append((full, tuple(names)))
        candidates.sort(reverse=True)
        best_score, best_combo = candidates[0]
        null = []
        cols = [name_index[name] for name in best_combo]
        x_best = values[:, cols] if best_combo != tuple(names) else values
        for draw in range(200):
            shuffled = rng.permutation(y)
            null.append(cv_auc(x_best, shuffled, draw + 1))
        null = np.asarray(null)
        results.append({
            "anti_target": target,
            "n_close": int(y.sum()),
            "best_descriptors": " + ".join(best_combo) if len(best_combo) <= 3 else f"all {len(names)}",
            "n_descriptors": len(best_combo),
            "cv_auc": best_score,
            "full_model_auc": full,
            "best_single": singles[0][1][0],
            "best_single_auc": singles[0][0],
            "permutation_mean_auc": float(null.mean()),
            "permutation_p": float((null >= best_score).mean()),
        })
        print(results[-1], flush=True)
        pd.DataFrame({"auc": [score for score, _ in candidates[:15]], "descriptors": [" + ".join(combo) if len(combo) <= 3 else "all" for _, combo in candidates[:15]]}).to_csv(
            OUT / f"descriptor_combinations_{target}.csv", index=False
        )
    table = pd.DataFrame(results)
    table.to_csv(OUT / "separation_summary.csv", index=False)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
