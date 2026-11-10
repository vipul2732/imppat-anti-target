"""Neighborhood pairs and splits for the hERG cliff question.

A neighbor pair has Morgan Tanimoto >= 0.80. Clusters are the connected
components of those pairs. Whole clusters are assigned to folds so a test
molecule does not leave its close analog in training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "input" / "hERG_Dataset.csv"
OUT_PAIRS = ROOT / "results" / "pairs"
OUT_SPLITS = ROOT / "results" / "splits"
CUTOFF = 0.80
N_FOLDS = 5
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_folds(group_ids):
    sizes = pd.Series(group_ids).value_counts()
    fold_counts = np.zeros(N_FOLDS, dtype=int)
    mapping = {}
    for group, size in sizes.sort_values(ascending=False).items():
        fold = int(fold_counts.argmin())
        mapping[group] = fold
        fold_counts[fold] += int(size)
    return mapping


def main():
    OUT_PAIRS.mkdir(parents=True, exist_ok=True)
    OUT_SPLITS.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(DATA)
    bitvects = []
    scaffolds = []
    valid = []
    for smiles in table["SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            bitvects.append(None)
            scaffolds.append("INVALID")
            valid.append(False)
            continue
        bitvects.append(FPGEN.GetFingerprint(mol))
        try:
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except ValueError:
            scaffolds.append("NO_SCAFFOLD")
        valid.append(True)

    n = len(table)
    y = table["PIC50"].to_numpy(dtype=float)
    uf = UnionFind(n)
    rows = []
    for i in range(n - 1):
        if bitvects[i] is None:
            continue
        later = [j for j in range(i + 1, n) if bitvects[j] is not None]
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(bitvects[i], [bitvects[j] for j in later]), dtype=float)
        hits = np.flatnonzero(sims >= CUTOFF)
        for hit in hits:
            j = later[int(hit)]
            uf.union(i, j)
            delta = abs(float(y[i] - y[j]))
            rows.append(
                {
                    "id_i": table.at[i, "SourceID"],
                    "id_j": table.at[j, "SourceID"],
                    "tanimoto": float(sims[hit]),
                    "y_i": float(y[i]),
                    "y_j": float(y[j]),
                    "abs_delta": delta,
                    "one_log": delta >= 1.0,
                    "two_log": delta >= 2.0,
                }
            )
        if i % 1000 == 0:
            print(f"paired {i}/{n} pairs {len(rows)}", flush=True)

    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT_PAIRS / "neighbor_pairs.csv", index=False)

    roots = [int(uf.find(i)) for i in range(n)]
    cluster_of = {}
    cluster_ids = []
    next_id = 0
    for root in roots:
        if root not in cluster_of:
            cluster_of[root] = next_id
            next_id += 1
        cluster_ids.append(cluster_of[root])
    cluster_fold_of = assign_folds(cluster_ids)
    scaffold_fold_of = assign_folds(scaffolds)
    folds = pd.DataFrame(
        {
            "SourceID": table["SourceID"],
            "SMILES": table["SMILES"],
            "PIC50": y,
            "USED_AS": table["USED_AS"],
            "valid_smiles": valid,
            "cluster_id": cluster_ids,
            "cluster_fold": [cluster_fold_of[c] for c in cluster_ids],
            "scaffold": scaffolds,
            "scaffold_fold": [scaffold_fold_of[s] for s in scaffolds],
        }
    )
    folds.to_csv(OUT_SPLITS / "compound_folds.csv", index=False)

    sizes = folds["cluster_id"].value_counts()
    summary = [
        f"Compounds: {n}",
        f"Neighbor pairs at Tanimoto >= {CUTOFF:.2f}: {len(pairs)}",
        f"One-log pairs: {int(pairs['one_log'].sum())}",
        f"Two-log pairs: {int(pairs['two_log'].sum())}",
        f"Clusters: {folds['cluster_id'].nunique()}",
        f"Largest cluster: {int(sizes.max())}",
        f"Singleton clusters: {int((sizes == 1).sum())}",
        "Cluster fold sizes: "
        + ", ".join(f"{int(k)}={int(v)}" for k, v in folds['cluster_fold'].value_counts().sort_index().items()),
        "Scaffold fold sizes: "
        + ", ".join(f"{int(k)}={int(v)}" for k, v in folds['scaffold_fold'].value_counts().sort_index().items()),
    ]
    text = "\n".join(summary) + "\n"
    (OUT_PAIRS / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
