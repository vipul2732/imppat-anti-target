"""Predict the hERG pIC50 change to the nearest Tanimoto >= 0.80 neighbor.

The training target is the signed difference. Whole 0.80-clusters are held
out together. Absolute-potency models are scored on the same pairs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "results" / "splits" / "compound_folds.csv"
GP_PRED = ROOT / "results" / "gp" / "cluster_predictions.csv"
RF_PRED = ROOT / "results" / "baseline" / "oof_predictions.csv"
OUT = ROOT / "results" / "delta"
CUTOFF = 0.80
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles):
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        arr = np.zeros((2048,), dtype=np.float32)
        fp = None
        if mol is not None:
            fp = FPGEN.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append((fp, arr))
    return rows


def nearest_neighbors(bitvects, y, names, folds):
    n = len(bitvects)
    best_sim = np.zeros(n)
    best_j = np.full(n, -1)
    for i in range(n - 1):
        if bitvects[i] is None:
            continue
        later_idx = [j for j in range(i + 1, n) if bitvects[j] is not None]
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(bitvects[i], [bitvects[j] for j in later_idx]), dtype=float)
        over = np.flatnonzero(sims >= CUTOFF)
        for hit in over:
            j = later_idx[int(hit)]
            sim = float(sims[hit])
            if sim > best_sim[i]:
                best_sim[i] = sim
                best_j[i] = j
            if sim > best_sim[j]:
                best_sim[j] = sim
                best_j[j] = i
        if i % 1000 == 0:
            print(f"nn {i}/{n}", flush=True)
    rows = []
    for i, j in enumerate(best_j):
        if j < 0:
            continue
        delta = float(y[i] - y[j])
        rows.append(
            {
                "id_i": names[i],
                "id_j": names[j],
                "i": i,
                "j": int(j),
                "tanimoto": float(best_sim[i]),
                "delta": delta,
                "abs_delta": abs(delta),
                "one_log": abs(delta) >= 1,
                "two_log": abs(delta) >= 2,
                "cluster_fold": int(folds[i]),
            }
        )
    return pd.DataFrame(rows)


def pair_features(fps, pairs):
    xi = fps[pairs["i"].to_numpy()]
    xj = fps[pairs["j"].to_numpy()]
    change = np.abs(xi - xj)
    tanimoto = pairs["tanimoto"].to_numpy()[:, None]
    return np.hstack([change, tanimoto])


def score(delta, pred, one_log, two_log):
    err = pred - delta
    out = {"delta_rmse": float(np.sqrt(np.mean(err ** 2)))}
    for name, flag, threshold in [("one_log", one_log, 1.0), ("two_log", two_log, 2.0)]:
        truth = flag.astype(bool)
        hit = truth & (np.abs(pred) >= threshold)
        out[f"{name}_n"] = int(truth.sum())
        out[f"{name}_recall"] = float(hit.sum() / truth.sum()) if truth.any() else np.nan
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(FOLDS)
    print("fingerprints", flush=True)
    packed = fingerprints(data["SMILES"])
    bitvects = [fp for fp, _ in packed]
    fps = np.vstack([arr for _, arr in packed])
    y = data["PIC50"].to_numpy(dtype=float)
    pairs = nearest_neighbors(bitvects, y, data["SourceID"].tolist(), data["cluster_fold"].to_numpy())
    pairs.to_csv(OUT / "nearest_neighbor_pairs.csv", index=False)
    print(f"nn pairs {len(pairs)} one_log {int(pairs.one_log.sum())} two_log {int(pairs.two_log.sum())}", flush=True)

    features = pair_features(fps, pairs)
    delta = pairs["delta"].to_numpy()
    folds = pairs["cluster_fold"].to_numpy()
    pred = np.full(len(pairs), np.nan)
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        model = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=0
        )
        model.fit(features[~test], delta[~test])
        pred[test] = model.predict(features[test])
        print(f"fold {fold}", flush=True)
    pairs["delta_pred"] = pred

    gp = pd.read_csv(GP_PRED).set_index("SourceID")["y_pred_cluster"]
    rf = pd.read_csv(RF_PRED)
    rf = rf.loc[rf["split"] == "cluster_fold"].set_index("SourceID")["y_pred"]
    gp_delta = pairs["id_i"].map(gp).to_numpy() - pairs["id_j"].map(gp).to_numpy()
    rf_delta = pairs["id_i"].map(rf).to_numpy() - pairs["id_j"].map(rf).to_numpy()
    zero = np.zeros(len(pairs))

    rows = []
    for name, values in [
        ("predict_zero", zero),
        ("absolute_random_forest", rf_delta),
        ("absolute_gaussian_process", gp_delta),
        ("neighbor_delta_forest", pred),
    ]:
        row = score(delta, values, pairs["one_log"].to_numpy(), pairs["two_log"].to_numpy())
        row["model"] = name
        rows.append(row)
    summary = pd.DataFrame(rows)[["model", "delta_rmse", "one_log_n", "one_log_recall", "two_log_n", "two_log_recall"]]
    summary.to_csv(OUT / "delta_comparison.csv", index=False)
    pairs.drop(columns=["i", "j"]).to_csv(OUT / "nearest_neighbor_predictions.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
