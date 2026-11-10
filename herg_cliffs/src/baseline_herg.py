"""Fingerprint baseline for absolute hERG pIC50.

Out-of-fold predictions use the 0.80-neighborhood split and the scaffold
split. Pair differences are scored on the neighborhood split, where both
members of a pair are held out together.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "results" / "splits" / "compound_folds.csv"
PAIRS = ROOT / "results" / "pairs" / "neighbor_pairs.csv"
OUT = ROOT / "results" / "baseline"
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles):
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        arr = np.zeros((2048,), dtype=np.float32)
        if mol is not None:
            DataStructs.ConvertToNumpyArray(FPGEN.GetFingerprint(mol), arr)
        rows.append(arr)
    return np.vstack(rows)


def metrics(y, pred):
    ok = np.isfinite(pred)
    err = pred[ok] - y[ok]
    ss_tot = np.sum((y[ok] - y[ok].mean()) ** 2)
    return {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / ss_tot),
    }


def oof(x, y, folds):
    pred = np.full(len(y), np.nan)
    for fold in sorted(pd.unique(folds)):
        test = folds == fold
        model = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=0
        )
        model.fit(x[~test], y[~test])
        pred[test] = model.predict(x[test])
        print(f"  fold {fold}", flush=True)
    return pred


def pair_scores(pairs, names, pred):
    lookup = dict(zip(names, pred))
    left = pairs["id_i"].map(lookup).to_numpy()
    right = pairs["id_j"].map(lookup).to_numpy()
    ok = np.isfinite(left) & np.isfinite(right)
    predicted = np.abs(left - right)
    observed = pairs["abs_delta"].to_numpy()
    err = predicted[ok] - observed[ok]
    out = {"n_pairs": int(ok.sum()), "delta_rmse": float(np.sqrt(np.mean(err ** 2)))}
    for flag, threshold in [("one_log", 1.0), ("two_log", 2.0)]:
        truth = pairs[flag].to_numpy().astype(bool) & ok
        hit = truth & (predicted >= threshold)
        out[f"{flag}_recall"] = float(hit.sum() / truth.sum()) if truth.any() else np.nan
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(FOLDS)
    print("fingerprints", flush=True)
    x = fingerprints(data["SMILES"])
    y = data["PIC50"].to_numpy(dtype=float)
    pred_rows = []
    metric_rows = []
    for split in ["cluster_fold", "scaffold_fold"]:
        print(split, flush=True)
        pred = oof(x, y, data[split].to_numpy())
        row = metrics(y, pred)
        row["split"] = split
        metric_rows.append(row)
        for name, fold, truth, value in zip(data["SourceID"], data[split], y, pred):
            pred_rows.append(
                {"SourceID": name, "split": split, "fold": int(fold), "y_true": float(truth), "y_pred": float(value)}
            )
    dev = data["USED_AS"].to_numpy() == "Dev-set"
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=0
    )
    model.fit(x[dev], y[dev])
    preset = np.full(len(y), np.nan)
    preset[~dev] = model.predict(x[~dev])
    row = metrics(y[~dev], preset[~dev])
    row["split"] = "preset_test"
    metric_rows.append(row)

    pairs = pd.read_csv(PAIRS)
    cluster_pred = [r["y_pred"] for r in pred_rows if r["split"] == "cluster_fold"]
    pair_row = pair_scores(pairs, data["SourceID"].tolist(), np.array(cluster_pred))
    pd.DataFrame(metric_rows).to_csv(OUT / "regression_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT / "oof_predictions.csv", index=False)
    pd.DataFrame([pair_row]).to_csv(OUT / "pair_delta_metrics.csv", index=False)
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(pair_row)


if __name__ == "__main__":
    main()
