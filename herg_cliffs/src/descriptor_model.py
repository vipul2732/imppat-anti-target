"""hERG potency from 2D descriptors, scored on the published test split.

Arab and Barakat (arXiv:2112.13467) report r2 = 0.67 and MSE = 0.59 for a
random forest on PaDEL descriptors over the same 499 Test-set compounds.
This uses the RDKit 2D descriptor block in place of PaDEL, and also reports
a cluster holdout so the number is comparable with the fingerprint models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "input" / "hERG_Dataset.csv"
FOLDS = ROOT / "results" / "splits" / "compound_folds.csv"
OUT = ROOT / "results" / "descriptors"


def descriptor_matrix(smiles_list):
    names = [name for name, _ in Descriptors.descList]
    rows = []
    valid = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            rows.append([np.nan] * len(names))
            valid.append(False)
            continue
        values = Descriptors.CalcMolDescriptors(mol)
        rows.append([values.get(name, np.nan) for name in names])
        valid.append(True)
    matrix = pd.DataFrame(rows, columns=names)
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    return matrix, np.asarray(valid)


def models(seed=0):
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=1, n_jobs=-1, random_state=seed
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
        ),
    }


def score(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(DATA)
    matrix, valid = descriptor_matrix(table["SMILES"])
    print(f"rows {len(table)} descriptors {matrix.shape[1]} parsed {int(valid.sum())}", flush=True)

    usable = valid & matrix.notna().all(axis=1).to_numpy()
    dropped = [name for name in matrix.columns if matrix.loc[usable, name].std() == 0]
    matrix = matrix.drop(columns=dropped)
    print(f"usable rows {int(usable.sum())} constant descriptors dropped {len(dropped)}", flush=True)

    table = table.loc[usable].reset_index(drop=True)
    x = matrix.loc[usable].to_numpy(dtype=float)
    y = table["PIC50"].to_numpy(dtype=float)

    rows = []
    preset_predictions = {}
    is_dev = (table["USED_AS"] == "Dev-set").to_numpy()
    for name, model in models().items():
        model.fit(x[is_dev], y[is_dev])
        pred = model.predict(x[~is_dev])
        rows.append({"split": "preset_test", "model": name, "n": int((~is_dev).sum()), **score(y[~is_dev], pred)})
        preset_predictions[name] = pred
        print(rows[-1], flush=True)

    folds = pd.read_csv(FOLDS)
    key = "SourceID" if "SourceID" in folds.columns else folds.columns[0]
    fold_column = "cluster_fold"
    mapping = dict(zip(folds[key].astype(str), folds[fold_column]))
    assigned = table["SourceID"].astype(str).map(mapping)
    has_fold = assigned.notna().to_numpy()
    fold_values = assigned.to_numpy()
    for name in models():
        oof = np.full(len(y), np.nan)
        for fold in sorted(set(fold_values[has_fold])):
            test = has_fold & (fold_values == fold)
            train = has_fold & (fold_values != fold)
            model = models()[name]
            model.fit(x[train], y[train])
            oof[test] = model.predict(x[test])
        mask = ~np.isnan(oof)
        rows.append({"split": "cluster_holdout", "model": name, "n": int(mask.sum()), **score(y[mask], oof[mask])})
        print(rows[-1], flush=True)
        preset_predictions[f"oof_{name}"] = oof

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "regression_metrics.csv", index=False)
    saved = table.loc[~is_dev, ["SourceID", "SMILES", "PIC50"]].copy()
    for name, pred in preset_predictions.items():
        if name.startswith("oof_"):
            continue
        saved[f"pred_{name}"] = pred
    saved.to_csv(OUT / "preset_test_predictions.csv", index=False)
    print(metrics.to_string(index=False), flush=True)
    print("published reference: r2 0.67, MSE 0.59 on N=499", flush=True)


if __name__ == "__main__":
    main()
