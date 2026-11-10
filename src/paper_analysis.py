"""Anti-target neighborhood of IMPPAT. Computes result tables only.

Figures are drawn by figures.py from these tables.

For every phytochemical, similarity to hERG, CYP, and PXR is the lower of
the Morgan and atom-pair Tanimoto values to the same measured neighbor.
A compound is close at 0.80 only under that rule. No potency is predicted.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
IMPPAT = INPUT
HERG_PATH = INPUT / "hERG_Dataset.csv"
CYP_PATH = INPUT / "cyp-challenge-TRAIN_inhibition.csv"
PXR_PATH = INPUT / "pxr-challenge_TRAIN.csv"
PXR_UNBLINDED = (
    INPUT / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    INPUT / "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
)
OUT = ROOT / "results" / "paper"
CUTOFFS = (0.70, 0.80, 0.90)
N_BOOT = 2000
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ATOM_PAIR = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
BASIC_N = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(N-a)]")
CYP_ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def safe_filename(name):
    name = re.sub(r'[\/\\:*?"<>|]', "_", str(name))
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] if name else "compound"


def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None and "|" in str(smiles):
        mol = Chem.MolFromSmiles(str(smiles).split("|", 1)[0])
    return mol


def parent_mol(mol):
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return None
    parent = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
    return parent if parent.GetNumAtoms() else None


def pack(mol):
    parent = parent_mol(mol)
    if parent is None:
        return None
    try:
        inchikey = Chem.MolToInchiKey(parent)
    except Exception:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent)
    except ValueError:
        scaffold = ""
    return {
        "morgan": MORGAN.GetFingerprint(parent),
        "atom_pair": ATOM_PAIR.GetFingerprint(parent),
        "inchikey": inchikey,
        "scaffold": scaffold,
        "mw": Descriptors.MolWt(parent),
        "logp": Crippen.MolLogP(parent),
        "tpsa": rdMolDescriptors.CalcTPSA(parent),
        "rings": Lipinski.RingCount(parent),
        "basic_n": len(parent.GetSubstructMatches(BASIC_N)),
    }


def nearest(query, morgan_fps, atom_fps):
    morgan = np.asarray(DataStructs.BulkTanimotoSimilarity(query["morgan"], morgan_fps), dtype=float)
    atom_pair = np.asarray(DataStructs.BulkTanimotoSimilarity(query["atom_pair"], atom_fps), dtype=float)
    consensus = np.minimum(morgan, atom_pair)
    morgan_j = int(morgan.argmax())
    consensus_j = int(consensus.argmax())
    return float(morgan[morgan_j]), float(consensus[consensus_j]), consensus_j


def within_neighbor(records):
    scores = np.zeros(len(records))
    fps = [item["morgan"] for item in records]
    for i, fp in enumerate(fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps), dtype=float)
        sims[i] = -1.0
        scores[i] = sims.max() if len(sims) > 1 else np.nan
    return scores


def potency_text(record):
    if record["set"] == "hERG":
        return f"{record['pic50']:.2f}", ""
    if record["set"] == "PXR":
        return f"{record['pec50']:.2f}", f"{record['ci_low']:.2f}-{record['ci_high']:.2f}"
    parts = []
    intervals = []
    for isoform in CYP_ISOFORMS:
        value = record["pic50"][isoform]
        if np.isnan(value):
            continue
        low, high = record["ci"][isoform]
        parts.append(f"{isoform}:{value:.2f}")
        intervals.append(f"{isoform}:{low:.2f}-{high:.2f}")
    return "; ".join(parts), "; ".join(intervals)


def load_references():
    herg = pd.read_csv(HERG_PATH)
    cyp = pd.read_csv(CYP_PATH)
    pxr = pd.read_csv(PXR_PATH)
    measured = cyp[[f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]].notna().any(axis=1)
    cyp = cyp.loc[measured]
    sets = {"hERG": [], "CYP": [], "PXR": []}
    seen = {name: set() for name in sets}
    for _, row in herg.iterrows():
        packed = pack(mol_from_smiles(row["SMILES"]))
        if packed is None or packed["inchikey"] in seen["hERG"]:
            continue
        seen["hERG"].add(packed["inchikey"])
        packed.update({"set": "hERG", "id": str(row["SourceID"]), "pic50": float(row["PIC50"])})
        sets["hERG"].append(packed)
    for _, row in cyp.iterrows():
        packed = pack(mol_from_smiles(row["SMILES"]))
        if packed is None or packed["inchikey"] in seen["CYP"]:
            continue
        seen["CYP"].add(packed["inchikey"])
        pic50, ci = {}, {}
        for isoform in CYP_ISOFORMS:
            pic50[isoform] = float(row[f"{isoform}_pIC50_direct_inhibition"])
            ci[isoform] = (
                float(row[f"{isoform}_pIC50_direct_inhibition_conf_low"]),
                float(row[f"{isoform}_pIC50_direct_inhibition_conf_high"]),
            )
        packed.update({"set": "CYP", "id": str(row["Molecule_Name"]), "pic50": pic50, "ci": ci})
        sets["CYP"].append(packed)
    for _, row in pxr.iterrows():
        packed = pack(mol_from_smiles(row["SMILES"]))
        if packed is None or packed["inchikey"] in seen["PXR"]:
            continue
        seen["PXR"].add(packed["inchikey"])
        packed.update({
            "set": "PXR",
            "id": str(row["OCNT_ID"]),
            "pec50": float(row["pEC50"]),
            "ci_low": float(row["pEC50_ci.lower (-log10(molarity))"]),
            "ci_high": float(row["pEC50_ci.upper (-log10(molarity))"]),
        })
        sets["PXR"].append(packed)
    return sets


def load_imppat():
    catalog = pd.read_csv(IMPPAT / "imppat.csv", usecols=["Phytochemical name", "IMPPAT Phytochemical identifier"])
    catalog = catalog.dropna(subset=["Phytochemical name"]).drop_duplicates("Phytochemical name")
    name_by_file = {
        safe_filename(name) + ".sdf": (name, identifier)
        for name, identifier in zip(catalog["Phytochemical name"], catalog["IMPPAT Phytochemical identifier"])
    }
    plant_counts = (
        pd.read_csv(IMPPAT / "imppat.csv", usecols=["IMPPAT Phytochemical identifier", "Indian medicinal plant"])
        .groupby("IMPPAT Phytochemical identifier")["Indian medicinal plant"]
        .nunique()
    )
    rows = []
    seen = set()
    paths = sorted((IMPPAT / "sdf_files").glob("*.sdf"))
    for number, path in enumerate(paths, start=1):
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
        mol = next((item for item in supplier if item is not None), None)
        packed = pack(mol)
        if packed is None or packed["inchikey"] in seen:
            continue
        seen.add(packed["inchikey"])
        name, identifier = name_by_file.get(path.name, (path.stem, ""))
        packed["name"] = name
        packed["imppat_id"] = identifier
        packed["n_plants"] = int(plant_counts.get(identifier, 0))
        rows.append(packed)
        if number % 2000 == 0:
            print(f"parsed {number}", flush=True)
    print(f"unique IMPPAT {len(rows)}", flush=True)
    return rows


def fraction_ci(mask, rng):
    values = np.asarray(mask, dtype=float)
    draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    stats = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def mean_diff_ci(close_vals, far_vals, rng):
    close_vals = np.asarray(close_vals, dtype=float)
    far_vals = np.asarray(far_vals, dtype=float)
    observed = float(close_vals.mean() - far_vals.mean())
    close_draw = close_vals[rng.integers(0, len(close_vals), size=(N_BOOT, len(close_vals)))].mean(axis=1)
    far_draw = far_vals[rng.integers(0, len(far_vals), size=(N_BOOT, len(far_vals)))].mean(axis=1)
    diffs = close_draw - far_draw
    return observed, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("references", flush=True)
    sets = load_references()
    fp_lists = {
        name: ([item["morgan"] for item in records], [item["atom_pair"] for item in records])
        for name, records in sets.items()
    }
    within, scaffolds = {}, {}
    for name, records in sets.items():
        print(f"within {name} {len(records)}", flush=True)
        within[name] = within_neighbor(records)
        scaffolds[name] = len({item["scaffold"] for item in records if item["scaffold"]})
    print("imppat", flush=True)
    molecules = load_imppat()
    rows = []
    for number, molecule in enumerate(molecules, start=1):
        row = {
            "phytochemical_name": molecule["name"],
            "imppat_id": molecule["imppat_id"],
            "inchikey": molecule["inchikey"],
            "n_plants": molecule["n_plants"],
            "mw": molecule["mw"],
            "logp": molecule["logp"],
            "tpsa": molecule["tpsa"],
            "rings": molecule["rings"],
            "basic_n": molecule["basic_n"],
        }
        for name, records in sets.items():
            morgan_fps, atom_fps = fp_lists[name]
            morgan, consensus, consensus_j = nearest(molecule, morgan_fps, atom_fps)
            potency, interval = potency_text(records[consensus_j])
            row[f"morgan_{name}"] = morgan
            row[f"consensus_{name}"] = consensus
            row[f"neighbor_{name}"] = records[consensus_j]["id"]
            row[f"neighbor_potency_{name}"] = potency
            row[f"neighbor_ci_{name}"] = interval
        rows.append(row)
        if number % 1000 == 0:
            print(f"scored {number}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "imppat_similarity.csv", index=False)

    rng = np.random.default_rng(0)
    robust_rows = []
    for name in ("hERG", "CYP", "PXR"):
        for rule, column in (("morgan", f"morgan_{name}"), ("consensus", f"consensus_{name}")):
            for cutoff in CUTOFFS:
                fraction, low, high = fraction_ci(frame[column] >= cutoff, rng)
                count = int((frame[column] >= cutoff).sum())
                robust_rows.append({
                    "anti_target": name, "rule": rule, "cutoff": cutoff,
                    "n": count, "fraction": fraction, "ci_low": low, "ci_high": high,
                })
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(OUT / "cutoff_fractions.csv", index=False)

    close = {
        "hERG": frame["consensus_hERG"] >= 0.80,
        "CYP": frame["consensus_CYP"] >= 0.80,
        "PXR": frame["consensus_PXR"] >= 0.80,
    }
    far = ~close["hERG"] & ~close["CYP"] & ~close["PXR"]
    effect_rows = []
    for name in ("hERG", "CYP", "PXR"):
        for column in ("mw", "logp", "tpsa", "rings", "basic_n"):
            left = frame.loc[close[name], column]
            right = frame.loc[far, column]
            if column == "basic_n":
                left = left.ge(1).astype(float)
                right = right.ge(1).astype(float)
            diff, low, high = mean_diff_ci(left, right, rng)
            effect_rows.append({"group": name, "descriptor": column, "difference": diff, "ci_low": low, "ci_high": high})
    effect = pd.DataFrame(effect_rows)
    effect.to_csv(OUT / "descriptor_differences.csv", index=False)

    catalog = pd.read_csv(IMPPAT / "imppat.csv", usecols=["IMPPAT Phytochemical identifier", "Indian medicinal plant"])
    summary_rows = []
    for name in ("hERG", "CYP", "PXR"):
        mask = close[name]
        ids = set(frame.loc[mask, "imppat_id"])
        plants = catalog.loc[catalog["IMPPAT Phytochemical identifier"].isin(ids), "Indian medicinal plant"].nunique()
        summary_rows.append({
            "group": name,
            "n_reference": len(sets[name]),
            "internal_median_nn": float(np.median(within[name])),
            "n_scaffolds": scaffolds[name],
            "n_close_consensus_0.80": int(mask.sum()),
            "fraction_close": float(mask.mean()),
            "fraction_ci_low": float(robust.loc[(robust.anti_target == name) & (robust.rule == "consensus") & (robust.cutoff == 0.80), "ci_low"].iloc[0]),
            "fraction_ci_high": float(robust.loc[(robust.anti_target == name) & (robust.rule == "consensus") & (robust.cutoff == 0.80), "ci_high"].iloc[0]),
            "n_plants": int(plants),
            "n_single_plant_compounds": int((frame.loc[mask, "n_plants"] == 1).sum()),
            "median_plants_per_compound": float(frame.loc[mask, "n_plants"].median()) if mask.any() else np.nan,
        })
    intersections = {
        "hERG∩CYP": close["hERG"] & close["CYP"],
        "hERG∩PXR": close["hERG"] & close["PXR"],
        "CYP∩PXR": close["CYP"] & close["PXR"],
        "all three": close["hERG"] & close["CYP"] & close["PXR"],
    }
    for label, mask in intersections.items():
        ids = set(frame.loc[mask, "imppat_id"])
        plants = catalog.loc[catalog["IMPPAT Phytochemical identifier"].isin(ids), "Indian medicinal plant"].nunique()
        summary_rows.append({
            "group": label,
            "n_reference": np.nan,
            "internal_median_nn": np.nan,
            "n_scaffolds": np.nan,
            "n_close_consensus_0.80": int(mask.sum()),
            "fraction_close": float(mask.mean()),
            "fraction_ci_low": np.nan,
            "fraction_ci_high": np.nan,
            "n_plants": int(plants),
            "n_single_plant_compounds": int((frame.loc[mask, "n_plants"] == 1).sum()),
            "median_plants_per_compound": float(frame.loc[mask, "n_plants"].median()) if mask.any() else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "table1_summary.csv", index=False)
    called = frame.loc[close["hERG"] | close["CYP"] | close["PXR"]].copy()
    called.to_csv(OUT / "close_compounds.csv", index=False)
    cache_rows = []
    for name, records in sets.items():
        for record, score in zip(records, within[name]):
            if name == "CYP":
                potency = float(np.nanmax(list(record["pic50"].values())))
            elif name == "PXR":
                potency = record["pec50"]
            else:
                potency = record["pic50"]
            cache_rows.append({
                "anti_target": name,
                "nn": score,
                "potency": potency,
                "scaffold": record["scaffold"],
            })
    pd.DataFrame(cache_rows).to_csv(OUT / "within_set_similarity.csv", index=False)
    score_unblinded(molecules)
    print(summary.to_string(index=False), flush=True)
    print(robust.to_string(index=False), flush=True)


def score_unblinded(molecules):
    extra = pd.concat([pd.read_csv(path) for path in PXR_UNBLINDED], ignore_index=True)
    records, seen = [], set()
    for _, row in extra.iterrows():
        packed = pack(mol_from_smiles(row["SMILES"]))
        if packed is None or packed["inchikey"] in seen:
            continue
        seen.add(packed["inchikey"])
        packed["id"] = str(row["Molecule Name"])
        packed["pec50"] = float(row["pEC50"])
        records.append(packed)
    morgan = [item["morgan"] for item in records]
    atom_pair = [item["atom_pair"] for item in records]
    rows = []
    for molecule in molecules:
        _, consensus, neighbor = nearest(molecule, morgan, atom_pair)
        rows.append({
            "phytochemical_name": molecule["name"],
            "imppat_id": molecule["imppat_id"],
            "inchikey": molecule["inchikey"],
            "consensus": consensus,
            "neighbor": records[neighbor]["id"],
            "neighbor_potency": records[neighbor]["pec50"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "pxr_unblinded_similarity.csv", index=False)
    print(f"unblinded PXR structures {len(records)} max consensus {out['consensus'].max():.3f}", flush=True)


if __name__ == "__main__":
    main()
