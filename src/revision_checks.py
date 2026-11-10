"""Robustness checks requested for the coverage claim.

The 0.80 consensus rule stays the definition used for plants. This script adds
the dataset-relative comparison, the mean of the two fingerprints, CYP isoform
reference sets, the strongest neighbor at or above 0.80, ubiquitous-cutoff
sensitivity, and a curation of the close matches.
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from rdkit import DataStructs

from paper_analysis import (
    CYP_ISOFORMS,
    CYP_PATH,
    IMPPAT,
    PXR_PATH,
    load_imppat,
    load_references,
    pack,
    mol_from_smiles,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "paper"
CUTOFF = 0.80
POTENT = 5.0
N_DRAWS = 20000
TARGETS = ("hERG", "CYP", "PXR")

# Compounds in the close set that are not credible plant metabolites.
# Xenobiotics are drugs or synthetic reagents. Contaminants are industrial or
# combustion chemicals. The rest of the close set stays in the headline counts.
XENOBIOTIC = {
    "1-(4-Methanesulfonamidophenoxy)-3-(N-methyl-3,4-dichlorophenylethylamino)-2-propanol",
    "4-(2-Aminopropyl)phenol",
    "4-Aminopyridine",
    "Amiloxate",
    "Chlorobutanol",
    "Cyproheptadine hydrochloride",
    "Dipyridamole",
    "Fenretinide",
    "Guaifenesin",
    "Homatropine",
    "Methyl isobutyl ketone",
    "Morpholine",
    "Musk ambrette",
    "Quinacrine",
    "Triphenylphosphine",
    "Triprolidine",
    "Vinpocetine",
    "alpha-Hexylcinnamaldehyde",
    "Phorbol 12-myristate 13-acetate",
    "Phorbol acetate, caprate",
}
CONTAMINANT = {
    "1,2-Dimethylnaphthalene",
    "4-Nonylphenol",
    "Cumene",
    "Dibutyl phthalate",
    "Diisobutyl phthalate",
    "Fluoranthene",
    "N-Phenyl-2-naphthylamine",
    "Phenanthrene",
}
UNCERTAIN = {
    "17alpha-Estradiol",
    "Androstenedione",
    "Cotinine",
    "Deoxycholic acid",
    "Estrone",
    "Menadione",
    "Pregnenolone",
    "Progesterone",
}


def classify(name):
    if name in XENOBIOTIC:
        return "xenobiotic"
    if name in CONTAMINANT:
        return "contaminant"
    if name in UNCERTAIN:
        return "uncertain"
    return "credible_phytochemical"


def scalar_potency(record):
    if record["set"] == "hERG":
        return float(record["pic50"])
    if record["set"] == "PXR":
        return float(record["pec50"])
    return float(np.nanmax(list(record["pic50"].values())))


def score_query(query, morgan_fps, atom_fps, potency):
    morgan = np.asarray(DataStructs.BulkTanimotoSimilarity(query["morgan"], morgan_fps), dtype=float)
    atom = np.asarray(DataStructs.BulkTanimotoSimilarity(query["atom_pair"], atom_fps), dtype=float)
    consensus = np.minimum(morgan, atom)
    mean = 0.5 * (morgan + atom)
    eligible = consensus >= CUTOFF
    max_potency = float(potency[eligible].max()) if eligible.any() else np.nan
    return float(consensus.max()), float(mean.max()), max_potency, int(eligible.sum())


def relative_coverage():
    imp = pd.read_csv(PAPER / "imppat_similarity.csv")
    within = pd.read_csv(PAPER / "within_set_similarity.csv")
    rows = []
    for name in TARGETS:
        reference = np.sort(within.loc[within.anti_target == name, "nn"].to_numpy())
        consensus = imp[f"consensus_{name}"].to_numpy()
        morgan = imp[f"morgan_{name}"].to_numpy()
        percentile = np.searchsorted(reference, consensus, side="right") / len(reference)
        median = float(np.median(reference))
        rows.append({
            "anti_target": name,
            "reference_median_nn": median,
            "reference_p10_nn": float(np.quantile(reference, 0.10)),
            "n_consensus_ge_0.80": int((consensus >= CUTOFF).sum()),
            "n_consensus_ge_reference_median": int((consensus >= median).sum()),
            "n_morgan_ge_reference_median": int((morgan >= median).sum()),
            "median_percentile_of_consensus": float(np.median(percentile)),
            "fraction_below_reference_p05": float((percentile < 0.05).mean()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(PAPER / "relative_coverage.csv", index=False)
    print(table.to_string(index=False), flush=True)
    return table


def shared_neighbor_overlap():
    imp = pd.read_csv(PAPER / "imppat_similarity.csv")
    cyp_ids = set(pd.read_csv(CYP_PATH, usecols=["Molecule_Name"])["Molecule_Name"].astype(str))
    pxr_ids = set(pd.read_csv(PXR_PATH, usecols=["OCNT_ID"])["OCNT_ID"].astype(str))
    shared = cyp_ids & pxr_ids
    both = imp[(imp.consensus_CYP >= CUTOFF) & (imp.consensus_PXR >= CUTOFF)]
    either = both.neighbor_CYP.astype(str).isin(shared) | both.neighbor_PXR.astype(str).isin(shared)
    print(f"shared reference IDs {len(shared)}; close to both {len(both)}; neighbor is a shared ID {int(either.sum())}", flush=True)
    return len(shared), len(both), int(either.sum())


def unmapped_bias():
    failed_src = Path("/home/vipul.k/targetscape/IMPPAT/failed_pubchem_downloads.csv")
    failed_dst = IMPPAT / "failed_pubchem_downloads.csv"
    if failed_src.exists() and not failed_dst.exists():
        shutil.copy(failed_src, failed_dst)
    failed = pd.read_csv(failed_dst if failed_dst.exists() else failed_src)
    catalog = pd.read_csv(
        IMPPAT / "imppat.csv",
        usecols=["Phytochemical name", "IMPPAT Phytochemical identifier", "Indian medicinal plant"],
    )
    plants = catalog.groupby("Phytochemical name")["Indian medicinal plant"].nunique()
    failed_names = set(failed["Phytochemical name"])
    mapped = plants.index.difference(failed_names)
    summary = pd.DataFrame({
        "set": ["mapped", "unmapped"],
        "n_names": [len(mapped), int(plants.index.isin(failed_names).sum())],
        "median_plants": [float(plants.loc[mapped].median()), float(plants.loc[plants.index.isin(failed_names)].median())],
        "fraction_in_one_plant": [
            float((plants.loc[mapped] == 1).mean()),
            float((plants.loc[plants.index.isin(failed_names)] == 1).mean()),
        ],
    })
    summary.to_csv(PAPER / "unmapped_name_summary.csv", index=False)
    failed.to_csv(PAPER / "unmapped_names.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def ubiquitous_sensitivity():
    from potent_split import max_potency

    imp = pd.read_csv(PAPER / "imppat_similarity.csv")
    catalog = pd.read_csv(
        IMPPAT / "imppat.csv",
        usecols=["IMPPAT Phytochemical identifier", "Indian medicinal plant"],
    ).rename(columns={"IMPPAT Phytochemical identifier": "imppat_id", "Indian medicinal plant": "plant"})
    scored = imp.dropna(subset=["imppat_id"]).drop_duplicates("imppat_id")
    membership = catalog.merge(scored, on="imppat_id", how="inner").drop_duplicates(["plant", "imppat_id"])
    compounds = scored.set_index("imppat_id")
    nearest = {name: compounds[f"neighbor_potency_{name}"].map(max_potency) for name in TARGETS}
    rng = np.random.default_rng(0)
    rows = []
    for limit in (10, 20, 50):
        for name in TARGETS:
            rules = {
                "all_close": (compounds[f"consensus_{name}"] >= CUTOFF) & (compounds["n_plants"] <= limit),
                "nearest_potent": (
                    (compounds[f"consensus_{name}"] >= CUTOFF)
                    & (nearest[name] >= POTENT)
                    & (compounds["n_plants"] <= limit)
                ),
            }
            for rule, indicator in rules.items():
                _append_enrichment(rows, membership, indicator, limit, name, rule, rng)
    table = pd.DataFrame(rows)
    table.to_csv(PAPER / "ubiquitous_sensitivity.csv", index=False)
    return table


def _append_enrichment(rows, membership, indicator, limit, name, rule, rng):
    values = membership["imppat_id"].map(indicator).fillna(False).to_numpy()
    frame = pd.DataFrame({"plant": membership["plant"].to_numpy(), "hit": values})
    observed = frame.groupby("plant")["hit"].sum()
    sizes = frame.groupby("plant")["hit"].size()
    pool = indicator.reindex(membership["imppat_id"].unique()).fillna(False).to_numpy()
    total, population = int(pool.sum()), len(pool)
    tail = {
        int(size): rng.hypergeometric(total, max(population - total, 0), int(size), size=N_DRAWS)
        for size in np.unique(sizes.to_numpy())
    }
    records = []
    for plant, count in observed.items():
        if count <= 0 or total == 0:
            continue
        draws = tail[int(sizes[plant])]
        records.append((plant, int(count), float((draws >= count).mean())))
    result = pd.DataFrame(records, columns=["plant", "n_close", "p_value"])
    if result.empty:
        n_sig = 0
        named = ""
    else:
        result = result.sort_values("p_value")
        q = result["p_value"].to_numpy() * len(result) / np.arange(1, len(result) + 1)
        result["q_value"] = np.minimum.accumulate(q[::-1])[::-1]
        kept = result[(result.q_value < 0.05) & (result.n_close >= 2)]
        n_sig = int((result.q_value < 0.05).sum())
        named = "; ".join(f"{row.plant} ({int(row.n_close)})" for row in kept.itertuples())
    rows.append({
        "max_plants": limit,
        "anti_target": name,
        "rule": rule,
        "n_compounds": total,
        "n_plants_q_lt_0.05": n_sig,
        "plants_q_lt_0.05_at_least_two": named,
    })
    print(name, rule, limit, "compounds", total, "plants", n_sig, flush=True)


def curation():
    close = pd.read_csv(PAPER / "close_compounds.csv")
    close["curation"] = close["phytochemical_name"].map(classify)
    close["excluded_as_artifact"] = close["curation"].isin(["xenobiotic", "contaminant"])
    columns = ["phytochemical_name", "imppat_id", "n_plants", "curation", "excluded_as_artifact"]
    for name in TARGETS:
        columns.append(f"consensus_{name}")
    out = close[columns].sort_values(["curation", "phytochemical_name"])
    out.to_csv(PAPER / "close_compound_curation.csv", index=False)
    print(out.curation.value_counts().to_string(), flush=True)
    rows = []
    for name in TARGETS:
        mask = close[f"consensus_{name}"] >= CUTOFF
        kept = mask & ~close["excluded_as_artifact"]
        rows.append({
            "anti_target": name,
            "n_close": int(mask.sum()),
            "n_after_excluding_artifacts": int(kept.sum()),
            "n_xenobiotic_or_contaminant": int((mask & close["excluded_as_artifact"]).sum()),
            "n_uncertain_retained": int((mask & close["curation"].eq("uncertain")).sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(PAPER / "coverage_after_curation.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return close.loc[close.excluded_as_artifact, "imppat_id"].astype(str).tolist(), summary


def rescore():
    print("loading references", flush=True)
    sets = load_references()
    groups = {}
    for name, records in sets.items():
        groups[name] = {
            "morgan": [item["morgan"] for item in records],
            "atom": [item["atom_pair"] for item in records],
            "potency": np.asarray([scalar_potency(item) for item in records], dtype=float),
            "n": len(records),
        }
    for isoform in CYP_ISOFORMS:
        records = [item for item in sets["CYP"] if not np.isnan(item["pic50"][isoform])]
        groups[isoform] = {
            "morgan": [item["morgan"] for item in records],
            "atom": [item["atom_pair"] for item in records],
            "potency": np.asarray([item["pic50"][isoform] for item in records], dtype=float),
            "n": len(records),
        }
        print(f"{isoform} reference {len(records)}", flush=True)
    molecules = load_imppat()
    rows = []
    for number, molecule in enumerate(molecules, start=1):
        row = {"imppat_id": molecule["imppat_id"], "phytochemical_name": molecule["name"]}
        for name, bundle in groups.items():
            consensus, mean, max_potency, n_neighbors = score_query(
                molecule, bundle["morgan"], bundle["atom"], bundle["potency"]
            )
            row[f"consensus_{name}"] = consensus
            row[f"mean_{name}"] = mean
            row[f"max_potency_{name}"] = max_potency
            row[f"n_neighbors_{name}"] = n_neighbors
        rows.append(row)
        if number % 2000 == 0:
            print(f"rescored {number}", flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(PAPER / "alternative_similarity.csv", index=False)

    agg_rows = []
    for name in TARGETS:
        agg_rows.append({
            "anti_target": name,
            "n_reference": groups[name]["n"],
            "n_min_ge_0.80": int((detail[f"consensus_{name}"] >= CUTOFF).sum()),
            "n_mean_ge_0.80": int((detail[f"mean_{name}"] >= CUTOFF).sum()),
        })
    aggregation = pd.DataFrame(agg_rows)
    aggregation.to_csv(PAPER / "fingerprint_aggregation.csv", index=False)
    print(aggregation.to_string(index=False), flush=True)

    iso_rows = []
    for isoform in CYP_ISOFORMS:
        iso_rows.append({
            "isoform": isoform,
            "n_reference": groups[isoform]["n"],
            "n_close_0.80": int((detail[f"consensus_{isoform}"] >= CUTOFF).sum()),
            "fraction": float((detail[f"consensus_{isoform}"] >= CUTOFF).mean()),
        })
    isoforms = pd.DataFrame(iso_rows)
    isoforms.to_csv(PAPER / "cyp_isoform_coverage.csv", index=False)
    print(isoforms.to_string(index=False), flush=True)

    similarity = pd.read_csv(PAPER / "imppat_similarity.csv")
    from potent_split import max_potency
    potency_rows = []
    for name in TARGETS:
        nearest = similarity[f"neighbor_potency_{name}"].map(max_potency)
        close = similarity[f"consensus_{name}"] >= CUTOFF
        both = similarity[["imppat_id"]].copy()
        both["nearest"] = nearest.to_numpy()
        both = both.merge(detail[["imppat_id", f"max_potency_{name}"]], on="imppat_id", how="left")
        both = both.rename(columns={f"max_potency_{name}": "strongest"})
        sub = both.loc[close.to_numpy()]
        potency_rows.append({
            "anti_target": name,
            "n_close": int(close.sum()),
            "n_nearest_ge_5": int((close & (nearest >= POTENT)).sum()),
            "n_any_neighbor_ge_5": int((sub["strongest"] >= POTENT).sum()),
            "n_flips_to_potent": int(((sub["nearest"] < POTENT) & (sub["strongest"] >= POTENT)).sum()),
        })
    potency = pd.DataFrame(potency_rows)
    potency.to_csv(PAPER / "neighbor_potency_rules.csv", index=False)
    print(potency.to_string(index=False), flush=True)
    return detail


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    relative_coverage()
    shared_neighbor_overlap()
    unmapped_bias()
    curation()
    ubiquitous_sensitivity()
    rescore()


if __name__ == "__main__":
    main()
