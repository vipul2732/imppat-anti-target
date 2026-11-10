"""Publication figures, drawn from the saved result tables.

Each figure is multi-panel. Annotation inside axes is kept short and placed in
a boxed legend rather than as free text.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "paper"
CACHE = PAPER / "within_set_similarity.csv"
INPUT = ROOT / "input"
HERG_PATH = INPUT / "hERG_Dataset.csv"
CYP_PATH = INPUT / "cyp-challenge-TRAIN_inhibition.csv"
PXR_PATH = INPUT / "pxr-challenge_TRAIN.csv"
TARGETS = ("hERG", "CYP", "PXR")
COLORS = {"hERG": "#0B6FB5", "CYP": "#E08A00", "PXR": "#00876C", "Far": "#7A7A7A"}
CUTOFF = 0.80
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
CYP_ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
BOX = dict(frameon=True, framealpha=1.0, edgecolor="0.3", fancybox=False, borderpad=0.25, handlelength=1.2, handletextpad=0.4, labelspacing=0.25)


def style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12.5,
        "axes.linewidth": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.fontsize": 7.5,
        "legend.title_fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })


def panel_label(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")


def parent_mol(mol):
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    parent = max(frags, key=lambda f: f.GetNumAtoms())
    return parent if parent.GetNumAtoms() else None


def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None and "|" in str(smiles):
        mol = Chem.MolFromSmiles(str(smiles).split("|", 1)[0])
    return mol


def build_within_cache():
    if CACHE.exists():
        return pd.read_csv(CACHE)
    frames = []
    sources = {
        "hERG": (pd.read_csv(HERG_PATH), "SMILES", lambda row: float(row["PIC50"])),
        "CYP": (pd.read_csv(CYP_PATH), "SMILES", lambda row: float(np.nanmax([row[f"{iso}_pIC50_direct_inhibition"] for iso in CYP_ISOFORMS]))),
        "PXR": (pd.read_csv(PXR_PATH), "SMILES", lambda row: float(row["pEC50"])),
    }
    for name, (table, column, potency) in sources.items():
        print(f"within {name}", flush=True)
        fps, scaffolds, values = [], [], []
        seen = set()
        for _, row in table.iterrows():
            parent = parent_mol(mol_from_smiles(row[column]))
            if parent is None:
                continue
            try:
                key = Chem.MolToInchiKey(parent)
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            fps.append(MORGAN.GetFingerprint(parent))
            try:
                scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=parent))
            except ValueError:
                scaffolds.append("")
            values.append(potency(row))
        nn = np.zeros(len(fps))
        for i, fp in enumerate(fps):
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps), dtype=float)
            sims[i] = -1.0
            nn[i] = sims.max()
        frames.append(pd.DataFrame({"anti_target": name, "nn": nn, "potency": values, "scaffold": scaffolds}))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(CACHE, index=False)
    return out


def _smooth(grid):
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(grid, ((0, 0), (2, 2)), mode="edge")
    tmp = sum(kernel[k] * padded[:, k:k + grid.shape[1]] for k in range(5))
    padded = np.pad(tmp, ((2, 2), (0, 0)), mode="edge")
    return sum(kernel[k] * padded[k:k + grid.shape[0], :] for k in range(5))


def _density_contour(ax, x, y, color, xedges, yedges):
    hist, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    hist = _smooth(hist.T)
    total = hist.sum()
    if total <= 0:
        return
    hist = hist / total
    peak = hist.max()
    levels = peak * np.array([0.15, 0.35, 0.55, 0.78])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    for level, alpha in zip(levels, (0.10, 0.16, 0.24, 0.34)):
        ax.contourf(xc, yc, hist, levels=[level, peak * 1.01], colors=[color], alpha=alpha)
    ax.contour(xc, yc, hist, levels=levels, colors=[color], linewidths=[1.0, 1.25, 1.55, 1.9])


def figure1(within):
    fig = plt.figure(figsize=(12.6, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, letter, name in zip(axes, "ABC", TARGETS):
        subset = within[within.anti_target == name]
        median = float(subset["nn"].median())
        ax.hist(subset["nn"], bins=np.linspace(0, 1, 26), color=COLORS[name], edgecolor="white", linewidth=0.6)
        ax.axvline(median, color="black", linewidth=2.2, linestyle="--")
        ax.set_title(name, color=COLORS[name])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Nearest neighbor (Tanimoto)")
        legend_corner = "upper left" if name == "hERG" else "lower right"
        ax.legend(handles=[Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label=f"median {median:.2f}")], loc=legend_corner, **BOX)
        panel_label(ax, letter)
    axes[0].set_ylabel("Compounds")

    ax = fig.add_subplot(gs[1, 0])
    counts = within.groupby("anti_target").agg(n=("nn", "size"), scaffolds=("scaffold", lambda s: s.replace("", np.nan).nunique()))
    counts = counts.reindex(list(TARGETS))
    x = np.arange(3)
    ax.bar(x - 0.2, counts["n"], width=0.4, color=[COLORS[t] for t in TARGETS], edgecolor="black", linewidth=1.2, label="Compounds")
    ax.bar(x + 0.2, counts["scaffolds"], width=0.4, color=[COLORS[t] for t in TARGETS], alpha=0.45, edgecolor="black", linewidth=1.2, hatch="//", label="Scaffolds")
    ax.set_xticks(x, list(TARGETS))
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", **BOX)
    panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1])
    data = [within.loc[within.anti_target == name, "nn"].to_numpy() for name in TARGETS]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, name in zip(parts["bodies"], TARGETS):
        body.set_facecolor(COLORS[name])
        body.set_alpha(0.75)
        body.set_edgecolor("black")
        body.set_linewidth(1.2)
    box = ax.boxplot(data, widths=0.14, showfliers=False, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 2}, boxprops={"facecolor": "black"},
                     whiskerprops={"linewidth": 1.4}, capprops={"linewidth": 1.4})
    ax.set_xticks([1, 2, 3], list(TARGETS))
    ax.set_ylabel("Nearest neighbor (Tanimoto)")
    ax.set_ylim(0, 1)
    panel_label(ax, "E")

    ax = fig.add_subplot(gs[1, 2])
    xedges = np.linspace(0, 1, 31)
    yedges = np.linspace(0, 10, 31)
    for name in TARGETS:
        subset = within[within.anti_target == name]
        _density_contour(ax, subset["nn"].to_numpy(), subset["potency"].to_numpy(), COLORS[name], xedges, yedges)
    handles = [Line2D([0], [0], color=COLORS[name], linewidth=2.2, label=name) for name in TARGETS]
    ax.legend(handles=handles, loc="lower left", **BOX)
    ax.set_xlabel("Nearest neighbor (Tanimoto)")
    ax.set_ylabel("Measured potency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    panel_label(ax, "F")
    fig.savefig(PAPER / "figure1_within_set_diversity.png")
    plt.close(fig)


def figure2(frame, unblinded):
    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)

    ax = fig.add_subplot(gs[0, :2])
    for name in TARGETS:
        values = np.sort(frame[f"consensus_{name}"].to_numpy())
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, color=COLORS[name], linewidth=2.8, label=name)
    ax.axvline(CUTOFF, color="black", linewidth=1.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Similarity to nearest measured compound")
    ax.set_ylabel("Cumulative fraction of IMPPAT")
    ax.legend(loc="upper left", title="Anti-target", **BOX)
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 2])
    for name in TARGETS:
        values = np.sort(frame[f"consensus_{name}"].to_numpy())
        y = 1.0 - np.arange(1, len(values) + 1) / len(values)
        ax.semilogy(values, np.clip(y, 1e-5, None), color=COLORS[name], linewidth=2.6)
    ax.axvline(CUTOFF, color="black", linewidth=1.8, linestyle="--")
    ax.set_xlim(0.5, 1)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Fraction at or above")
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 26)
    for name in TARGETS:
        ax.hist(frame[f"consensus_{name}"], bins=bins, histtype="step", linewidth=2.6, color=COLORS[name], label=name)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Phytochemicals")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", **BOX)
    panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    close = {name: frame[f"consensus_{name}"] >= CUTOFF for name in TARGETS}
    sizes = [
        int((close["hERG"] & ~close["CYP"] & ~close["PXR"]).sum()),
        int((close["CYP"] & ~close["hERG"] & ~close["PXR"]).sum()),
        int((close["PXR"] & ~close["hERG"] & ~close["CYP"]).sum()),
        int((close["hERG"] & close["CYP"] & ~close["PXR"]).sum()),
        int((close["hERG"] & close["PXR"] & ~close["CYP"]).sum()),
        int((close["CYP"] & close["PXR"] & ~close["hERG"]).sum()),
        int((close["hERG"] & close["CYP"] & close["PXR"]).sum()),
    ]
    names = ["hERG", "CYP", "PXR", "hERG\nCYP", "hERG\nPXR", "CYP\nPXR", "all\nthree"]
    bar_colors = [COLORS["hERG"], COLORS["CYP"], COLORS["PXR"], "#6E5AA8", "#2F8FA8", "#9C8A20", "#444444"]
    ax.bar(np.arange(len(sizes)), sizes, color=bar_colors, edgecolor="black", linewidth=1.1)
    for index, value in enumerate(sizes):
        ax.text(index, value + 2, str(value), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(len(names)), names, fontsize=9)
    ax.set_ylabel("Phytochemicals close")
    ax.set_ylim(0, max(sizes) * 1.25)
    panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 2])
    merged = frame[["imppat_id", "consensus_PXR"]].merge(unblinded[["imppat_id", "consensus"]], on="imppat_id", how="inner")
    for values, color, label in (
        (merged["consensus_PXR"].to_numpy(), COLORS["PXR"], "Training set"),
        (merged["consensus"].to_numpy(), "#6B3FA0", "Unblinded test"),
    ):
        ordered = np.sort(values)
        y = 1.0 - np.arange(1, len(ordered) + 1) / len(ordered)
        ax.semilogy(ordered, np.clip(y, 1e-5, None), color=color, linewidth=2.6, label=label)
    ax.axvline(CUTOFF, color="black", linewidth=1.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Similarity to PXR")
    ax.set_ylabel("Fraction at or above")
    ax.legend(loc="upper right", **BOX)
    panel_label(ax, "E")
    fig.savefig(PAPER / "figure2_imppat_similarity.png")
    plt.close(fig)


def figure3(frame, robust, relative):
    fig = plt.figure(figsize=(12.6, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    cutoffs = sorted(robust["cutoff"].unique())
    for index, name in enumerate(TARGETS):
        for rule, marker, face in (("morgan", "o", "none"), ("consensus", "o", COLORS[name])):
            rows = robust[(robust.anti_target == name) & (robust.rule == rule)].sort_values("cutoff")
            x = np.arange(len(cutoffs)) + (index - 1) * 0.16 + (0.05 if rule == "consensus" else -0.05)
            y = rows["fraction"].to_numpy() * 100
            err = np.vstack([(y - rows["ci_low"] * 100), (rows["ci_high"] * 100 - y)])
            ax.errorbar(x, y, yerr=err, fmt=marker, markerfacecolor=face, markeredgecolor=COLORS[name],
                        color=COLORS[name], markersize=9, markeredgewidth=2, capsize=4, elinewidth=1.8)
    ax.set_xticks(np.arange(len(cutoffs)), [f"{c:.2f}" for c in cutoffs])
    ax.set_xlabel("Similarity cutoff")
    ax.set_ylabel("IMPPAT called close (%)")
    handles = [Patch(facecolor=COLORS[name], edgecolor="black", label=name) for name in TARGETS]
    handles += [
        Line2D([0], [0], marker="o", color="0.2", markerfacecolor="none", markeredgewidth=2, linestyle="None", markersize=9, label="Morgan"),
        Line2D([0], [0], marker="o", color="0.2", markerfacecolor="0.2", linestyle="None", markersize=9, label="Consensus"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=1, **BOX)
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    subset = frame.sample(n=min(6000, len(frame)), random_state=0)
    ax.scatter(subset["morgan_hERG"], subset["consensus_hERG"], s=6, alpha=0.2, color=COLORS["hERG"], linewidths=0, rasterized=True)
    flagged = frame[(frame["morgan_hERG"] >= CUTOFF) & (frame["consensus_hERG"] < CUTOFF)]
    ax.scatter(flagged["morgan_hERG"], flagged["consensus_hERG"], s=30, color="#C0392B", edgecolor="black", linewidth=0.7, zorder=3)
    ax.plot([0, 1], [0, 1], color="0.35", linewidth=1.4, linestyle=":")
    ax.axhline(CUTOFF, color="black", linewidth=1.5, linestyle="--")
    ax.axvline(CUTOFF, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Morgan similarity")
    ax.set_ylabel("Consensus similarity")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#C0392B", markeredgecolor="black", markersize=9, label=f"Reclassified (n={len(flagged)})")],
              loc="upper left", **BOX)
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    width = 0.36
    morgan = [int(robust[(robust.anti_target == n) & (robust.rule == "morgan") & (robust.cutoff == CUTOFF)]["n"].iloc[0]) for n in TARGETS]
    consensus = [int(robust[(robust.anti_target == n) & (robust.rule == "consensus") & (robust.cutoff == CUTOFF)]["n"].iloc[0]) for n in TARGETS]
    x = np.arange(3)
    ax.bar(x - width / 2, morgan, width=width, color=[COLORS[n] for n in TARGETS], alpha=0.42, edgecolor="black", linewidth=1.2, hatch="//")
    ax.bar(x + width / 2, consensus, width=width, color=[COLORS[n] for n in TARGETS], edgecolor="black", linewidth=1.2)
    for index, (left, right) in enumerate(zip(morgan, consensus)):
        ax.text(index - width / 2, left + 3, str(left), ha="center", fontsize=10, fontweight="bold")
        ax.text(index + width / 2, right + 3, str(right), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, list(TARGETS))
    ax.set_ylabel(f"Close at {CUTOFF:.2f}")
    ax.set_ylim(0, max(morgan) * 1.3)
    ax.legend(handles=[Patch(facecolor="0.7", edgecolor="black", hatch="//", label="Morgan"),
                       Patch(facecolor="0.3", edgecolor="black", label="Consensus")], loc="upper right", **BOX)
    panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    relative = relative.set_index("anti_target").reindex(list(TARGETS))
    width = 0.36
    fixed = relative["n_consensus_ge_0.80"].to_numpy()
    own = relative["n_consensus_ge_reference_median"].to_numpy()
    ax.bar(x - width / 2, fixed, width=width, color=[COLORS[n] for n in TARGETS], edgecolor="black", linewidth=1.2)
    ax.bar(x + width / 2, own, width=width, color=[COLORS[n] for n in TARGETS], alpha=0.45, edgecolor="black", linewidth=1.2, hatch="//")
    for index, (left, right) in enumerate(zip(fixed, own)):
        ax.text(index - width / 2, left + max(own) * 0.02, str(int(left)), ha="center", fontsize=10, fontweight="bold")
        ax.text(index + width / 2, right + max(own) * 0.02, str(int(right)), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, list(TARGETS))
    ax.set_ylabel("Phytochemicals")
    ax.set_ylim(0, max(own) * 1.25)
    ax.legend(handles=[
        Patch(facecolor="0.35", edgecolor="black", label="Consensus ≥ 0.80"),
        Patch(facecolor="0.75", edgecolor="black", hatch="//", label="At or above reference median"),
    ], loc="upper right", **BOX)
    panel_label(ax, "D")
    fig.savefig(PAPER / "figure3_cutoff_robustness.png")
    plt.close(fig)


def figure4(frame, effect, separation):
    groups = ("hERG", "CYP", "PXR", "Far")
    masks = {name: frame[f"consensus_{name}"] >= CUTOFF for name in TARGETS}
    masks["Far"] = (frame[[f"consensus_{n}" for n in TARGETS]] < CUTOFF).all(axis=1)

    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)
    panels = [("mw", "Molecular weight"), ("logp", "Calculated logP"), ("tpsa", "Polar surface area (Å²)")]
    for index, (column, label) in enumerate(panels):
        ax = fig.add_subplot(gs[0, index])
        data = [frame.loc[masks[g], column].to_numpy() for g in groups]
        box = ax.boxplot(data, widths=0.62, patch_artist=True, showfliers=False,
                         medianprops={"color": "white", "linewidth": 2.2},
                         boxprops={"linewidth": 1.3}, whiskerprops={"linewidth": 1.4}, capprops={"linewidth": 1.4})
        for patch, group in zip(box["boxes"], groups):
            patch.set_facecolor(COLORS[group])
        ax.set_xticks(np.arange(1, 5), list(groups))
        ax.set_ylabel(label)
        panel_label(ax, "ABC"[index])

    ax = fig.add_subplot(gs[1, 0])
    fractions = [float(frame.loc[masks[g], "basic_n"].ge(1).mean()) * 100 for g in groups]
    ax.bar(np.arange(4), fractions, color=[COLORS[g] for g in groups], edgecolor="black", linewidth=1.3)
    for index, value in enumerate(fractions):
        ax.text(index, value + 1.2, f"{value:.0f}%", ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(np.arange(4), list(groups))
    ax.set_ylabel("With basic nitrogen (%)")
    ax.set_ylim(0, max(fractions) * 1.3)
    panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1])
    order = ["mw", "logp", "tpsa", "rings", "basic_n"]
    pretty = {"mw": "MW", "logp": "logP", "tpsa": "TPSA", "rings": "Rings", "basic_n": "Basic N"}
    positions = []
    tick_labels = []
    row = 0
    for descriptor in order:
        for name in TARGETS:
            item = effect[(effect.group == name) & (effect.descriptor == descriptor)].iloc[0]
            scale = np.nanstd(frame[descriptor].to_numpy()) if descriptor != "basic_n" else 1.0
            center = item["difference"] / scale
            low = item["ci_low"] / scale
            high = item["ci_high"] / scale
            ax.errorbar(center, row, xerr=[[center - low], [high - center]], fmt="o", color=COLORS[name],
                        markersize=8, capsize=4, elinewidth=2, markeredgecolor="black", markeredgewidth=0.8)
            positions.append(row)
            tick_labels.append(pretty[descriptor] if name == "CYP" else "")
            row += 1
        row += 0.6
    ax.axvline(0, color="black", linewidth=1.6, linestyle="--")
    ax.set_yticks(positions, tick_labels)
    ax.set_xlabel("Standardized difference vs far set")
    ax.invert_yaxis()
    ax.legend(handles=[Patch(facecolor=COLORS[n], edgecolor="black", label=n) for n in TARGETS], loc="lower right", **BOX)
    panel_label(ax, "E")

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(3)
    observed = [float(separation[separation.anti_target == n]["cv_auc"].iloc[0]) for n in TARGETS]
    singles = [float(separation[separation.anti_target == n]["best_single_auc"].iloc[0]) for n in TARGETS]
    null = [float(separation[separation.anti_target == n]["permutation_mean_auc"].iloc[0]) for n in TARGETS]
    ax.bar(x - 0.22, singles, width=0.22, color=[COLORS[n] for n in TARGETS], alpha=0.45, edgecolor="black", linewidth=1.2, hatch="//")
    ax.bar(x, observed, width=0.22, color=[COLORS[n] for n in TARGETS], edgecolor="black", linewidth=1.2)
    ax.bar(x + 0.22, null, width=0.22, color="0.75", edgecolor="black", linewidth=1.2)
    for index, value in enumerate(observed):
        ax.text(index, value + 0.015, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="black", linewidth=1.5, linestyle=":")
    ax.set_xticks(x, list(TARGETS))
    ax.set_ylabel("Cross-validated AUC")
    ax.set_ylim(0.4, 1.0)
    ax.legend(handles=[Patch(facecolor="0.7", edgecolor="black", hatch="//", label="Best single"),
                       Patch(facecolor="0.3", edgecolor="black", label="All 24"),
                       Patch(facecolor="0.75", edgecolor="black", label="Permuted")], loc="upper left", ncol=1, **BOX)
    panel_label(ax, "F")
    fig.savefig(PAPER / "figure4_close_vs_far.png")
    plt.close(fig)


def figure5(enrichment):
    restricted = enrichment[enrichment.compound_set == "restricted"]
    every = enrichment[enrichment.compound_set == "all"]
    fig = plt.figure(figsize=(12.8, 8.0))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.55, height_ratios=[1.35, 1])

    for index, name in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, index])
        subset = restricted[(restricted.anti_target == name) & (restricted.q_value < 0.05)].sort_values("n_close").tail(10)
        if subset.empty:
            ax.set_axis_off()
            continue
        y = np.arange(len(subset))
        ax.barh(y, subset["n_close"], color=COLORS[name], edgecolor="black", linewidth=1.2, height=0.68)
        ax.plot(subset["expected"], y, marker="d", linestyle="None", color="#B3200E", markersize=8,
                markeredgecolor="black", markeredgewidth=0.8, zorder=3)
        ax.set_yticks(y, subset["plant"], fontsize=9.5, fontstyle="italic")
        ax.set_xlabel("Close constituents")
        ax.set_title(name, color=COLORS[name])
        ax.set_xlim(0, subset["n_close"].max() * 1.2)
        if index == 0:
            ax.legend(handles=[Line2D([0], [0], marker="d", color="none", markerfacecolor="#B3200E",
                                      markeredgecolor="black", markersize=9, label="Expected")], loc="lower right", **BOX)
            panel_label(ax, "A")

    ax = fig.add_subplot(gs[1, 0])
    width = 0.36
    x = np.arange(3)
    before = [int((every[(every.anti_target == n)]["q_value"] < 0.05).sum()) for n in TARGETS]
    after = [int((restricted[(restricted.anti_target == n)]["q_value"] < 0.05).sum()) for n in TARGETS]
    ax.bar(x - width / 2, before, width=width, color="0.72", edgecolor="black", linewidth=1.2)
    ax.bar(x + width / 2, after, width=width, color=[COLORS[n] for n in TARGETS], edgecolor="black", linewidth=1.2)
    for index, (left, right) in enumerate(zip(before, after)):
        ax.text(index - width / 2, left + 12, str(left), ha="center", fontsize=10, fontweight="bold")
        ax.text(index + width / 2, right + 12, str(right), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, list(TARGETS))
    ax.set_ylabel("Plants at q < 0.05")
    ax.set_ylim(0, max(before) * 1.25)
    ax.legend(handles=[Patch(facecolor="0.72", edgecolor="black", label="All compounds"),
                       Patch(facecolor="0.3", edgecolor="black", label="≤20 plants")], loc="upper left", **BOX)
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 1])
    for name in TARGETS:
        subset = restricted[restricted.anti_target == name]
        ax.scatter(subset["expected"], subset["n_close"], s=46, color=COLORS[name], alpha=0.78,
                   edgecolor="black", linewidth=0.7)
    limit = max(restricted["n_close"].max(), restricted["expected"].max()) * 1.1
    ax.plot([0, limit], [0, limit], color="0.35", linestyle=":", linewidth=1.6)
    ax.set_xlabel("Expected under null")
    ax.set_ylabel("Observed close")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.legend(handles=[Patch(facecolor=COLORS[n], edgecolor="black", label=n) for n in TARGETS], loc="upper left", **BOX)
    panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 2])
    for name in TARGETS:
        subset = restricted[restricted.anti_target == name]
        values = np.sort(subset["q_value"].to_numpy())
        y = np.arange(1, len(values) + 1)
        ax.step(values, y, where="post", color=COLORS[name], linewidth=2.6, label=name)
    ax.axvline(0.05, color="black", linewidth=1.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("q value")
    ax.set_ylabel("Plants (cumulative)")
    ax.legend(loc="upper left", **BOX)
    panel_label(ax, "D")
    fig.savefig(PAPER / "figure5_plant_enrichment.png")
    plt.close(fig)


def main():
    style()
    frame = pd.read_csv(PAPER / "imppat_similarity.csv")
    unblinded = pd.read_csv(PAPER / "pxr_unblinded_similarity.csv")
    robust = pd.read_csv(PAPER / "cutoff_fractions.csv")
    relative = pd.read_csv(PAPER / "relative_coverage.csv")
    effect = pd.read_csv(PAPER / "descriptor_differences.csv")
    separation = pd.read_csv(PAPER / "separation_summary.csv")
    enrichment = pd.read_csv(PAPER / "plant_enrichment.csv")
    within = build_within_cache()
    for name, function, args in (
        ("figure1", figure1, (within,)),
        ("figure2", figure2, (frame, unblinded)),
        ("figure3", figure3, (frame, robust, relative)),
        ("figure4", figure4, (frame, effect, separation)),
        ("figure5", figure5, (enrichment,)),
    ):
        function(*args)
        print(f"{name} done", flush=True)
    old = PAPER / "figure2_imppat_ecdf.png"
    if old.exists():
        old.unlink()


if __name__ == "__main__":
    main()
