# Most phytochemicals of Indian medicinal plants lack a close structural precedent in measured hERG, CYP, and PXR sets

Draft, 24 September 2026. Numbers are from `results/paper/`; every value below is
reproduced by the scripts in `src/`.

## Abstract

Safety assessment of traditional medicinal plants is usually framed as a
prediction problem: estimate an anti-target liability for each constituent.
We show that for the Indian Medicinal Plants, Phytochemistry and Therapeutics
(IMPPAT) collection this framing is unsupported, because almost none of the
constituents are structurally represented in the assays that would license a
prediction. Using three OpenADMET and ChEMBL-derived reference sets with
measured potencies — hERG blockade (8,877 compounds), cytochrome P450
inhibition (4,905), and pregnane-X receptor induction (4,138) — we compute for
each of 14,956 unique IMPPAT structures the similarity to its nearest measured
compound, requiring agreement between Morgan and atom-pair fingerprints. Only
0.92%, 0.30%, and 0.46% of phytochemicals have a neighbor at Tanimoto 0.80 or
above for hERG, CYP, and PXR respectively. This close minority is a
distinguishable chemical group: a logistic model over 24 two-dimensional
descriptors separates close from far structures with cross-validated AUC
0.73–0.80, against 0.50 under label permutation. At the plant level, a
hypergeometric test that holds each plant's constituent count fixed identifies
12, 6, and 4 significantly enriched species after compounds recorded from more
than 20 plants are removed; without that removal the counts are 566, 864, and
860. Requiring the nearest measured neighbor to reach pIC50 or pEC50 of at
least 5 leaves 51, 16, and 11 phytochemicals. The plants that remain are
*Rauvolfia*, *Sinopodophyllum*, *Dysosma*, and *Cinchona* for hERG, the two
lignan plants for CYP, and *Catharanthus roseus* for PXR. *Digitalis lanata*
and the CYP signal for *Cinchona* do not survive that potency requirement. We
conclude that anti-target concern in this pharmacopeia is chemically justified
for a narrow and already-known set of taxa, and that for the remaining species
the correct statement is absence of evidence rather than evidence of safety.

## 1. Introduction

Cytochrome P450 inhibition, hERG blockade, and PXR induction are the
liabilities most often invoked when traditional remedies are screened
computationally. The usual workflow trains a quantitative structure–activity
model on a public dataset and applies it to a phytochemical library. The
implicit assumption is that the model's training chemistry and the plant
chemistry overlap.

We test that assumption directly rather than assuming it. The question is not
which phytochemicals are predicted to be liabilities, but which phytochemicals
are close enough to a measured compound for any prediction to be meaningful.
This reframing matters because the failure mode of the standard workflow is
silent: a regression model returns a plausible number for every input,
including inputs far outside its domain.

Two results motivate the reframing. First, on the hERG reference set used here,
we measured how prediction error varies with distance to the training set
(Supplementary Section S1): error falls monotonically as the nearest training
neighbor approaches Tanimoto 0.80 within the set, but on a held-out external
split the relationship disappears, and compounds with a close training neighbor
still carry a median absolute error near 0.5 log units. Second, the
architecture does not rescue this. Morgan-fingerprint random forests, a
Tanimoto Gaussian process, and RDKit two-dimensional descriptor models reach
r² of 0.19, 0.29, and 0.41 on the published 499-compound external test, against
0.67 for the published PaDEL descriptor model on the identical split. Every
variant leaves 0.85–0.93 log units of error on external compounds.

## 2. Methods

### 2.1 Reference sets

**hERG.** 8,879 compounds with pIC50 from the ToxTree dataset (Arab and
Barakat, arXiv:2112.13467), retaining the authors' Dev-set and Test-set
assignment. 8,877 unique structures after standardization. No per-measurement
uncertainty is distributed with this set.

**CYP.** The OpenADMET CYP Challenge training table
(`openadmet/cyp-challenge-train-test`), 4,905 compounds with direct-inhibition
pIC50 for CYP1A2, CYP2C9, CYP2D6, and CYP3A4, each with a 95% confidence
interval and a standard deviation. Labels are sparse: every compound has at
least one isoform measured, 41 have all four. A phytochemical called CYP-close
is close to a compound measured on at least one of those isoforms. It is not
a claim of isoform-independent inhibition. Coverage within each isoform is
reported separately.

**PXR.** The OpenADMET PXR Challenge training table
(`openadmet/pxr-challenge-train-test`), 4,139 compounds with pEC50 for receptor
induction and confidence intervals; 4,138 unique structures. 1,370 of these
compounds also appear in the CYP table, so CYP and PXR results are not
independent, and we report their intersection explicitly. The 513-compound
test set, released with labels after the challenge (253 in phase 1 and 260 in
phase 2), was scored with the same consensus rule and kept separate from this
reference set.

The endpoints differ in kind — channel blockade, enzyme inhibition, and
receptor induction — and are never pooled.

### 2.2 Query set

IMPPAT records 189,386 plant–part–compound associations covering 4,010 plants
and 17,967 phytochemicals. Structures were retrieved from PubChem by compound
name, yielding 15,119 SDF files and 14,956 unique structures after taking the
largest covalent fragment, which drops salts and other smaller components, and
deduplicating on the full InChIKey, which keeps stereoisomers distinct.
Tautomers were not canonicalized, and formal charges were not normalized.
2,835 names failed retrieval and are excluded. Those unmapped names are
recorded from a median of one plant, and 80% occur in a single plant, compared
with 58% of mapped names, so the missing structures are concentrated among
sparsely reported entries rather than among widespread metabolites. The names
are listed in `results/paper/unmapped_names.csv`.

### 2.3 Consensus similarity

For every IMPPAT structure and every reference set we compute the Tanimoto
similarity to all reference compounds under two independent fingerprints:
Morgan (radius 2, 2,048 bits) and hashed atom-pair (2,048 bits). We take the
elementwise minimum of the two and retain the maximum over reference compounds.

The second fingerprint is not decorative. Morgan fingerprints at radius 2
cannot distinguish long unbranched aliphatic chains: oleic and palmitoleic
acid, which differ by two methylene units, receive Tanimoto 1.0. Thirty
distinct IMPPAT fatty acids collapse onto a single hERG reference compound
under Morgan alone. Atom-pair similarity encodes topological distance and
assigns the same pair 0.79. Requiring both fingerprints to agree removes this
class of artifact. Across the three reference sets it reduces the count of
close structures from 212, 85, and 147 to 137, 45, and 69.

A structure is called *close* when the consensus similarity is at least 0.80.
The cutoff is inherited from the analog-pair analysis of the hERG set and is
not tuned on IMPPAT, which has no anti-target labels on which to optimize it.
Sensitivity at 0.70 and 0.90 is reported in Figure 3. Replacing the minimum
with the mean of the two fingerprints raises the 0.80 counts from 137, 45,
and 69 to 188, 67, and 108. Both rules leave more than 98% of the library
without a close neighbor.

The same absolute cutoff sits in a different part of each reference set.
Within-set Morgan nearest-neighbor similarity has median 0.79 for hERG, 0.45
for CYP, and 0.39 for PXR, so 0.80 is near the middle of the hERG set and in
the far tail of the CYP and PXR sets. Figure 3D therefore also counts IMPPAT
structures whose consensus similarity reaches that set-specific median. The
median IMPPAT consensus similarity itself falls at the 1st percentile of the
hERG within-set distribution, the 0.6th percentile of CYP, and the 2nd
percentile of PXR.

### 2.4 Uncertainty

Fractions of the library called close carry 95% bootstrap confidence intervals
over 2,000 resamples of the 14,956 structures. Descriptor mean differences
between close and far groups carry 95% intervals from independent bootstrap
resampling of each group. For a compound that passes the cutoff we report the
measured potency of its nearest reference neighbor together with that
measurement's assay confidence interval where available (CYP and PXR; the hERG
set does not distribute one). This interval describes the reference
measurement. It is not a predicted potency for the phytochemical.

### 2.5 Descriptor separation

We computed 24 two-dimensional descriptors spanning size, lipophilicity,
polarity, hydrogen bonding, ring systems, saturation, topological complexity,
and ionizable nitrogen. For each anti-target we scored every single descriptor,
every pair and triple among the eight strongest singles, and the full set, by
five-fold cross-validated AUC using L2-regularized logistic regression with
balanced class weights and per-fold standardization. The selected model was
compared against 200 permutations of the class labels.

### 2.6 Plant enrichment

For each plant we count constituents called close and compare against a
hypergeometric null that holds the plant's recorded constituent count fixed and
resamples which compounds it carries (20,000 draws per distinct count).
Benjamini–Hochberg correction is applied across plants. We report the analysis
twice: over all close compounds, and restricted to close compounds recorded
from at most 20 plants. The same test at caps of 10 and 50 plants is reported
alongside that choice. The restriction is essential — ubiquitous metabolites
are recorded across hundreds of species and carry no information about any
particular source.

A second pass keeps only those close constituents whose nearest measured
neighbor has pIC50 or pEC50 of at least 5 (10 µM). For CYP the value is the
maximum across isoforms that were actually measured, so activity on any one
isoform is enough. The same pass is repeated using the highest potency among
all reference neighbors at consensus similarity ≥ 0.80, rather than only the
nearest neighbor. Plants whose only signal
is a single such constituent are reported in the table and are not treated as
separate discoveries.

All analyses use RDKit 2026.03.6, scikit-learn 1.9.1, and Python 3.11.

## 3. Results

### 3.1 The three reference sets differ in internal diversity (Figure 1)

The hERG set is analog-rich: median nearest-neighbor similarity 0.79 across
8,877 compounds in 4,427 Murcko scaffolds, consistent with accumulated
medicinal-chemistry series. The CYP and PXR sets are diversity libraries, with
medians of 0.45 and 0.39 and scaffold counts of 4,519 and 3,669 for 4,905 and
4,138 compounds — nearly one scaffold per compound.

This difference constrains interpretation downstream. A phytochemical distant
from the CYP set is distant from a deliberately diverse sampling of chemical
space; the same distance from the hERG set may only mean it falls outside the
particular series pharmaceutical programs happened to pursue.

### 3.2 Phytochemicals are overwhelmingly distant from all three sets (Figure 2, Table 1)

Median nearest-neighbor similarity from IMPPAT is 0.28 to hERG, 0.26 to CYP,
and comparable to PXR. The cumulative distributions reach 0.99 before the 0.80
threshold for all three. Against the 513 unblinded PXR test compounds the
highest consensus similarity in IMPPAT is 0.42 (Figure 2E).

| Anti-target | Reference compounds | Internal median NN | Close at 0.80 | Fraction (95% CI) |
|---|---:|---:|---:|---|
| hERG | 8,877 | 0.79 | 137 | 0.92% (0.78–1.07) |
| CYP | 4,905 | 0.45 | 45 | 0.30% (0.22–0.39) |
| PXR | 4,138 | 0.39 | 69 | 0.46% (0.35–0.58) |
| hERG ∩ CYP | — | — | 23 | 0.15% |
| hERG ∩ PXR | — | — | 20 | 0.13% |
| CYP ∩ PXR | — | — | 9 | 0.06% |
| All three | — | — | 7 | 0.05% |

Intersections are small. All 9 phytochemicals that are close to both CYP and
PXR have a nearest neighbor whose identifier is one of the 1,370 compounds
present in both reference tables, so that intersection is shared chemistry
rather than two independent matches.

### 3.3 The consensus requirement removes a systematic artifact (Figure 3)

At every cutoff the two-fingerprint rule returns fewer close structures than
Morgan alone, and the gap widens as the cutoff rises. At 0.90 Morgan alone
calls 140 hERG-close structures, of which only 106 survive. The discarded
structures are dominated by aliphatic chain compounds. Because Morgan
fingerprints are the default in this literature, this artifact would inflate
apparent phytochemical–anti-target overlap in any study using them alone.
Panel D compares the absolute cutoff of 0.80 with each reference set's own
median nearest-neighbor similarity.

### 3.4 Close phytochemicals are a distinguishable chemical group (Figure 4)

The full 24-descriptor model separates close from far structures with
cross-validated AUC 0.77 (hERG), 0.73 (CYP), and 0.80 (PXR). Label permutation
gives 0.49–0.50 in all three cases, with no permutation reaching the observed
value in 200 draws.

No small combination matches the full model. The strongest single descriptors
are aromatic ring count for hERG (AUC 0.67), fraction of sp3 carbons for CYP
(0.62), and Bertz complexity for PXR (0.59). The best triples reach 0.75, 0.65,
and 0.70. Separation is therefore distributed across many weakly informative
descriptors rather than driven by one property.

Directionally, hERG-close phytochemicals carry more rings (difference 0.86,
95% CI 0.45–1.28) and more often an ionizable nitrogen (0.30, 0.22–0.38) —
the recognized hERG pharmacophore of a basic amine with aromatic bulk.
PXR-close phytochemicals are less lipophilic than the far set (logP difference
−0.77, −1.40 to −0.18) and also nitrogen-enriched (0.20, 0.10–0.32). CYP-close
phytochemicals are smaller (molecular weight −42, −77 to −6) and less polar
(TPSA −17, −27 to −6).

An AUC of 0.73–0.80 means substantial overlap remains. The descriptor profile
describes a tendency, not a decision rule. Because a structure is labeled
close by its similarity to the reference compounds, the model is describing
the chemistry those assays have already measured. It is not an independent
demonstration that the close phytochemicals form a biological class.

### 3.5 A narrow set of plants is enriched, and the correction is decisive (Figure 5)

Without restricting ubiquitous compounds, 566, 864, and 860 plants reach
q < 0.05 for hERG, CYP, and PXR. Restricting to compounds recorded from at most
20 plants reduces this to 12, 6, and 4. A cap of 10 plants gives 8, 3, and 7.
A cap of 50 gives 16, 6, and 5. The collapse from several hundred plants does
not depend on the particular cap of 20. The overwhelming majority of apparent
enrichment reflects widely distributed metabolites, not plant-specific
chemistry.

The surviving species are recognized producers of pharmacologically active
scaffolds:

| Anti-target | Plant | Constituents | Close | Expected | q |
|---|---|---:|---:|---:|---:|
| hERG | *Catharanthus roseus* | 384 | 15 | 3.05 | <0.001 |
| hERG | *Rauvolfia serpentina* | 70 | 13 | 0.57 | <0.001 |
| hERG | *Rauvolfia tetraphylla* | 33 | 9 | 0.27 | <0.001 |
| hERG | *Sinopodophyllum hexandrum* | 48 | 7 | 0.39 | <0.001 |
| hERG | *Dysosma aurantiocaulis* | 10 | 3 | 0.08 | <0.001 |
| hERG | *Uncaria macrophylla* | 3 | 3 | 0.02 | <0.001 |
| hERG | *Ephedra gerardiana* | 11 | 3 | 0.08 | 0.008 |
| CYP | *Sinopodophyllum hexandrum* | 48 | 9 | 0.12 | <0.001 |
| CYP | *Dysosma aurantiocaulis* | 10 | 3 | 0.02 | <0.001 |
| CYP | *Cinchona calisaya* | 27 | 3 | 0.06 | 0.001 |
| CYP | *Cinchona hybrida* | 4 | 2 | 0.01 | 0.001 |
| PXR | *Digitalis lanata* | 60 | 5 | 0.22 | <0.001 |
| PXR | *Catharanthus roseus* | 384 | 11 | 1.40 | <0.001 |
| PXR | *Rauvolfia serpentina* | 70 | 4 | 0.25 | 0.011 |

Full listings are in `plant_enrichment.csv`; per-compound assignments with
plant part in `plant_close_compounds.csv`.

These plants are the close set before any requirement on the neighbor's potency. *Catharanthus* and *Rauvolfia* contribute
monoterpene indole alkaloids (vinca alkaloids, ajmaline, reserpine analogs).
*Sinopodophyllum* and *Dysosma* contribute aryltetralin lignans of the
podophyllotoxin class, and appear under both hERG and CYP. *Cinchona* species
contribute quinoline alkaloids. *Digitalis lanata* contributes cardiac
glycosides. *Ephedra* contributes phenylethylamines. Section 3.6 keeps only the plants whose neighbors are potent.

That the method recovers these taxa from structure alone, using no activity
information about any plant compound, is the internal validation of the
negative result. The approach is capable of finding known cases; the
remaining ~3,900 species are not flagged because their chemistry is genuinely
unrepresented in these assays.

### 3.6 Most close neighbors are not potent

Closeness does not imply that the measured neighbor is active. Of the 137, 45,
and 69 consensus-close phytochemicals, 51, 16, and 11 have a neighbor at pIC50
or pEC50 of at least 5. The median neighbor potency among all close compounds
is 4.85 for hERG, 4.70 for CYP, and 2.50 for PXR. The PXR close set is mostly
a match to weak inducers. Replacing the nearest neighbor with the most potent
neighbor that still has consensus similarity ≥ 0.80 leaves the hERG and PXR
counts at 51 and 11. For CYP it raises the count from 16 to 19. The three
additions are quinine, quinidine, and epiquinine: their nearest CYP neighbor
is below 5, and another neighbor above 0.80 has pIC50 7.47. Under the
nearest-neighbor rule, *Cinchona* does not remain in the CYP plant list.

Repeating the plant test on this potent subset, still excluding compounds
recorded from more than 20 plants, changes who is named.

| Anti-target | Close | Neighbor ≥ 5 | Plants at q < 0.05 with at least two such constituents |
|---|---:|---:|---|
| hERG | 137 | 51 | *Rauvolfia serpentina* (5 vs 0.22), *R. tetraphylla* (4 vs 0.10), *Sinopodophyllum hexandrum* (7 vs 0.15), *Dysosma aurantiocaulis* (3 vs 0.03), *Cinchona calisaya* (3 vs 0.09), *Alangium salviifolium* (3 vs 0.28), *Carapichea ipecacuanha*, *Croton tiglium*, *Camellia sinensis* |
| CYP | 45 | 16 | *Sinopodophyllum hexandrum* (7 vs 0.04), *Dysosma aurantiocaulis* (3 vs 0.01) |
| PXR | 69 | 11 | *Catharanthus roseus* (9 vs 0.28) |

*Cinchona* no longer stands out for CYP, and *Digitalis lanata* no longer
stands out for PXR. Nine of the eleven potent PXR neighbors are vinca
alkaloids from *Catharanthus*, all matching one reference compound at pEC50
5.64. Cytisine (neighbor pEC50 6.09) accounts for the remaining single-compound
rows under several *Sophora* species; one constituent in a short plant list
meets q < 0.05 by arithmetic and is not a separate result. The full potent
table is `plant_enrichment_potent.csv`.

### 3.7 Records that are chemically close but biologically implausible

All 206 unique structures within 0.80 of at least one reference set were
classified as a credible phytochemical (170), a xenobiotic or drug recorded
under a plant name (20), an industrial or combustion contaminant (8), or
uncertain (8). Examples of the second and third classes are quinacrine,
dipyridamole, cyproheptadine, triphenylphosphine, and fluoranthene. Removing
the 28 xenobiotics and contaminants lowers the close counts from 137, 45, and
69 to 113, 40, and 65. The uncertain records, including several mammalian
steroids, are retained. The classifications are in
`results/paper/close_compound_curation.csv`. The headline counts above keep
every close structure, so the coverage fraction is not improved by deleting
awkward rows.

### 3.8 The coverage claim does not depend on one absolute cutoff

Figure 3D compares two definitions. Consensus similarity ≥ 0.80 calls 137, 45,
and 69 phytochemicals. Reaching the reference set's own median nearest-neighbor
similarity calls 142, 859, and 1,643. For hERG the two bars match, because
0.80 is essentially that set's median. For CYP and PXR, 0.80 is far above the
similarity those diversity sets have internally, so the 99% figure is partly
the cutoff. What does not move is the location of a typical IMPPAT structure:
its consensus similarity is at about the 1st percentile of each reference
set's own nearest-neighbor distribution, and 80%, 76%, and 65% of IMPPAT fall
below the 5th percentile of the hERG, CYP, and PXR distributions.

The CYP total splits by isoform, using only compounds with a measurement on
that enzyme: 7 of 1,412 for CYP1A2, 25 of 1,285 for CYP2C9, 11 of 1,493 for
CYP2D6, and 21 of 2,335 for CYP3A4. The union of those four calls is the same
45 structures as the pooled analysis.

## 4. Discussion

The central claim is negative and quantitative: under the 0.80 consensus rule,
99% or more of this phytochemical collection has no close structural precedent
in the compounds these three assays have measured. A typical structure remains
in the extreme tail of each reference set when the cutoff is taken from that
set's own nearest-neighbor distribution. Applying a trained anti-target
model to the full library would return a number for every compound while
carrying no close experimental precedent for the overwhelming majority.

This is not an argument against modeling. It is an argument about where a model
applies. For the 137, 45, and 69 phytochemicals with a genuine close neighbor,
the measured potency of that neighbor is the informative quantity, and only
51, 16, and 11 of those neighbors reach 10 µM. For the rest, distance to the measured set is the
honest reportable quantity.

The asymmetry between reference sets deserves emphasis. The CYP and PXR sets
are diverse by construction, so a fixed similarity of 0.80 is a stricter bar
for them than for the series-dominated hERG set. Figure 3D is the comparison
that uses each set's own median instead.

**Limitations.** Structures were retrieved by compound name and 2,835 names
failed, so coverage is incomplete. The unmapped names are disproportionately
single-plant entries. IMPPAT records literature occurrence, not concentration,
bioavailability, or dose, so a plant's constituent list reflects investigation
effort as much as composition, and a close in vitro neighbor is not a clinical
toxicity. The fixed-count null addresses list length. It does not supply the
missing exposure. The 0.80 cutoff is a convention from the hERG analog analysis.
Figure 3 shows the library-level conclusion at 0.70, at 0.90, and at each
reference set's own median nearest-neighbor similarity. Two-dimensional
fingerprints do not capture protonation, conformational preference, or a shared
binding geometry between structures that are distant in the fingerprint. A
phytochemical without a close neighbor in these assays may still block hERG,
inhibit a CYP isoform, or induce PXR. Nothing here licenses a safety claim
about any plant or preparation.

**Implication for practice.** Studies that dock or score entire phytochemical
libraries against anti-target proteins should report the distance of each
compound to the relevant measured set alongside the score. Without it, a
ranking over compounds that are uniformly far from any training or reference
chemistry conveys precision the underlying data cannot support.

## 5. Conclusion

Across 14,956 phytochemical structures and three measured anti-target sets,
fewer than 1% have a consensus neighbor at 0.80, and a typical structure still
sits at about the 1st percentile of each reference set's own nearest-neighbor
distribution. The close minority is the portion of IMPPAT that resembles
chemistry the assays have already measured. Once the neighbor is required to
be potent, that minority concentrates in *Rauvolfia*, the podophyllotoxin
plants, and *Catharanthus roseus*. For the remainder of this pharmacopeia,
the appropriate conclusion is that a close experimental precedent for an
anti-target assessment does not yet exist.

## Data and code

| Item | Path |
|---|---|
| Per-compound similarity and descriptors | `results/paper/imppat_similarity.csv` |
| Similarity to the 513 unblinded PXR test compounds | `results/paper/pxr_unblinded_similarity.csv` |
| Close compounds with neighbors and assay intervals | `results/paper/close_compounds.csv` |
| Table 1 | `results/paper/table1_summary.csv` |
| Cutoff sensitivity | `results/paper/cutoff_fractions.csv` |
| Coverage relative to each reference set's own nearest neighbor | `results/paper/relative_coverage.csv` |
| Minimum versus mean of the two fingerprints, and CYP isoforms | `results/paper/fingerprint_aggregation.csv`, `results/paper/cyp_isoform_coverage.csv` |
| Strongest neighbor at consensus ≥ 0.80 | `results/paper/neighbor_potency_rules.csv` |
| Ubiquitous-compound caps of 10, 20, and 50 | `results/paper/ubiquitous_sensitivity.csv` |
| Curation of the 206 close structures | `results/paper/close_compound_curation.csv` |
| Names that failed structure retrieval | `results/paper/unmapped_names.csv` |
| Descriptor separation | `results/paper/separation_summary.csv`, `descriptor_combinations_*.csv` |
| Descriptor group differences | `results/paper/descriptor_differences.csv` |
| Plant enrichment | `results/paper/plant_enrichment.csv` |
| Potent-neighbor plant enrichment | `results/paper/plant_enrichment_potent.csv` |
| Per-plant close constituents | `results/paper/plant_close_compounds.csv` |
| Figures 1–5 | `results/paper/figure[1-5]_*.png` |

Scripts: `src/paper_analysis.py`, `src/figures.py`,
`src/separate_close_far.py`, `src/plant_enrichment.py`, `src/potent_split.py`,
`src/revision_checks.py`.
`reproduce.ipynb` runs these in order from the files in `input/`.
Supporting hERG modeling in `../herg_cliffs/src/`. The Morgan-only overlap that
preceded the two-fingerprint rule is in `archive_superseded/` and is not used.

## Supplementary

**S1. Prediction error against distance to the training set (hERG).** Under
cluster holdout, Gaussian-process RMSE falls from 0.86 for compounds whose
nearest training neighbor is below Tanimoto 0.30 to 0.60 in the 0.70–0.80 bin.
On the published external test split the relationship does not hold: binned
RMSE is 0.91, 1.07, 0.79, 0.76, 1.09, 0.89, and 0.68 across ascending
similarity bins, and the 29 test compounds with a training neighbor at 0.80 or
above retain a median absolute error of 0.50 log units.
Source: `../herg_cliffs/results/applicability/error_by_similarity.csv`.

**S2. Model comparison on the published hERG split (N = 499).**

| Model | r² | MSE |
|---|---:|---:|
| Published PaDEL random forest (Arab and Barakat) | 0.67 | 0.59 |
| RDKit 2D descriptors, gradient boosting | 0.41 | 0.72 |
| RDKit 2D descriptors, random forest | 0.40 | 0.74 |
| Morgan fingerprint Gaussian process | 0.29 | 0.87 |
| Morgan fingerprint random forest | 0.19 | 0.99 |

Descriptors outperform fingerprints on this endpoint, and no variant tested
here reproduces the published result.
Source: `../herg_cliffs/results/descriptors/regression_metrics.csv`.