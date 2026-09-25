# Third-party notices and data-license boundaries

This repository contains the study's computational implementation and reproduction utilities. The authors' MIT grant is in `LICENSE`, with material and path exclusions in `LICENSE_NOTES.md`. It applies only to rights they may grant and does not relicense upstream datasets, third-party code, fitted artifacts or outputs subject to other terms. This notice provides attribution and scope information, not an additional licence grant.

Dependencies should be installed from their original distributions and retain their licenses. Copies of upstream notices in `third_party_licenses/` document sources used by the reproduction workflow; their presence does not mean that the entire corresponding upstream software project is bundled here.

## Public morphology resources

- **EU-OPENSCREEN profiles:** [version 3.0.0, DOI 10.5281/zenodo.19347244](https://doi.org/10.5281/zenodo.19347244), CC-BY-4.0. Upstream analysis software is separately MIT licensed. Retain the dataset attribution and note preprocessing/derivation when regenerating outputs.
- **JUMP / Cell Painting Gallery:** Gallery image and profile data are CC0-1.0; cite the original JUMP resource and Gallery. [JUMP repository metadata and source](https://github.com/jump-cellpainting/datasets) carry BSD-3-Clause notices. The microscopy workflow records original image and illumination URLs rather than presenting the images as newly acquired.
- **LINCS Cell Painting:** [upstream repository](https://github.com/broadinstitute/lincs-cell-painting), data/results/figures CC0-1.0 and code BSD-3-Clause; dataset DOI [10.5281/zenodo.5008187](https://doi.org/10.5281/zenodo.5008187).

## RxRx3-core attribution and restrictions

We used the RxRx3-core dataset, available from Recursion Pharmaceuticals at www.rxrx.ai, pursuant to Recursion Pharmaceutical's licensing terms at [this Agreement](https://huggingface.co/datasets/recursionpharma/rxrx3-core/blob/89aedc798bf33e6f51cc8c6363009d9ffed69e31/LICENSE). Under this license, Recursion Pharmaceuticals disclaims all representations and warranties with respect to such dataset.

The study selected and transformed licensed profiles, fitted and evaluated models, and produced derived predictions, residuals, statistics and figures. These are study-created derivatives, not Recursion's original measurements or an endorsement by Recursion. The full pinned license is included unchanged at `third_party_licenses/Recursion_RxRx3_core_EULA.txt`.

The custom EULA's definition of derivative technology is broad. Sharing is subject to section 7, including essentially equivalent terms and section 3 purpose restrictions; it is not unrestricted CC-BY/MIT reuse. Users acquiring RxRx3-core must read and comply with the upstream terms. Acquisition utilities must not silently represent that another user or institution has accepted them.

The original-code MIT grant does not resolve the treatment of RxRx3-specific modifications or trained artifacts. Dataset-specific adaptation paths reserved from that grant are listed in `LICENSE_NOTES.md`; equivalent terms for applicable derivatives remain to be resolved. This distinguishes the original-software grant from source-dependent rights without making a legal determination that the whole method is derivative.

## Separate chemical annotation terms

Biological-annotation download and preparation code must retain the original source, not treat all annotations as CC0. LINCS target/MoA fields trace to the public LINCS bundle's `repurposing_info.tsv`; the original [Drug Repurposing Hub](https://repo-hub.broadinstitute.org/repurposing) identifies CC-BY-4.0 compound metadata but retains a commercial-repackaging restriction elsewhere in its FAQ. Cite Corsello et al., Nature Medicine 23, 405–408 (2017), and distinguish source metadata from downstream Cell Painting measurements. A CLUE platform/software license is not interchangeable with this dataset provenance.

The RxRx3 relationship builder reads EFAAR's `compound_gene_interactions.csv`. Its local directory license specifies other annotation datasets, not this file. The root Apache-2.0 software license does not by itself clear BindingDB/ChEMBL-derived rows for redistribution. The two original notices are retained in `third_party_licenses/EFAAR_*`. Any public raw annotation or target-matrix release needs source-scope clarification. ChEMBL's [database license is CC-BY-SA-3.0](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions); do not replace it with the license of this repository. See the data archive's notices for the corresponding staged-file scope.

## Files not reclassified as study-owned software

Third-party licenses and notices, acquired data, dataset metadata, pretrained/trained weights, generated data and separately licensed manuscript/template files are outside any blanket grant over original source code. Do not copy journal `.cls` or `.bst` files into a permissively licensed code tree without preserving their original LPPL conditions. Do not bundle proprietary font files merely because a figure-generation environment used them.

See `third_party_licenses/README.md` for exact license-source URLs. The corresponding data archive supplies file-level provenance and additional data notices.
