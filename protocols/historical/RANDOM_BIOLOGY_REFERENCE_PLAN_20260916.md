# Matched random-reference control for biological radial borrowing

Recorded before execution, 16 September 2026. This is a development attribution
experiment on the existing 1,188 opened LINCS objects. It changes neither CORE,
world-model weights, measurement endpoint, protected data nor the main rule.

## Question and fixed comparison

Does target/MoA-based radial-error borrowing outperform equally supported random
borrowing after matching the donor amplitude strata? Retain the completed
`biology_borrowing_diagnostic_20260916_v2` experiment's ten query cells, 40 actual
radial reference representatives per cell, STATE50 means/scatter, amplitude
conditioning, empirical radial kernels, ADD_TWO cost .02, lambda .2, stable-ID
ties and original per-cell budgets. Target and MoA remain separate. Evaluate
alpha .25, .5 and 1 with the existing exact distribution-mixture identities.

## Random reference construction, fixed before reading its scores

Run 20 randomizations with seeds derived from 2026091607, replicate index, cell
index and relation. All randomizations are reported; none is selected.

For each supported query and relation, retain exactly the same positive weight
multiset, donor count and ESS as real biological borrowing. Candidate donors
must come from the same original reference bank, have that relation annotated,
match the original cell/time/dose requirements and not share chemical-group
identity with the query. Query support is the original biological-support mask.

Define five amplitude bins using log first-well norm quintiles fitted only on
the existing MODEL_FIT objects. Within each bin, uniformly permute the legal
donor positions, and transport the original nonzero weights with that
permutation. This preserves each query's weighted donor-amplitude-bin histogram
exactly without matching on residuals, future wells, Gamma or predictive scores.
An original biological donor can remain selected by chance; related references
are not artificially forbidden in the random control. Save all donor mappings.

Bins with one legal donor have no exchangeability and remain unchanged. Record
their count and mass, exact/changed donors, actual within-bin amplitude mismatch,
and accidental retained biological support. Do not relax hard conditions or
silently widen bins to manufacture a comparison. Any illegal original positive
weight or loss of exact count/weight/bin invariants stops the experiment.

This is a matched random-reference control, not an exact target-label permutation
test: query-specific legal pools/strata and donor reuse are part of the design.

## Evaluation and uncertainty

Use the existing full nine-coordinate geometry evaluator with 100,000 complete
draws per endpoint. Preserve original normal/radial seeds and query positions.
Unsupported objects reuse saved CORE values bit for bit. Mixture CRPS/energy
include cross-component terms; neither scores nor sample vectors are averaged
as a substitute for mixing distributions. All five joint coverage levels,
single/pair/average functional scores, Gamma CRPS, NULL Brier, radial NLL and
energy are reported. Full-cohort selected value/NULL counts use the unchanged
original budget; supported-subgroup distribution scores are primary.

Compare real BIO with CORE, each randomized arm with CORE, and real BIO with
the mean per-object performance across 20 randomized controls. That last mean
is expected performance of the randomized construction, not an ensemble
distribution or a deployable ensemble policy. Report between-randomization
spread separately. Chemistry-group and layout paired bootstraps operate on
object records, never treat donor pairs as independent. Their intervals are
conditional on the fitted CORE/reference banks and do not include model
refitting, repeated-development selection or formal deployment certification.

## Interpretation

* BIO better than random but worse than CORE: relation supplies information
  relative to this matched comparator, insufficient to beat amplitude CORE.
* BIO and random unresolved: this comparison does not distinguish their
  information value; it does not establish equivalence or no biological signal.
* BIO worse than random: current relationship retrieval introduces detrimental
  bias relative to matched random borrowing in this interface.
* Very low exchangeable mass: attribution is weakly identified by the available
  reference bank; report this limitation rather than overinterpreting scores.

No SE, new kernel, learned gate, JEPA or replacement mean is trained here.
LKCP remains the prior hard-context non-applicability case; no mismatched donor
is admitted and it is not relabelled an independent efficacy validation.
