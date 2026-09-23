"""Observed-role sensitivity and exact finite-table variation accounting.

The decomposition here is algebra on the supplied observed table. It is not a
random-effects variance estimate, causal batch attribution, reliability bound,
or a ceiling for any prediction algorithm. The same physical wells are reused
across role assignments; their outputs are deliberately not treated as IID.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import MeasurementDataset, load_dataset, _indices
from .utility import cosine


@dataclass(frozen=True)
class RoleAssignment:
    name: str
    initial: int
    additional: tuple[int, ...]
    validation: int

    def __post_init__(self):
        values = (self.initial,*self.additional,self.validation)
        if not self.name or any(not isinstance(x,(int,np.integer)) or isinstance(x,(bool,np.bool_)) or x < 0 for x in values):
            raise ValueError("Role assignments need a name and exact nonnegative integer well indices")
        if not self.additional or len(self.additional) > 2 or len(set(values)) != len(values):
            raise ValueError("Initial, acquired, and validation roles must be distinct physical slots")
        object.__setattr__(self,"initial",int(self.initial))
        object.__setattr__(self,"validation",int(self.validation))
        object.__setattr__(self,"additional",tuple(int(x) for x in self.additional))


def fixed_x_three_roles():
    return tuple(RoleAssignment(f"V_slot_{v}",0,tuple(x for x in (1,2,3) if x != v),v)
                 for v in (1,2,3))


def finite_two_way_ss(values):
    """Exact equal-cell-weight SS for a complete compound × role table.

    There is one scalar in each cell. Consequently the final component combines
    interaction, stochastic errors and nonlinear endpoint effects inseparably.
    No independence or Gaussian assumption is needed for this identity.
    """
    x = np.asarray(values,float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite-table SS requires a complete finite N×R table with both dimensions >=2")
    n,r = x.shape
    # Translation stabilizes a constant or nearly constant table without
    # imposing a numerical threshold that could erase genuinely small effects.
    origin=float(x[0,0])
    centered=x-origin
    grand,row,column = centered.mean(),centered.mean(1),centered.mean(0)
    residual = centered-row[:,None]-column[None,:]+grand
    components = {
        "compound_average":float(r*np.square(row-grand).sum()),
        "role_column":float(n*np.square(column-grand).sum()),
        "interaction_plus_residual":float(np.square(residual).sum())}
    total = float(np.square(centered-grand).sum())
    difference = sum(components.values())-total
    if not np.isclose(sum(components.values()),total,rtol=1e-10,atol=1e-12):
        raise ArithmeticError("Finite-table sums of squares failed their exact additive identity")
    return {
        "n_compounds":n,"n_role_columns":r,"grand_mean":float(grand+origin),
        "sum_of_squares":components,"total_sum_of_squares":total,
        "fraction_of_total_SS":{k:v/total if total>0 else None for k,v in components.items()},
        "degrees_of_freedom":{"compound_average":n-1,"role_column":r-1,
                              "interaction_plus_residual":(n-1)*(r-1),"total":n*r-1},
        "identity_absolute_error":abs(difference),
        "finite_cell_variance_ddof0":float(centered.var()),
        "mean_within_compound_variance_ddof0":float(centered.var(1).mean()),
        "variance_of_compound_means_ddof0":float(row.var()),
        "mean_within_compound_variance_ddof1_descriptive":float(centered.var(1,ddof=1).mean()),
        "within_compound_component_fraction":float(centered.var(1).mean()/centered.var()) if total>0 else None,
        "interaction_and_measurement_noise_separable":False,
        "reason":"One observed value per compound×role cell; all role columns reuse the same physical measurements",
        "inferential_pvalues":None,"prediction_ceiling":None,
        "is_random_effects_variance_estimation":False,"is_causal_decomposition":False}


def _condition_signature(ds,compound,assignment,level):
    # Include ancestry so reused local batch/plate IDs cannot join environments.
    def key(well):
        return tuple(int(x) for x in ds.groups[compound,well,:level+1])
    return (key(assignment.initial),tuple(sorted(key(w) for w in assignment.additional)),key(assignment.validation))


def _layout_report(ds,indices,roles,values):
    result,records = {},[]
    for level,name in ((1,"batch"),(2,"plate")):
        signatures = [[_condition_signature(ds,int(i),role,level) for role in roles] for i in indices]
        unique = sorted(set(sig for row in signatures for sig in row))
        known = all(all(x>=0 for point in (sig[0],*sig[1],sig[2]) for x in point) for sig in unique)
        observed = sum(len(set(row)) for row in signatures)
        complete = known and all(len(row)==len(set(row)) and set(row)==set(unique) for row in signatures)
        role_counts = {role.name:len(set(row[j] for row in signatures)) for j,role in enumerate(roles)}
        result[name] = {
            "distinct_condition_signatures":len(unique),"observed_compound_condition_cells":observed,
            "full_factorial_cells_required":len(indices)*len(unique),
            "compound_condition_complete_balanced":bool(complete),
            "metadata_identifiers_complete":known,
            "distinct_condition_signatures_per_role_column":role_counts,
            "role_column_is_one_fixed_condition":all(x==1 for x in role_counts.values()),
            "identified_biological_or_technical_variance":False,
            "incomplete_design_reason":None if complete else
                "Compound×actual-condition table is incomplete, repeats a condition within a compound, or has unknown IDs; no orthogonal batch/plate attribution is computed"}
        if complete:
            lookup = {v:j for j,v in enumerate(unique)}
            aligned = np.empty((len(indices),len(unique)))
            for i,row in enumerate(signatures):
                for j,signature in enumerate(row):
                    aligned[i,lookup[signature]]=values[i,j]
            result[name]["aligned_finite_condition_SS"] = finite_two_way_ss(aligned)
            result[name]["interpretation"] = "Descriptive condition-signature table only; shared wells and observational condition assignment preclude a causal attribution"
        else:
            result[name]["aligned_finite_condition_SS"] = None
        for i,index in enumerate(indices):
            for j,role in enumerate(roles):
                records.append({"compound_id":str(ds.ids[index]),"role":role.name,"condition_level":name,
                                "condition_signature":json.dumps(signatures[i][j],separators=(",",":"))})
    return result,records


def observed_role_diagnostics(dataset: MeasurementDataset, *, indices=None, assignments=None,
                              cost_per_well=.01, positive_margin=.005):
    """Use already observed outcomes; incomplete objects are explicitly excluded.

    This function is a diagnostic, not an action policy. It intentionally reads
    all allowed repetitions to describe role sensitivity, never labels them as
    deployable decision information, and never replaces a registered endpoint.
    """
    if dataset.metadata.get("train_scaler_applied"):
        raise ValueError("Observed utility must use the fixed endpoint space, not the model training transform")
    if not np.isfinite(cost_per_well) or cost_per_well < 0 or not np.isfinite(positive_margin) or positive_margin <= 0:
        raise ValueError("Declare finite nonnegative well cost and positive event margin")
    roles = tuple(fixed_x_three_roles() if assignments is None else assignments)
    if len(roles) < 2 or len({r.name for r in roles}) != len(roles):
        raise ValueError("At least two uniquely named role assignments are required")
    if len({r.initial for r in roles}) != 1 or len({len(r.additional) for r in roles}) != 1:
        raise ValueError("Role sensitivity holds X and acquired-well count fixed")
    required = sorted({w for r in roles for w in (r.initial,*r.additional,r.validation)})
    if max(required) >= dataset.Y.shape[1]:
        raise ValueError("Role table requests an unavailable slot; no extra repetition is opened")
    ix = np.arange(len(dataset)) if indices is None else _indices(indices,len(dataset))
    if len(ix) < 2:
        raise ValueError("At least two compounds are required")
    available = dataset.observed_mask[ix][:,required] & dataset.well_mask[ix][:,required]
    finite = np.isfinite(dataset.Y[np.ix_(ix,required,np.arange(dataset.Y.shape[-1]))]).all(-1)
    keep = (available & finite).all(1)
    excluded = dataset.ids[ix[~keep]].tolist()
    ix = ix[keep]
    if len(ix)<2:
        raise ValueError("Fewer than two complete compounds for the declared diagnostic role table")
    values = np.empty((len(ix),len(roles)))
    for j,role in enumerate(roles):
        x,v = dataset.Y[ix,role.initial],dataset.Y[ix,role.validation]
        aggregate = (x+dataset.Y[ix][:,role.additional].sum(1))/(1+len(role.additional))
        values[:,j] = .5*(cosine(aggregate,v)-cosine(x,v))-cost_per_well*len(role.additional)
    null = values<=0
    positive = values>=positive_margin
    flip = null.any(1)&~null.all(1)
    per_role = []
    for j,role in enumerate(roles):
        g=values[:,j]
        per_role.append({"role":role.name,"n":len(g),"mean_gamma":float(g.mean()),
                         "sd_gamma_ddof0":float(g.std()),"median_gamma":float(np.median(g)),
                         "null_count":int(null[:,j].sum()),"null_fraction":float(null[:,j].mean()),
                         "positive_count":int(positive[:,j].sum()),"positive_fraction":float(positive[:,j].mean()),
                         "ambiguous_fraction":float((~null[:,j]&~positive[:,j]).mean())})
    pairs=[]
    for left,right in combinations(range(len(roles)),2):
        a,b=values[:,left],values[:,right]
        rho=float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None
        pairs.append({"left_role":roles[left].name,"right_role":roles[right].name,
                      "spearman_rank_correlation":rho,
                      "null_label_disagreement_fraction":float(np.mean(null[:,left]!=null[:,right])),
                      "paired_mean_gamma_difference":float(np.mean(a-b)),
                      "paired_sd_gamma_difference_ddof0":float(np.std(a-b)),
                      "pvalue":None})
    layout,layout_records = _layout_report(dataset,ix,roles,values)
    summary = {
        "scope":"OBSERVED_DEVELOPMENT_ROLE_DIAGNOSTIC_NOT_CERTIFICATION",
        "dataset":dataset.metadata.get("dataset","supplied measured dataset"),"space":dataset.metadata.get("space"),
        "assignments":[asdict(r) for r in roles],"cost_per_well":cost_per_well,"positive_margin":positive_margin,
        "requested_compounds":len(keep),"complete_compounds":len(ix),"excluded_missing_compound_ids":excluded,
        "missing_handling":"Complete-case diagnostic only; excluded objects are reported, not silently given zero or reclassified",
        "utility":"Original fixed-space half-cosine difference minus actual additional-well cost",
        "original_registered_endpoint_changed":False,"role_average_is_diagnostic_not_replacement_endpoint":True,
        "finite_compound_by_role_SS":finite_two_way_ss(values),"per_role":per_role,"rank_stability":pairs,
        "null_flip_count":int(flip.sum()),"null_flip_fraction":float(flip.mean()),
        "mean_within_compound_gamma_range":float(np.ptp(values,axis=1).mean()),
        "median_within_compound_gamma_range":float(np.median(np.ptp(values,axis=1))),
        "physical_condition_layout":layout,
        "interpretation":[
            "Compound-average SS includes shared measurement noise, fixed-position effects and biological differences; these are not identified separately",
            "Role-column SS is not batch SS when a role column contains several actual batch combinations",
            "Interaction and residual measurement noise cannot be separated without additional independent observations in compound×role cells",
            "Role outputs reuse the same physical wells; compounds can also share environments, so no IID standard errors, F tests or confidence ceilings are claimed",
            "No SS fraction is a ceiling on decision-time learnability, and none identifies a causal effect of sending a measurement to another site"],
        "fifth_repeat_read":False if dataset.Y.shape[1]==4 else None,
        "old_final_opened":dataset.metadata.get("final_profiles_read"),
        "external_data_discovered":False,
        "access_flags_scope":"This diagnostic consumes only its supplied dataset and requested existing slots; no external files or new data are discovered"}
    per_compound = {"compound_id":dataset.ids[ix],"null_label_flips":flip,
                    "gamma_mean_across_roles":values.mean(1),"gamma_sd_across_roles_ddof0":values.std(1),
                    "gamma_range_across_roles":np.ptp(values,axis=1),"empirical_role_null_fraction":null.mean(1)}
    for j,role in enumerate(roles):
        per_compound[role.name+"__gamma"] = values[:,j]
    return summary,pd.DataFrame(per_compound),pd.DataFrame(layout_records)


def write_role_diagnostics(dataset,output,**kwargs):
    directory=Path(output)
    if directory.exists():
        raise FileExistsError("Use a fresh observed-diagnostic result directory")
    summary,compounds,layout=observed_role_diagnostics(dataset,**kwargs)
    directory.mkdir(parents=True)
    (directory/"diagnostic.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    compounds.to_csv(directory/"compound_role_values.tsv",sep="\t",index=False)
    layout.to_csv(directory/"physical_condition_layout.tsv",sep="\t",index=False)
    pd.DataFrame(summary["per_role"]).to_csv(directory/"role_summary.tsv",sep="\t",index=False)
    pd.DataFrame(summary["rank_stability"]).to_csv(directory/"rank_stability.tsv",sep="\t",index=False)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,help="Explicit portable observed measurement NPZ")
    parser.add_argument("--output",required=True)
    parser.add_argument("--roles",help="Optional JSON list of name,initial,additional,validation assignments")
    parser.add_argument("--cost-per-well",type=float,default=.01)
    args=parser.parse_args(argv)
    roles=None if args.roles is None else tuple(RoleAssignment(**item) for item in json.loads(Path(args.roles).read_text()))
    summary=write_role_diagnostics(load_dataset(args.dataset),args.output,assignments=roles,cost_per_well=args.cost_per_well)
    print(json.dumps({"output":str(Path(args.output).resolve()),"n":summary["complete_compounds"],
                      "null_flip_fraction":summary["null_flip_fraction"],
                      "finite_table_fractions":summary["finite_compound_by_role_SS"]["fraction_of_total_SS"],
                      "not_causal_or_learning_ceiling":True},indent=2),flush=True)


if __name__=="__main__":
    main()
