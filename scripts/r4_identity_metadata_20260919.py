"""R4 identity preparation only; leaves every existing reservation unchanged.

Reads an explicit allowlist of chemical identity tables. PubChem calls contain
only public Broad identifiers, CIDs, or InChIKeys. No profile, X, or outcome is
read. All outputs remain under the new preparation/identity directory.
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "reports/r4_preparation_20260919_v1/identity"
OLD = PROJECT / "reports/new_data_assignment_20260917_v1/identity_resolution"
QUAL = PROJECT / "reports/new_data_qualification_20260917_v1"
ASSIGN = PROJECT / "reports/new_data_assignment_20260917_v1"
BIO = PROJECT / "reports/lincs_biology_preflight_20260915/biology_audit/public_metadata/metadata/moa"


def read(path, delim=",", encoding="utf-8-sig"):
    with open(path, encoding=encoding, newline="") as f:
        return list(csv.DictReader((line for line in f if not line.startswith("!")), delimiter=delim))


def write(name, rows, fields):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fetch(label, url):
    path = OUT / "sources" / (label + ".json")
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "R4-chemical-identity-metadata-audit/1.0"})
            with urllib.request.urlopen(req, timeout=25) as response:
                result = dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                              http_status=response.status, data=json.load(response))
            break
        except urllib.error.HTTPError as exc:
            result = dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                          http_status=exc.code, error=exc.read().decode("utf-8", errors="replace"))
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except Exception as exc:
            result = dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                          http_status=None, error=repr(exc))
        time.sleep(1 + attempt)
    path.write_text(json.dumps(result, indent=2) + "\n")
    time.sleep(0.25)
    return result


def geo_identity_rows(targets):
    """Retrieve only the official perturbagen identity table, never a profile."""
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz"
    path = OUT / "sources/GSE92742_identity_matches.json"
    if path.exists():
        return json.loads(path.read_text())
    wanted = {r["object_id"][:13] for r in targets}
    with urllib.request.urlopen(url, timeout=40) as response:
        data = gzip.decompress(response.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(data), delimiter="\t")
    rows = list(reader)
    selected = [r for r in rows if r.get("pert_id", "") in wanted]
    result = dict(source_url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                  scope="chemical_identity_only", source_rows=len(rows), columns=reader.fieldnames,
                  retained_rows=selected)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources").mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("previous_identity_rules", OLD / "resolve_metadata_only.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    targets = read(OLD / "remaining_unresolved_historical.csv")
    geo = geo_identity_rows(targets)
    assignments = read(ASSIGN / "identity_assignments.csv")
    candidates = read(QUAL / "candidate_identity_overlap.csv")
    history_rows = read(QUAL / "historical_identity_ledger.csv")
    history = {(r["dataset"], r["object_id"]): r for r in history_rows}
    catalog = read(BIO / "repurposing_info.tsv", "\t")
    evidence, resolutions, components, checks = [], [], [], []
    for target in targets:
        print("Checking", target["object_id"], flush=True)
        historical = history[target["dataset"], target["object_id"]]
        object_id, original = target["object_id"], target["original_id"]
        base_id = object_id[:13]
        options = []
        geo_matches = [r for r in geo["retained_rows"] if r["pert_id"] == base_id]
        checks.append(dict(object_id=object_id, check="official_GSE92742_compound_id", query=base_id,
            status="matched" if geo_matches else "no_match", source=geo["source_url"], hits=len(geo_matches)))
        for row in geo_matches:
            options.append(dict(source="GEO_GSE92742_official_compound_id", stable_id=row["pert_id"],
                query=base_id, cid="", smiles=row.get("canonical_smiles", ""),
                key=previous.declared_key(row.get("inchi_key", "")),
                source_url=geo["source_url"], exact_synonym=True, title=row.get("pert_iname", ""),
                source_file="sources/GSE92742_identity_matches.json"))
        for row in catalog:
            deprecated = row.get("deprecated_broad_id", "")
            if base_id in deprecated:
                options.append(dict(source="local_broad_deprecated_id", stable_id=row["broad_id"],
                    query=original, cid="", smiles=row["smiles"], key=row["InChIKey"],
                    source_url="https://github.com/broadinstitute/lincs-cell-painting/blob/master/metadata/moa/repurposing_info.tsv",
                    exact_synonym=True, title=row["pert_iname"], source_file=str(BIO / "repurposing_info.tsv")))
        checks.append(dict(object_id=object_id, check="local_deprecated_broad_id", query=base_id,
                           status="matched" if options else "no_match", source="repurposing_info.tsv", hits=len(options)))
        queries = [original]
        # Check the compound identifier as a separate namespace even if an exact
        # sample synonym succeeds; disagreement remains visible, never overwritten.
        if base_id != original:
            queries.append(base_id)
        for query in queries:
            url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + urllib.parse.quote(query) + "/property/InChIKey,CanonicalSMILES,IsomericSMILES,IUPACName/JSON"
            response = fetch("name_" + query, url)
            props = response.get("data", {}).get("PropertyTable", {}).get("Properties", [])
            checks.append(dict(object_id=object_id, check="pubchem_exact_identifier_lookup", query=query,
                status=str(response["http_status"]), source=url, hits=len(props)))
            for prop in props:
                cid = prop["CID"]
                synurl = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
                synresponse = fetch(f"cid_{cid}_synonyms", synurl)
                synonyms = [s for info in synresponse.get("data", {}).get("InformationList", {}).get("Information", []) for s in info.get("Synonym", [])]
                exact = query in synonyms
                options.append(dict(source="PubChem_exact_sample_synonym" if query == original else "PubChem_compound_id_synonym",
                    stable_id=query, query=query, cid=cid, smiles=prop.get("SMILES", prop.get("IsomericSMILES", prop.get("ConnectivitySMILES", ""))),
                    key=prop.get("InChIKey", ""), source_url=url, exact_synonym=exact, title=prop.get("IUPACName", ""),
                    source_file="sources/name_" + query + ".json", synonyms_source="sources/cid_" + str(cid) + "_synonyms.json"))
        # Existing mixed-record graph is authoritative historical input and is
        # retained alongside additional representations, without selecting a parent.
        if historical["smiles"]:
            options.append(dict(source="existing_historical_structure", stable_id=original, query=original,
                cid="", smiles=historical["smiles"], key=historical["supplied_inchikey"], source_url=historical["source"],
                exact_synonym=True, title=historical["name"], source_file=str(QUAL / "historical_identity_ledger.csv")))
        if not options:
            for query in queries:
                substance_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/name/" + urllib.parse.quote(query) + "/sids/JSON"
                substance = fetch("substance_" + query, substance_url)
                sids = substance.get("data", {}).get("IdentifierList", {}).get("SID", [])
                checks.append(dict(object_id=object_id, check="pubchem_substance_identifier_lookup", query=query,
                    status=str(substance["http_status"]), source=substance_url, hits=len(sids)))
                if sids:
                    raise RuntimeError("New substance identity evidence needs explicit parsing before completion: " + query)
        all_components, all_keys, accepted_sources = set(), set(), []
        sets = []
        for option in options:
            parsed = previous.structure(option["smiles"])
            accepted = option["exact_synonym"] and bool(parsed["components"])
            component_keys = sorted({c["connectivity"] for c in parsed["components"]})
            evidence.append(dict(dataset=target["dataset"], object_id=object_id, original_id=original,
                **option, computed_raw_key=parsed["raw_key"], parse_status=parsed["status"],
                source_key_matches_computed=option["key"] == parsed["raw_key"],
                normalized_connectivities="|".join(component_keys), accepted_for_conservative_exclusion=accepted))
            if not accepted:
                continue
            accepted_sources.append(option["source"] + ":" + option["stable_id"])
            sets.append(set(component_keys))
            all_components.update(component_keys)
            all_keys.update(k for k in (option["key"], parsed["raw_key"]) if k)
            for comp in parsed["components"]:
                components.append(dict(dataset=target["dataset"], object_id=object_id, original_id=original,
                    source=option["source"], source_stable_id=option["stable_id"], cid=option["cid"],
                    **comp))
        if not all_components:
            status = "unresolved_no_verified_structure"
        elif len(all_components) == 1:
            status = "resolved_connectivity_for_conservative_isolation"
        else:
            status = "component_or_source_union_for_conservative_exclusion"
        resolutions.append(dict(dataset=target["dataset"], object_id=object_id, original_id=original,
            previous_status=target["resolution_status"], status=status,
            normalized_connectivities="|".join(sorted(all_components)), raw_source_keys="|".join(sorted(all_keys)),
            accepted_evidence="|".join(accepted_sources), all_accepted_component_sets_agree=all(s == sets[0] for s in sets) if sets else False,
            name_used_for_matching=False, exact_reagent_or_biological_equivalence_claimed=False,
            author_decision="retain_unknown_overlap_warning_and_obtain_original_structure" if not all_components else
                "review_conservative_alias_union_without_promoting_or_releasing_candidates"))
    resolution_by = {r["object_id"]: r for r in resolutions}
    parents, keys = defaultdict(list), defaultdict(list)
    for r in resolutions:
        for key in filter(None, r["normalized_connectivities"].split("|")):
            parents[key].append(r)
        for key in filter(None, r["raw_source_keys"].split("|")):
            keys[key[:14]].append(r)
    hits = []
    assignment_by = {(r["dataset"], r["object_id"]): r for r in assignments}
    for candidate in candidates:
        for basis, key, lookup in [("normalized_component", candidate["connectivity"], parents),
                                  ("raw_source_connectivity", candidate["raw_inchikey"][:14], keys),
                                  ("supplied_source_connectivity", candidate["supplied_inchikey"][:14], keys)]:
            if not key:
                continue
            for historical in lookup.get(key, []):
                role = assignment_by.get((candidate["dataset"], candidate["object_id"]), {}).get("identity_role", "")
                hits.append(dict(candidate_dataset=candidate["dataset"], candidate_id=candidate["object_id"],
                    candidate_name=candidate["name"], candidate_connectivity=candidate["connectivity"], current_role=role,
                    historical_dataset=historical["dataset"], historical_id=historical["object_id"],
                    basis=basis, matched_key=key, proposed_action="exclude_from_new_confirmation_no_release_to_fit"))
    new_eu = {r["candidate_id"] for r in hits if r["candidate_dataset"] == "EU_OPENSCREEN" and r["current_role"] == "EU_CONFIRMATION_RESERVED"}
    eu_proposal = []
    unresolved = [r["object_id"] for r in resolutions if r["status"] == "unresolved_no_verified_structure"]
    for row in assignments:
        if row["dataset"] != "EU_OPENSCREEN" or row["identity_role"] not in ("EU_CONFIRMATION_RESERVED", "EU_RESERVED_HISTORICAL_OVERLAP_INELIGIBLE"):
            continue
        prior_exclusion = row["identity_role"] == "EU_RESERVED_HISTORICAL_OVERLAP_INELIGIBLE"
        added_exclusion = row["object_id"] in new_eu
        eu_proposal.append(dict(object_id=row["object_id"], name=row["name"], connectivity=row["connectivity"],
            current_role=row["identity_role"], original_protection_retained=True,
            proposed_status="prior_overlap_ineligible" if prior_exclusion else "new_overlap_proposed_ineligible" if added_exclusion else "candidate_pending_unknown_identity_and_protocol_decisions",
            has_detected_new_overlap=added_exclusion,
            independent_identity_certified=False, measurement_access_released=False,
            unresolved_historical_count=len(unresolved)))
    write("historical_resolution.csv", resolutions, list(resolutions[0]))
    write("identity_evidence.csv", evidence, list(dict.fromkeys(k for r in evidence for k in r)))
    write("component_evidence.csv", components, list(dict.fromkeys(k for r in components for k in r)))
    write("lookup_checks.csv", checks, list(checks[0]))
    write("candidate_overlap_pairs.csv", hits, ["candidate_dataset", "candidate_id", "candidate_name", "candidate_connectivity", "current_role", "historical_dataset", "historical_id", "basis", "matched_key", "proposed_action"])
    write("EU_original1549_proposed_qualification.csv", eu_proposal, list(eu_proposal[0]))
    write("EU_additional_overlap_proposed_exclusion.csv", [r for r in eu_proposal if r["has_detected_new_overlap"]], list(eu_proposal[0]))
    write("EU_remaining_candidate_proposal.csv", [r for r in eu_proposal if r["proposed_status"] == "candidate_pending_unknown_identity_and_protocol_decisions"], list(eu_proposal[0]))
    # Recheck the complete legacy ledger and both identity overlays, so the new
    # shortlist is not justified by testing the 20 outstanding records alone.
    known_parents, known_keys = set(), set()
    for row in history_rows:
        if row.get("connectivity"):
            known_parents.add(row["connectivity"])
        known_keys.update(row[k][:14] for k in ("raw_inchikey", "supplied_inchikey") if row.get(k))
    for row in read(OLD / "component_connectivity_overlay.csv"):
        if row["scope"] == "historical":
            known_parents.add(row["connectivity"])
    for row in read(OLD / "source_key_overlay.csv"):
        if row["scope"] == "historical":
            known_keys.add(row["key_connectivity"])
    known_parents.update(parents)
    known_keys.update(keys)
    eu_original_ids = {row["object_id"] for row in eu_proposal}
    full_known_overlap = {row["object_id"] for row in candidates if row["dataset"] == "EU_OPENSCREEN"
        and row["object_id"] in eu_original_ids and (row["connectivity"] in known_parents
        or any(row[k][:14] in known_keys for k in ("raw_inchikey", "supplied_inchikey") if row.get(k)))}
    prior_ids = {row["object_id"] for row in eu_proposal if row["proposed_status"] == "prior_overlap_ineligible"}
    reservation = json.loads((ASSIGN / "reservation_manifest.json").read_text())
    validation = dict(
        historical_target_rows=len(targets), historical_resolution_rows=len(resolutions),
        unique_historical_targets=len({(r["dataset"], r["object_id"]) for r in resolutions}),
        full_legacy_ledger_rows_rechecked=len(history_rows),
        full_legacy_plus_old_and_new_overlay_detected_overlap_ids=sorted(full_known_overlap),
        original_1549_groups_equal_existing_protection={r["connectivity"] for r in eu_proposal} == set(reservation["reserved_connectivity_groups"]),
        full_recheck_equals_6_prior_plus_2_new=full_known_overlap == prior_ids | new_eu,
        no_candidate_release=all(not r["measurement_access_released"] and r["original_protection_retained"] for r in eu_proposal),
        proposed_candidates_disjoint_known_overlap=not ({r["object_id"] for r in eu_proposal if r["proposed_status"] == "candidate_pending_unknown_identity_and_protocol_decisions"} & full_known_overlap),
        unresolved_cannot_be_screened_structurally=True)
    assert len(targets) == len(resolutions) == validation["unique_historical_targets"] == 20
    assert validation["original_1549_groups_equal_existing_protection"]
    assert validation["full_recheck_equals_6_prior_plus_2_new"]
    assert validation["proposed_candidates_disjoint_known_overlap"]
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    summary = dict(status="METADATA_ONLY_PROPOSAL_NOT_APPLIED", input_missing_structure=19, input_multicomponent=1,
        status_counts=dict(Counter(r["status"] for r in resolutions)), unresolved_historical_ids=unresolved,
        EU_original_protected_groups=len(eu_proposal), EU_current_candidate_groups=1543,
        EU_prior_exclusions=6, EU_additional_proposed_exclusions=len(new_eu),
        EU_additional_proposed_exclusion_ids=sorted(new_eu), EU_remaining_proposed_candidates=1543-len(new_eu),
        all_original_1549_protection_retained=True, old_manifests_modified=False, independence_certified=False,
        measurements_read=False, confirmation_X_read=False, measurement_access_released=False,
        rdkit_version=previous.rdBase.rdkitVersion, sources_retrieved_utc=datetime.now(timezone.utc).isoformat(),
        limitation="Stable-ID/depositor-synonym links justify conservative exclusion, not sample purity, stereochemical or biological equivalence. Unresolved identities prevent unconditional zero-overlap claims.")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
