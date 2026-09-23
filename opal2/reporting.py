"""Generate reports from completed artifacts, without hard-coded conclusions."""
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


def number(value,digits=3):
    return "not available" if value is None else f"{value:.{digits}f}"


def write_development_report(root,output,*,tests_summary):
    root=Path(root)
    lines=["# OPAL 2.0 R2: recorded execution checks and development runs","",
           f"Generated: {datetime.now(timezone.utc).isoformat()}","",tests_summary,"",
           "Software correctness, real-data execution, effectiveness and independent certification are different outcomes.",
           "An already-open DEV partition is not new confirmatory evidence.","",
           "| Artifact | Evaluation n | ADD_TWO r | POSITIVE AUC | R² | Scope |",
           "| --- | ---: | ---: | ---: | ---: | --- |"]
    rows=[]; paths=[]
    for path in sorted((root/"runs").rglob("evaluation.json")):
        report=json.loads(path.read_text())
        if "gain_metrics" not in report:continue
        metrics=report["gain_metrics"]
        metrics=metrics.get("add_0_1",metrics)
        scope=report.get("purpose",report.get("evidence_scope","not supplied"))
        for parent in (path.parent,path.parent.parent):
            manifest=parent/"run_manifest.json"
            if manifest.exists():
                scope=json.loads(manifest.read_text()).get("purpose",scope);break
        relative=str(path.relative_to(root));paths.append(relative)
        rows.append({"artifact":relative,"scope":scope,**metrics})
        lines.append(f"| {relative} | {metrics.get('n','?')} | {number(metrics.get('pearson_r'))} | "
                     f"{number(metrics.get('positive_auc'))} | {number(metrics.get('r2'))} | {scope} |")
    if not paths:lines += ["","No completed evaluation was found; no effectiveness result is inferred from code."]
    lines += ["","## Interpretation boundaries","",
              "- Model-implied measurement parameters are not identified biological effects.",
              "- Model-based utility and assurance are not observed policy benefit.",
              "- Coverage checks do not certify the full conditional distribution.",
              "- Compound counts do not remove shared-batch dependence.",
              "- New-source/Target-2 adaptation and biological usefulness need separate real experiments.",
              "- The original OPAL 1.0 results and contract are not rewritten by these runs.",""]
    Path(output).write_text("\n".join(lines))
    (root/"runs").mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(root/"runs"/"first_test_metrics.tsv",sep="\t",index=False)
    (root/"runs"/"results_index.json").write_text(json.dumps({"authoritative_evaluations":paths,
        "test_results":"runs/software_test_results.xml","generated_from_completed_outputs":True},indent=2)+"\n")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=".")
    p.add_argument("--output",default="RECORDED_RUNS.md")
    p.add_argument("--tests-summary",required=True)
    a=p.parse_args()
    write_development_report(a.root,a.output,tests_summary=a.tests_summary)
