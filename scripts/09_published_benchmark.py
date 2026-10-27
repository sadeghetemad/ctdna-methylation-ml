"""
09_published_benchmark.py: compare the step 08 final model with the published ELSA-seq
classifier on the same test samples and task (Stage I vs control).

The publication's per-sample calls ('predicted' in Supplementary Table 7) are used only
here, as a benchmark, never as a feature. The step 08 selected pipeline is refit on the
development group with its final hyperparameters and scored on the test group.

With --all-models every single pipeline in 04_heldout_test.tsv is refit the same way and its sensitivity at the
published specificity is written to benchmark_models.tsv.

Writes to --results: test_predictions.tsv and benchmark.json (read by 10_report.py).

Example:
    python scripts/09_published_benchmark.py --data data/ml_dataset/nfcore --results results/ml/nfcore_paper
"""

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SUPPLEMENTARY = ROOT / "metadata/publication/ELSA_supplementary_tables.xlsx"


def load_step08():
    spec = importlib.util.spec_from_file_location("step08", ROOT / "scripts/08_ml_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def published_calls():
    raw = pd.read_excel(SUPPLEMENTARY, sheet_name="Supplementary Table 7", header=None)
    cols = ["sample_id", "predicted", "ground_truth", "stage", "cancer_type", "max_diameter_cm", "model_group",
            "age", "gender", "dna_input_ng", "dna_concentration_ng_ul"]
    cancer, control = raw.iloc[2:311, :11].copy(), raw.iloc[313:, :11].copy()
    cancer.columns = control.columns = cols
    t = pd.concat([cancer, control])
    t = t[t["sample_id"].astype(str).str.match(r"^BMD\d+$", na=False)]
    return t.set_index("sample_id")["predicted"].astype(str).str.lower().eq("tumor").astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml_dataset/nfcore")
    ap.add_argument("--results", default="results/ml/nfcore_paper")
    ap.add_argument("--all-models", action="store_true")
    args = ap.parse_args()
    data = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    res = Path(args.results) if Path(args.results).is_absolute() else ROOT / args.results
    R = json.loads((res / "results.json").read_text())
    dev, test_group = R["data"]["train_group"], R["data"]["test_group"]
    sel = R["selection"]
    final = pd.read_csv(res / "04_heldout_test.tsv", sep="\t").query("role == 'selected'").iloc[0]
    params = json.loads(final["final_params"])

    info = json.loads((data / "dataset_info.json").read_text())
    samples = pd.read_csv(data / "samples.tsv", sep="\t").set_index("sample_id")
    X = pd.read_csv(data / "X_methylation.tsv", sep="\t", index_col="sample_id")
    samples = samples.loc[X.index]
    inputs = {"Region methylation": X}
    if sel["best_input"] != "Region methylation" or args.all_models:
        parts = [X.add_prefix("meth04:")]
        parts += [pd.read_csv(data / f"X_{n}.tsv", sep="\t", index_col="sample_id").add_prefix(f"{n}:")
                  for n in info["read_feature_sets"]]
        parts.append(pd.read_csv(data / "X_sample.tsv", sep="\t", index_col="sample_id").add_prefix("sample:"))
        inputs["Methylation + fragment features"] = pd.concat(parts, axis=1).astype(float)
    Xi = inputs[sel["best_input"]]

    step08 = load_step08()
    pipe = clone(step08.candidates(Xi.shape[1])[sel["best_candidate"]][0]).set_params(**params)
    y = samples["ml_label"].astype(int)
    tr = (samples["model_group"] == dev).to_numpy()
    te = (samples["model_group"] == test_group).to_numpy()
    p = pipe.fit(Xi[tr], y[tr]).predict_proba(Xi[te])[:, 1]
    yt = y[te].to_numpy()
    batch = samples.loc[te, "submission_series"].to_numpy()

    pub = published_calls().reindex(samples.index[te])
    assert pub.notna().all(), "published calls missing for some test samples"
    pub = pub.to_numpy()
    pd.DataFrame({"sample_id": samples.index[te], "ml_label": yt, "batch": batch, "our_score": p,
                  "published_call": pub}).to_csv(res / "test_predictions.tsv", sep="\t", index=False)

    def rates(mask):
        ca, co = mask & (yt == 1), mask & (yt == 0)
        pub_sens, pub_spec = pub[ca].mean(), 1 - pub[co].mean()
        thr = np.quantile(p[co], pub_spec) if co.any() else np.nan
        return {"n": int(mask.sum()), "cancers": int(ca.sum()), "published_sens": float(pub_sens),
                "published_spec": float(pub_spec), "our_auc": float(roc_auc_score(yt[mask], p[mask])) if 0 < ca.sum() < mask.sum() else None,
                "our_sens_at_published_spec": float((p[ca] > thr).mean()) if co.any() else None}

    out = {"test_group": test_group, "selected": f"{sel['best_candidate']} · {sel['best_input']}",
           "note": "Our threshold here is set on the test controls to match the published specificity; comparison only.",
           "all": rates(np.ones(len(yt), bool))}
    for b in sorted(set(batch)):
        out[b] = rates(batch == b)
    (res / "benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    if args.all_models:
        co = yt == 0
        pub_spec = 1 - pub[co].mean()
        rows = []
        refit = pd.read_csv(res / "04_heldout_test.tsv", sep="\t")
        for _, r in refit[refit["input"].isin(list(inputs))].iterrows():
            Xr = inputs[r["input"]]
            m = clone(step08.candidates(Xr.shape[1])[r["candidate"]][0]).set_params(**json.loads(r["final_params"]))
            q = m.fit(Xr[tr], y[tr]).predict_proba(Xr[te])[:, 1]
            thr = np.quantile(q[co], pub_spec)
            rows.append({"input": r["input"], "candidate": r["candidate"], "n_features": Xr.shape[1],
                         "auc_test": roc_auc_score(yt, q), "sens_at_published_spec": (q[yt == 1] > thr).mean()})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(res / "benchmark_models.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
