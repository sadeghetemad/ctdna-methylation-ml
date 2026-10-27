"""Shared data loading, splits and metrics for the modelling scripts 08-13."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CTDNA_ML_DATA", ROOT / "data/ml_dataset/nfcore"))
OUT = Path(os.environ.get("CTDNA_ML_OUT", ROOT / "results/modelling"))
SEED = 42
MAX_FPR = 0.05
FAMILIES = {
    "methylation": "Region methylation (all CpG calls)",
    "meth": "Region methylation (artifact-free)",
    "frac_meth": "Methylated fragments",
    "frac_meth_short": "Methylated short fragments",
    "mhl": "Methylation haplotype load",
    "pdr": "Discordant reads (PDR)",
    "entropy": "Pattern entropy",
    "frac_short": "Short-fragment share",
    "sample": "Genome-wide fragment size",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def out_dir(name):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load():
    """samples (index sample_id), X (all feature families, columns '<family>:<region>'), regions."""
    info = json.loads((DATA / "dataset_info.json").read_text())
    samples = pd.read_csv(DATA / "samples.tsv", sep="\t").set_index("sample_id")
    read = lambda n: pd.read_csv(DATA / f"X_{n}.tsv", sep="\t", index_col="sample_id").loc[samples.index]  # noqa: E731
    parts = [read("methylation").add_prefix("methylation:")]
    parts += [read(n).add_prefix(f"{n}:") for n in info["read_feature_sets"]]
    parts.append(read("sample").add_prefix("sample:"))
    X = pd.concat(parts, axis=1).astype(float)
    regions = pd.read_csv(DATA / "regions.tsv", sep="\t").set_index("region_id")
    return samples, X, regions


def family(col):
    return col.split(":", 1)[0]


def region(col):
    return col.split(":", 1)[1]


def age_band(age):
    band = pd.cut(age.astype(float), [0, 50, 60, 200], labels=["<50", "50-59", "60+"], right=False)
    return band.astype(object).where(band.notna(), "unknown").astype(str)


def strata(samples, with_demographics=True):
    """Stratification key: label x sequencing batch (x sex x age band)."""
    key = samples["ml_label"].astype(str) + "|" + samples["submission_series"].astype(str)
    if with_demographics:
        key = key + "|" + samples["gender"].fillna("unknown").astype(str) + "|" + age_band(samples["age"])
    return key


def balanced_split(samples, test_size=0.3, seed=SEED):
    """Hold-out split stratified on label x batch x sex x age band; rare strata fall back to label x batch."""
    base = strata(samples, with_demographics=False)
    key = strata(samples)
    for coarse in (base + "|" + samples["gender"].astype(str), base, samples["ml_label"].astype(str)):
        small = key.map(key.value_counts()) < 4
        key[small] = coarse[small]
    # whatever is still too small joins the largest stratum of its own label
    for i in key.index[key.map(key.value_counts()) < 4]:
        same = key[(samples["ml_label"] == samples.at[i, "ml_label"]) & (key.map(key.value_counts()) >= 4)]
        key[i] = same.value_counts().index[0]
    tr, te = train_test_split(samples.index, test_size=test_size, random_state=seed, stratify=key)
    return pd.Index(tr), pd.Index(te)


DESIGNS = ("balanced", "study", "lobo")
SUPPLEMENTARY = ROOT / "metadata/publication/ELSA_supplementary_tables.xlsx"


def designs(samples):
    """name -> (train index, test index).

    balanced   30% hold-out stratified on label x batch x sex x age band
    study      the publication's groups, TV -> SBT
    lobo       leave one batch out: train on the larger sequencing batch, test on the smaller one
    """
    tr, te = balanced_split(samples)
    g, b = samples["model_group"], samples["submission_series"]
    big = b.value_counts().index[0]
    return {"balanced": (tr, te), "study": (samples.index[g == "TV"], samples.index[g == "SBT"]),
            "lobo": (samples.index[b == big], samples.index[b != big])}


def published_calls():
    """Per-sample calls of the published classifier (Supplementary Table 7): 1 = tumour."""
    raw = pd.read_excel(SUPPLEMENTARY, sheet_name="Supplementary Table 7", header=None)
    cols = ["sample_id", "predicted", "ground_truth", "stage", "cancer_type", "max_diameter_cm", "model_group",
            "age", "gender", "dna_input_ng", "dna_concentration_ng_ul"]
    cancer, control = raw.iloc[2:311, :11].copy(), raw.iloc[313:, :11].copy()
    cancer.columns = control.columns = cols
    t = pd.concat([cancer, control])
    t = t[t["sample_id"].astype(str).str.match(r"^BMD\d+$", na=False)]
    return t.set_index("sample_id")["predicted"].astype(str).str.lower().eq("tumor").astype(int)


def cv_folds(y, key, n_splits=5, seed=SEED):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), key))


def metrics(y, p, thr=None):
    y = np.asarray(y)
    out = {"auc": roc_auc_score(y, p), "pauc": roc_auc_score(y, p, max_fpr=MAX_FPR)}
    if thr is not None:
        out["sens"] = float((p[y == 1] > thr).mean())
        out["spec"] = float((p[y == 0] <= thr).mean())
        out["bal_acc"] = (out["sens"] + out["spec"]) / 2
    return out


def threshold_at_spec(y, p, spec=0.95):
    y = np.asarray(y)
    return float(np.quantile(p[y == 0], spec))


def bootstrap_ci(y, p, fn=roc_auc_score, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def subgroups(samples):
    """name -> boolean Series, for the fairness audit."""
    s = samples
    return {
        "Female": s["gender"].astype(str).str.lower().str.startswith("f"),
        "Male": s["gender"].astype(str).str.lower().str.startswith("m"),
        "Age <50": age_band(s["age"]) == "<50",
        "Age 50-59": age_band(s["age"]) == "50-59",
        "Age 60+": age_band(s["age"]) == "60+",
        **{f"Batch {b}": s["submission_series"] == b for b in sorted(s["submission_series"].unique())},
    }


_NEW_LR = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)


def logreg(l1_ratio=0.0, C=1.0, solver=None, **kw):
    """Logistic regression with an L1 / L2 / elastic-net mix, for old and new scikit-learn."""
    kw = {"C": C, "class_weight": "balanced", "max_iter": 5000, **kw}
    if _NEW_LR:
        return LogisticRegression(l1_ratio=l1_ratio, solver=solver or ("liblinear" if l1_ratio == 1 else "saga"
                                                                       if 0 < l1_ratio < 1 else "lbfgs"), **kw)
    penalty = "l2" if l1_ratio == 0 else "l1" if l1_ratio == 1 else "elasticnet"
    extra = {"l1_ratio": l1_ratio} if penalty == "elasticnet" else {}
    return LogisticRegression(penalty=penalty, solver=solver or ("liblinear" if penalty == "l1" else "saga"
                                                                 if penalty == "elasticnet" else "lbfgs"),
                              **extra, **kw)


def outer_folds(samples_tr, n_splits=5, repeats=1, seed=SEED):
    """(repeat, fold, train positions, validation positions) on a design's training part, label x batch strata."""
    y = samples_tr["ml_label"].to_numpy()
    key = strata(samples_tr, with_demographics=False).to_numpy()
    return [(r, f, a, b) for r in range(repeats) for f, (a, b) in enumerate(cv_folds(y, key, n_splits, seed + r))]


def fold_vote(scores, val_scores, val_y, spec=0.95):
    """1 where a fold model calls cancer, with that model's own 95%-specificity threshold on its validation fold."""
    return (np.asarray(scores) > threshold_at_spec(val_y, val_scores, spec)).astype(float)


def summarise(name, family_name, y_tr, oof, y_te, p_te, samples_te, oof_vote=None, te_vote=None):
    """One result row plus the subgroup audit.

    AUCs use the probabilities. Sensitivity and specificity use the fold-model votes when given (a patient is
    called positive when most fold models call it), because averaged probabilities are compressed and a
    threshold from out-of-fold scores would not transfer to them.
    """
    cv, te = metrics(y_tr, oof), metrics(y_te, p_te)
    if oof_vote is None:
        thr = threshold_at_spec(y_tr, oof)
        oq, tq = oof, p_te
    else:
        thr, oq, tq = 0.5, np.asarray(oof_vote), np.asarray(te_vote)
    for d, yy, pp in ((cv, y_tr, oq), (te, y_te, tq)):
        d.update({k: v for k, v in metrics(yy, pp, thr).items() if k in ("sens", "spec", "bal_acc")})
    lo, hi = bootstrap_ci(y_te, p_te)
    row = {"model": name, "family": family_name, "cv_auc": cv["auc"], "cv_pauc": cv["pauc"],
           "test_auc": te["auc"], "test_auc_lo": lo, "test_auc_hi": hi, "test_pauc": te["pauc"],
           "test_sens": te["sens"], "test_spec": te["spec"], "test_bal_acc": te["bal_acc"]}
    sub = []
    for g, m in subgroups(samples_te).items():
        m = m.to_numpy()
        if 0 < y_te[m].sum() < m.sum():
            sub.append({"model": name, "group": g, "n": int(m.sum()), "cancers": int(y_te[m].sum()),
                        "auc": roc_auc_score(y_te[m], p_te[m]),
                        "sens": float((tq[m][y_te[m] == 1] > thr).mean()),
                        "spec": float((tq[m][y_te[m] == 0] <= thr).mean())})
    return row, sub, thr
