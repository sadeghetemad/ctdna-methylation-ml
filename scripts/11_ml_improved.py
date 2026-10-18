"""
11_ml_improved.py: model variants aimed at the weaknesses found by step 08
(batch transfer, high-specificity sensitivity, threshold transfer).

Evaluation designs:
    pooled      nested CV on all samples: outer 5-fold x repeats, stratified by label x
                sequencing batch, so both batches contribute cancers to every training set
    lobo        leave one batch out: train on one sequencing batch, test on the other
    paper       the study's split (TV -> SBT), for comparison with step 08

Feature inputs and pipelines:
    base        L2 logistic regression on region methylation, and on methylation + all
                step 05 features (the step 08 selection), as reference
    clean       `meth` (CpG calls without the end-repair artifact), technical covariates
                regressed out and each batch centred on its own training controls
    burden      tumour-fragment evidence per region: methylated fragments (and CpG calls)
                above the training-control rate, one-sided and overdispersion-aware;
                used per region, or summed over the regions that are most informative in
                the training data
    clean+burden both together

Each pipeline is tuned by ROC-AUC and, separately, by partial AUC at FPR <= 5%
(the screening operating point). Every fit also gets a transferable 95%-specificity
threshold: the inner-CV fold models are kept and averaged, and the threshold comes from
their out-of-fold control scores, so scores and threshold come from the same models.
The refit-on-all model with the same threshold is reported as well.

Nothing learned from labels sees the held-out samples. When a test batch has no training
samples (lobo, and SRR151 cancers in paper), batch centring falls back to the label-free
median of that batch in the test data, against the median of all training samples.

Example:
    python scripts/11_ml_improved.py --data notebooks/data/nfcore_full --features features/nfcore \
        --out results/ml/improved
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin, clone  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
MAX_FPR = 0.05
TECH = ["median_region_depth", "n_fragments", "dna_input_ng", "r2_end_ch_meth", "ch_filtered_fraction"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "axes.axisbelow": True, "font.size": 9})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cols(X, prefix):
    return [c for c in X.columns if c.startswith(prefix)]


# ============================================================ transformers
# Inputs are DataFrames with prefixed columns: "m:" continuous features, "Kf:"/"Nf:" methylated /
# all fragments, "Kc:"/"Nc:" methylated / all CpG calls, "tech:" covariates, "batch".

class Continuous(BaseEstimator, TransformerMixin):
    """The "m:" columns, optionally with technical covariates regressed out and each batch centred.

    residualise: a linear fit of every feature on the technical covariates, estimated on the
    training controls only, is subtracted. batch_centre: every batch is shifted by the median
    of its training controls; a batch without training controls uses the median of its own
    rows at transform time (label-free).
    """

    def __init__(self, residualise=False, batch_centre=False):
        self.residualise = residualise
        self.batch_centre = batch_centre

    def _tech(self, X):
        T = X[cols(X, "tech:")].to_numpy(float)
        T = np.where(np.isnan(T), self.tech_median_, T)
        return np.c_[np.ones(len(T)), (T - self.tech_median_) / self.tech_scale_]

    def _resid(self, X):
        M = X[self.m_cols_].to_numpy(float, copy=True)
        if self.residualise:
            M = M - self._tech(X)[:, 1:] @ self.beta_[1:]
        return M

    def fit(self, X, y):
        y = np.asarray(y)
        self.m_cols_ = cols(X, "m:")
        T = X[cols(X, "tech:")].to_numpy(float)
        self.tech_median_ = np.nanmedian(T, axis=0)
        self.tech_scale_ = np.nanstd(T, axis=0) + 1e-9
        if self.residualise:
            c = y == 0
            A = self._tech(X)[c]
            M = X.loc[c, self.m_cols_].to_numpy(float)
            M = np.where(np.isnan(M), np.nanmedian(M, axis=0), M)
            self.beta_ = np.linalg.lstsq(A, np.nan_to_num(M), rcond=None)[0]
        if self.batch_centre:
            M = self._resid(X)
            b = X["batch"].to_numpy()
            ref = np.nanmedian(M[y == 0], axis=0)
            self.centre_ = {k: np.nanmedian(M[(b == k) & (y == 0)], axis=0) - ref
                            for k in np.unique(b) if ((b == k) & (y == 0)).sum() >= 5}
            self.ref_all_ = np.nanmedian(M, axis=0)
        return self

    def transform(self, X):
        M = self._resid(X)
        if self.batch_centre:
            b = X["batch"].to_numpy()
            for k in np.unique(b):
                m = b == k
                shift = self.centre_.get(k)
                if shift is None:
                    shift = np.nanmedian(M[m], axis=0) - self.ref_all_
                M[m] = M[m] - shift
        return M


class TumourBurden(BaseEstimator, TransformerMixin):
    """One-sided evidence for tumour-derived methylated fragments, per region.

    For counts K of N (methylated fragments, or methylated CpG calls), the training controls give
    a pooled rate p0 and a between-sample variance s2 beyond binomial noise. A sample's evidence
    in a region is z = (K/N - p0) / sqrt(p0 (1 - p0) / N + s2), clipped at 0: only an excess over
    the controls counts, as early-stage ctDNA adds methylated fragments to regions that are
    unmethylated in normal plasma. per_batch uses the controls of the sample's own batch.
    top: None returns z per region; an integer returns, per count type, the mean z over the top
    regions ranked by their training AUC.
    """

    def __init__(self, kinds=("f", "c"), per_batch=False, top=None):
        self.kinds = kinds
        self.per_batch = per_batch
        self.top = top

    @staticmethod
    def _rates(K, N):
        p0 = (K.sum(0) + 0.5) / (N.sum(0) + 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(N > 0, K / np.maximum(N, 1), np.nan)
            binom = np.where(N > 0, p0 * (1 - p0) / np.maximum(N, 1), np.nan)
        s2 = np.clip(np.nanvar(r, axis=0) - np.nanmean(binom, axis=0), 0, None)
        return p0, np.nan_to_num(s2)

    def _z(self, X, kind):
        K = X[self.k_cols_[kind]].to_numpy(float)
        N = X[self.n_cols_[kind]].to_numpy(float)
        K, N = np.nan_to_num(K), np.nan_to_num(N)
        b = X["batch"].to_numpy()
        p0 = np.tile(self.p0_[kind]["all"], (len(X), 1))
        s2 = np.tile(self.s2_[kind]["all"], (len(X), 1))
        if self.per_batch:
            for k, (pk, sk) in self.batch_ref_[kind].items():
                p0[b == k], s2[b == k] = pk, sk
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (K / np.maximum(N, 1) - p0) / np.sqrt(p0 * (1 - p0) / np.maximum(N, 1) + s2)
        return np.clip(np.where(N > 0, z, 0), 0, 20)

    def fit(self, X, y):
        y = np.asarray(y)
        c = y == 0
        b = X["batch"].to_numpy()
        self.k_cols_, self.n_cols_, self.p0_, self.s2_, self.batch_ref_, self.sel_ = {}, {}, {}, {}, {}, {}
        for kind in self.kinds:
            self.k_cols_[kind], self.n_cols_[kind] = cols(X, f"K{kind}:"), cols(X, f"N{kind}:")
            K = np.nan_to_num(X.loc[c, self.k_cols_[kind]].to_numpy(float))
            N = np.nan_to_num(X.loc[c, self.n_cols_[kind]].to_numpy(float))
            p0, s2 = self._rates(K, N)
            self.p0_[kind], self.s2_[kind] = {"all": p0}, {"all": s2}
            self.batch_ref_[kind] = {}
            for k in np.unique(b):
                m = (b[c] == k)
                if m.sum() >= 5:
                    self.batch_ref_[kind][k] = self._rates(K[m], N[m])
            if self.top:
                z = self._z(X, kind)
                auc = column_auc(z, y)
                self.sel_[kind] = np.argsort(-np.nan_to_num(auc, nan=0.5))[: self.top]
        return self

    def transform(self, X):
        out = []
        for kind in self.kinds:
            z = self._z(X, kind)
            out.append(z[:, self.sel_[kind]].mean(axis=1, keepdims=True) if self.top else z)
        return np.hstack(out)


def column_auc(Z, y):
    y = np.asarray(y) == 1
    R = pd.DataFrame(Z).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (R[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def lr(C=0.01):
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced")


def pauc_scorer(est, X, y):
    return roc_auc_score(y, est.predict_proba(X)[:, 1], max_fpr=MAX_FPR)


def pipelines(quick=False):
    """name -> (input, pipeline, grid)."""
    Cs = [0.001, 0.01, 0.1] if quick else [0.001, 0.003, 0.01, 0.03, 0.1]
    cont = lambda **kw: ("cont", Continuous(**kw))  # noqa: E731
    std = [("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler())]
    p = {
        "base · methylation": ("methylation", Pipeline([cont(), *std, ("clf", lr())]), {"clf__C": Cs}),
        "base · methylation + fragment features": ("meth_frag", Pipeline([cont(), *std, ("clf", lr())]), {"clf__C": Cs}),
        "clean · meth, residualised, batch-centred": (
            "clean", Pipeline([cont(residualise=True, batch_centre=True), *std, ("clf", lr())]),
            {"clf__C": Cs, "cont__residualise": [False, True]}),
        "burden · per region": (
            "counts", Pipeline([("tb", TumourBurden()), ("sc", StandardScaler()), ("clf", lr())]),
            {"clf__C": Cs, "tb__per_batch": [False, True]}),
        "burden · summed over top regions": (
            "counts", Pipeline([("tb", TumourBurden(top=200)), ("sc", StandardScaler()), ("clf", lr(1.0))]),
            {"tb__top": [50, 200, 800], "tb__per_batch": [False, True]}),
        "clean + burden": (
            "clean_counts",
            Pipeline([("u", FeatureUnion([("m", Pipeline([cont(residualise=True, batch_centre=True),
                                                         ("imp", SimpleImputer(strategy="median",
                                                                               keep_empty_features=True))])),
                                          ("tb", TumourBurden())])),
                      ("sc", StandardScaler()), ("clf", lr())]),
            {"clf__C": Cs, "u__tb__per_batch": [False, True]}),
    }
    if quick:
        p = {k: v for k, v in p.items() if k in ("base · methylation", "clean · meth, residualised, batch-centred",
                                                 "burden · summed over top regions")}
    return p


# ============================================================ fitting

def fit_with_threshold(pipe, grid, scoring, X, y, strata, seed):
    """Tune on (X, y); return the refit model, the inner fold models and their 95%-spec threshold."""
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), strata))
    gs = GridSearchCV(pipe, grid, scoring=scoring, cv=splits, refit=True, n_jobs=1).fit(X, y)
    oof = np.full(len(y), np.nan)
    models = []
    for tr, va in splits:
        m = clone(pipe).set_params(**gs.best_params_).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
    thr = float(np.quantile(oof[y == 0], 0.95))
    return gs.best_estimator_, models, thr, gs.best_params_


def predict(fitted, X):
    refit, models, thr, params = fitted
    return refit.predict_proba(X)[:, 1], np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def load_input(cache, name):
    with open(Path(cache) / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


SCORING = {"auc": "roc_auc", "pauc": pauc_scorer}


def run(cache, quick, name, s, y, strata, tr, te, seed):
    inp, pipe, grid = pipelines(quick)[name]
    X = load_input(cache, inp)
    fitted = fit_with_threshold(pipe, grid, SCORING[s], X.iloc[tr], y[tr], strata[tr], seed)
    p_refit, p_bag = predict(fitted, X.iloc[te])
    return p_refit, p_bag, fitted[2], fitted[3]


def permuted_auc(cache, quick, name, s, yp, batch01, folds):
    inp, pipe, grid = pipelines(quick)[name]
    X = load_input(cache, inp)
    p = np.full(len(yp), np.nan)
    for f, (tr, te) in enumerate(folds):
        fitted = fit_with_threshold(pipe, grid, SCORING[s], X.iloc[tr], yp[tr], yp[tr] * 2 + batch01[tr], SEED + f)
        p[te] = predict(fitted, X.iloc[te])[1]
    return roc_auc_score(yp, p)


def metrics(y, p, batch=None):
    y = np.asarray(y)
    out = {"auc": roc_auc_score(y, p), "pauc": roc_auc_score(y, p, max_fpr=MAX_FPR)}
    if batch is not None:
        for b in np.unique(batch):
            m = batch == b
            out[f"auc_{b}"] = roc_auc_score(y[m], p[m]) if 0 < y[m].sum() < m.sum() else np.nan
    return out


def threshold_rates(y, p, thr):
    y = np.asarray(y)
    return float((p[y == 1] > thr).mean()), float((p[y == 0] <= thr).mean())


# ============================================================ main

def main():
    ap = argparse.ArgumentParser(description="Model variants for batch transfer and high-specificity sensitivity.")
    ap.add_argument("--data", default="notebooks/data/nfcore_full")
    ap.add_argument("--features", default="features/nfcore", help="Step 05 count tables (read_n_frag, ...).")
    ap.add_argument("--out", default="results/ml/improved")
    ap.add_argument("--outer-repeats", type=int, default=3)
    ap.add_argument("--permutations", type=int, default=50, help="Label shuffles for the best pooled pipeline.")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="Smoke test: 3 pipelines, 1 repeat, 5 permutations.")
    args = ap.parse_args()
    if args.quick:
        args.outer_repeats, args.permutations = 1, 5
    rel = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p  # noqa: E731
    data, feats, out = rel(args.data), rel(args.features), rel(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------------- inputs
    info = json.loads((data / "dataset_info.json").read_text())
    samples = pd.read_csv(data / "samples.tsv", sep="\t").set_index("sample_id")
    y = samples["ml_label"].astype(int).to_numpy()
    batch = samples["submission_series"].to_numpy()
    idx = samples.index

    def matrix(path):
        df = pd.read_csv(path, sep="\t")
        return df.set_index("sample_id").drop(columns=[c for c in ("run_id", "ml_label") if c in df.columns]).loc[idx]

    side = pd.DataFrame({"batch": batch}, index=idx).join(samples[TECH].add_prefix("tech:").astype(float))
    meth04 = matrix(data / "X_methylation.tsv")
    read_sets = [matrix(data / f"X_{n}.tsv").add_prefix(f"{n}:") for n in info["read_feature_sets"]]
    x_sample = matrix(data / "X_sample.tsv").add_prefix("sample:")
    cohort = info["cohort"]
    raw = {n: matrix(feats / f"{cohort}_read_{n}.tsv") for n in ("frac_meth", "n_frag", "meth", "n_calls")}
    regions = raw["n_frag"].columns
    counts = pd.concat([
        (raw["frac_meth"][regions] * raw["n_frag"][regions]).round().add_prefix("Kf:"),
        raw["n_frag"][regions].add_prefix("Nf:"),
        (raw["meth"][regions] * raw["n_calls"][regions]).round().add_prefix("Kc:"),
        raw["n_calls"][regions].add_prefix("Nc:"),
    ], axis=1).fillna(0)
    inputs = {
        "methylation": pd.concat([meth04.add_prefix("m:"), side], axis=1),
        "meth_frag": pd.concat([pd.concat([meth04.add_prefix("meth04:"), *read_sets, x_sample], axis=1)
                                .add_prefix("m:"), side], axis=1),
        "clean": pd.concat([matrix(data / "X_meth.tsv").add_prefix("m:"), side], axis=1),
        "counts": pd.concat([counts, side], axis=1),
    }
    inputs["clean_counts"] = pd.concat([inputs["clean"], counts], axis=1)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    for k, v in inputs.items():
        assert v.index.equals(idx), k
        with open(cache / f"{k}.pkl", "wb") as fh:
            pickle.dump(v, fh, protocol=pickle.HIGHEST_PROTOCOL)
    del inputs, read_sets, raw, counts
    batches = sorted(np.unique(batch))
    strata = y * 2 + (batch == batches[0]).astype(int)
    log(f"{len(y)} samples ({y.sum()} cancers); batches " + ", ".join(f"{b} {(batch == b).sum()}" for b in batches))

    PIPES = pipelines(args.quick)
    cache_s = str(cache)
    configs = [(name, s) for name in PIPES for s in SCORING]
    R = {"settings": vars(args), "max_fpr": MAX_FPR, "configs": [f"{n} | tuned by {s}" for n, s in configs]}

    # ---------------------------------------------------------------- pooled nested CV
    outer = [(r, f, tr, te) for r in range(args.outer_repeats)
             for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + r)
                                          .split(np.zeros(len(y)), strata))]
    log(f"pooled nested CV: {len(configs)} configs x {len(outer)} outer folds")
    tasks = [(c, o) for c in range(len(configs)) for o in range(len(outer))]
    res = Parallel(n_jobs=args.jobs)(
        delayed(run)(cache_s, args.quick, *configs[c], y, strata, outer[o][2], outer[o][3], SEED + 100 * outer[o][0] + outer[o][1])
        for c, o in tasks)
    pooled_rows, oof_store = [], {}
    for c, (name, s) in enumerate(configs):
        P_ref = np.full((args.outer_repeats, len(y)), np.nan)
        P_bag = np.full_like(P_ref, np.nan)
        hit_bag = np.full_like(P_ref, np.nan)
        hit_ref = np.full_like(P_ref, np.nan)
        params = []
        for (cc, o), (p_ref, p_bag, thr, bp) in zip(tasks, res):
            if cc != c:
                continue
            r, f, tr, te = outer[o]
            P_ref[r, te], P_bag[r, te] = p_ref, p_bag
            hit_bag[r, te], hit_ref[r, te] = p_bag > thr, p_ref > thr
            params.append(json.dumps(bp, sort_keys=True))
        per_rep = [metrics(y, P_bag[r], batch) for r in range(args.outer_repeats)]
        row = {"pipeline": name, "tuned_by": s}
        for k in per_rep[0]:
            v = [m[k] for m in per_rep]
            row[k], row[f"{k}_sd"] = float(np.mean(v)), float(np.std(v))
        row["auc_refit"] = float(np.mean([roc_auc_score(y, P_ref[r]) for r in range(args.outer_repeats)]))
        for lab, H in (("bagged", hit_bag), ("refit", hit_ref)):
            row[f"sens_95_{lab}"] = float(np.nanmean(H[:, y == 1]))
            row[f"spec_95_{lab}"] = float(1 - np.nanmean(H[:, y == 0]))
        for b in batches:
            m = batch == b
            row[f"sens_95_bagged_{b}"] = float(np.nanmean(hit_bag[:, (y == 1) & m]))
            row[f"spec_95_bagged_{b}"] = float(1 - np.nanmean(hit_bag[:, (y == 0) & m]))
        row["typical_params"] = pd.Series(params).value_counts().index[0]
        pooled_rows.append(row)
        oof_store[(name, s)] = np.nanmean(P_bag, axis=0)
        log(f"   {name:45s} {s:5s} AUC {row['auc']:.3f} ± {row['auc_sd']:.3f}  pAUC {row['pauc']:.3f}  "
            f"sens@95 {row['sens_95_bagged']:.2f} (spec {row['spec_95_bagged']:.2f})  "
            + "  ".join(f"{b} {row[f'auc_{b}']:.2f}" for b in batches))
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(out / "01_pooled_cv.tsv", sep="\t", index=False)
    pd.DataFrame({"sample_id": idx, "ml_label": y, "batch": batch,
                  **{f"{n} | {s}": v for (n, s), v in oof_store.items()}}).to_csv(out / "01_pooled_oof.tsv", sep="\t", index=False)

    # ---------------------------------------------------------------- leave one batch out and paper split
    designs = [(f"train {a} -> test {b}", batch == a, batch == b) for a in batches for b in batches if a != b]
    groups = samples["model_group"].to_numpy()
    designs.append(("paper: TV -> SBT", groups == "TV", groups == "SBT"))
    log(f"transfer designs: {', '.join(d[0] for d in designs)}")
    tasks2 = [(c, d) for c in range(len(configs)) for d in range(len(designs))]
    res2 = Parallel(n_jobs=args.jobs)(
        delayed(run)(cache_s, args.quick, *configs[c], y, strata, np.where(designs[d][1])[0], np.where(designs[d][2])[0], SEED)
        for c, d in tasks2)
    transfer_rows = []
    for (c, d), (p_ref, p_bag, thr, bp) in zip(tasks2, res2):
        name, s = configs[c]
        dname, tr, te = designs[d]
        yt, bt = y[te], batch[te]
        m = metrics(yt, p_bag, bt if dname.startswith("paper") else None)
        sens, spec = threshold_rates(yt, p_bag, thr)
        sens_r, spec_r = threshold_rates(yt, p_ref, thr)
        transfer_rows.append({"design": dname, "pipeline": name, "tuned_by": s, "n_train": int(tr.sum()),
                              "cancers_train": int(y[tr].sum()), "n_test": int(te.sum()), "cancers_test": int(yt.sum()),
                              **m, "auc_refit": roc_auc_score(yt, p_ref), "sens_95_bagged": sens, "spec_95_bagged": spec,
                              "sens_95_refit": sens_r, "spec_95_refit": spec_r, "params": json.dumps(bp, sort_keys=True)})
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv(out / "02_transfer.tsv", sep="\t", index=False)
    for dname in transfer["design"].unique():
        t = transfer[transfer["design"] == dname]
        log(f"   {dname}: best AUC {t['auc'].max():.3f} ({t.loc[t['auc'].idxmax(), 'pipeline']}), "
            f"base methylation {t[t['pipeline'] == 'base · methylation']['auc'].max():.3f}")

    # ---------------------------------------------------------------- permutation test, best pooled config by pAUC
    best = pooled.sort_values("pauc", ascending=False).iloc[0]
    best_cfg = (best["pipeline"], best["tuned_by"])
    R["best_by_pauc"] = {"pipeline": best_cfg[0], "tuned_by": best_cfg[1], "auc": best["auc"], "pauc": best["pauc"]}
    if args.permutations:
        log(f"permutation test ({args.permutations} shuffles, 1 repeat): {best_cfg[0]} | {best_cfg[1]}")
        rng = np.random.default_rng(SEED)
        folds = [(tr, te) for r, f, tr, te in outer if r == 0]
        batch01 = (batch == batches[0]).astype(int)
        null = np.array(Parallel(n_jobs=args.jobs)(
            delayed(permuted_auc)(cache_s, args.quick, *best_cfg, rng.permutation(y), batch01, folds)
            for _ in range(args.permutations)))
        R["permutation"] = {"real_auc": float(best["auc"]), "null_mean": float(null.mean()), "null_max": float(null.max()),
                            "p": float((1 + (null >= best["auc"]).sum()) / (1 + len(null))), "n": int(len(null))}
        log(f"   real {best['auc']:.3f}, null {null.mean():.3f} (max {null.max():.3f}), p = {R['permutation']['p']:.4f}")

    # ---------------------------------------------------------------- figures
    lab = [f"{n}\n(tuned by {s})" for n, s in zip(pooled["pipeline"], pooled["tuned_by"])]
    yy = np.arange(len(pooled))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.42 * len(pooled) + 1.5), sharey=True)
    for ax, k, title in [(axes[0], "auc", "Pooled CV AUC"), (axes[1], "pauc", f"Partial AUC (FPR ≤ {MAX_FPR:.0%})")]:
        ax.barh(yy, pooled[k], xerr=pooled[f"{k}_sd"], color=["#2a78d6" if t == "auc" else "#1baf7a" for t in pooled["tuned_by"]])
        ax.set_xlim(0.5, 1)
        ax.set_title(title, fontsize=10)
    for b, col in zip(batches, ["#8a5cc2", "#eda100"]):
        axes[2].scatter(pooled[f"sens_95_bagged_{b}"], yy, color=col, label=f"batch {b}", zorder=3)
    axes[2].scatter(pooled["sens_95_bagged"], yy, color="#eb6834", marker="D", label="all", zorder=3)
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Sensitivity at the transferable 95%-spec threshold", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8)
    axes[0].set_yticks(yy, lab, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig01_pooled.png", bbox_inches="tight")
    plt.close(fig)

    piv = transfer.assign(cfg=transfer["pipeline"] + " | " + transfer["tuned_by"]).pivot(index="cfg", columns="design", values="auc")
    piv = piv.loc[[f"{n} | {s}" for n, s in configs]]
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(piv) + 1.5))
    w = 0.8 / piv.shape[1]
    for j, d in enumerate(piv.columns):
        ax.barh(np.arange(len(piv))[::-1] + (j - (piv.shape[1] - 1) / 2) * w, piv[d], height=w, label=d)
    ax.axvline(0.5, color="#8a8985", linestyle=":")
    ax.set_yticks(np.arange(len(piv))[::-1], piv.index, fontsize=7.5)
    ax.set_xlim(0.3, 1)
    ax.set_xlabel("Test AUC")
    ax.set_title("Transfer between sequencing batches and the paper's split", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig02_transfer.png", bbox_inches="tight")
    plt.close(fig)

    R["runtime_min"] = (time.time() - t0) / 60
    (out / "results.json").write_text(json.dumps(R, indent=2, default=str))
    log(f"done in {R['runtime_min']:.1f} min -> {out}")


if __name__ == "__main__":
    main()
