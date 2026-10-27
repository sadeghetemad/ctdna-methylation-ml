"""
08_ml_analysis.py: modelling and evaluation on the ML dataset of step 07
(Stage I lung cancer vs non-cancer control).

Design (no information from the test set reaches any choice):
    * The paper's split (Supplementary Table 6): model_group TV ("Training/Validation") is the
      development set, SBT ("Single-blind Test") is the test set (--train-group, --test-group).
    * Every candidate pipeline (scaling, imputation, dimensionality reduction, feature
      selection, model) is tuned with an inner grid search and scored with an outer,
      repeated, label x batch stratified CV on the development set only ("nested CV").
    * The candidate with the best nested-CV AUC is the pre-registered final model; an
      ensemble of the three best candidates is the second pre-specified model.
    * Both are refit on the whole development set and scored once on the test set: ROC-AUC
      and sensitivity at 95% specificity (threshold from development out-of-fold scores), with
      bootstrap CIs and AUC within each sequencing batch. All other candidates are exploratory.

Sections and outputs (--out):
    01 confounders          covariates and the batch as single predictors
    02 univariate           per-region AUC, Mann-Whitney + BH-FDR, batch check, PCA
    03 nested CV (dev)      candidate pipelines x feature inputs
    04 held-out test        selected model, ensemble, all candidates (exploratory)
    05 permutation test     selected configuration, label-shuffled, on all samples
    06 robustness           covariates-only, methylation + covariates, within-batch CV
    07 top regions          univariate effect sizes and final-model weights
    results.json + figures (*.png), read by scripts/09_published_benchmark.py and scripts/10_report.py

Examples:
    python scripts/08_ml_analysis.py --data data/ml_dataset/nfcore --out results/ml/nfcore_paper
    python scripts/08_ml_analysis.py --data data/ml_dataset/nfcore --out /tmp/ml_smoke --quick
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin, clone  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
CANCER, CONTROL, MUTED = "#eb6834", "#2a78d6", "#8a8985"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "axes.axisbelow": True, "font.size": 9})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================ building blocks

class OutlierCount(BaseEstimator, TransformerMixin):
    """Tumour-signal summary: share of regions far above / below the training controls.

    The training controls define a robust centre (median) and spread (MAD, floored at its
    median) per region; a sample is described by the share of regions with z > z_hi and
    z < -z_hi, and its mean z. Rare hypermethylated tumour signal spreads over many regions;
    this turns it into three numbers without any label-driven feature selection.
    """

    def __init__(self, z_hi=3.0):
        self.z_hi = z_hi

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        c = X[np.asarray(y) == 0]
        self.center_ = np.nanmedian(c, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(c - self.center_), axis=0)
        floor = np.nanmedian(mad[mad > 0]) if np.any(mad > 0) else 1.0
        self.scale_ = np.where(np.isfinite(mad), np.fmax(mad, floor), floor)
        self.center_ = np.nan_to_num(self.center_)
        return self

    def transform(self, X):
        z = (np.asarray(X, dtype=float) - self.center_) / self.scale_
        n = np.isfinite(z).sum(axis=1).clip(min=1)
        return np.c_[np.nansum(z > self.z_hi, axis=1) / n, np.nansum(z < -self.z_hi, axis=1) / n,
                     np.nanmean(np.clip(z, -10, 10), axis=1)]


def imp():
    return SimpleImputer(strategy="median", keep_empty_features=True)


def lr(**kw):
    return LogisticRegression(max_iter=5000, class_weight="balanced", **kw)


def candidates(n_features):
    """(pipeline, grid) per candidate; inputs with many columns skip the slow models."""
    heavy = n_features > 5000
    ks = sorted({min(k, n_features) for k in (50, 200, 800)})
    c = {
        "LogReg L2": (Pipeline([("imp", imp()), ("sc", StandardScaler()), ("clf", lr())]),
                      {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1]}),
        "LogReg L1 (sparse)": (Pipeline([("imp", imp()), ("sc", StandardScaler()),
                                         ("clf", lr(penalty="l1", solver="liblinear"))]),
                               {"clf__C": [0.01, 0.03, 0.1, 0.3]}),
        "Top-k ANOVA + LogReg": (Pipeline([("imp", imp()), ("sel", SelectKBest(f_classif)), ("sc", StandardScaler()),
                                           ("clf", lr())]),
                                 {"sel__k": ks, "clf__C": [0.01, 0.1]}),
        "PCA + LogReg": (Pipeline([("imp", imp()), ("sc", StandardScaler()), ("pca", PCA(random_state=SEED)),
                                   ("clf", lr())]),
                         {"pca__n_components": [10, 30, 80], "clf__C": [0.01, 0.1, 1.0]}),
        "Outlier count + LogReg": (Pipeline([("imp", imp()), ("out", OutlierCount()), ("sc", StandardScaler()),
                                             ("clf", lr())]),
                                   {"out__z_hi": [2.0, 3.0, 4.0], "clf__C": [0.1, 1.0]}),
        "Gradient boosting": (Pipeline([("clf", HistGradientBoostingClassifier(
                                  max_iter=300, l2_regularization=1.0, class_weight="balanced", random_state=SEED))]),
                              {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [7, 15]}),
    }
    if not heavy:
        c["SVM (RBF)"] = (Pipeline([("imp", imp()), ("sc", StandardScaler()), ("pca", PCA(n_components=80, random_state=SEED)),
                                    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))]),
                          {"clf__C": [0.3, 1.0, 3.0], "clf__gamma": ["scale", 0.003]})
        c["Random forest"] = (Pipeline([("imp", imp()), ("clf", RandomForestClassifier(
                                  n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2,
                                  random_state=SEED, n_jobs=1))]),
                              {"clf__max_features": ["sqrt", 0.05]})
    return c


def tuned(pipe, grid, inner_seed):
    return GridSearchCV(pipe, grid, scoring="roc_auc", n_jobs=1, refit=True,
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_seed))


# ============================================================ metrics

def column_auc(X, y):
    R = X.rank().to_numpy()
    seen = ~np.isnan(R)
    R = np.nan_to_num(R)
    pos = np.asarray(y) == 1
    n1, n0 = seen[pos].sum(0), seen[~pos].sum(0)
    U = R[pos].sum(0) - n1 * (n1 + 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = U / (n1 * n0)
        z = (U - n1 * n0 / 2) / np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12)
    return pd.Series(auc, X.columns), pd.Series(2 * stats.norm.sf(np.abs(z)), X.columns)


def sens_at_threshold(y, p, thr):
    y = np.asarray(y)
    return float((p[y == 1] > thr).mean())


def _cv_pred(pipe, Xs, yv, seed):
    p = np.zeros(len(yv))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, yv):
        p[va] = clone(pipe).fit(Xs.iloc[tr], yv[tr]).predict_proba(Xs.iloc[va])[:, 1]
    return p


# ============================================================ main

def main():
    ap = argparse.ArgumentParser(description="Modelling and evaluation on the step 07 ML dataset.")
    ap.add_argument("--data", default="data/ml_dataset/nfcore")
    ap.add_argument("--out", default="results/ml/nfcore_paper")
    ap.add_argument("--outer-repeats", type=int, default=3, help="Repeats of the outer 5-fold CV on the development set.")
    ap.add_argument("--train-group", default="TV", help="model_group used for development (paper: TV).")
    ap.add_argument("--test-group", default="SBT", help="model_group used once as test set (paper: SBT).")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="Smoke test: 1 outer repeat, 10 permutations, 100 boots.")
    ap.add_argument("--max-features", type=int, default=0, help="Keep only the first N columns per set (testing).")
    args = ap.parse_args()
    if args.quick:
        args.outer_repeats, args.permutations, args.boot = 1, 10, 100

    data = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = {"settings": vars(args)}

    # ------------------------------------------------ load
    info = json.loads((data / "dataset_info.json").read_text())
    samples = pd.read_csv(data / "samples.tsv", sep="\t").set_index("sample_id")
    regions = pd.read_csv(data / "regions.tsv", sep="\t").set_index("region_id")
    sets = {"methylation": pd.read_csv(data / "X_methylation.tsv", sep="\t", index_col="sample_id")}
    for name in info["read_feature_sets"]:
        sets[name] = pd.read_csv(data / f"X_{name}.tsv", sep="\t", index_col="sample_id")
    X_sample = pd.read_csv(data / "X_sample.tsv", sep="\t", index_col="sample_id")
    if args.max_features:
        sets = {k: v.iloc[:, :args.max_features] for k, v in sets.items()}
    idx = sets["methylation"].index
    for X in [*sets.values(), X_sample]:
        assert X.index.equals(idx), "sample order differs between matrices"
    samples = samples.loc[idx]
    y = samples["ml_label"].astype(int)
    X = sets["methylation"]
    BATCH = "submission_series"
    big_batch = samples[BATCH].value_counts().index[0]
    batch_bin = (samples[BATCH] == big_batch).astype(int)
    strata = (y * 2 + batch_bin).to_numpy()
    train = (samples["model_group"] == args.train_group).to_numpy()
    test = (samples["model_group"] == args.test_group).to_numpy()
    DEV, TEST = args.train_group, args.test_group
    R["data"] = {"train_group": DEV, "test_group": TEST, "samples": len(y), "cancer": int(y.sum()), "control": int((y == 0).sum()), "regions": X.shape[1],
                 "train": int(train.sum()), "train_cancer": int(y[train].sum()),
                 "test": int(test.sum()), "test_cancer": int(y[test].sum()),
                 "batches": samples[BATCH].value_counts().to_dict(), "big_batch": big_batch,
                 "batch_by_label": pd.crosstab(samples[BATCH], y).to_dict(),
                 "stage": samples["stage"].value_counts().to_dict()}
    log(f"{len(y)} samples; development {DEV} {train.sum()} / test {TEST} {test.sum()}; sets: {', '.join(sets)}")

    # ------------------------------------------------ 01 confounders
    log("01 confounders")
    covs = {"Sequencing batch": batch_bin.astype(float), "Age": samples["age"], "Gender (male)": (samples["gender"] == "M").astype(float),
            "DNA input (ng)": samples["dna_input_ng"], "Median region depth": samples["median_region_depth"],
            "Fragments": samples["n_fragments"], "End-repair artifact": samples["r2_end_ch_meth"],
            "Mean region methylation": samples["mean_region_methylation"]}
    covs.update({f"Fragment {c.replace('frag_', '').replace('_', ' ')}": X_sample[c] for c in X_sample.columns})
    rows = []
    for name, v in covs.items():
        v = v.astype(float)
        m = v.notna().to_numpy()
        a = roc_auc_score(y[m], v[m])
        p = stats.mannwhitneyu(v[m & (y == 1).to_numpy()], v[m & (y == 0).to_numpy()]).pvalue
        rows.append({"variable": name, "auc": a, "abs_auc": max(a, 1 - a), "p": p,
                     "control_median": v[y == 0].median(), "cancer_median": v[y == 1].median()})
    conf = pd.DataFrame(rows).set_index("variable").sort_values("abs_auc", ascending=False)
    conf.to_csv(out / "01_confounders.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(conf.index[::-1], conf["abs_auc"][::-1] - 0.5, left=0.5,
            color=[CANCER if v == "Mean region methylation" else MUTED for v in conf.index[::-1]])
    ax.axvline(0.5, color="black", linewidth=1)
    ax.set_xlim(0.45, 0.8)
    ax.set_xlabel("AUC of the variable alone (0.5 = no information)")
    ax.set_title("How well does each variable alone separate cancer from control?", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig01_confounders.png")
    plt.close(fig)

    # ------------------------------------------------ 02 univariate + PCA
    log("02 univariate and PCA")
    big = batch_bin.to_numpy() == 1
    uni_rows, uni = [], {}
    for name, Xs in {**sets, "sample fragment features": X_sample}.items():
        a, p = column_auc(Xs, y)
        q = multipletests(p.fillna(1), method="fdr_bh")[1]
        ab, _ = column_auc(Xs[big], y[big])
        t = pd.DataFrame({"auc": a, "p": p, "fdr": q, "auc_big_batch": ab})
        uni[name] = t
        sig = t["fdr"] < 0.05
        uni_rows.append({"feature_set": name, "features": Xs.shape[1], "fdr_sig": int(sig.sum()),
                         "pct_sig": 100 * sig.mean(), "up_in_cancer": int((sig & (t["auc"] > 0.5)).sum()),
                         "down_in_cancer": int((sig & (t["auc"] < 0.5)).sum()),
                         "best_auc": float(t["auc"].sub(0.5).abs().max() + 0.5),
                         "batch_consistency": float(np.corrcoef(t["auc"].fillna(.5), t["auc_big_batch"].fillna(.5))[0, 1])})
    univ = pd.DataFrame(uni_rows).set_index("feature_set")
    univ.to_csv(out / "02_univariate_summary.tsv", sep="\t")
    regcols = [c for c in ("chrom", "start0", "end", "median_depth") if c in regions.columns]
    top_uni = uni["methylation"].join(regions[regcols])
    top_uni.index.name = "region_id"
    top_uni["control_mean"], top_uni["cancer_mean"] = X[y == 0].mean(), X[y == 1].mean()
    top_uni["delta"] = top_uni["cancer_mean"] - top_uni["control_mean"]
    top_uni = top_uni.reindex(top_uni["auc"].sub(0.5).abs().sort_values(ascending=False).index)
    top_uni.to_csv(out / "02_univariate_methylation_regions.tsv", sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sig = top_uni["fdr"] < 0.05
    axes[0].scatter(top_uni["delta"], -np.log10(top_uni["p"].clip(lower=1e-300)), s=6, alpha=0.5,
                    c=np.where(sig, CANCER, MUTED), linewidth=0)
    axes[0].axvline(0, color=MUTED, linewidth=1)
    axes[0].set_xlabel("Methylation difference (cancer − control)")
    axes[0].set_ylabel("−log10 p")
    axes[0].set_title(f"{int(sig.sum())} of {len(sig)} regions differ (FDR < 5%), almost all higher in cancer", fontsize=9)
    axes[1].scatter(top_uni["auc"], top_uni["auc_big_batch"], s=6, alpha=0.5, color=CONTROL, linewidth=0)
    axes[1].plot([0.25, 0.85], [0.25, 0.85], color=MUTED, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Region AUC, all samples")
    axes[1].set_ylabel(f"Region AUC, batch {big_batch} only")
    axes[1].set_title(f"Same signal inside one batch (r = {univ.loc['methylation', 'batch_consistency']:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig02_univariate.png")
    plt.close(fig)

    Z = StandardScaler().fit_transform(imp().fit_transform(X))
    pca = PCA(n_components=5, random_state=SEED).fit(Z)
    pc = pca.transform(Z)
    pc_auc = [max(a, 1 - a) for a in (roc_auc_score(y, pc[:, i]) for i in range(5))]
    pc_batch = [max(a, 1 - a) for a in (roc_auc_score(batch_bin, pc[:, i]) for i in range(5))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for lab, col, nm in [(0, CONTROL, "Control"), (1, CANCER, "Stage I cancer")]:
        m = (y == lab).to_numpy()
        axes[0].scatter(pc[m, 0], pc[m, 1], s=10, alpha=0.6, color=col, label=f"{nm} (n={m.sum()})", linewidth=0)
    for b, col in zip(sorted(samples[BATCH].unique()), ["#1baf7a", "#8a5cc2"]):
        m = (samples[BATCH] == b).to_numpy()
        axes[1].scatter(pc[m, 0], pc[m, 1], s=10, alpha=0.6, color=col, label=f"{b} (n={m.sum()})", linewidth=0)
    for ax, ttl in zip(axes, ["Coloured by label", "Coloured by sequencing batch"]):
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_title(ttl, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    fig.tight_layout()
    fig.savefig(out / "fig03_pca.png")
    plt.close(fig)
    R["pca"] = {"explained": pca.explained_variance_ratio_.tolist(), "auc_label": pc_auc, "auc_batch": pc_batch}

    # ------------------------------------------------ 03 nested CV on the development set
    read_sets = list(info["read_feature_sets"])
    inputs = {
        "Region methylation": X,
        "Methylation + fragment features": pd.concat(
            [X.add_prefix("meth04:"), *[sets[n].add_prefix(f"{n}:") for n in read_sets],
             X_sample.add_prefix("sample:")], axis=1),
    }
    X_tr_by_input = {k: v[train] for k, v in inputs.items()}
    y_tr, s_tr = y[train], strata[train]
    outer = [(r, f, tr, va) for r in range(args.outer_repeats)
             for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(np.zeros(len(y_tr)), s_tr))]
    jobs = []
    for in_name, Xi in X_tr_by_input.items():
        for c_name, (pipe, grid) in candidates(Xi.shape[1]).items():
            jobs.append((in_name, c_name, pipe, grid))
    log(f"03 nested CV on {DEV}: {len(jobs)} candidate x input pairs, {len(outer)} outer folds")

    def run_fold(in_name, pipe, grid, r, f, tr, va):
        Xi = X_tr_by_input[in_name]
        gs = tuned(pipe, grid, SEED + 100 * r + f).fit(Xi.iloc[tr], y_tr.iloc[tr])
        return gs.predict_proba(Xi.iloc[va])[:, 1], gs.best_params_

    tasks = [(j, o) for j in range(len(jobs)) for o in range(len(outer))]
    res = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(run_fold)(jobs[j][0], jobs[j][2], jobs[j][3], *outer[o]) for j, o in tasks)
    oof = {j: np.full((args.outer_repeats, len(y_tr)), np.nan) for j in range(len(jobs))}
    chosen = {j: [] for j in range(len(jobs))}
    for (j, o), (p, bp) in zip(tasks, res):
        r, f, tr, va = outer[o]
        oof[j][r, va] = p
        chosen[j].append(bp)
    nested_rows = []
    for j, (in_name, c_name, _, _) in enumerate(jobs):
        aucs = [roc_auc_score(y_tr, oof[j][r]) for r in range(args.outer_repeats)]
        mean_p = np.nanmean(oof[j], axis=0)
        thr = np.quantile(mean_p[y_tr.to_numpy() == 0], 0.95)
        common = pd.Series([json.dumps(b, sort_keys=True) for b in chosen[j]]).value_counts().index[0]
        nested_rows.append({"input": in_name, "candidate": c_name, "nested_auc": np.mean(aucs),
                            "nested_auc_sd": np.std(aucs), "sens_at_95spec": sens_at_threshold(y_tr, mean_p, thr),
                            "typical_params": common})
    nested = pd.DataFrame(nested_rows).sort_values("nested_auc", ascending=False).reset_index(drop=True)
    nested.to_csv(out / "03_nested_cv_dev.tsv", sep="\t", index=False)
    for _, rw in nested.iterrows():
        log(f"   {rw['input']:34s} {rw['candidate']:26s} nested AUC {rw['nested_auc']:.3f} ± {rw['nested_auc_sd']:.3f}")

    # selection: best single candidate; ensemble of the three best (distinct candidates)
    job_index = {(jobs[j][0], jobs[j][1]): j for j in range(len(jobs))}
    best = nested.iloc[0]
    top3 = nested.drop_duplicates("candidate").head(3)
    ens_oof = np.mean([np.nanmean(oof[job_index[(r["input"], r["candidate"])]], axis=0) for _, r in top3.iterrows()], axis=0)
    ens_auc = roc_auc_score(y_tr, ens_oof)
    R["selection"] = {"best_input": best["input"], "best_candidate": best["candidate"], "best_nested_auc": best["nested_auc"],
                      "ensemble_members": [f"{r['candidate']} · {r['input']}" for _, r in top3.iterrows()],
                      "ensemble_oof_auc_dev": ens_auc}
    log(f"   selected: {best['candidate']} on {best['input']} (nested AUC {best['nested_auc']:.3f}); ensemble OOF AUC {ens_auc:.3f}")

    plot = nested.copy()
    plot["label"] = plot["candidate"] + "  ·  " + plot["input"].str.replace("Methylation + fragment features", "meth + fragment")
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(plot) + 1.2))
    yy = np.arange(len(plot))[::-1]
    ax.errorbar(plot["nested_auc"], yy, xerr=plot["nested_auc_sd"], fmt="none", ecolor="#c9c8c3", elinewidth=2)
    ax.scatter(plot["nested_auc"], yy, s=30, zorder=3,
               c=[CANCER if i == 0 else (CONTROL if "fragment" in inp else "#1baf7a") for i, inp in enumerate(plot["input"])])
    ax.set_yticks(yy, plot["label"], fontsize=7.5)
    ax.axvline(0.5, color=MUTED, linestyle=":")
    ax.set_xlabel(f"Nested cross-validated AUC on the development set ({DEV})")
    ax.set_title("Model selection (orange: selected; green: methylation only; blue: + fragment features)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig04_model_selection.png")
    plt.close(fig)

    # ------------------------------------------------ 04 held-out test set
    log(f"04 held-out test set {TEST}")
    y_te = y[test].to_numpy()
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, test.sum(), test.sum()) for _ in range(args.boot)]
    batches = sorted(samples[BATCH].unique())
    te_batch = samples.loc[test, BATCH].to_numpy()

    def final_fit(in_name, c_name):
        pipe, grid = candidates(inputs[in_name].shape[1])[c_name]
        gs = tuned(pipe, grid, SEED).fit(X_tr_by_input[in_name], y_tr)
        return gs.predict_proba(inputs[in_name][test])[:, 1], gs.best_params_

    def evaluate(p, thr):
        auc_b = [roc_auc_score(y_te[b], p[b]) for b in boots if 0 < y_te[b].sum() < len(b)]
        sens_b = [sens_at_threshold(y_te[b], p[b], thr) for b in boots]
        row = {"auc_test": roc_auc_score(y_te, p), "auc_ci_low": np.percentile(auc_b, 2.5), "auc_ci_high": np.percentile(auc_b, 97.5),
               "sens_at_95spec": sens_at_threshold(y_te, p, thr), "sens_ci_low": np.percentile(sens_b, 2.5),
               "sens_ci_high": np.percentile(sens_b, 97.5), "spec_reached": float((p[y_te == 0] <= thr).mean())}
        for b in batches:
            m = te_batch == b
            row[f"auc_test_{b}"] = roc_auc_score(y_te[m], p[m]) if 0 < y_te[m].sum() < m.sum() else np.nan
        return row

    fits = Parallel(n_jobs=args.jobs)(delayed(final_fit)(inp, cand) for inp, cand, _, _ in jobs)
    ho_rows, ho_probs = [], {}
    for j, ((inp, cand, _, _), (p, bp)) in enumerate(zip(jobs, fits)):
        mean_oof = np.nanmean(oof[j], axis=0)
        thr = np.quantile(mean_oof[y_tr.to_numpy() == 0], 0.95)
        ho_probs[(inp, cand)] = (p, thr)
        role = "selected" if (inp, cand) == (best["input"], best["candidate"]) else "exploratory"
        ho_rows.append({"input": inp, "candidate": cand, "role": role, "final_params": json.dumps(bp), **evaluate(p, thr)})
    ens_p = np.mean([ho_probs[(r["input"], r["candidate"])][0] for _, r in top3.iterrows()], axis=0)
    ens_thr = np.quantile(ens_oof[y_tr.to_numpy() == 0], 0.95)
    ho_rows.append({"input": "top-3 ensemble", "candidate": "Mean of 3 best", "role": "pre-specified ensemble",
                    "final_params": "", **evaluate(ens_p, ens_thr)})
    ho = pd.DataFrame(ho_rows)
    ho.to_csv(out / "04_heldout_test.tsv", sep="\t", index=False)
    sel_row = ho[ho["role"] == "selected"].iloc[0]
    ens_row = ho[ho["role"] == "pre-specified ensemble"].iloc[0]
    R["heldout"] = {"selected": sel_row.drop(["final_params"]).to_dict(), "ensemble": ens_row.drop(["final_params"]).to_dict(),
                    "best_exploratory_auc": float(ho["auc_test"].max())}
    log(f"   selected test AUC {sel_row['auc_test']:.3f}, sens@95spec {sel_row['sens_at_95spec']:.2f}; "
        f"ensemble test AUC {ens_row['auc_test']:.3f}")

    sel_p, sel_thr = ho_probs[(best["input"], best["candidate"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
    curves = [("Selected: " + best["candidate"], sel_p, CANCER), ("Top-3 ensemble", ens_p, "#8a5cc2")]
    for _, r in top3.iterrows():
        if (r["input"], r["candidate"]) != (best["input"], best["candidate"]):
            curves.append((r["candidate"], ho_probs[(r["input"], r["candidate"])][0], MUTED))
    for nm, p, col in curves:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, color=col, drawstyle="steps-post", linewidth=2 if col != MUTED else 1.2,
                label=f"{nm} (AUC {roc_auc_score(y_te, p):.2f})")
    ax.axvline(0.05, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"Held-out test set ({TEST}, n={len(y_te)}), trained on {DEV}", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax = axes[1]
    for lab, col, nm in [(0, CONTROL, "Control"), (1, CANCER, "Stage I cancer")]:
        v = sel_p[y_te == lab]
        ax.scatter(np.full(len(v), lab) + np.random.default_rng(lab).uniform(-0.2, 0.2, len(v)), v, s=10, alpha=0.6, color=col)
        ax.boxplot(v, positions=[lab], widths=0.5, showfliers=False, medianprops={"color": "black"})
    ax.axhline(sel_thr, color="black", linestyle="--", linewidth=1)
    ax.text(1.45, sel_thr, " 95% specificity\n threshold", fontsize=7, va="center")
    ax.set_xticks([0, 1], ["Control", "Stage I cancer"])
    ax.set_xlim(-0.6, 1.9)
    ax.set_ylabel("Predicted cancer score")
    ax.set_title("Scores of the selected model on the test set", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig05_heldout.png")
    plt.close(fig)

    # ------------------------------------------------ 05 permutation test
    log(f"05 permutation test ({args.permutations})")
    sel_params = json.loads(sel_row["final_params"])
    sel_pipe = clone(candidates(inputs[best["input"]].shape[1])[best["candidate"]][0]).set_params(**sel_params)
    Xa = inputs[best["input"]]

    def cv_auc(labels, seed):
        return roc_auc_score(labels, _cv_pred(sel_pipe, Xa, labels, seed))

    yv = y.to_numpy()
    real = float(np.mean(Parallel(n_jobs=args.jobs)(delayed(cv_auc)(yv, SEED + r) for r in range(3))))
    null = np.array(Parallel(n_jobs=args.jobs)(
        delayed(cv_auc)(np.random.default_rng(SEED + 1000 + i).permutation(yv), SEED) for i in range(args.permutations)))
    perm_p = (1 + (null >= real).sum()) / (1 + len(null))
    R["permutation"] = {"real_auc": real, "null_mean": float(null.mean()), "null_max": float(null.max()), "p": perm_p, "n": len(null)}
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(null, bins=20, color=MUTED, edgecolor="white", label="Shuffled labels")
    ax.axvline(real, color=CANCER, linewidth=2, label=f"Real labels (AUC {real:.2f})")
    ax.set_xlabel("Cross-validated AUC")
    ax.set_title(f"Permutation test of the selected model (p = {perm_p:.3f}, {len(null)} shuffles)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig06_permutation.png")
    plt.close(fig)

    # ------------------------------------------------ 06 robustness
    log("06 robustness")
    cov = samples[["age", "dna_input_ng", "median_region_depth", "n_fragments", "r2_end_ch_meth"]].astype(float).copy()
    cov["batch"] = batch_bin.astype(float)
    cov["male"] = (samples["gender"] == "M").astype(float)
    lr2 = Pipeline([("imp", imp()), ("sc", StandardScaler()), ("clf", lr(C=1.0))])

    def ho_simple(pipe, Xs):
        cvp = np.zeros(int(train.sum()))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs[train], s_tr):
            cvp[va] = clone(pipe).fit(Xs[train].iloc[tr], y_tr.iloc[tr]).predict_proba(Xs[train].iloc[va])[:, 1]
        thr = np.quantile(cvp[y_tr.to_numpy() == 0], 0.95)
        return evaluate(clone(pipe).fit(Xs[train], y_tr).predict_proba(Xs[test])[:, 1], thr)

    rob = [{"analysis": "Covariates only (batch, age, gender, DNA input, depth, fragments, artifact)", **ho_simple(lr2, cov)},
           {"analysis": "Selected model + covariates", **ho_simple(sel_pipe, pd.concat([Xa, cov.add_prefix('cov:')], axis=1))}]
    for b in batches:
        m = (samples[BATCH] == b).to_numpy()
        if pd.Series(yv[m]).value_counts().min() >= 15:
            aucs = [roc_auc_score(yv[m], _cv_pred(sel_pipe, Xa[m], yv[m], SEED + r)) for r in range(3)]
            rob.append({"analysis": f"CV within batch {b} only (n={m.sum()}, {int(yv[m].sum())} cancers)", "cv_auc": float(np.mean(aucs))})
    robust = pd.DataFrame(rob).set_index("analysis")
    robust.to_csv(out / "06_robustness.tsv", sep="\t")
    R["robustness"] = json.loads(robust.to_json(orient="index"))

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    names = [f"Test, batch {b}" for b in batches] + ["Test, all"]
    vals = [sel_row[f"auc_test_{b}"] for b in batches] + [sel_row["auc_test"]]
    ax.barh(names[::-1], np.array(vals[::-1]) - 0.5, left=0.5, color=[CANCER] + [CONTROL] * len(batches))
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(0.5, color="black", linewidth=1)
    ax.axvline(robust.iloc[0]["auc_test"], color=MUTED, linestyle="--", linewidth=1)
    ax.text(robust.iloc[0]["auc_test"], len(names) - 0.4, " covariates only", fontsize=7, color=MUTED)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC of the selected model")
    ax.set_title("Selected model overall and per sequencing batch", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig07_batch.png")
    plt.close(fig)

    # ------------------------------------------------ 07 top regions
    log("07 top regions")
    top_uni.head(100).to_csv(out / "07_top_regions_univariate.tsv", sep="\t")
    final = clone(sel_pipe).fit(Xa[train], y_tr)
    clf = final.steps[-1][1]
    if hasattr(clf, "coef_") and "pca" not in final.named_steps and "out" not in final.named_steps:
        cols = Xa.columns[final.named_steps["sel"].get_support()] if "sel" in final.named_steps else Xa.columns
        weights = pd.Series(clf.coef_[0], index=cols, name="weight")
        weights = weights.reindex(weights.abs().sort_values(ascending=False).index)
        weights.head(100).to_csv(out / "07_final_model_weights.tsv", sep="\t")
    shown = top_uni.head(20)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(shown.index[::-1], shown["delta"][::-1] * 100, color=CANCER)
    for i, (rid, r) in enumerate(shown[::-1].iterrows()):
        ax.text(r["delta"] * 100, i, f"  AUC {r['auc']:.2f}", va="center", fontsize=7)
    ax.set_xlabel("Methylation difference, cancer − control (percentage points)")
    ax.set_title("20 most discriminative regions (univariate)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig08_top_regions.png")
    plt.close(fig)
    R["top_regions"] = shown[["auc", "fdr", "delta", *regcols]].reset_index().to_dict(orient="records")
    R["univariate"] = json.loads(univ.to_json(orient="index"))
    R["confounders"] = json.loads(conf.to_json(orient="index"))
    R["nested"] = nested.to_dict(orient="records")
    R["heldout_all"] = json.loads(ho.drop(columns="final_params").to_json(orient="records"))
    R["runtime_min"] = (time.time() - t0) / 60
    (out / "results.json").write_text(json.dumps(R, indent=2, default=float))
    log(f"done in {R['runtime_min']:.1f} min -> {out}")


if __name__ == "__main__":
    main()
