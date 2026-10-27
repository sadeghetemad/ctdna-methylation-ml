"""
11_classical_ml.py: classical machine-learning models with in-fold feature selection, SHAP explanations
and a fairness audit.

Every model is one pipeline: in-fold feature selection (the steps of 10: quality filter, single-feature
AUC ranking, correlation pruning, top k) -> imputation -> scaling -> classifier, with class-balanced
weights. On the training part of a design:

    outer CV     5 folds x 2 repeats, stratified by label x batch; inner 3-fold grid search (ROC-AUC)
    test score   mean of the outer-fold models (a CV ensemble), so the scores on the test part come from
                 the same models whose out-of-fold scores set the 95%-specificity threshold
    explanation  the best tree model and the best linear model are refit on the whole training part;
                 SHAP values on the test part (TreeExplainer / LinearExplainer), or permutation importance
                 when the shap package is not installed

Models: L2 and elastic-net logistic regression, SVM (RBF), random forest, extra trees, histogram gradient
boosting, and XGBoost / LightGBM when installed. Two controls:

    covariates only   the same evaluation for a logistic model on technical and demographic covariates
                      (batch, age, sex, DNA input, depth, fragments, artifact strength): what a model could get
                      without any methylation signal (row family "reference")
    permutation test  the best model's CV AUC against label-shuffled refits (5 folds, its typical settings)

Outputs (results/modelling/11_classical/<design>/): cv_folds.tsv, results.tsv, subgroups.tsv,
predictions.tsv, permutation.json, fig_permutation.png, importance_<model>.tsv, fig_shap_<model>.png, fig_family_importance.png, results.json
"""

import argparse
import importlib.util
import json
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Memory, Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from ml_common import (DESIGNS, FAMILIES, OUT, SEED, fold_vote, designs, family, load, log, logreg, out_dir,  # noqa: E402
                    outer_folds, region, strata, summarise)
from ml_selection import FoldSelector  # noqa: E402

warnings.filterwarnings("ignore")
HAS = {m: importlib.util.find_spec(m) is not None for m in ("xgboost", "lightgbm", "shap")}


def candidates(memory):
    imp = lambda: SimpleImputer(strategy="median", keep_empty_features=True)  # noqa: E731
    sel = lambda: FoldSelector()  # noqa: E731
    lin = lambda clf: Pipeline([("sel", sel()), ("imp", imp()), ("sc", StandardScaler()), ("clf", clf)], memory=memory)  # noqa: E731
    tree = lambda clf: Pipeline([("sel", sel()), ("imp", imp()), ("clf", clf)], memory=memory)  # noqa: E731
    c = {
        "Logistic regression (L2)": ("linear", lin(logreg(0.0)), {"sel__k": [50, 200, 1000], "clf__C": [0.01, 0.1, 1.0]}),
        "Elastic-net logistic": ("linear", lin(logreg(0.5, solver="saga")),
                                 {"sel__k": [200, 1000], "clf__C": [0.05, 0.2, 1.0]}),
        "SVM (RBF kernel)": ("kernel", lin(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
                             {"sel__k": [50, 200], "clf__C": [0.3, 1.0, 3.0]}),
        "Random forest": ("tree", tree(RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=1,
                                                              class_weight="balanced_subsample", random_state=SEED)),
                          {"sel__k": [200, 1000], "clf__max_features": ["sqrt", 0.1]}),
        "Extra trees": ("tree", tree(ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=1,
                                                          class_weight="balanced_subsample", random_state=SEED)),
                        {"sel__k": [200, 1000], "clf__max_features": ["sqrt", 0.1]}),
        "Gradient boosting": ("tree", tree(HistGradientBoostingClassifier(max_iter=300, l2_regularization=1.0,
                                                                          class_weight="balanced", random_state=SEED)),
                              {"sel__k": [200, 1000], "clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [7, 15]}),
    }
    if HAS["xgboost"]:
        from xgboost import XGBClassifier
        c["XGBoost"] = ("tree", tree(XGBClassifier(n_estimators=400, subsample=0.8, colsample_bytree=0.5, n_jobs=1,
                                                   eval_metric="logloss", random_state=SEED)),
                        {"sel__k": [200, 1000], "clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1]})
    if HAS["lightgbm"]:
        from lightgbm import LGBMClassifier
        c["LightGBM"] = ("tree", tree(LGBMClassifier(n_estimators=400, subsample=0.8, subsample_freq=1,
                                                     colsample_bytree=0.5, class_weight="balanced", n_jobs=1,
                                                     verbose=-1, random_state=SEED)),
                         {"sel__k": [200, 1000], "clf__num_leaves": [7, 15], "clf__learning_rate": [0.03, 0.1]})
    return c


def fit_fold(pipe, grid, X, y, key, tr, seed):
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(tr)), key[tr]))
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True).fit(X[tr], y[tr])
    return gs.best_estimator_, gs.best_params_


def explain(name, kind, est, X_tr, y_tr, X_te, y_te, cols, out, max_display=20):
    """SHAP (or permutation importance) of a refit pipeline, on the test part."""
    sel = est.named_steps["sel"]
    names = np.asarray(cols)[sel.support_]
    pre = Pipeline(est.steps[:-1])
    A_te = pre.transform(X_te)
    clf = est.named_steps["clf"]
    method = "permutation"
    if HAS["shap"] and kind in ("tree", "linear"):
        import shap
        A_tr = pre.transform(X_tr)
        try:
            if kind == "tree":
                sv = shap.TreeExplainer(clf).shap_values(A_te)
            else:
                sv = shap.LinearExplainer(clf, A_tr).shap_values(A_te)
            sv = sv[1] if isinstance(sv, list) else sv
            sv = sv[..., 1] if sv.ndim == 3 else sv
            imp = np.abs(sv).mean(0)
            method = "shap"
            order = np.argsort(-imp)[:max_display]
            plt.figure()
            shap.summary_plot(sv[:, order], A_te[:, order], feature_names=[pretty(n) for n in names[order]],
                              show=False, max_display=max_display, plot_size=(8, 6))
            plt.title(f"SHAP values on the test set: {name}", fontsize=10)
            plt.tight_layout()
            plt.savefig(out / f"fig_shap_{slug(name)}.png", dpi=170)
            plt.close("all")
        except Exception as e:  # SHAP version quirks: fall back to permutation importance
            log(f"   SHAP failed for {name} ({e}); using permutation importance")
            method = "permutation"
    if method == "permutation":
        r = permutation_importance(clf, A_te, y_te, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=-1)
        imp = np.clip(r.importances_mean, 0, None)
    tab = pd.DataFrame({"feature": names, "importance": imp, "family": [family(n) for n in names],
                        "region": [region(n) for n in names]}).sort_values("importance", ascending=False)
    tab["method"] = method
    return tab


def covariates(samples):
    """Technical and demographic covariates, numeric (batch and sex as 0 / 1)."""
    s = samples
    return pd.DataFrame({"batch": (s["submission_series"] == s["submission_series"].value_counts().index[0]).astype(float),
                         "male": s["gender"].astype(str).str.upper().str.startswith("M").astype(float),
                         **{c: s[c].astype(float) for c in ("age", "dna_input_ng", "median_region_depth", "n_fragments",
                                                            "r2_end_ch_meth", "ch_filtered_fraction")}}, index=s.index)


def pretty(col):
    f, r = col.split(":", 1)
    return f"{FAMILIES.get(f, f).split(' (')[0]} · {r}"


def slug(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def main():
    ap = argparse.ArgumentParser(description="Classical ML with in-fold selection, SHAP and a fairness audit.")
    ap.add_argument("--design", default="balanced", choices=list(DESIGNS))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="Smoke test: 2 models, 1 repeat, small grids.")
    args = ap.parse_args()
    out = out_dir(f"11_classical/{args.design}")
    samples, X, _ = load()
    tr_idx, te_idx = designs(samples)[args.design]
    s_tr, s_te = samples.loc[tr_idx], samples.loc[te_idx]
    Xtr, Xte = X.loc[tr_idx].to_numpy(), X.loc[te_idx].to_numpy()
    y_tr, y_te = s_tr["ml_label"].to_numpy(), s_te["ml_label"].to_numpy()
    key = strata(s_tr, with_demographics=False).to_numpy()
    folds = outer_folds(s_tr, repeats=1 if args.quick else args.repeats)
    memory = Memory(tempfile.mkdtemp(prefix="ctdna_sel_"), verbose=0)
    C = candidates(memory)
    if args.quick:
        C = {k: (kind, p, {g: v[:1] for g, v in grid.items()}) for k, (kind, p, grid) in list(C.items())[:2]}
    log(f"design {args.design}: train {len(tr_idx)} ({y_tr.sum()} cancers), test {len(te_idx)} ({y_te.sum()}); "
        f"{len(C)} models x {len(folds)} outer folds; optional packages {HAS}")

    tasks = [(m, o) for m in C for o in range(len(folds))]
    res = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(fit_fold)(C[m][1], C[m][2], Xtr, y_tr, key, folds[o][2], SEED + 10 * folds[o][0] + folds[o][1])
        for m, o in tasks)

    rows, subs, pred, fold_rows = [], [], {"sample_id": te_idx, "ml_label": y_te}, []
    n_rep = max(f[0] for f in folds) + 1
    for m, (kind, _, _) in C.items():
        oof = np.zeros((n_rep, len(y_tr)))
        oof_q = np.zeros_like(oof)
        p_te, q_te = [], []
        for (mm, o), (est, bp) in zip(tasks, res):
            if mm != m:
                continue
            r, f, a, b = folds[o]
            oof[r, b] = est.predict_proba(Xtr[b])[:, 1]
            p_te.append(est.predict_proba(Xte)[:, 1])
            oof_q[r, b] = fold_vote(oof[r, b], oof[r, b], y_tr[b])
            q_te.append(fold_vote(p_te[-1], oof[r, b], y_tr[b]))
            fold_rows.append({"model": m, "repeat": r, "fold": f, "auc": roc_auc_score(y_tr[b], oof[r, b]),
                              "params": json.dumps(bp, sort_keys=True, default=str)})
        oof_m, p = oof.mean(0), np.mean(p_te, axis=0)
        row, sub, thr = summarise(m, kind, y_tr, oof_m, y_te, p, s_te, oof_q.mean(0), np.mean(q_te, axis=0))
        row["cv_auc_sd"] = float(np.std([roc_auc_score(y_tr, oof[r]) for r in range(n_rep)]))
        rows.append(row)
        subs += sub
        pred[m] = p
        log(f"   {m:28s} CV AUC {row['cv_auc']:.3f}  test AUC {row['test_auc']:.3f} "
            f"[{row['test_auc_lo']:.2f}-{row['test_auc_hi']:.2f}]  pAUC {row['test_pauc']:.3f}  "
            f"sens {row['test_sens']:.2f} @ spec {row['test_spec']:.2f}")
    # covariates-only reference: same folds, threshold vote and test
    cov = covariates(samples).loc[list(tr_idx) + list(te_idx)]
    Ctr, Cte = cov.loc[tr_idx].to_numpy(), cov.loc[te_idx].to_numpy()
    cpipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", logreg(0.0, C=1.0))])
    oof, oq, p_te, q_te = np.zeros(len(y_tr)), np.zeros(len(y_tr)), [], []
    for r, f, a, b in [x for x in folds if x[0] == 0]:
        m = clone(cpipe).fit(Ctr[a], y_tr[a])
        oof[b] = m.predict_proba(Ctr[b])[:, 1]
        oq[b] = fold_vote(oof[b], oof[b], y_tr[b])
        p_te.append(m.predict_proba(Cte)[:, 1])
        q_te.append(fold_vote(p_te[-1], oof[b], y_tr[b]))
    row, sub, _ = summarise("Covariates only (reference)", "reference", y_tr, oof, y_te, np.mean(p_te, 0), s_te,
                            oq, np.mean(q_te, 0))
    rows.append(row)
    subs += sub
    pred["Covariates only (reference)"] = np.mean(p_te, 0)
    log(f"   {'Covariates only (reference)':28s} CV AUC {row['cv_auc']:.3f}  test AUC {row['test_auc']:.3f}")
    results = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    results.to_csv(out / "results.tsv", sep="\t", index=False)
    pd.DataFrame(subs).to_csv(out / "subgroups.tsv", sep="\t", index=False)
    pd.DataFrame(pred).to_csv(out / "predictions.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "cv_folds.tsv", sep="\t", index=False)

    # explanations: best tree model and best linear model by CV AUC, refit on the whole training part
    cols = list(X.columns)
    explained = {}
    for kind in ("tree", "linear"):
        best = results[results["family"] == kind]
        if best.empty:
            continue
        m = best.iloc[0]["model"]
        _, pipe, grid = C[m]
        inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y_tr)), key))
        est = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=args.jobs).fit(Xtr, y_tr).best_estimator_
        tab = explain(m, kind, est, Xtr, y_tr, Xte, y_te, cols, out)
        ann = OUT / "08_annotation/region_genes.tsv"
        if ann.exists():
            tab = tab.join(pd.read_csv(ann, sep="\t", index_col="region_id")[["gene", "promoter"]], on="region")
        tab.to_csv(out / f"importance_{slug(m)}.tsv", sep="\t", index=False)
        explained[m] = tab.groupby("family")["importance"].sum() / tab["importance"].sum()
        log(f"   explained {m} ({tab['method'].iloc[0]}); top: " + ", ".join(tab["feature"].head(5)))
    if explained:
        fam = pd.DataFrame(explained).fillna(0)
        fam = fam.loc[[f for f in FAMILIES if f in fam.index]]
        ax = fam.T.plot.barh(stacked=True, figsize=(8, 2.2 + 0.5 * len(explained)), colormap="tab10", width=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of total importance on the test set")
        ax.legend([FAMILIES[f] for f in fam.index], fontsize=7.5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(out / "fig_family_importance.png", dpi=170)
        plt.close("all")
    # permutation test of the best model (its most frequent settings, 5-fold CV, shuffled labels)
    perm = {}
    if args.permutations:
        m = results[results["family"] != "reference"].iloc[0]["model"]
        fr = pd.DataFrame(fold_rows)
        bp = json.loads(fr[fr["model"] == m]["params"].value_counts().index[0])
        pipe = clone(C[m][1]).set_params(**bp)
        f5 = [(a, b) for r, f, a, b in folds if r == 0]

        def cv_auc(labels):
            p = np.zeros(len(labels))
            for a, b in f5:
                p[b] = clone(pipe).fit(Xtr[a], labels[a]).predict_proba(Xtr[b])[:, 1]
            return roc_auc_score(labels, p)

        real = cv_auc(y_tr)
        rng = np.random.default_rng(SEED)
        null = np.array(Parallel(n_jobs=args.jobs)(delayed(cv_auc)(rng.permutation(y_tr)) for _ in range(args.permutations)))
        perm = {"model": m, "params": bp, "real_auc": float(real), "null_mean": float(null.mean()),
                "null_max": float(null.max()), "p": float((1 + (null >= real).sum()) / (1 + len(null))), "n": len(null)}
        (out / "permutation.json").write_text(json.dumps(perm, indent=2, default=str))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(null, bins=30, color="#9CA3AF")
        ax.axvline(real, color="#E4572E", linewidth=2.5, label=f"real labels {real:.2f}")
        ax.set_xlabel(f"CV AUC with shuffled labels ({len(null)} permutations)")
        ax.set_ylabel("Count")
        ax.set_title(f"Permutation test: {m} (p = {perm['p']:.3f})", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "fig_permutation.png", dpi=170)
        plt.close(fig)
        log(f"   permutation test {m}: real {real:.3f}, null {null.mean():.3f} (max {null.max():.3f}), p = {perm['p']:.4f}")
    (out / "results.json").write_text(json.dumps({"design": args.design, "n_train": len(tr_idx), "n_test": len(te_idx),
                                                  "repeats": n_rep, "packages": HAS, "permutation": perm,
                                                  "explained": list(explained)}, indent=2, default=str))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
