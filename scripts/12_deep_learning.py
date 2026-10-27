"""
12_deep_learning.py: deep-learning models on the same designs, folds and evaluation as 11.

    MLP                  two hidden layers on the in-fold selected features (FoldSelector, k = 200)
    FT-Transformer       every selected feature becomes a token (per-feature linear embedding), a CLS
                         token and a Transformer encoder (Gorishniy et al. 2021), k = 64
    Region Transformer   every panel region becomes a token whose channels are its 8 region-level feature
                         families (+ missing-value masks), plus a learned region embedding; the encoder
                         attends across the top 128 regions of the training fold. The CLS attention gives
                         a region-importance ranking.

Small data (a few hundred samples): AdamW with weight decay, dropout, token dropout for the Region
Transformer, early stopping on a stratified 20% validation split of each training fold, and an ensemble
of seeds. Test scores are the mean over the outer-fold models, as in 11.

Outputs (results/modelling/12_deep/<design>/): results.tsv, subgroups.tsv, predictions.tsv,
region_attention.tsv, fig_curves.png, fig_region_attention.png, results.json
"""

import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ml_common import DESIGNS, OUT, SEED, fold_vote, designs, family, load, log, out_dir, outer_folds, region, summarise  # noqa: E402
from ml_selection import FoldSelector  # noqa: E402

warnings.filterwarnings("ignore")
REGION_FAMILIES = ["methylation", "meth", "frac_meth", "frac_meth_short", "mhl", "pdr", "entropy", "frac_short"]


# ============================================================ models

class MLP(nn.Module):
    def __init__(self, d_in, hidden=(128, 32), dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Block(nn.Module):
    """Pre-norm Transformer block that also returns its attention weights."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n1(x)
        a, w = self.att(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x))), w


class FTTransformer(nn.Module):
    """Feature tokenizer: token_j = x_j * W_j + b_j; CLS token; encoder; head on CLS."""

    def __init__(self, n_features, d=32, heads=4, layers=2, ff=64, dropout=0.2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_features, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_features, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, return_attention=False):
        t = x.unsqueeze(-1) * self.w + self.b
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        for blk in self.blocks:
            t, w = blk(t)
        out = self.head(t[:, 0]).squeeze(-1)
        return (out, w[:, 0, 1:]) if return_attention else out


class RegionTransformer(nn.Module):
    """Tokens are regions: channels = feature families (+ masks), plus a learned region embedding."""

    def __init__(self, n_regions, n_channels, n_sample, d=32, heads=4, layers=2, ff=64, dropout=0.2, token_drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_channels, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Embedding(n_regions, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d + n_sample), nn.Linear(d + n_sample, 16), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(16, 1))
        self.token_drop = token_drop

    def forward(self, x, s, return_attention=False):
        t = self.proj(x) + self.pos.weight.unsqueeze(0)
        if self.training and self.token_drop > 0:
            keep = (torch.rand(t.shape[:2], device=t.device) > self.token_drop).float().unsqueeze(-1)
            t = t * keep
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        for blk in self.blocks:
            t, w = blk(t)
        out = self.head(torch.cat([t[:, 0], s], 1)).squeeze(-1)
        return (out, w[:, 0, 1:]) if return_attention else out


# ============================================================ training

def train(model, inputs, y, val_inputs, y_val, epochs, lr=1e-3, wd=1e-2, batch=32, patience=30, seed=SEED):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    yt = torch.tensor(y, dtype=torch.float32)
    best, best_state, wait, curve = -1, None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            b = order[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(*[t[b] for t in inputs]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(*val_inputs)).numpy()
        auc = roc_auc_score(y_val, pv) if 0 < y_val.sum() < len(y_val) else 0.5
        curve.append(auc)
        if auc > best + 1e-4:
            best, wait = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, curve


def tabular_inputs(Xtr, ytr, Xother, k):
    sel = FoldSelector(k=k).fit(Xtr, ytr)
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(sel.transform(Xtr))
    sc = StandardScaler().fit(imp.transform(sel.transform(Xtr)))
    f = lambda A: np.clip(sc.transform(imp.transform(sel.transform(A))), -6, 6).astype(np.float32)  # noqa: E731
    return f(Xtr), [f(A) for A in Xother], sel.support_


def region_inputs(T, M, S, tr, others, n_regions):
    """Standardise per (region, family) on the training rows; keep the top regions by single-feature AUC."""
    mu = np.nanmean(T[tr], 0)
    sd = np.nanstd(T[tr], 0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = np.nan_to_num((T - mu) / sd)
    y = M
    R = np.zeros(T.shape[1])
    for c in range(T.shape[2]):
        v = pd.DataFrame(T[tr][:, :, c]).rank().to_numpy()
        seen = ~np.isnan(v)
        v = np.where(seen, v, np.nanmean(v, 0))
        pos = y[tr] == 1
        n1, n0 = seen[pos].sum(0), seen[~pos].sum(0)
        auc = (v[pos].sum(0) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)
        R = np.maximum(R, np.nan_to_num(np.abs(auc - 0.5)))
    top = np.argsort(-R)[:n_regions]
    mask = (~np.isnan(T[:, top])).astype(np.float32)
    X = np.concatenate([np.clip(Z[:, top], -6, 6), mask], -1).astype(np.float32)
    s_mu, s_sd = np.nanmean(S[tr], 0), np.nanstd(S[tr], 0) + 1e-9
    Sz = np.nan_to_num((S - s_mu) / s_sd).astype(np.float32)
    return X, Sz, top


def run_fold(kind, data, tr, va, te_rows, cfg, seed):
    torch.set_num_threads(cfg["threads"])
    y = data["y"]
    a, v = train_test_split(tr, test_size=0.2, random_state=seed, stratify=y[tr])
    scores_va, scores_te, att = [], [], None
    for k in range(cfg["seeds"]):
        sd = seed + 101 * k
        torch.manual_seed(sd)
        if kind in ("MLP", "FT-Transformer"):
            kk = 200 if kind == "MLP" else 64
            Xa, (Xv, Xva, Xte), support = tabular_inputs(data["X"][a], y[a], [data["X"][v], data["X"][va],
                                                                              data["X"][te_rows]], kk)
            model = MLP(Xa.shape[1]) if kind == "MLP" else FTTransformer(Xa.shape[1])
            to = lambda A: [torch.tensor(A)]  # noqa: E731
        else:
            X, Sz, top = region_inputs(data["T"], y, data["S"], a, None, cfg["regions"])
            model = RegionTransformer(len(top), X.shape[2], Sz.shape[1])
            Xa, Xv, Xva, Xte = (X[a], Sz[a]), (X[v], Sz[v]), (X[va], Sz[va]), (X[te_rows], Sz[te_rows])
            to = lambda A: [torch.tensor(t) for t in A]  # noqa: E731
        model, curve = train(model, to(Xa), y[a], to(Xv), y[v], cfg["epochs"],
                             lr=1e-3 if kind == "MLP" else 5e-4, seed=sd)
        model.eval()
        with torch.no_grad():
            scores_va.append(torch.sigmoid(model(*to(Xva))).numpy())
            if kind == "Region Transformer":
                o, w = model(*to(Xte), return_attention=True)
                scores_te.append(torch.sigmoid(o).numpy())
                a_reg = np.zeros(data["T"].shape[1])
                a_reg[top] = w.mean(0).numpy()
                att = a_reg if att is None else att + a_reg
            else:
                scores_te.append(torch.sigmoid(model(*to(Xte))).numpy())
    return np.mean(scores_va, 0), np.mean(scores_te, 0), curve, att


def main():
    ap = argparse.ArgumentParser(description="Deep-learning models: MLP, FT-Transformer, Region Transformer.")
    ap.add_argument("--design", default="balanced", choices=list(DESIGNS))
    ap.add_argument("--seeds", type=int, default=5, help="Ensemble members per fold.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--regions", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="Smoke test: 1 seed, 30 epochs, 2 folds.")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs = 1, 30
    out = out_dir(f"12_deep/{args.design}")
    samples, X, regions = load()
    tr_idx, te_idx = designs(samples)[args.design]
    order = list(tr_idx) + list(te_idx)
    samples, X = samples.loc[order], X.loc[order]
    n_tr = len(tr_idx)
    y = samples["ml_label"].to_numpy()
    reg_ids = list(regions.index[regions["kept"]]) if "kept" in regions else list(regions.index)
    T = np.full((len(order), len(reg_ids), len(REGION_FAMILIES)), np.nan, dtype=np.float32)
    pos = {r: i for i, r in enumerate(reg_ids)}
    for c in X.columns:
        f, r = family(c), region(c)
        if f in REGION_FAMILIES and r in pos:
            T[:, pos[r], REGION_FAMILIES.index(f)] = X[c].to_numpy()
    S = X[[c for c in X.columns if family(c) == "sample"]].to_numpy(dtype=np.float32)
    data = {"X": X.to_numpy(dtype=np.float32), "T": T, "S": S, "y": y}
    folds = outer_folds(samples.iloc[:n_tr], repeats=1)
    if args.quick:
        folds = folds[:2]
    te_rows = np.arange(n_tr, len(order))
    cpu = os.cpu_count() or 4
    cfg = {"seeds": args.seeds, "epochs": args.epochs, "regions": args.regions,
           "threads": max(1, cpu // min(args.jobs, len(folds)))}
    kinds = ["MLP", "FT-Transformer", "Region Transformer"]
    log(f"design {args.design}: train {n_tr}, test {len(te_rows)}; tensor {T.shape}; {len(folds)} folds; {cfg}")
    tasks = [(k, f) for k in kinds for f in range(len(folds))]
    res = Parallel(n_jobs=min(args.jobs, len(tasks)))(
        delayed(run_fold)(k, data, folds[f][2], folds[f][3], te_rows, cfg, SEED + f) for k, f in tasks)
    rows, subs, pred, curves = [], [], {"sample_id": samples.index[te_rows], "ml_label": y[te_rows]}, {}
    att_total = np.zeros(len(reg_ids))
    for k in kinds:
        oof = np.full(n_tr, np.nan)
        oof_q = np.full(n_tr, np.nan)
        p_te, q_te = [], []
        for (kk, f), (pv, pt, curve, att) in zip(tasks, res):
            if kk != k:
                continue
            yv = y[:n_tr][folds[f][3]]
            oof[folds[f][3]] = pv
            oof_q[folds[f][3]] = fold_vote(pv, pv, yv)
            p_te.append(pt)
            q_te.append(fold_vote(pt, pv, yv))
            curves.setdefault(k, curve)
            if att is not None:
                att_total += att
        ok = ~np.isnan(oof)
        row, sub, _ = summarise(k, "deep", y[:n_tr][ok], oof[ok], y[te_rows], np.mean(p_te, 0), samples.iloc[n_tr:],
                                oof_q[ok], np.mean(q_te, 0))
        rows.append(row)
        subs += sub
        pred[k] = np.mean(p_te, 0)
        log(f"   {k:20s} CV AUC {row['cv_auc']:.3f}  test AUC {row['test_auc']:.3f} "
            f"[{row['test_auc_lo']:.2f}-{row['test_auc_hi']:.2f}]  pAUC {row['test_pauc']:.3f}  "
            f"sens {row['test_sens']:.2f} @ spec {row['test_spec']:.2f}")
    pd.DataFrame(rows).to_csv(out / "results.tsv", sep="\t", index=False)
    pd.DataFrame(subs).to_csv(out / "subgroups.tsv", sep="\t", index=False)
    pd.DataFrame(pred).to_csv(out / "predictions.tsv", sep="\t", index=False)
    att = pd.DataFrame({"region": reg_ids, "attention": att_total / max(att_total.sum(), 1e-12)})
    ann = OUT / "08_annotation/region_genes.tsv"
    if ann.exists():
        att = att.join(pd.read_csv(ann, sep="\t", index_col="region_id")[["gene", "promoter"]], on="region")
    att = att.sort_values("attention", ascending=False)
    att.to_csv(out / "region_attention.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, c in curves.items():
        ax.plot(range(1, len(c) + 1), c, label=k)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC (first fold)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_curves.png", dpi=170)
    plt.close(fig)
    top = att.head(20)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(range(len(top))[::-1], top["attention"], color="#00A19A")
    ax.set_yticks(range(len(top))[::-1], [f"{r} ({g})" if isinstance(g, str) else r
                                          for r, g in zip(top["region"], top.get("gene", [None] * len(top)))],
                  fontsize=8)
    ax.set_xlabel("Share of CLS attention (Region Transformer, test set)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_region_attention.png", dpi=170)
    plt.close(fig)
    (out / "results.json").write_text(json.dumps({"design": args.design, "cfg": cfg, "folds": len(folds),
                                                  "torch": torch.__version__}, indent=2))
    log(f"-> {out}")


if __name__ == "__main__":
    main()
