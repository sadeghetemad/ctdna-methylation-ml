"""In-fold feature selection shared by the modelling scripts (the steps of 10_feature_selection.py)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FoldSelector(BaseEstimator, TransformerMixin):
    """Quality filter, single-feature AUC ranking and greedy correlation pruning; keeps the top k."""

    def __init__(self, k=200, corr=0.9, max_missing=0.2, screen=3000):
        self.k, self.corr, self.max_missing, self.screen = k, corr, max_missing, screen

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        ok = (np.isnan(X).mean(0) <= self.max_missing) & (np.nanstd(X, 0) > 1e-9)
        cols = np.flatnonzero(ok)
        R = pd.DataFrame(X[:, cols]).rank().to_numpy()
        seen = ~np.isnan(R)
        R = np.where(seen, R, np.nanmean(R, 0))
        pos = y == 1
        n1, n0 = seen[pos].sum(0), seen[~pos].sum(0)
        auc = (R[pos].sum(0) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)
        order = np.argsort(-np.abs(auc - 0.5))[: self.screen]
        Z = R[:, order]
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1) / np.sqrt(len(Z))
        kept = []
        for j in range(Z.shape[1]):
            if kept and np.max(np.abs(Z[:, kept].T @ Z[:, j])) >= self.corr:
                continue
            kept.append(j)
            if len(kept) == self.k:
                break
        self.support_ = cols[order[kept]]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]
