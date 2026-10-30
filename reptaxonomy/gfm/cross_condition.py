"""Cross-condition representation analysis (splits / countries / seasons / ecoregions).

Each condition is a rep_taxonomy dataset whose embeddings live at
``embed_dir/<condition>/<model>/test/embd_<stem>.npy`` (C x h x w). The driver
also writes, per condition,

* ``<work_dir>/manifests/<condition>.json``  stems, sample_keys, pair_keys
* ``<work_dir>/labels/<condition>.npz``      nearest-downsampled label grids

For every model and every Comparison(a, b) this module computes

Unpaired (distribution level, pooled per-chip embeddings, PCA-reduced jointly)
  mmd2_rbf, mmd_pvalue     kernel two-sample test (median-heuristic RBF, permutations)
  c2st_acc                 classifier two-sample test: CV accuracy of a logistic
                           domain classifier (0.5 = indistinguishable)
  frechet                  Frechet distance between Gaussian fits
  centroid_shift           mean over shared classes of ||mu_a,c - mu_b,c|| / within-class sd
  class_geometry_rho       Spearman rho between the class-centroid distance matrices
  knn_acc_in / knn_acc_transfer / knn_transfer_drop
                           token-level kNN probe trained on A, tested on held-out A vs on B

Paired (only when the two conditions share pair_keys, e.g. FTW window A vs B)
  paired_cka_pooled, paired_cka_tokens   linear CKA on matched chips / matched grid cells
  paired_mutual_knn@k                    neighbourhood overlap of matched chips
  paired_retrieval_top1                  fraction of chips whose nearest cross-condition
                                         neighbour is their own pair

Across models
  model_cka/<condition>.csv     model x model linear CKA on pooled chips
  taxonomy_stability.csv        Spearman rho between the model-CKA matrices of a and b
                                (does the inter-model taxonomy survive the shift?)

Outputs: ``<out_dir>/cross_condition/<dataset>/`` (pairwise_metrics.csv,
condition_summary.csv, model_cka/, taxonomy_stability.csv, heatmaps/).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from reptaxonomy.intercompare.reference_similarity import linear_cka

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- io
def _embed_path(embed_dir: str, cond: str, model: str, stem: str, split: str = "test") -> str:
    return os.path.join(embed_dir, cond, model, split, f"embd_{stem}.npy")


def load_manifest(work_dir: str, cond: str) -> Dict[str, Any]:
    return json.loads(Path(work_dir, "manifests", f"{cond}.json").read_text())


def load_labels(work_dir: str, cond: str) -> Optional[Dict[str, np.ndarray]]:
    p = Path(work_dir, "labels", f"{cond}.npz")
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def _resize_nearest(lab: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = lab.shape
    ys = (np.arange(h) * H / h + H / (2 * h)).astype(int).clip(0, H - 1)
    xs = (np.arange(w) * W / w + W / (2 * w)).astype(int).clip(0, W - 1)
    return lab[np.ix_(ys, xs)]


class CondEmbeddings:
    """Pooled + token embeddings of one (condition, model)."""

    def __init__(self, embed_dir: str, work_dir: str, cond: str, model: str,
                 max_tokens_per_chip: int = 64, ignore_index: int = 255, seed: int = 0) -> None:
        man = load_manifest(work_dir, cond)
        labels = load_labels(work_dir, cond) or {}
        rng = np.random.default_rng(seed)
        pooled, tok, tok_lab, tok_pos, keys, pkeys = [], [], [], [], [], []
        for stem, pk in zip(man["stems"], man["pair_keys"]):
            p = _embed_path(embed_dir, cond, model, stem)
            if not os.path.exists(p):
                continue
            e = np.load(p).astype(np.float32)  # C, h, w
            C, h, w = e.shape
            pooled.append(e.reshape(C, -1).mean(1))
            keys.append(stem); pkeys.append(pk)
            flat = e.reshape(C, -1).T
            lab = labels.get(stem)
            lab = _resize_nearest(lab, h, w).ravel() if lab is not None else np.full(h * w, ignore_index)
            sel = rng.choice(h * w, size=min(max_tokens_per_chip, h * w), replace=False)
            sel.sort()
            tok.append(flat[sel]); tok_lab.append(lab[sel]); tok_pos.append(np.stack([np.full(len(sel), len(keys) - 1), sel], 1))
        self.cond, self.model = cond, model
        self.keys, self.pair_keys = keys, pkeys
        self.pooled = np.stack(pooled) if pooled else np.zeros((0, 0), np.float32)
        self.tokens = np.concatenate(tok) if tok else np.zeros((0, 0), np.float32)
        self.token_labels = np.concatenate(tok_lab) if tok_lab else np.zeros(0, int)
        self.token_pos = np.concatenate(tok_pos) if tok_pos else np.zeros((0, 2), int)
        self.ignore_index = ignore_index
        self._grid_cache: Dict[str, np.ndarray] = {}
        self._embed_dir = embed_dir

    def __len__(self) -> int:
        return len(self.keys)

    def grid(self, stem: str) -> np.ndarray:
        if stem not in self._grid_cache:
            self._grid_cache[stem] = np.load(_embed_path(self._embed_dir, self.cond, self.model, stem)).astype(np.float32)
        return self._grid_cache[stem]


# ----------------------------------------------------------------------------- metrics
def _joint_pca(X: np.ndarray, Y: np.ndarray, k: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    Z = np.concatenate([X, Y]).astype(np.float64)
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Z = (Z - mu) / sd
    k = min(k, Z.shape[0] - 1, Z.shape[1])
    if k < Z.shape[1]:
        U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        Z = Z @ Vt[:k].T
    return Z[: len(X)], Z[len(X):]


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0)


def mmd2_rbf(X: np.ndarray, Y: np.ndarray, n_perm: int = 200, seed: int = 0) -> Tuple[float, float]:
    Z = np.concatenate([X, Y])
    D = _sqdist(Z, Z)
    med = np.median(D[np.triu_indices_from(D, 1)]) if len(Z) > 2 else 1.0
    K = np.exp(-D / max(med, 1e-12))
    n = len(X)

    def stat(idx: np.ndarray) -> float:
        a, b = idx[:n], idx[n:]
        kxx = K[np.ix_(a, a)]; kyy = K[np.ix_(b, b)]; kxy = K[np.ix_(a, b)]
        m, l = len(a), len(b)
        return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (l * (l - 1))
                     - 2 * kxy.mean())

    base = np.arange(len(Z))
    obs = stat(base)
    rng = np.random.default_rng(seed)
    null = [stat(rng.permutation(base)) for _ in range(n_perm)]
    p = (1 + sum(v >= obs for v in null)) / (1 + n_perm)
    return obs, float(p)


def c2st(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    n = min(len(X), len(Y))
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), n, replace=False)]
    Ys = Y[rng.choice(len(Y), n, replace=False)]
    Z = np.concatenate([Xs, Ys]); y = np.r_[np.zeros(n), np.ones(n)]
    folds = min(5, n)
    if folds < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    return float(cross_val_score(clf, Z, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed)).mean())


def frechet(X: np.ndarray, Y: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    m1, m2 = X.mean(0), Y.mean(0)
    s1, s2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    cs = sqrtm(s1 @ s2)
    cs = cs.real if np.iscomplexobj(cs) else cs
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def class_metrics(A: CondEmbeddings, Bm: CondEmbeddings, min_tokens: int = 20) -> Dict[str, float]:
    ign = A.ignore_index
    ta, la = A.tokens, A.token_labels
    tb, lb = Bm.tokens, Bm.token_labels
    classes = sorted(set(np.unique(la)) & set(np.unique(lb)) - {ign})
    cls = [c for c in classes if (la == c).sum() >= min_tokens and (lb == c).sum() >= min_tokens]
    out: Dict[str, float] = {"n_shared_classes": float(len(cls))}
    if not cls:
        return out
    shifts, mu_a, mu_b = [], [], []
    for c in cls:
        xa, xb = ta[la == c], tb[lb == c]
        sd = np.sqrt(0.5 * (xa.var(0).sum() + xb.var(0).sum())) + 1e-8
        shifts.append(np.linalg.norm(xa.mean(0) - xb.mean(0)) / sd)
        mu_a.append(xa.mean(0)); mu_b.append(xb.mean(0))
    out["centroid_shift"] = float(np.mean(shifts))
    if len(cls) >= 3:
        from scipy.stats import spearmanr
        da = _sqdist(np.stack(mu_a), np.stack(mu_a)); db = _sqdist(np.stack(mu_b), np.stack(mu_b))
        iu = np.triu_indices(len(cls), 1)
        out["class_geometry_rho"] = float(spearmanr(da[iu], db[iu]).statistic)
    return out


def knn_transfer(A: CondEmbeddings, Bm: CondEmbeddings, k: int = 10, max_train: int = 20000,
                 seed: int = 0) -> Dict[str, float]:
    from sklearn.neighbors import KNeighborsClassifier
    ign = A.ignore_index
    ma, mb = A.token_labels != ign, Bm.token_labels != ign
    if ma.sum() < 2 * k or mb.sum() < k:
        return {}
    Xa, ya = A.tokens[ma], A.token_labels[ma]
    chips = A.token_pos[ma, 0]
    rng = np.random.default_rng(seed)
    uniq = np.unique(chips)
    test_chips = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False).tolist())
    te = np.array([c in test_chips for c in chips])
    tr_idx = np.where(~te)[0]
    if len(tr_idx) > max_train:
        tr_idx = rng.choice(tr_idx, max_train, replace=False)
    mu, sd = Xa[tr_idx].mean(0), Xa[tr_idx].std(0) + 1e-8
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit((Xa[tr_idx] - mu) / sd, ya[tr_idx])
    acc_in = float((clf.predict((Xa[te] - mu) / sd) == ya[te]).mean()) if te.any() else float("nan")
    Xb, yb = Bm.tokens[mb], Bm.token_labels[mb]
    acc_tr = float((clf.predict((Xb - mu) / sd) == yb).mean())
    return {"knn_acc_in": acc_in, "knn_acc_transfer": acc_tr, "knn_transfer_drop": acc_in - acc_tr}


def paired_metrics(A: CondEmbeddings, Bm: CondEmbeddings, k: int = 10, max_cells: int = 20000,
                   seed: int = 0) -> Dict[str, float]:
    ia = {p: i for i, p in enumerate(A.pair_keys)}
    ib = {p: i for i, p in enumerate(Bm.pair_keys)}
    shared = sorted(set(ia) & set(ib))
    out: Dict[str, float] = {"n_pairs": float(len(shared))}
    if len(shared) < 3:
        return out
    Pa = A.pooled[[ia[p] for p in shared]]; Pb = Bm.pooled[[ib[p] for p in shared]]
    out["paired_cka_pooled"] = linear_cka(Pa, Pb)
    na = Pa / (np.linalg.norm(Pa, axis=1, keepdims=True) + 1e-8)
    nb = Pb / (np.linalg.norm(Pb, axis=1, keepdims=True) + 1e-8)
    S = na @ nb.T
    out["paired_retrieval_top1"] = float((S.argmax(1) == np.arange(len(shared))).mean())
    kk = min(k, len(shared) - 1)
    Sa, Sb = na @ na.T, nb @ nb.T
    np.fill_diagonal(Sa, -np.inf); np.fill_diagonal(Sb, -np.inf)
    Na = np.argsort(-Sa, 1)[:, :kk]; Nb = np.argsort(-Sb, 1)[:, :kk]
    out[f"paired_mutual_knn@{kk}"] = float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(Na, Nb)]))
    rng = np.random.default_rng(seed)
    cells_a, cells_b = [], []
    per = max(1, max_cells // len(shared))
    for p in shared:
        ga, gb = A.grid(A.keys[ia[p]]), Bm.grid(Bm.keys[ib[p]])
        if ga.shape[1:] != gb.shape[1:]:
            continue
        fa, fb = ga.reshape(ga.shape[0], -1).T, gb.reshape(gb.shape[0], -1).T
        sel = rng.choice(len(fa), size=min(per, len(fa)), replace=False)
        cells_a.append(fa[sel]); cells_b.append(fb[sel])
    if cells_a:
        out["paired_cka_tokens"] = linear_cka(np.concatenate(cells_a), np.concatenate(cells_b))
    return out


def compare_conditions(A: CondEmbeddings, Bm: CondEmbeddings, paired: bool, n_perm: int = 200,
                       pca_dim: int = 64, seed: int = 0) -> Dict[str, float]:
    out: Dict[str, float] = {"n_a": float(len(A)), "n_b": float(len(Bm))}
    if len(A) < 3 or len(Bm) < 3:
        return out
    Xa, Xb = _joint_pca(A.pooled, Bm.pooled, pca_dim)
    out["mmd2_rbf"], out["mmd_pvalue"] = mmd2_rbf(Xa, Xb, n_perm=n_perm, seed=seed)
    out["c2st_acc"] = c2st(Xa, Xb, seed=seed)
    try:
        out["frechet"] = frechet(Xa, Xb)
    except Exception as exc:  # singular covariance on tiny sets
        logger.debug("frechet failed: %s", exc)
    out.update(class_metrics(A, Bm))
    out.update(knn_transfer(A, Bm, seed=seed))
    if paired:
        out.update(paired_metrics(A, Bm, seed=seed))
    return out


def model_cka_matrix(embs: Dict[str, CondEmbeddings]) -> pd.DataFrame:
    models = [m for m, e in embs.items() if len(e)]
    if not models:
        return pd.DataFrame()
    common = set(embs[models[0]].keys)
    for m in models[1:]:
        common &= set(embs[m].keys)
    common = sorted(common)
    M = np.eye(len(models))
    if len(common) >= 3:
        mats = {m: embs[m].pooled[[embs[m].keys.index(s) for s in common]] for m in models}
        for i, a in enumerate(models):
            for j in range(i + 1, len(models)):
                M[i, j] = M[j, i] = linear_cka(mats[a], mats[models[j]])
    else:
        M[:] = np.nan
    return pd.DataFrame(M, index=models, columns=models)


def _heatmaps(df: pd.DataFrame, out: Path, formats: Sequence[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out.mkdir(parents=True, exist_ok=True)
    pw = df[df["kind"] == "pairwise"]
    for (setup, model, metric), g in pw.groupby(["setup", "model", "metric"]):
        if metric not in ("mmd2_rbf", "c2st_acc", "centroid_shift", "knn_transfer_drop"):
            continue
        names = sorted(set(g["a"]) | set(g["b"]))
        M = pd.DataFrame(np.nan, index=names, columns=names)
        for _, r in g.iterrows():
            M.loc[r["a"], r["b"]] = M.loc[r["b"], r["a"]] = r["value"]
        fig, ax = plt.subplots(figsize=(1 + 0.45 * len(names), 1 + 0.45 * len(names)))
        im = ax.imshow(M.values, cmap="viridis")
        short = [n.split("__")[-2] if n.count("__") >= 2 else n for n in names]
        ax.set_xticks(range(len(names)), short, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), short, fontsize=7)
        ax.set_title(f"{metric} | {model}", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        stem = f"{setup.replace(':', '_')}__{model}__{metric}".replace("/", "_")
        for fmt in formats:
            fig.savefig(out / f"{stem}.{fmt}", dpi=200)
        plt.close(fig)


def run_cross_condition(embed_dir: str, work_dir: str, out_dir: str, dataset: str,
                        comparisons: Sequence[Dict[str, Any]], models: Sequence[str],
                        conditions: Sequence[str], n_perm: int = 200, pca_dim: int = 64,
                        max_tokens_per_chip: int = 64, seed: int = 0,
                        figure_formats: Sequence[str] = ("png",)) -> pd.DataFrame:
    out = Path(out_dir, "cross_condition", dataset)
    (out / "model_cka").mkdir(parents=True, exist_ok=True)
    cache: Dict[Tuple[str, str], CondEmbeddings] = {}

    def get(cond: str, model: str) -> CondEmbeddings:
        if (cond, model) not in cache:
            cache[(cond, model)] = CondEmbeddings(embed_dir, work_dir, cond, model, max_tokens_per_chip, seed=seed)
        return cache[(cond, model)]

    rows = []
    for comp in comparisons:
        for m in models:
            try:
                res = compare_conditions(get(comp["a"], m), get(comp["b"], m), bool(comp.get("paired")),
                                         n_perm=n_perm, pca_dim=pca_dim, seed=seed)
            except Exception as exc:
                logger.warning("cross-condition %s vs %s (%s) failed: %s", comp["a"], comp["b"], m, exc)
                continue
            for k, v in res.items():
                rows.append({"setup": comp["setup"], "kind": comp["kind"], "a": comp["a"], "b": comp["b"],
                             "model": m, "metric": k, "value": v})
    df = pd.DataFrame(rows)
    df.to_csv(out / "pairwise_metrics.csv", index=False)

    summ, mats = [], {}
    for cond in conditions:
        embs = {m: get(cond, m) for m in models}
        M = model_cka_matrix(embs)
        if not M.empty:
            M.to_csv(out / "model_cka" / f"{cond}.csv")
            mats[cond] = M
        for m, e in embs.items():
            summ.append({"condition": cond, "model": m, "n_chips": len(e),
                         "embed_dim": int(e.pooled.shape[1]) if len(e) else 0})
    pd.DataFrame(summ).to_csv(out / "condition_summary.csv", index=False)

    stab = []
    from scipy.stats import spearmanr
    for comp in comparisons:
        A, Bm = mats.get(comp["a"]), mats.get(comp["b"])
        if A is None or Bm is None:
            continue
        common = [m for m in A.index if m in Bm.index]
        if len(common) < 3:
            continue
        iu = np.triu_indices(len(common), 1)
        a = A.loc[common, common].values[iu]; b = Bm.loc[common, common].values[iu]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 3:
            stab.append({"setup": comp["setup"], "kind": comp["kind"], "a": comp["a"], "b": comp["b"],
                         "n_models": len(common), "spearman_rho": float(spearmanr(a[ok], b[ok]).statistic)})
    pd.DataFrame(stab).to_csv(out / "taxonomy_stability.csv", index=False)
    if not df.empty:
        _heatmaps(df, out / "heatmaps", figure_formats)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="plan.json written by reptaxonomy.pipelines.gfm_experiment")
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    for ds, p in plan["datasets"].items():
        run_cross_condition(plan["embed_dir"], plan["work_dir"], plan["out_dir"], ds, p["comparisons"],
                            plan["models"], [c["name"] for c in p["conditions"] if c["analyze"]],
                            **plan.get("cross_condition", {}))


if __name__ == "__main__":
    main()
