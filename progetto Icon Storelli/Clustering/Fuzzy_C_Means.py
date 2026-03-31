"""
Fuzzy C-Means per factor loadings.

Uso:
    python Clustering/Fuzzy_C_Means.py --dataset Ontologia/Autism-Dataset.csv --out Clustering/Autism-Dataset-factors.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "Ontologia" / "Autism-Dataset.csv"
DEFAULT_FACTORS_OUT = Path(__file__).resolve().parent / "Autism-Dataset-factors.csv"

FEATURES = [
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score",
    "A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic",
    "screening_score","PDD_parent","Class/ASD"
]

# Feature da escludere sempre (es. identificativi)
EXCLUDE_FEATURES = {
    "id"
}


def _load_raw_dataset_for_fcm(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(dataset_path, sep=",")
    if "Class/ASD" not in dataset.columns:
        dataset = pd.read_csv(
            dataset_path,
            sep=",",
            names=FEATURES,
            dtype={name: object for name in FEATURES}
        )
        dataset = dataset[
            dataset["Class/ASD"].notna()
            & (dataset["Class/ASD"].astype(str).str.strip().str.lower() != "class/asd")
        ].reset_index(drop=True)
    else:
        cols = [c for c in FEATURES if c in dataset.columns]
        dataset = dataset[cols]

    y = dataset["Class/ASD"].astype(str)
    y_map = y.map({"1": 1, "0": 0})
    mask = y_map.notna()
    if not mask.all():
        dropped = (~mask).sum()
        print(f"[WARN] Rimosse {dropped} righe con target non valido")
        dataset = dataset.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    X_raw = dataset.drop(columns=["Class/ASD"], errors="ignore")
    if EXCLUDE_FEATURES:
        X_raw = X_raw.drop(columns=[c for c in X_raw.columns if c in EXCLUDE_FEATURES], errors="ignore")

    X_num = pd.DataFrame(index=X_raw.index)
    X_cat = pd.DataFrame(index=X_raw.index)
    for col in X_raw.columns:
        col_numeric = pd.to_numeric(X_raw[col], errors="coerce")
        if col_numeric.notna().sum() == X_raw[col].notna().sum():
            X_num[col] = col_numeric
        else:
            X_cat[col] = X_raw[col].astype(str)

    for col in X_num.columns:
        mode_val = X_num[col].mode()
        X_num[col] = X_num[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)
    X_num = X_num.astype(float)

    return X_num, X_cat, y


def _fuzzy_c_means(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n_points, n_dims = X.shape
    if n_clusters <= 1 or n_clusters >= n_points:
        raise ValueError("n_clusters must be between 2 and n_points-1")

    U = rng.random((n_clusters, n_points))
    U = U / U.sum(axis=0, keepdims=True)

    for _ in range(max_iter):
        U_prev = U.copy()
        um = U ** m
        centers = (um @ X) / um.sum(axis=1, keepdims=True)

        D = np.linalg.norm(X[None, :, :] - centers[:, None, :], axis=2)
        power = 2.0 / (m - 1.0)

        zero_mask = D == 0
        if np.any(zero_mask):
            U = np.zeros_like(D)
            zero_cols = zero_mask.any(axis=0)
            for j in np.where(zero_cols)[0]:
                zero_clusters = zero_mask[:, j]
                U[zero_clusters, j] = 1.0 / zero_clusters.sum()
            non_zero_cols = ~zero_cols
            if np.any(non_zero_cols):
                D_nz = D[:, non_zero_cols]
                inv = D_nz ** (-power)
                U[:, non_zero_cols] = inv / inv.sum(axis=0, keepdims=True)
        else:
            inv = D ** (-power)
            U = inv / inv.sum(axis=0, keepdims=True)

        if np.max(np.abs(U - U_prev)) < tol:
            break

    D = np.linalg.norm(X[None, :, :] - centers[:, None, :], axis=2)
    objective = np.sum((U ** m) * (D ** 2))
    return U, centers, float(objective)


def _choose_elbow(k_values: list[int], obj_values: list[float]) -> int:
    if len(k_values) == 1:
        return k_values[0]
    knee = None
    try:
        kl = KneeLocator(k_values, obj_values, curve="convex", direction="decreasing")
        knee = kl.knee
    except Exception:
        knee = None
    if knee is not None:
        return int(knee)

    if len(obj_values) >= 3:
        second_diff = np.diff(obj_values, n=2)
        idx = int(np.argmax(second_diff)) + 1
        return k_values[idx]
    return k_values[0]


def _plot_elbow(
    k_values: list[int],
    obj_values: list[float],
    best_k: int | None,
    out_path: Path,
    show: bool = False
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, obj_values, marker="o")
    plt.xlabel("Numero cluster (k)")
    plt.ylabel("Objective (FCM)")
    plt.title("Regola del gomito - Fuzzy C-Means")
    plt.grid(True, alpha=0.3)
    if best_k is not None and best_k in k_values:
        best_idx = k_values.index(best_k)
        plt.scatter([best_k], [obj_values[best_idx]], color="red", zorder=3, label=f"k={best_k}")
        plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def build_factor_dataset(
    dataset_path: Path,
    out_path: Path,
    seed: int = 42,
    c_min: int = 3,
    c_max: int | None = None,
    plot_path: Path | None = None,
    show_plot: bool = False
) -> tuple[Path, int]:
    X_df, X_cat, target_series = _load_raw_dataset_for_fcm(dataset_path)
    if X_df.shape[1] == 0:
        raise ValueError("Nessuna feature numerica disponibile per il Fuzzy C-Means.")
    X = X_df.to_numpy()
    feature_matrix = X.T
    n_features = feature_matrix.shape[0]

    if c_max is None:
        c_max = n_features // 2 + 1
    if c_max < c_min:
        raise ValueError("Numero di feature insufficiente per la ricerca dei cluster.")

    k_values = list(range(c_min, c_max + 1))
    obj_values: list[float] = []
    memberships: dict[int, np.ndarray] = {}

    print(f"[INFO] Fuzzy C-Means su {n_features} feature: cluster da {c_min} a {c_max}")
    for k in k_values:
        U, _, obj = _fuzzy_c_means(feature_matrix, n_clusters=k, seed=seed)
        obj_values.append(obj)
        memberships[k] = U
        print(f"[INFO] k={k}: objective={obj:.4f}")

    best_k = _choose_elbow(k_values, obj_values)
    print(f"[INFO] Cluster scelto (gomito): k={best_k}")

    if plot_path is None:
        plot_path = out_path.parent / "fcm_elbow.png"
    _plot_elbow(k_values, obj_values, best_k, plot_path, show=show_plot)

    U_best = memberships[best_k]
    denom = U_best.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    weights = U_best / denom
    factors = X @ weights.T

    factors_df = pd.DataFrame(
        factors,
        columns=[f"Factor_{i+1}" for i in range(best_k)]
    )
    if not X_cat.empty:
        factors_df = pd.concat([factors_df, X_cat.reset_index(drop=True)], axis=1)
    factors_df["Class/ASD"] = target_series.values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    factors_df.to_csv(out_path, index=False)
    print(f"[INFO] Dataset fattori salvato in: {out_path}")
    return out_path, best_k


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuzzy C-Means per factor loadings")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET), help="Path CSV input")
    parser.add_argument("--out", type=str, default=str(DEFAULT_FACTORS_OUT), help="Path CSV output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-min", type=int, default=3, help="Numero minimo di cluster")
    parser.add_argument("--c-max", type=int, default=None, help="Numero massimo di cluster")
    parser.add_argument("--elbow-plot", type=str, default=None, help="Path PNG per grafico gomito")
    parser.add_argument("--show-plot", action="store_true", help="Mostra il grafico del gomito")
    args = parser.parse_args()

    build_factor_dataset(
        dataset_path=Path(args.dataset),
        out_path=Path(args.out),
        seed=args.seed,
        c_min=args.c_min,
        c_max=args.c_max,
        plot_path=Path(args.elbow_plot) if args.elbow_plot else None,
        show_plot=args.show_plot
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
