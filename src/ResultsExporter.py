"""
ResultsExporter.py
───────────────────
Utility centralizzata per il salvataggio di grafici, metriche e tabelle
prodotti durante l'esecuzione della pipeline KBS Airbnb.

Struttura cartelle prodotta
───────────────────────────
results/
├── plots/
│   ├── clustering/          ← K-Means PCA, GMM Certainty
│   ├── regression/          ← Actual vs Predicted, Feature Importance
│   └── classification/      ← ROC Curve, Feature Importance CV
├── metrics/
│   ├── cv_results_availability.csv
│   ├── clustering_analysis_kmeans.csv
│   ├── clustering_analysis_gmm.csv
│   └── summary_metrics.csv  ← riepilogo unico di tutti i task
└── README.txt               ← legenda automatica dei file
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend non-interattivo — salva senza aprire finestre
import matplotlib.pyplot as plt
from datetime import datetime


# ── Directory di output ───────────────────────────────────────────────────────
RESULTS_ROOT   = "results"
PLOTS_ROOT     = os.path.join(RESULTS_ROOT, "plots")
METRICS_ROOT   = os.path.join(RESULTS_ROOT, "metrics")

PLOT_DIRS = {
    "clustering":     os.path.join(PLOTS_ROOT, "clustering"),
    "regression":     os.path.join(PLOTS_ROOT, "regression"),
    "classification": os.path.join(PLOTS_ROOT, "classification"),
}


def _ensure_dirs():
    for d in [METRICS_ROOT] + list(PLOT_DIRS.values()):
        os.makedirs(d, exist_ok=True)


def _savefig(fig: plt.Figure, category: str, filename: str, dpi: int = 150) -> str:
    """Salva una figura matplotlib e restituisce il path."""
    _ensure_dirs()
    path = os.path.join(PLOT_DIRS[category], filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Export] Grafico salvato: {path}")
    return path


def _save_csv(df: pd.DataFrame, filename: str) -> str:
    """Salva un DataFrame CSV nella cartella metrics."""
    _ensure_dirs()
    path = os.path.join(METRICS_ROOT, filename)
    df.to_csv(path, index=False)
    print(f"[Export] CSV salvato: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════

def save_kmeans_plot(X_pca: np.ndarray, labels: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=labels, cmap="viridis", s=40, alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    ax.set_title("Guest Preference Clusters — K-Means (PCA 2D)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    fig.tight_layout()
    return _savefig(fig, "clustering", "kmeans_pca_clusters.png")


def save_gmm_certainty_plot(X_pca: np.ndarray, certainty: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                    c=certainty, cmap="coolwarm", s=40, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="Certainty")
    ax.set_title("GMM Clustering — Certainty Levels (PCA 2D)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    fig.tight_layout()
    return _savefig(fig, "clustering", "gmm_certainty.png")


def save_clustering_analysis(df: pd.DataFrame, kind: str = "kmeans") -> str:
    """kind: 'kmeans' o 'gmm'"""
    fname = f"clustering_analysis_{kind}.csv"
    return _save_csv(df, fname)


# ═════════════════════════════════════════════════════════════════════════════
# REGRESSIONE
# ═════════════════════════════════════════════════════════════════════════════

def save_regression_scatter(y_test, y_pred,
                             model_name: str = "Model") -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, color="#378ADD")
    lim = (0, max(y_test.max(), y_pred.max()) * 1.05)
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfetto")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Actual Price vs Predicted Price — {model_name}")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fname = f"regression_scatter_{model_name.replace(' ', '_').lower()}.png"
    return _savefig(fig, "regression", fname)


def save_regression_metrics(metrics: dict, model_name: str) -> str:
    """
    metrics: dict con chiavi mse, r2, std_pred, std_actual
    """
    row = {"model": model_name, "timestamp": datetime.now().isoformat(), **metrics}
    summary_path = os.path.join(METRICS_ROOT, "summary_regression.csv")
    _ensure_dirs()
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(summary_path, index=False)
    print(f"[Export] Metriche regressione salvate: {summary_path}")
    return summary_path


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICAZIONE
# ═════════════════════════════════════════════════════════════════════════════

def save_roc_curve(fpr, tpr, auc_score: float, model_name: str = "Model") -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#1D9E75", lw=2,
            label=f"ROC curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fname = f"roc_curve_{model_name.replace(' ', '_').lower()}.png"
    return _savefig(fig, "classification", fname)


def save_feature_importance(importances: np.ndarray,
                             feature_names: list,
                             model_name: str = "Model",
                             top_n: int = 15) -> str:
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1D9E75" if n.startswith("kb_") else "#B4B2A9"
              for n in np.array(feature_names)[idx]]
    ax.barh(np.array(feature_names)[idx], importances[idx], color=colors)
    ax.set_title(f"Top-{top_n} Feature Importance — {model_name}\n"
                 "(verde = feature KB-derived)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fname = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
    return _savefig(fig, "classification", fname)


def save_cv_results(cv_df: pd.DataFrame,
                    model_name: str = "XGBoost") -> str:
    fname = f"cv_results_{model_name.replace(' ', '_').lower()}.csv"
    return _save_csv(cv_df, fname)


def save_cv_barplot(cv_df: pd.DataFrame,
                    model_name: str = "XGBoost") -> str:
    """Barplot con mean ± std per ogni metrica della CV."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    means   = cv_df[metrics].mean()
    stds    = cv_df[metrics].std()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color="#378ADD", alpha=0.85, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"10-Fold CV — Mean ± Std — {model_name}")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fname = f"cv_barplot_{model_name.replace(' ', '_').lower()}.png"
    return _savefig(fig, "classification", fname)


# ═════════════════════════════════════════════════════════════════════════════
# README automatico
# ═════════════════════════════════════════════════════════════════════════════

def write_readme():
    _ensure_dirs()
    content = f"""Airbnb KBS — Risultati Pipeline
Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}

results/
├── plots/
│   ├── clustering/
│   │   ├── kmeans_pca_clusters.png     ← K-Means: scatter PCA 2D con label cluster
│   │   └── gmm_certainty.png           ← GMM: scatter PCA 2D con livelli di certezza
│   ├── regression/
│   │   ├── regression_scatter_*.png    ← Actual vs Predicted per ogni modello
│   └── classification/
│       ├── roc_curve_*.png             ← Curva ROC con AUC
│       ├── feature_importance_*.png    ← Top-15 feature (verde = KB-derived)
│       └── cv_barplot_*.png            ← Mean ± Std per Accuracy/Precision/Recall/F1/AUC
└── metrics/
    ├── cv_results_*.csv                ← Risultati per fold + Mean + Std
    ├── clustering_analysis_kmeans.csv  ← Centroidi K-Means
    ├── clustering_analysis_gmm.csv     ← Centroidi GMM
    └── summary_regression.csv         ← MSE, R² per ogni modello di regressione
"""
    path = os.path.join(RESULTS_ROOT, "README.txt")
    with open(path, "w") as f:
        f.write(content)
    print(f"[Export] README scritto: {path}")
