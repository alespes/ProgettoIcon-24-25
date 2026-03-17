import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.ResultsExporter import (
    save_gmm_certainty_plot, save_clustering_analysis
)


class EM_Implementation:

    N_COMPONENTS = 3

    def __init__(self, data: pd.DataFrame):
        self.data          = data
        self.featured_data = None
        self._cluster_labels: np.ndarray | None = None

    def call(self) -> np.ndarray:
        self.featured_data = self.data[[
            'neighbourhood group', 'host_identity_verified', 'room type',
            'minimum nights', 'instant_bookable',
            'cancellation_policy', 'availability 365', 'reviews per month'
        ]].dropna().reset_index(drop=True)

        scaler       = StandardScaler()
        scaled_nums  = scaler.fit_transform(
            self.featured_data.select_dtypes(include=['float64', 'int64'])
        )
        encoder      = OneHotEncoder(sparse_output=False)
        encoded_cats = encoder.fit_transform(
            self.featured_data.select_dtypes(include='object')
        )
        X_prepared = np.hstack((scaled_nums, encoded_cats))

        gmm = GaussianMixture(
            n_components=self.N_COMPONENTS,
            covariance_type='full', random_state=42
        )
        gmm.fit(X_prepared)

        cluster_probs  = gmm.predict_proba(X_prepared)
        cluster_labels = gmm.predict(X_prepared)

        print(f"[GMM] BIC: {gmm.bic(X_prepared):.2f}  |  AIC: {gmm.aic(X_prepared):.2f}")

        self.featured_data['cluster'] = cluster_labels

        # Analisi cluster
        num_cols  = self.featured_data.select_dtypes(include='number').columns
        cat_cols  = self.featured_data.select_dtypes(include='object').columns
        num_means = self.featured_data.groupby('cluster')[num_cols].mean()
        cat_modes = self.featured_data.groupby('cluster')[cat_cols].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        cluster_analysis = pd.concat([num_means, cat_modes], axis=1)
        print("\n[GMM] Analisi dei cluster:")
        print(cluster_analysis.to_string())

        # ── Salvataggio CSV ───────────────────────────────────────────────
        save_clustering_analysis(cluster_analysis, kind="gmm")

        # ── PCA 2D + salvataggio grafico ──────────────────────────────────
        pca      = PCA(n_components=2)
        X_pca    = pca.fit_transform(X_prepared)
        z_scores = np.abs(zscore(X_pca))
        non_out  = (z_scores < 3).all(axis=1)

        certainty          = cluster_probs.max(axis=1)
        certainty_filtered = certainty[non_out]

        save_gmm_certainty_plot(X_pca[non_out], certainty_filtered)

        self._cluster_labels = cluster_labels
        return self._cluster_labels