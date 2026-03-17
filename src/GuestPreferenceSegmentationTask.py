import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from src.ResultsExporter import (
    save_kmeans_plot, save_clustering_analysis, write_readme
)


class GuestPreferenceSegmentationTask:

    N_CLUSTERS = 3

    def __init__(self, data: pd.DataFrame):
        self.data          = data
        self.featured_data = None
        self.X             = None
        self._cluster_labels: np.ndarray | None = None

    def preproccessing(self):
        self.featured_data = self.data[[
            'neighbourhood group', 'host_identity_verified', 'room type',
            'price', 'minimum nights', 'instant_bookable',
            'cancellation_policy', 'availability 365', 'reviews per month'
        ]].dropna().reset_index(drop=True)

        numerical_features   = ['price', 'minimum nights', 'availability 365', 'reviews per month']
        categorical_features = ['room type', 'instant_bookable', 'cancellation_policy',
                                'neighbourhood group', 'host_identity_verified']

        scaler       = StandardScaler()
        scaled_nums  = scaler.fit_transform(self.featured_data[numerical_features])
        encoder      = OneHotEncoder(sparse_output=False)
        encoded_cats = encoder.fit_transform(self.featured_data[categorical_features].values)
        self.X = np.column_stack((scaled_nums, encoded_cats))

    def apply_Kmeans(self):
        kmeans   = KMeans(n_clusters=self.N_CLUSTERS, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(self.X)
        self.featured_data['cluster'] = clusters

        pca   = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        # Rimozione outlier
        z_scores     = np.abs(zscore(X_pca))
        non_outliers = (z_scores < 3).all(axis=1)
        X_pca_clean  = X_pca[non_outliers]
        feat_clean   = self.featured_data[non_outliers].reset_index(drop=True)

        # ── Salvataggio grafico ───────────────────────────────────────────
        save_kmeans_plot(X_pca_clean, feat_clean['cluster'].values)

        # Analisi cluster
        num_cols  = self.featured_data.select_dtypes(include='number').columns
        cat_cols  = self.featured_data.select_dtypes(include='object').columns
        num_means = self.featured_data.groupby('cluster')[num_cols].mean()
        cat_modes = self.featured_data.groupby('cluster')[cat_cols].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        cluster_analysis = pd.concat([num_means, cat_modes], axis=1)
        print("\n[K-Means] Analisi dei cluster:")
        print(cluster_analysis.to_string())

        # ── Salvataggio CSV ───────────────────────────────────────────────
        save_clustering_analysis(cluster_analysis, kind="kmeans")

        self._cluster_labels = self.featured_data['cluster'].values

    def call(self) -> np.ndarray:
        self.preproccessing()
        self.apply_Kmeans()
        write_readme()
        return self._cluster_labels