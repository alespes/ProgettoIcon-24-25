"""
UnsupervisedTrainingManager.py
────────────────────────────────
Coordina i task di clustering (K-Means e GMM/EM) e persiste le label
dei cluster su disco per l'uso successivo della Knowledge Base.
"""

import os
import numpy as np
import pandas as pd

from src.EM_Implementation import EM_Implementation
from src.GuestPreferenceSegmentationTask import GuestPreferenceSegmentationTask


CLUSTER_LABELS_PATH = os.path.join('data', 'cluster_labels.npy')


class UnsupervisedTrainingManager:

    def __init__(self, input_file_path: str):
        self.file_path = input_file_path
        self.data: pd.DataFrame | None = None
        self._load_data()

    def _load_data(self):
        try:
            self.data = pd.read_csv(self.file_path, low_memory=False)
            print(f"[Unsupervised] Dati caricati: {self.data.shape}")
        except FileNotFoundError:
            print(f"[Unsupervised] File non trovato: {self.file_path}")
        except Exception as exc:
            print(f"[Unsupervised] Errore caricamento: {exc}")


def call() -> np.ndarray | None:
    """
    Esegue K-Means e GMM, salva le label del clustering su disco
    e restituisce le label K-Means (usate dalla KB per l'arricchimento
    del dataset supervisionato).

    Returns
    -------
    np.ndarray o None
        Label cluster K-Means per le righe del dataset dopo il dropna.
    """
    original_file_path = os.path.join('data', 'Post_PreProcessing', 'Airbnb_Processed_Data.csv')
    manager = UnsupervisedTrainingManager(original_file_path)

    if manager.data is None:
        print("[Unsupervised] Impossibile procedere senza dati.")
        return None

    kmeans_labels: np.ndarray | None = None

    # ── Hard clustering (K-Means) ─────────────────────────────────────────
    hard_clustering_task = True
    if hard_clustering_task:
        print("\n[Unsupervised] ── K-Means ──────────────────────────")
        hard_clustering = GuestPreferenceSegmentationTask(manager.data)
        kmeans_labels   = hard_clustering.call()
        print(f"[Unsupervised] K-Means: {len(kmeans_labels)} label generate.")

    # ── Soft clustering (GMM / EM) ────────────────────────────────────────
    soft_clustering_task = True
    if soft_clustering_task:
        print("\n[Unsupervised] ── Gaussian Mixture Model ───────────")
        soft_clustering = EM_Implementation(manager.data)
        soft_clustering.call()

    # ── Persiste le label K-Means per la KB ──────────────────────────────
    if kmeans_labels is not None:
        os.makedirs('data', exist_ok=True)
        np.save(CLUSTER_LABELS_PATH, kmeans_labels)
        print(f"\n[Unsupervised] Label cluster salvate in '{CLUSTER_LABELS_PATH}'.")

    return kmeans_labels
