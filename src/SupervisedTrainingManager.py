"""
SupervisedTrainingManager.py
─────────────────────────────
Coordina i task supervisionati integrando l'arricchimento KB prima del training.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.KnowledgeBase import AirbnbKnowledgeBase
from src.AvailabilityPredictionTask import AvailabilityPredictionTask
from src.PricePredictionTask import PricePredictionTask
from src.UnsupervisedTrainingManager import CLUSTER_LABELS_PATH


class SupervisedTrainingManager:

    def __init__(self, input_file_path: str):
        self.file_path = input_file_path
        self.data: pd.DataFrame | None = None
        self._load_data()

    def _load_data(self):
        try:
            self.data = pd.read_csv(self.file_path, low_memory=False)
            print(f"[Supervised] Dati caricati: {self.data.shape}")
        except FileNotFoundError:
            print(f"[Supervised] File non trovato: {self.file_path}")
        except Exception as exc:
            print(f"[Supervised] Errore caricamento: {exc}")

    def _load_cluster_labels(self) -> np.ndarray | None:
        if os.path.exists(CLUSTER_LABELS_PATH):
            labels = np.load(CLUSTER_LABELS_PATH)
            print(f"[Supervised] Label cluster caricate: {len(labels)} record.")
            return labels
        print("[Supervised] Label cluster non trovate — feature kb_cluster omesse.")
        return None

    def _enrich_with_kb(self, cluster_labels: np.ndarray | None) -> pd.DataFrame:
        """Istanzia la KB e arricchisce il dataset con le feature kb_*."""
        kb = AirbnbKnowledgeBase()
        kb.print_summary()
        return kb.enrich_dataset(self.data, cluster_labels=cluster_labels)


def call():
    original_file_path = os.path.join(
        'data', 'Post_PreProcessing', 'Airbnb_Processed_Data.csv'
    )
    manager = SupervisedTrainingManager(original_file_path)

    if manager.data is None:
        print("[Supervised] Impossibile procedere senza dati.")
        return

    cluster_labels = manager._load_cluster_labels()
    enriched_data  = manager._enrich_with_kb(cluster_labels)

    # ── Task regressione: Price Prediction ───────────────────────────────
    regression_task = True
    if regression_task:
        print("\n[Supervised] ── Price Prediction ──────────────────")
        price_predictor = PricePredictionTask(
            enriched_data, 'price',
            model=RandomForestRegressor(n_estimators=100, random_state=42)
        )
        price_predictor.call(preprocessing=True, validation=True,
                             train=True, show_results=True)

    # ── Task classificazione: Availability Prediction (10-Fold CV) ───────
    classification_task = True
    if classification_task:
        print("\n[Supervised] ── Availability Prediction (10-Fold CV) ──")
        availability_predictor = AvailabilityPredictionTask(
            data=enriched_data,
            target_column='instant_bookable',
            n_cv_folds=10
        )
        availability_predictor.call(preprocessing=True, validation=True,
                                    train=True, show_results=True)