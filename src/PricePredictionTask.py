"""
PricePredictionTask.py
──────────────────────
Regressione supervisionata per la predizione del prezzo degli annunci Airbnb.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from src.ResultsExporter import save_regression_scatter, save_regression_metrics


class PricePredictionTask:

    def __init__(self, data: pd.DataFrame, target_column: str,
                 model=RandomForestRegressor(n_estimators=100, random_state=42)):
        """
        Parameters
        ----------
        data          : DataFrame pre-processato
        target_column : colonna target ('price')
        model         : stimatore sklearn (default: RandomForestRegressor)
        """
        self.data          = data
        self.target_column = target_column
        self.model         = model
        self.model_name    = model.__class__.__name__
        self.trained       = False
        self.X_train = self.X_test = self.Y_train = self.Y_test = self.Y_pred = None

    # ─────────────────────────────────────────────────────────────────────────
    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42):
        Y = self.data['price']

        # Colonne da escludere per il task di regressione sul prezzo
        columns_to_drop = [
            'price',
            'host_identity_verified',
            'neighbourhood group',
            'lat', 'long',
            'last review',
            'cancellation_policy',
            'availability 365',
            'instant_bookable',
        ]
        X = self.data.drop(
            columns=[c for c in columns_to_drop if c in self.data.columns]
        )

        # One-hot encoding: 'neighbourhood' (granulare) e 'room type'
        # drop_first=True per evitare la trappola della multicollinearità
        ohe_cols = [c for c in ['neighbourhood', 'room type'] if c in X.columns]
        X = pd.get_dummies(X, columns=ohe_cols, drop_first=True)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        print(f"[PriceTask] Pre-processing OK — Feature: {X.shape[1]}")

    # ─────────────────────────────────────────────────────────────────────────
    def validate(self, num_folds: int = 10,
                 show_results: bool = False,
                 show_individual_results: bool = False):
        """
        GridSearchCV per ottimizzare gli iperparametri del modello.
        Al termine aggiorna self.model con il miglior stimatore trovato.

        NOTA: 'auto' è rimosso da max_features perché deprecato in
        scikit-learn >= 1.1 e rimosso dalla 1.3.
        """
        param_grid = {
            'n_estimators':     [300],
            'max_depth':        [15],
            'min_samples_split':[10],
            'min_samples_leaf': [1],
            'max_features':     ['sqrt'],
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=num_folds,
            scoring='r2',      # unica metrica per refit — evita l'accesso
            refit=True,        # a cv_results_ con chiavi multi-metrica
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(self.X_train, self.Y_train)

        print("[PriceTask] Risultati GridSearchCV:")
        print(f"  Best estimator : {grid_search.best_estimator_}")
        print(f"  Best R²        : {grid_search.best_score_:.4f}")
        print(f"  Best params    : {grid_search.best_params_}")

        self.model = grid_search.best_estimator_

    # ─────────────────────────────────────────────────────────────────────────
    def train(self):
        print("[PriceTask] Addestramento sul training set…")
        self.model.fit(self.X_train, self.Y_train)
        self.trained = True
        print("[PriceTask] Addestramento completato.")

    # ─────────────────────────────────────────────────────────────────────────
    def generate_prediction(self, show_results: bool = False):
        if not self.trained:
            print("[PriceTask] Modello non ancora addestrato.")
            return

        self.Y_pred = self.model.predict(self.X_test)

        mse        = mean_squared_error(self.Y_test, self.Y_pred)
        r2         = r2_score(self.Y_test, self.Y_pred)
        std_pred   = np.std(self.Y_pred)
        std_actual = np.std(self.Y_test)

        print(f"\n[PriceTask] Performance finale — {self.model_name}")
        print(f"  MSE              : {mse:.4f}")
        print(f"  R²               : {r2:.4f}")
        print(f"  Std pred         : {std_pred:.4f}")
        print(f"  Std actual       : {std_actual:.4f}")

        # Salva scatter plot e riga nel CSV riepilogativo
        save_regression_scatter(self.Y_test, self.Y_pred,
                                model_name=self.model_name)
        save_regression_metrics(
            {"mse": mse, "r2": r2,
             "std_pred": std_pred, "std_actual": std_actual},
            model_name=self.model_name,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def call(self, preprocessing: bool = True, validation: bool = False,
             train: bool = True, show_results: bool = False,
             show_individual_results: bool = False):
        if preprocessing:
            self.preprocess_data()
        if validation:
            self.validate(num_folds=10, show_results=show_results,
                          show_individual_results=show_individual_results)
        if train:
            self.train()
        self.generate_prediction(show_results=show_results)