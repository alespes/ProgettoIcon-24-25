"""
AvailabilityPredictionTask.py
─────────────────────────────
Classificazione binaria: predizione di instant_bookable.
Produce automaticamente grafici e CSV nella cartella results/.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

from src.ResultsExporter import (
    save_roc_curve, save_feature_importance,
    save_cv_results, save_cv_barplot
)


_DEFAULT_XGB = XGBClassifier(
    alpha=0, base_score=0.5, booster='gbtree',
    colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.6,
    eval_metric='logloss', gamma=0.2, learning_rate=0.1,
    max_delta_step=0, max_depth=7, min_child_weight=5,
    missing=np.nan, n_estimators=300, n_jobs=-1,
    objective='binary:logistic', random_state=42,
    reg_alpha=0.5, reg_lambda=1, scale_pos_weight=1,
    subsample=0.8, verbosity=0
)


class AvailabilityPredictionTask:

    def __init__(self, data: pd.DataFrame, target_column: str,
                 model: XGBClassifier = _DEFAULT_XGB, n_cv_folds: int = 10):
        self.data          = data
        self.data_dmatrix  = None
        self.target_column = target_column
        self.model         = model
        self.model_name    = model.__class__.__name__
        self.n_cv_folds    = n_cv_folds
        self.trained       = False
        self.X_train = self.X_test = self.Y_train = self.Y_test = self.Y_pred = None
        self.cv_results: pd.DataFrame | None = None

    # ─────────────────────────────────────────────────────────────────────────
    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42):
        self.data[self.target_column] = self.data[self.target_column].astype(int)
        Y = self.data[self.target_column]

        hard_drop = ['instant_bookable', 'host id', 'id', 'neighbourhood', 'lat', 'long']
        X = self.data.drop(columns=[c for c in hard_drop if c in self.data.columns])

        cat_cols = [c for c in ['host_identity_verified', 'neighbourhood group',
                                'cancellation_policy', 'room type'] if c in X.columns]
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        if 'last review' in X.columns:
            ref = pd.to_datetime('2012-01-01')
            X['last review'] = pd.to_datetime(X['last review'], format='mixed', errors='coerce')
            X['last review'] = X['last review'].fillna(X['last review'].median())
            X['last review'] = (X['last review'] - ref).dt.days

        self.data_dmatrix = xgb.DMatrix(data=X, label=Y)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, stratify=Y
        )
        kb_n = sum(1 for c in X.columns if c.startswith('kb_'))
        print(f"[Task] Pre-processing OK — Feature totali: {X.shape[1]}  KB: {kb_n}")

    # ─────────────────────────────────────────────────────────────────────────
    def run_cross_validation(self):
        print(f"\n[CV] {self.n_cv_folds}-Fold Stratified CV ({self.model_name})…")
        skf     = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=42)
        records = []

        for fold_idx, (tr_idx, val_idx) in enumerate(
            skf.split(self.X_train, self.Y_train), start=1
        ):
            X_tr, X_val = self.X_train.iloc[tr_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.Y_train.iloc[tr_idx], self.Y_train.iloc[val_idx]

            m = self.model.__class__(**self.model.get_params())
            m.fit(X_tr, y_tr, verbose=False)
            y_hat = m.predict(X_val)

            records.append({
                'Fold':      fold_idx,
                'Accuracy':  accuracy_score(y_val, y_hat),
                'Precision': precision_score(y_val, y_hat, zero_division=0),
                'Recall':    recall_score(y_val, y_hat, zero_division=0),
                'F1':        f1_score(y_val, y_hat, zero_division=0),
                'AUC':       roc_auc_score(y_val, y_hat),
            })
            r = records[-1]
            print(f"  Fold {fold_idx:>2}  Acc={r['Accuracy']:.4f}  "
                  f"Prec={r['Precision']:.4f}  Rec={r['Recall']:.4f}  "
                  f"F1={r['F1']:.4f}  AUC={r['AUC']:.4f}")

        self.cv_results = pd.DataFrame(records)
        self._print_and_save_cv_summary()

    def _print_and_save_cv_summary(self):
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        means   = self.cv_results[metrics].mean()
        stds    = self.cv_results[metrics].std()

        print(f"\n{'═'*65}")
        print(f"  {self.n_cv_folds}-Fold CV — {self.model_name} — Mean ± Std")
        print(f"{'═'*65}")
        for m in metrics:
            print(f"  {m:<12}: {means[m]:.4f} ± {stds[m]:.4f}")
        print(f"{'═'*65}\n")

        full_df = pd.concat([
            self.cv_results,
            pd.DataFrame([{'Fold': 'Mean', **means.to_dict()}]),
            pd.DataFrame([{'Fold': 'Std',  **stds.to_dict()}]),
        ], ignore_index=True)

        # ── Salvataggio CSV e barplot ─────────────────────────────────────
        save_cv_results(full_df, model_name=self.model_name)
        save_cv_barplot(self.cv_results, model_name=self.model_name)

    # ─────────────────────────────────────────────────────────────────────────
    def train(self):
        print("[Task] Addestramento finale sul training set completo…")
        self.model.fit(self.X_train, self.Y_train, verbose=False)
        self.trained = True

    def tune(self):
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5],
        }
        rs = RandomizedSearchCV(
            XGBClassifier(objective='binary:logistic', random_state=42, verbosity=0),
            param_grid, n_iter=50, scoring='f1', cv=5,
            verbose=1, random_state=42, n_jobs=-1
        )
        rs.fit(self.X_train, self.Y_train)
        print("Best params:", rs.best_params_)
        self.model = rs.best_estimator_

    # ─────────────────────────────────────────────────────────────────────────
    def generate_prediction(self, show_results: bool = False):
        if not self.trained:
            print("[Task] Modello non addestrato.")
            return

        self.Y_pred = self.model.predict(self.X_test)

        acc  = accuracy_score(self.Y_test, self.Y_pred)
        prec = precision_score(self.Y_test, self.Y_pred, zero_division=0)
        rec  = recall_score(self.Y_test, self.Y_pred, zero_division=0)
        f1   = f1_score(self.Y_test, self.Y_pred, zero_division=0)
        auc  = roc_auc_score(self.Y_test, self.Y_pred)

        print(f"\n{'═'*50}")
        print(f"  Test Set — {self.model_name}")
        print(f"{'═'*50}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  AUC-ROC   : {auc:.4f}")
        print(f"{'═'*50}\n")

        # ── ROC Curve ─────────────────────────────────────────────────────
        fpr, tpr, _ = roc_curve(self.Y_test, self.Y_pred)
        save_roc_curve(fpr, tpr, auc, model_name=self.model_name)

        # ── Feature Importance ────────────────────────────────────────────
        if isinstance(self.model, XGBClassifier):
            importances   = self.model.feature_importances_
            feature_names = list(self.X_train.columns)
            save_feature_importance(importances, feature_names,
                                    model_name=self.model_name)

    # ─────────────────────────────────────────────────────────────────────────
    def call(self, preprocessing=True, validation=False,
             train=True, show_results=True, show_individual_results=False):
        if preprocessing:
            self.preprocess_data()
        if validation:
            self.tune()
        self.run_cross_validation()
        if train:
            self.train()
        self.generate_prediction(show_results=show_results)