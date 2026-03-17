"""
main.py
────────
Entry point del sistema KBS Airbnb.

Pipeline completa
  1. DatasetPreProcessing  — pulizia e normalizzazione raw data
  2. DataAnalyzer          — analisi esplorativa e visualizzazioni
  3. UnsupervisedTraining  — K-Means + GMM → label cluster (salvate su disco)
  4. SupervisedTraining    — KB enrichment + XGBoost + 10-Fold CV
"""

import sys
import os

# Garantisce che la root del progetto sia nel path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import DatasetPreProcessing
import DataAnalyzer
import SupervisedTrainingManager
import UnsupervisedTrainingManager


def main():
    print("=" * 65)
    print("  AIRBNB KBS — Knowledge-Based System")
    print("  Ingegneria della Conoscenza")
    print("=" * 65)

    # ── Step 1: Pre-processing ────────────────────────────────────────────
    print("\n[STEP 1] Pre-processing dataset…")
    DatasetPreProcessing.call()

    # ── Step 2: Analisi esplorativa ───────────────────────────────────────
    print("\n[STEP 2] Analisi esplorativa…")
    DataAnalyzer.call()

    # ── Step 3: Clustering (genera label per la KB) ───────────────────────
    print("\n[STEP 3] Clustering unsupervised (K-Means + GMM)…")
    UnsupervisedTrainingManager.call()   # salva cluster_labels.npy

    # ── Step 4: Supervised + KB enrichment ───────────────────────────────
    print("\n[STEP 4] Training supervisionato con KB enrichment…")
    SupervisedTrainingManager.call()

    print("\n" + "=" * 65)
    print("  Pipeline completata.")
    print("=" * 65)


if __name__ == "__main__":
    main()
