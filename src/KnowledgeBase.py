"""
KnowledgeBase.py
────────────────
Knowledge Base ibrida per il sistema KBS Airbnb.

Approccio semplificato:
  - owlready2 definisce lo schema OWL (classi + proprietà) e lo salva
    su file .owl per la documentazione formale del progetto
  - Python applica le regole logiche e produce le feature kb_* per XGBoost
  - Nessuna SWRL, nessun HermiT, nessuna Java dependency
"""

import os
import numpy as np
import pandas as pd

try:
    from owlready2 import (
        get_ontology, Thing, DataProperty,
        FunctionalProperty, AllDisjoint
    )
    OWLREADY2_AVAILABLE = True
except ImportError:
    OWLREADY2_AVAILABLE = False
    print("[KB] owlready2 non trovato — pip install owlready2")


# ── Soglie semantiche ─────────────────────────────────────────────────────────
PRICE_BUDGET_THR     = 80.0
PRICE_LUXURY_THR     = 250.0
PRICE_ANOMALY_THR    = 50.0
MIN_NIGHTS_BIZ       = 2
AVAIL_BIZ            = 200
MIN_NIGHTS_FAMILY    = 3
REVIEWS_HIGH_DEMAND  = 3.0
CLUSTER_IB_THRESHOLD = 0.5


class AirbnbKnowledgeBase:

    ONTOLOGY_IRI  = "http://airbnb-kbs.org/ontology#"
    ONTOLOGY_FILE = os.path.join("data", "airbnb_ontology.owl")

    def __init__(self):
        self.onto = None
        if OWLREADY2_AVAILABLE:
            self._build_ontology()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ontology(self):
        """Schema OWL puro: classi e proprietà. Nessuna SWRL, nessun reasoner."""
        self.onto = get_ontology(self.ONTOLOGY_IRI)

        with self.onto:

            class Listing(Thing): pass

            class BudgetOption(Listing): pass
            class MidRangeOption(Listing): pass
            class LuxuryOption(Listing): pass
            AllDisjoint([BudgetOption, MidRangeOption, LuxuryOption])

            class BusinessFriendly(Listing): pass
            class FamilyFriendly(Listing): pass
            class HighDemandListing(Listing): pass
            class ConsistencyAnomaly(Listing): pass

            class BudgetClusterMember(Listing): pass
            class MidRangeClusterMember(Listing): pass
            class PremiumClusterMember(Listing): pass

            class hasMinNights(DataProperty, FunctionalProperty):
                domain = [Listing]; range = [int]
            class hasAvailability(DataProperty, FunctionalProperty):
                domain = [Listing]; range = [int]
            class hasReviewsPerMonth(DataProperty, FunctionalProperty):
                domain = [Listing]; range = [float]
            class hasRoomType(DataProperty, FunctionalProperty):
                domain = [Listing]; range = [str]
            class hasNeighbourhoodGroup(DataProperty, FunctionalProperty):
                domain = [Listing]; range = [str]
            class belongsToCluster(DataProperty, FunctionalProperty):
                domain = [Listing]; range = [int]
            class isInstantBookable(DataProperty, FunctionalProperty):
                domain = [Listing]; range = [bool]

        try:
            os.makedirs(os.path.dirname(self.ONTOLOGY_FILE), exist_ok=True)
            self.onto.save(file=self.ONTOLOGY_FILE, format="rdfxml")
            print(f"[KB] Ontologia salvata in '{self.ONTOLOGY_FILE}'.")
        except Exception as exc:
            print(f"[KB] Salvataggio ontologia fallito: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    def enrich_dataset(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Produce le feature kb_* applicando le regole logiche al DataFrame.
        Ogni colonna corrisponde a una classe dell'ontologia OWL.
        """
        df = df.copy()
        is_entire = df["room type"].str.contains("Entire", na=False, case=False)

        df["kb_is_budget"]      = (df["price"] < PRICE_BUDGET_THR).astype(np.int8)
        df["kb_is_luxury"]      = (df["price"] > PRICE_LUXURY_THR).astype(np.int8)
        df["kb_is_midrange"]    = ((df["price"] >= PRICE_BUDGET_THR) & (df["price"] <= PRICE_LUXURY_THR)).astype(np.int8)
        df["kb_is_anomaly"]     = ((df["price"] < PRICE_ANOMALY_THR) & is_entire).astype(np.int8)
        df["kb_is_business"]    = ((df["minimum nights"] <= MIN_NIGHTS_BIZ) & (df["availability 365"] > AVAIL_BIZ)).astype(np.int8)
        df["kb_is_family"]      = (is_entire & (df["minimum nights"] >= MIN_NIGHTS_FAMILY)).astype(np.int8)
        df["kb_is_high_demand"] = (df["reviews per month"] > REVIEWS_HIGH_DEMAND).astype(np.int8)

        if cluster_labels is not None:
            n = len(df)
            aligned = np.full(n, -1, dtype=np.int8)
            aligned[:min(len(cluster_labels), n)] = cluster_labels[:n]
            df["kb_cluster"] = aligned

            mask = df["kb_cluster"] >= 0
            ib_mean = (df.loc[mask, "instant_bookable"].astype(int)
                         .groupby(df.loc[mask, "kb_cluster"]).mean())
            df["kb_cluster_ib_mean"] = df["kb_cluster"].map(ib_mean).fillna(-1.0).round(4)

            ng = df.get("neighbourhood group", pd.Series([""] * n, index=df.index))
            df["kb_manhattan_cluster_flag"] = (
                (ng == "Manhattan") & (df["kb_cluster_ib_mean"] > CLUSTER_IB_THRESHOLD)
            ).astype(np.int8)

        kb_cols = [c for c in df.columns if c.startswith("kb_")]
        print(f"[KB] Dataset arricchito con {len(kb_cols)} feature KB-derived.")

        n_anomaly = int(df["kb_is_anomaly"].sum())
        if n_anomaly:
            print(f"[KB] Rilevate {n_anomaly} ConsistencyAnomaly (case intere < {PRICE_ANOMALY_THR}$).")

        return df

    # ─────────────────────────────────────────────────────────────────────────
    def print_summary(self):
        print("\n" + "=" * 58)
        print("  AIRBNB KNOWLEDGE BASE")
        print("=" * 58)
        print(f"  owlready2  : {'si' if OWLREADY2_AVAILABLE else 'no (solo logica Python)'}")
        print(f"  File OWL   : {self.ONTOLOGY_FILE}")
        print(f"  Classi     : BudgetOption, MidRangeOption, LuxuryOption,")
        print(f"               BusinessFriendly, FamilyFriendly,")
        print(f"               HighDemandListing, ConsistencyAnomaly,")
        print(f"               BudgetClusterMember, MidRangeClusterMember,")
        print(f"               PremiumClusterMember")
        print(f"  Soglie     : budget<{PRICE_BUDGET_THR}, luxury>{PRICE_LUXURY_THR},")
        print(f"               anomaly<{PRICE_ANOMALY_THR}, biz_nights<={MIN_NIGHTS_BIZ},")
        print(f"               biz_avail>{AVAIL_BIZ}, family_nights>={MIN_NIGHTS_FAMILY},")
        print(f"               high_demand>{REVIEWS_HIGH_DEMAND}")
        print("=" * 58 + "\n")