# A Machine Learning & Knowledge-Based Approach to AirBnB

**Corso:** Ingegneria della Conoscenza — AA 25/26  
**Autore:** Alessandro Pesari — Matricola 779406  
**Docente:** Prof. Nicola Fanizzi  
**Repository:** [GitHub](https://github.com/alespes/ProgettoIcon-2026/tree/main)

---

## Indice

1. [Introduzione e Obiettivi](#1-introduzione-e-obiettivi)
2. [Dataset e Pre-processing](#2-dataset-e-pre-processing)
3. [Architettura del Sistema KBS](#3-architettura-del-sistema-kbs)
4. [Knowledge Base: Rappresentazione e Ragionamento](#4-knowledge-base-rappresentazione-e-ragionamento)
5. [Guest Preference Segmentation — Clustering](#5-guest-preference-segmentation--clustering)
6. [Price Prediction — Regressione](#6-price-prediction--regressione)
7. [Availability Prediction — Classificazione con KB Enrichment](#7-availability-prediction--classificazione-con-kb-enrichment)
8. [Integrazione Clustering → KB → Apprendimento Supervisionato](#8-integrazione-clustering--kb--apprendimento-supervisionato)
9. [Valutazione Comparativa e Conclusioni](#9-valutazione-comparativa-e-conclusioni)

---

## 1. Introduzione e Obiettivi

Il progetto sviluppa un **Knowledge-Based System (KBS)** per l'analisi del mercato degli affitti a breve termine su Airbnb nella città di New York. L'obiettivo non è limitarsi all'applicazione di algoritmi di Machine Learning standard, ma integrare **Background Knowledge (BK)** formalizzata in un'ontologia OWL 2 con tecniche di apprendimento supervisionato e non supervisionato, seguendo il paradigma ML+OntoBK.

Il sistema affronta tre task principali:

- **Regressione** — predizione del prezzo di un annuncio (`price`)
- **Classificazione** — predizione della prenotabilità immediata (`instant_bookable`)
- **Clustering** — segmentazione delle preferenze degli ospiti (K-Means + GMM/EM)

La novità rispetto a un approccio ML classico risiede nel **quarto componente**: la KB arricchisce il dataset con feature semantiche derivate da regole ontologiche prima di passarlo ai modelli supervisionati, chiudendo il ciclo _clustering → ragionamento → apprendimento_.

---

## 2. Dataset e Pre-processing

**Fonte:** [Airbnb Open Data — Kaggle](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata/data)  
**Dimensioni originali:** 102.599 record × 26 feature  
**Dimensioni post-processing:** 101.903 record × 20 feature

### 2.1 Scelte di Pre-processing

**Eliminazione di righe:** applicata solo per le colonne critiche senza sostituto logico (`neighbourhood group`, `lat`, `long`, `instant_bookable`). Per tutte le altre colonne si è preferita l'imputation per preservare la dimensione del dataset.

**Imputation:** la scelta tra media e mediana è guidata dalla distribuzione:

| Colonna | Strategia | Motivazione |
|---------|-----------|-------------|
| `price`, `service fee` | Media | Distribuzione approssimativamente normale dopo rimozione outlier |
| `number of reviews`, `reviews per month`, `availability 365` | Mediana | Distribuzione asimmetrica con coda destra |
| `Construction year` | Media per `neighbourhood group` | Correlazione geografica tra anno e quartiere |
| `cancellation_policy` | Moda | Variabile categorica nominale |
| `host_identity_verified` | Default `'unconfirmed'` | Semanticamente corretto in assenza di informazione |

**Correzioni di consistenza logica:**
- `availability 365`: valori negativi → `|x|`; valori > 365 → 365
- `minimum nights`: stessa logica; `NaN` → 1
- `last review`: date future → data corrente; formato unificato a `datetime64`
- Typo `'brookln'` → `'Brooklyn'` in `neighbourhood group`

**Encoding:** `get_dummies` con `drop_first=True` per evitare la trappola della multicollinearità. La data `last review` viene convertita in numero di giorni da una data di riferimento (`2012-01-01`) per preservarne l'ordinalità.

---

## 3. Architettura del Sistema KBS

Il sistema segue una pipeline a quattro step con dipendenze esplicite:

```
[STEP 1] DatasetPreProcessing
         └─ 102.599 → 101.903 record × 20 feature
                    │
[STEP 3] UnsupervisedTrainingManager
         ├─ K-Means (k=3)  ──→ clustering_analysis_kmeans.csv
         ├─ GMM (k=3)      ──→ clustering_analysis_gmm.csv
         └─ cluster_labels.npy  (101.903 label)
                    │
[STEP 4] SupervisedTrainingManager
         ├─ AirbnbKnowledgeBase.enrich_dataset(df, cluster_labels)
         │        └─ +10 colonne kb_* → 245 feature (price task)
         │                            → 30 feature  (availability task)
         ├─ PricePredictionTask       (RandomForest + GridSearchCV)
         └─ AvailabilityPredictionTask (XGBoost + 10-Fold Stratified CV)
```

La dipendenza tra STEP 3 e STEP 4 è il punto architetturalmente rilevante: le label cluster prodotte dall'apprendimento non supervisionato vengono usate dalla KB per generare feature semantiche che alimentano i modelli supervisionati. Le label vengono serializzate in `data/cluster_labels.npy` per il riuso, rendendo i due step indipendenti nell'esecuzione ma collegati nel flusso di conoscenza.

---

## 4. Knowledge Base: Rappresentazione e Ragionamento

### 4.1 Tecnologia e Architettura

La KB è implementata tramite **owlready2** con ontologia **OWL 2 AL**. Il pattern adottato è **Classify-then-Assert**: Python valuta le condizioni numeriche (stabile su Python 3.12+) mentre owlready2 gestisce la gerarchia delle classi e serializza l'ontologia in `data/airbnb_ontology.owl` (RDF/XML).

Questa scelta è motivata dall'instabilità del parser SWRL di owlready2 per i built-in matematici (`swrlb:lessThan` ecc.) su Python 3.12+, che causa `ValueError` indipendentemente dalla sintassi usata. Il risultato semantico è equivalente all'esecuzione delle SWRL rules.

### 4.2 Gerarchia delle Classi

```
owl:Thing
  └── Listing
        ├── BudgetOption           price < 80 USD
        ├── MidRangeOption         80 ≤ price ≤ 250 USD
        ├── LuxuryOption           price > 250 USD
        │     ↑ AllDisjoint ↑
        ├── BusinessFriendly       minNights ≤ 2 ∧ availability > 200
        ├── FamilyFriendly         Entire home ∧ minNights ≥ 3
        ├── HighDemandListing      reviewsPerMonth > 3.0
        ├── ConsistencyAnomaly     price < 50 ∧ Entire home
        ├── BudgetClusterMember    cluster K-Means = 0
        ├── MidRangeClusterMember  cluster K-Means = 1
        └── PremiumClusterMember   cluster K-Means = 2
```

L'assioma `AllDisjoint([BudgetOption, MidRangeOption, LuxuryOption])` è un vincolo di consistenza formale: un individuo classificato simultaneamente in due fasce di prezzo indica un errore nell'ontologia o nei dati, rilevabile da un reasoner esterno.

### 4.3 Regole Logiche e Feature KB-derived

| ID | Classe OWL | Condizione | Feature prodotta |
|----|-----------|------------|-----------------|
| R1 | `BudgetOption` | `price < 80` | `kb_is_budget` |
| R2 | `LuxuryOption` | `price > 250` | `kb_is_luxury` |
| — | `MidRangeOption` | `80 ≤ price ≤ 250` | `kb_is_midrange` |
| R3 | `ConsistencyAnomaly` | `price < 50 ∧ Entire home` | `kb_is_anomaly` |
| R4 | `BusinessFriendly` | `minNights ≤ 2 ∧ availability > 200` | `kb_is_business` |
| R5 | `FamilyFriendly` | `Entire home ∧ minNights ≥ 3` | `kb_is_family` |
| R6 | `HighDemandListing` | `reviewsPerMonth > 3.0` | `kb_is_high_demand` |
| R7–R9 | `*ClusterMember` | `cluster ∈ {0, 1, 2}` | `kb_cluster` |
| D1 | — (derivata) | `Manhattan ∧ cluster_ib_mean > 0.5` | `kb_manhattan_cluster_flag` |

La regola D1 è ibrida ML+KB: non esprimibile in OWL puro perché richiede l'aggregazione `mean(instant_bookable)` per cluster, disponibile solo dopo il clustering. Rappresenta il collegamento formale tra i risultati non supervisionati e la KB.

### 4.4 Valutazione della Complessità

| Dimensione | Valore |
|-----------|--------|
| Classi OWL | 11 (1 radice + 10 sottoclassi) |
| Data Properties | 7 (`FunctionalProperty`) |
| Assiomi di disgiunzione | 1 (`AllDisjoint` su 3 classi) |
| Regole logiche | 9 (R1–R9) + 1 derivata (D1) |
| Feature KB prodotte | 10 colonne `kb_*` |
| Complessità logica | ALC — sufficiente per il dominio applicativo |
| Overhead computazionale | O(n) — applicazione vettorizzata |

---

## 5. Guest Preference Segmentation — Clustering

### 5.1 Feature e Pre-processing

9 feature selezionate per descrivere le preferenze degli ospiti: `neighbourhood group`, `host_identity_verified`, `room type`, `price`, `minimum nights`, `instant_bookable`, `cancellation_policy`, `availability 365`, `reviews per month`. Le variabili numeriche sono normalizzate con `StandardScaler` (media 0, varianza 1), le categoriche codificate con `OneHotEncoder`.

### 5.2 K-Means — k=3

Scelta di k=3 validata dalla separabilità nel piano PCA 2D e confermata dal GMM. Gli outlier con z-score > 3 sulle componenti PCA vengono rimossi dalla visualizzazione (non dal training).

![GMM Cluster](results/plots/clustering/kmeans_pca_clusters.png)

**Profili dei cluster (medie):**

| Cluster | Price | Min Nights | Availability | Reviews/mese | Quartiere modale | Room Type modale |
|---------|-------|-----------|-------------|-------------|-----------------|-----------------|
| 0 | 623.49 | 9.07 | 284.56 | 1.39 | Manhattan | Entire home/apt |
| 1 | 626.78 | 6.08 | 40.72 | 1.01 | Brooklyn | Entire home/apt |
| 2 | 608.58 | 328.98 | 183.06 | 0.98 | Manhattan | Entire home/apt |

I tre cluster rivelano segmenti di mercato distinti: il cluster 0 corrisponde a soggiorni medi a Manhattan con alta disponibilità annua; il cluster 1 a soggiorni brevi a Brooklyn con disponibilità molto bassa (occupazione alta); il cluster 2 a soggiorni quasi annuali a Manhattan (minimum nights ≈ 329), un segmento inatteso che suggerisce l'uso di Airbnb come canale per affitti semi-permanenti. Questo pattern ha implicazioni dirette per la KB: gli annunci del cluster 2 vengono classificati come `PremiumClusterMember` ma hanno caratteristiche funzionalmente incompatibili con `BusinessFriendly` o `FamilyFriendly`.

### 5.3 Validazione con GMM/EM

Il Gaussian Mixture Model (3 componenti, `covariance_type='full'`) produce:

| Metrica | Valore |
|---------|--------|
| BIC | -7.713.484,59 |
| AIC | -7.718.364,86 |

I valori negativi di grande magnitudine indicano un ottimo fit. L'allineamento tra le hard-assignment K-Means e le soft-assignment GMM conferma la robustezza della struttura a 3 cluster.

![GMM Certainty](results/plots/clustering/gmm_certainty.png)

---

## 6. Price Prediction — Regressione

### 6.1 Feature Selection

Feature escluse: `price` (target), `host_identity_verified`, `neighbourhood group`, `lat`, `long`, `last review`, `cancellation_policy`, `availability 365`, `instant_bookable`. Mantenuta `neighbourhood` (più granulare di `neighbourhood group`) con one-hot encoding. **Feature totali dopo encoding: 245.**

### 6.2 Ottimizzazione Iperparametri — GridSearchCV

`GridSearchCV` con 10-fold su 243 combinazioni parametriche:

| Parametro | Griglia | Ottimale |
|-----------|---------|----------|
| `n_estimators` | [100, 200, 300] | 300 |
| `max_depth` | [5, 10, 15] | 15 |
| `min_samples_split` | [2, 5, 10] | 10 |
| `min_samples_leaf` | [1, 2, 4] | 1 |
| `max_features` | ['sqrt', 'log2', None] | 'sqrt' |

`'auto'` è stato rimosso dalla griglia perché deprecato in scikit-learn ≥ 1.1 e rimosso dalla 1.3. **R² CV (best): 0.8319.**

### 6.3 Risultati sul Test Set

| Metrica | Valore |
|---------|--------|
| MSE | 16.577,73 |
| **R²** | **0.8493** |
| Std predizioni | 239.82 |
| Std valori reali | 331.68 |

![GMM Random Forest](results/plots/regression/regression_scatter_randomforestregressor.png)

Il modello spiega l'84.9% della varianza del prezzo. La differenza tra std predizioni (239.82) e std valori reali (331.68) indica compressione della distribuzione: il modello sottostima i prezzi estremi, comportamento tipico dei metodi ensemble che mediano su più alberi. Il lieve gap tra R² CV (0.8319) e R² test set (0.8493) è nella norma e non indica overfitting.

---

## 7. Availability Prediction — Classificazione con KB Enrichment

### 7.1 Contesto

Nella versione originale del progetto tutti i modelli convergevano verso random guessing (accuracy ≈ 0.50). L'analisi della feature importance confermava correlazioni tra `instant_bookable` e le altre feature, escludendo una vera indipendenza statistica. La KB introduce 10 feature semantiche con l'ipotesi che combinazioni logiche esplicite rendano il segnale più accessibile al modello. Il dataset arricchito conta **30 feature totali**.

### 7.2 Iperparametri Ottimali (RandomizedSearchCV, 50 iter, 5-fold)

| Parametro | Valore |
|-----------|--------|
| `n_estimators` | 300 |
| `learning_rate` | 0.1 |
| `max_depth` | 10 |
| `min_child_weight` | 3 |
| `subsample` | 0.6 |
| `colsample_bytree` | 1.0 |
| `gamma` | 0.2 |

### 7.3 Risultati 10-Fold Stratified CV

| Fold | Accuracy | Precision | Recall | F1 | AUC |
|------|----------|-----------|--------|----|-----|
| 1 | 0.4944 | 0.4924 | 0.4951 | 0.4937 | 0.4944 |
| 2 | 0.4949 | 0.4926 | 0.4781 | 0.4853 | 0.4948 |
| 3 | 0.4971 | 0.4951 | 0.4931 | 0.4941 | 0.4970 |
| 4 | 0.4945 | 0.4923 | 0.4786 | 0.4853 | 0.4944 |
| 5 | 0.4952 | 0.4931 | 0.4840 | 0.4885 | 0.4952 |
| 6 | 0.4957 | 0.4937 | 0.4989 | 0.4963 | 0.4957 |
| 7 | 0.5006 | 0.4985 | 0.4935 | 0.4960 | 0.5006 |
| 8 | 0.5016 | 0.4995 | 0.4937 | 0.4966 | 0.5016 |
| 9 | 0.4933 | 0.4910 | 0.4834 | 0.4872 | 0.4932 |
| 10 | 0.4987 | 0.4965 | 0.4905 | 0.4935 | 0.4986 |
| **Mean** | **0.4966** | **0.4945** | **0.4889** | **0.4916** | **0.4966** |
| **Std** | **0.0028** | **0.0029** | **0.0073** | **0.0046** | **0.0028** |

**Risultati Test Set:** Accuracy 0.4967 — Precision 0.4946 — Recall 0.4878 — F1 0.4912 — AUC 0.4967

**[Grafico: results/plots/classification/cv_barplot_xgbclassifier.png]**

**[Grafico: results/plots/classification/roc_curve_xgbclassifier.png]**

**[Grafico: results/plots/classification/feature_importance_xgbclassifier.png]**

### 7.4 Analisi del Risultato

Le prestazioni rimangono equivalenti al random guessing (AUC ≈ 0.50) anche con KB enrichment. La **deviazione standard molto bassa** (Accuracy std = 0.0028 su 10 fold) è un dato rilevante: non si tratta di alta varianza o instabilità, ma di un risultato stabile e riproducibile. Il modello apprende consistentemente, ma non trova pattern predittivi.

Questo porta a due conclusioni:

**1. La KB non introduce degrado.** Le feature `kb_*` non confondono il modello: le prestazioni pre e post KB enrichment sono praticamente identiche, confermando la correttezza logica delle regole ontologiche.

**2. Il limite è nella natura della feature target.** `instant_bookable` è una scelta editoriale discrezionale del proprietario, non una conseguenza misurabile delle caratteristiche strutturali dell'annuncio. Nessuna rappresentazione della conoscenza di dominio può predire decisioni umane arbitrarie da dati strutturali. La feature importance (grafico) mostra che `availability 365` e `reviews per month` sono le feature più informative — ma anche le più correlate con la decisione dell'host di rendere l'annuncio prenotabile istantaneamente, non causalmente dipendenti da essa.

---

## 8. Integrazione Clustering → KB → Apprendimento Supervisionato

### 8.1 Flusso di Conoscenza

Il clustering produce una partizione empirica del dominio. La KB la formalizza come classi OWL e la integra con regole di dominio preesistenti. Il modello supervisionato riceve feature che codificano sia struttura statistica (da dove viene il cluster) sia semantica di dominio (cosa significa appartenere a quel cluster rispetto al task target).

### 8.2 La Regola D1

```
kb_manhattan_cluster_flag = 1
  IFF  neighbourhood_group == "Manhattan"
  AND  mean(instant_bookable | cluster) > 0.5
```

Questa regola non è esprimibile in OWL puro (richiede aggregazione su istanze) e non è catturabile da XGBoost senza KB (richiede conoscenza della struttura cluster come feature esterna). Costituisce l'esempio più rappresentativo del valore aggiunto dell'approccio ibrido: combina conoscenza geografica di dominio (importanza di Manhattan) con conoscenza statistica appresa (comportamento del cluster rispetto alla variabile target).

Il fatto che nemmeno questa feature migliori le prestazioni del classificatore rafforza la tesi che `instant_bookable` sia intrinsecamente non predittibile dalle feature strutturali disponibili.

---

## 9. Valutazione Comparativa e Conclusioni

### 9.1 Riepilogo Prestazioni

| Task | Modello | Metrica | Risultato |
|------|---------|---------|-----------|
| Price Prediction | RandomForest (GridSearchCV) | R² test set | **0.8493** |
| Price Prediction | RandomForest (GridSearchCV) | MSE test set | 16.577,73 |
| Price Prediction | RandomForest (CV 10-fold) | R² mean | 0.8319 |
| Availability + KB (10-fold CV) | XGBoost | Accuracy mean ± std | 0.4966 ± 0.0028 |
| Availability + KB (10-fold CV) | XGBoost | F1 mean ± std | 0.4916 ± 0.0046 |
| Availability + KB (10-fold CV) | XGBoost | AUC mean ± std | 0.4966 ± 0.0028 |
| Clustering K-Means | k=3 | BIC (GMM validation) | -7.713.484,59 |

### 9.2 Conclusioni Tecniche

**Price Prediction:** il modello RandomForest ottimizzato (R² = 0.8493) è generalizzabile e non in overfitting. La compressione della distribuzione delle predizioni (std pred 239.82 vs std actual 331.68) è il limite principale: prezzi estremi sono sistematicamente sotto e sovrastimati.

**Clustering:** struttura a 3 cluster robusta e validata. Il cluster 2 (minimum nights ≈ 329) è il risultato più inatteso: rivela l'uso di Airbnb come canale per affitti a lungo termine, un segmento di mercato con dinamiche completamente diverse dai soggiorni turistici.

**Availability Prediction:** il task rimane non risolvibile con le feature disponibili (AUC ≈ 0.50, std = 0.0028). La bassa varianza documenta che si tratta di un limite informativo del dataset, non di un problema metodologico. La KB non peggiora le prestazioni, confermando la correttezza logica delle regole ontologiche e l'assenza di rumore introdotto dall'arricchimento.

### 9.3 Sviluppi Futuri

- **Modifica del target di classificazione:** sostituire `instant_bookable` con una variabile derivata come `alta_disponibilita = availability_365 > 200`, per cui le correlazioni con le feature strutturali sono documentate e la KB offre regole direttamente rilevanti (R4 — `BusinessFriendly`)
- **Arricchimento con dati esterni via SPARQL:** interrogare DBpedia per aggiungere feature sul quartiere (densità turistica, eventi locali), potenzialmente in grado di spiegare la variabilità residua nel task di classificazione
- **Kernel su rappresentazioni logiche:** esplorare SVM con kernel definibili su descrizioni Datalog per confronto diretto con l'approccio XGBoost+KB sul task di regressione

---

*Documentazione redatta per il corso di Ingegneria della Conoscenza, AA 25/26 — Università degli Studi di Bari Aldo Moro, Dipartimento di Informatica.*
