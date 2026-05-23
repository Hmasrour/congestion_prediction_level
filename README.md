# Bouznika Traffic Intelligence
**PFE — Master Exécutif Smart Cities en Afrique**
Centre Urban Systems — Université Mohammed 6 Polytechnique (2025–2026)

## Objectif
Prédire la congestion routière à Bouznika (routes RP3323, RN1, RN11, RN23)
à partir de données historiques TomTom Move (août 2024) via trois modèles d'IA.

---

## Structure du projet
```
bouznika_traffic/
│
├── main.py                        # Point d'entrée — lancer ce fichier
├── requirements.txt
│
├── config/
│   └── settings.py                # Tous les paramètres (API key, chemins, hyperparamètres)
│
├── data/
│   ├── collector.py               # Classe DataCollector   (Besoin 1)
│   ├── series_builder.py          # Classe TimeSeriesBuilder
│   ├── raw/                       # Données brutes TomTom (générées à l'exécution)
│   └── processed/                 # Données filtrées      (générées à l'exécution)
│
├── models/
│   └── predictors.py              # BaseModel + ARIMAModel + RandomForestModel + LSTMModel
│
├── results/
│   ├── exporter.py                # Classe ResultsExporter (Besoin 3)
│   ├── csv/                       # Fichiers CSV exportés
│   └── figures/                   # Graphiques PNG exportés
│
└── utils/
    └── helpers.py                 # Logger, haversine, métriques, features lag
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Ouvrir `config/settings.py` et renseigner la clé API TomTom Move :

```python
TOMTOM_API_KEY = "votre_clé_ici"   # → move.tomtom.com (trial gratuit)
```

## Lancement

```bash
# Collecte TomTom → CSV uniquement (recommandé en premier)
python main.py --collect-only

# Pipeline complet (collecte + modèles + export)
python main.py
```

---

## Résultats produits

| Fichier | Description |
|---------|-------------|
| `results/csv/comparaison_modeles.csv` | MAE, RMSE, MAPE, R² des 3 modèles |
| `results/csv/predictions_vs_reel.csv` | Prédictions heure par heure |
| `results/csv/stats_par_point.csv` | Vitesse moy/min/max par point GPS |
| `results/figures/resultats_comparatifs.png` | 4 graphiques comparatifs |
| `data/processed/tomtom_filtered.csv` | Données TomTom filtrées |
| `data/raw/tomtom_raw.csv` | Données TomTom brutes |

---

## Modèles & Justification

| Modèle | Ordre/Params | Pourquoi ce choix |
|--------|-------------|-------------------|
| **ARIMA** | (5,1,0) | Baseline statistique, léger, interprétable |
| **Random Forest** | 100 arbres, lag=6h | Robuste, non-linéaire, pas besoin de gros dataset |
| **LSTM** | 50 neurones, lookback=24h | Capture les dépendances temporelles longues |

---

## Points GPS étudiés

| ID | Latitude | Longitude |
|----|----------|-----------|
| RP3323 | 33.790780 | -7.158819 |
| RN1    | 33.790024 | -7.158405 |
| RN11   | 33.789480 | -7.159704 |
| RN23   | 33.789620 | -7.158738 |
