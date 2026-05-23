"""
config/settings.py
==================
Configuration centralisée du projet Bouznika Traffic Intelligence.
Modifier ce fichier pour adapter les paramètres sans toucher au code.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List


# ── Points GPS d'étude ─────────────────────────────────────────
ROAD_POINTS: Dict[str, Tuple[float, float]] = {
    "RP3323": (33.790780, -7.158819),
    "RN1":    (33.790024, -7.158405),
    "RN11":   (33.789480, -7.159704),
    "RN23":   (33.789620, -7.158738),
}

# ── TomTom Move API ────────────────────────────────────────────
TOMTOM_API_KEY    = "API_KEY"   # → move.tomtom.com
TOMTOM_BASE_URL   = "https://api.tomtom.com/traffic/trafficstats"
TOMTOM_DATE_FROM  = "2024-08-01"
TOMTOM_DATE_TO    = "2024-08-31"
TOMTOM_TIMEZONE   = "Africa/Casablanca"
TOMTOM_FRC_CLASSES: List[str] = ["0", "1", "2", "3", "4", "5"]

# ── Collecte ───────────────────────────────────────────────────
RADIUS_DEG   = 0.0018    # ~200m en degrés de latitude
FILTER_RADIUS_M = 250    # Rayon de filtrage en mètres
POLL_INTERVAL_S = 30     # Secondes entre chaque vérification du job

# ── Modèles ────────────────────────────────────────────────────
TEST_RATIO   = 0.20      # 20% des données pour le test
ARIMA_ORDER  = (5, 1, 0) # (p, d, q)
RF_N_TREES   = 100
RF_N_LAGS    = 6         # Nombre de lags horaires
LSTM_UNITS   = 50
LSTM_LOOKBACK= 24        # Heures de contexte
LSTM_EPOCHS  = 20
LSTM_BATCH   = 32
LSTM_DROPOUT = 0.2

# ── Chemins ────────────────────────────────────────────────────
DATA_RAW_DIR       = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
RESULTS_CSV_DIR    = "results/csv"
RESULTS_FIG_DIR    = "results/figures"

# ── Visualisation ──────────────────────────────────────────────
FIGURE_DPI    = 150
MODEL_COLORS  = {
    "ARIMA":        "#2196F3",
    "RandomForest": "#4CAF50",
    "LSTM":         "#FF5722",
}
