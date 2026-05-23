"""
models/predictors.py
====================
Besoin 2 — Modélisation prédictive
Classes : BaseModel (ABC) → ARIMAModel · RandomForestModel · LSTMModel

Justification des choix méthodologiques :
──────────────────────────────────────────
• ARIMA(5,1,0) : baseline statistique classique pour séries temporelles
  stationnaires. Choisi car la série de vitesse présente une tendance faible
  et une saisonnalité horaire régulière. Léger et interprétable.
  → Limite : ne capture pas les non-linéarités.

• Random Forest : agrège 100 arbres de décision sur des features lag horaires.
  Choisi car il gère les relations non-linéaires sans hypothèse distributionnelle,
  et reste robuste aux données manquantes/outliers.
  → Avantage sur ARIMA : meilleur sur patterns complexes.
  → Avantage sur LSTM : ne nécessite pas de grandes quantités de données.

• LSTM (Long Short-Term Memory) : réseau de neurones récurrent avec lookback=24h.
  Choisi car il mémorise les dépendances à long terme (cycles jour/nuit, semaine).
  → Avantage sur ARIMA/RF : modélise explicitement l'ordre temporel.
  → Limite : nécessite plus de données et de temps d'entraînement.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import config.settings as cfg
from utils import compute_metrics, make_lag_features, get_logger

warnings.filterwarnings("ignore")


# ================================================================
# Classe abstraite — contrat commun à tous les modèles
# ================================================================
class BaseModel(ABC):
    """
    Classe abstraite définissant l'interface commune des modèles prédictifs.
    Chaque modèle doit implémenter `fit()` et `predict()`.
    """

    def __init__(self):
        self.log     = get_logger(self.__class__.__name__)
        self._fitted = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom lisible du modèle."""

    @abstractmethod
    def fit(self, train: pd.Series) -> "BaseModel":
        """Entraîne le modèle sur la série d'entraînement."""

    @abstractmethod
    def predict(self, n_steps: int) -> np.ndarray:
        """Génère n_steps prédictions."""

    def evaluate(self, test: pd.Series) -> Dict[str, Any]:
        """
        Prédit sur la période de test et calcule les métriques.
        Retourne un dict avec predictions, actual et metrics.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name} : appelez fit() avant evaluate()")
        preds  = self.predict(len(test))
        actual = test.values
        n      = min(len(actual), len(preds))
        return {
            "name":        self.name,
            "predictions": preds[:n],
            "actual":      actual[:n],
            "metrics":     compute_metrics(actual[:n], preds[:n]),
        }

    @staticmethod
    def train_test_split(series: pd.Series, test_ratio: float = cfg.TEST_RATIO
                         ) -> Tuple[pd.Series, pd.Series]:
        """Découpe la série en train / test (pas de mélange aléatoire)."""
        split = int(len(series) * (1 - test_ratio))
        return series.iloc[:split], series.iloc[split:]


# ================================================================
# ARIMA
# ================================================================
class ARIMAModel(BaseModel):
    """
    ARIMA(p, d, q) — AutoRegressive Integrated Moving Average.
    Paramètres : p=5 (mémoire 5h), d=1 (différenciation), q=0.
    """

    def __init__(self, order: Tuple[int, int, int] = cfg.ARIMA_ORDER):
        super().__init__()
        self.order   = order
        self._model  = None
        self._fitted_model = None

    @property
    def name(self) -> str:
        return f"ARIMA{self.order}"

    def fit(self, train: pd.Series) -> "ARIMAModel":
        from statsmodels.tsa.arima.model import ARIMA
        self.log.info(f"Entraînement {self.name} sur {len(train)} observations...")
        self._train       = train
        self._fitted_model = ARIMA(train.values, order=self.order).fit()
        self._fitted      = True
        self.log.info(f"{self.name} entraîné ✓")
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        return self._fitted_model.forecast(steps=n_steps)


# ================================================================
# Random Forest
# ================================================================
class RandomForestModel(BaseModel):
    """
    Random Forest Regressor avec features lag horaires.
    n_estimators=100, lag=6h, features : lag_1…lag_6, hour, weekday.
    """

    def __init__(self, n_estimators: int = cfg.RF_N_TREES, n_lags: int = cfg.RF_N_LAGS):
        super().__init__()
        self.n_estimators = n_estimators
        self.n_lags       = n_lags
        self._rf          = None
        self._train       = None

    @property
    def name(self) -> str:
        return f"Random Forest ({self.n_estimators} arbres, lag={self.n_lags}h)"

    def fit(self, train: pd.Series) -> "RandomForestModel":
        from sklearn.ensemble import RandomForestRegressor
        self.log.info(f"Entraînement {self.name} sur {len(train)} observations...")
        self._train  = train
        feats        = make_lag_features(train, self.n_lags)
        X_train      = feats.drop("y", axis=1)
        y_train      = feats["y"]
        self._rf     = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=42, n_jobs=-1
        )
        self._rf.fit(X_train, y_train)
        self._feature_names = X_train.columns.tolist()
        self._fitted = True
        self.log.info(f"{self.name} entraîné ✓")
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        """Prédiction récursive : chaque prédiction devient un lag pour la suivante."""
        history = list(self._train.values[-self.n_lags:])
        preds   = []
        has_time = hasattr(self._train.index, "hour")

        for i in range(n_steps):
            row = {f"lag_{j+1}": history[-(j+1)] for j in range(self.n_lags)}
            if has_time:
                future_idx = self._train.index[-1] + pd.Timedelta(hours=i + 1)
                row["hour"]    = future_idx.hour
                row["weekday"] = future_idx.dayofweek
            X = pd.DataFrame([row])[self._feature_names]
            p = float(self._rf.predict(X)[0])
            preds.append(p)
            history.append(p)
        return np.array(preds)

    @property
    def feature_importance(self) -> Dict[str, float]:
        if not self._fitted:
            raise RuntimeError("Modèle non entraîné")
        return dict(zip(self._feature_names, self._rf.feature_importances_))


# ================================================================
# LSTM
# ================================================================
class LSTMModel(BaseModel):
    """
    LSTM bi-couche avec Dropout.
    Architecture : LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1)
    lookback = 24h (une journée de contexte).
    """

    def __init__(
        self,
        units: int    = cfg.LSTM_UNITS,
        lookback: int = cfg.LSTM_LOOKBACK,
        epochs: int   = cfg.LSTM_EPOCHS,
        batch_size: int = cfg.LSTM_BATCH,
        dropout: float  = cfg.LSTM_DROPOUT,
    ):
        super().__init__()
        self.units      = units
        self.lookback   = lookback
        self.epochs     = epochs
        self.batch_size = batch_size
        self.dropout    = dropout
        self._model     = None
        self._scaler    = None
        self._train     = None

    @property
    def name(self) -> str:
        return f"LSTM ({self.units} neurones, lookback={self.lookback}h)"

    def fit(self, train: pd.Series) -> "LSTMModel":
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        self.log.info(f"Entraînement {self.name} sur {len(train)} observations...")
        self._train   = train
        self._scaler  = MinMaxScaler()
        scaled        = self._scaler.fit_transform(train.values.reshape(-1, 1))

        X, y = self._make_sequences(scaled)
        X    = X.reshape(X.shape[0], X.shape[1], 1)

        self._model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(self.dropout),
            LSTM(self.units),
            Dropout(self.dropout),
            Dense(1),
        ])
        self._model.compile(optimizer="adam", loss="mse")
        self._model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
        )
        self._fitted = True
        self.log.info(f"{self.name} entraîné ✓")
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        """Prédiction récursive à partir des dernières valeurs d'entraînement."""
        scaled  = self._scaler.transform(self._train.values.reshape(-1, 1))
        history = list(scaled[-self.lookback:, 0])
        preds   = []
        for _ in range(n_steps):
            x = np.array(history[-self.lookback:]).reshape(1, self.lookback, 1)
            p = float(self._model.predict(x, verbose=0)[0, 0])
            preds.append(p)
            history.append(p)
        return self._scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()

    def _make_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
