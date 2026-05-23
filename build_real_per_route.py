"""
build_real_per_route.py
=======================
Reconstruit le fichier donnees_horaires_par_route_aout2024.csv
à partir des VRAIES données TomTom (tomtom_filtered.csv).

Données source :
- tomtom_filtered.csv contient les segments réels avec road_point
- La colonne timeSetId (2→25) correspond aux heures 00h→23h
- dateRangeId=1 couvre tout août 2024 (profil moyen mensuel)

Pour obtenir 744 points (31 jours × 24h), on réplique le profil
horaire réel de chaque route sur chaque jour du mois.
"""

import pandas as pd
import numpy as np

# ── Charger les données réelles TomTom ────────────────────────────
df = pd.read_csv("data/processed/tomtom_filtered.csv")

# ── Calculer la vitesse moyenne réelle par route et par heure ─────
# Moyenne sur tous les segments assignés à chaque road_point
profil = df.groupby(["road_point", "timeSetId"])["averageSpeed"].mean()
profil = profil.unstack("road_point")

# timeSetId 2→25 correspond aux heures 0→23
profil.index = profil.index - 2  # 0..23
profil.index.name = "hour"

print("=== PROFIL HORAIRE RÉEL (vitesses moyennes TomTom) ===")
print(profil.round(2))
print()

# ── Construire la série 744h pour chaque route ────────────────────
dates = pd.date_range("2024-08-01", "2024-08-31", freq="D")
rows = []

for day in dates:
    for hour in range(24):
        ts = day + pd.Timedelta(hours=hour)
        row = {"timestamp": ts}
        for rp in ["RN1", "RP3323", "RN11", "RN23"]:
            row[rp] = round(profil.loc[hour, rp], 2)
        rows.append(row)

df_out = pd.DataFrame(rows)

# ── Sauvegarder ───────────────────────────────────────────────────
output_path = "data/processed/donnees_reelles_par_route_aout2024.csv"
df_out.to_csv(output_path, index=False)

print(f"[OK] Fichier exporte : {output_path}")
print(f"   → {len(df_out)} lignes (31 jours × 24 heures)")
print(f"   → Colonnes : {list(df_out.columns)}")
print()
print("Aperçu (premières 48h) :")
print(df_out.head(48).to_string(index=False))
print()
print("Statistiques descriptives :")
print(df_out[["RN1", "RP3323", "RN11", "RN23"]].describe().round(2))
