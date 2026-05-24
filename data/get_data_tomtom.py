"""
tomtom_hourly.py
================
1. Télécharge les XLSX des 4 jobs existants (données agrégées)
2. Soumet 4 nouveaux jobs avec 24 timeSets horaires (00:00-01:00 ... 23:00-23:59)
3. Attend leur complétion et télécharge les XLSX horaires

Lancement : python tomtom_hourly.py
"""

import requests
import os
import time
import gzip
import zipfile
import io

API_KEY    = "U8dmlGS4eUfOgOigAwuemyIftDEmRfKT"
BASE_URL   = "https://api.tomtom.com/traffic/trafficstats"
HEADERS    = {"Content-Type": "application/json"}
OUTPUT_DIR = "tomtom_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DONE_STATUSES   = {"DONE", "COMPLETED", "SUCCESS"}
FAILED_STATUSES = {"FAILED", "CANCELLED", "ERROR"}

# Jobs existants (données agrégées AllDay)
EXISTING_JOBS = {
    "RP3323": "9254663",
    "RN1":    "9254664",
    "RN11":   "9254666",
    "RN23":   "9254667",
}

# Coordonnées des routes
ROUTES = [
    {
        "road_id": "RP3323",
        "name":    "RP3323_Bouznika",
        "start":   {"latitude": 33.790780 - 0.001, "longitude": -7.158819},
        "end":     {"latitude": 33.790780 + 0.001, "longitude": -7.158819},
    },
    {
        "road_id": "RN1",
        "name":    "RN1_Bouznika",
        "start":   {"latitude": 33.790024, "longitude": -7.158405 - 0.001},
        "end":     {"latitude": 33.790024, "longitude": -7.158405 + 0.001},
    },
    {
        "road_id": "RN11",
        "name":    "RN11_Bouznika",
        "start":   {"latitude": 33.789480 - 0.001, "longitude": -7.159704},
        "end":     {"latitude": 33.789480 + 0.001, "longitude": -7.159704},
    },
    {
        "road_id": "RN23",
        "name":    "RN23_Bouznika",
        "start":   {"latitude": 33.789620, "longitude": -7.158738 - 0.001},
        "end":     {"latitude": 33.789620, "longitude": -7.158738 + 0.001},
    },
]

# 24 timeSets horaires
HOURLY_TIMESETS = [
    {
        "name": f"Hour_{h:02d}",
        "timeGroups": [{
            "days": ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
            "times": [f"{h:02d}:00-{h:02d}:59"] if h < 23 else ["23:00-23:59"],
        }]
    }
    for h in range(24)
]


# ════════════════════════════════════════════════════════
# Utilitaires
# ════════════════════════════════════════════════════════

def get_job(job_id: str) -> dict:
    url = f"{BASE_URL}/routeanalysis/1/{job_id}?key={API_KEY}"
    return requests.get(url, headers=HEADERS).json()


def find_xlsx_url(results_urls) -> str | None:
    """Trouve l'URL xlsx dans resultsUrls (liste ou dict)."""
    if isinstance(results_urls, list):
        return next((u for u in results_urls if ".xlsx" in u), None)
    elif isinstance(results_urls, dict):
        return results_urls.get("xlsx")
    return None


def download_xlsx(url: str, path: str):
    """Télécharge un fichier xlsx (gzip ou zip si nécessaire)."""
    r = requests.get(url)

    # Dézipper si c'est un zip
    if url.endswith(".zip") or r.headers.get("Content-Type", "").startswith("application/zip"):
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for name in z.namelist():
            if name.endswith(".xlsx"):
                content = z.read(name)
                break
        else:
            content = r.content
    elif r.headers.get("Content-Encoding") == "gzip":
        content = gzip.decompress(r.content)
    else:
        content = r.content

    with open(path, "wb") as f:
        f.write(content)
    print(f"    ✅ {path}  ({len(content)/1024:.1f} KB)")


# ════════════════════════════════════════════════════════
# ÉTAPE 1 — Télécharger xlsx des jobs existants
# ════════════════════════════════════════════════════════

def download_existing_xlsx():
    print("\n" + "="*55)
    print("  ÉTAPE 1 — Téléchargement XLSX jobs existants")
    print("="*55)

    for road_id, job_id in EXISTING_JOBS.items():
        print(f"\n  {road_id} | jobId: {job_id}")
        data   = get_job(job_id)
        status = data.get("status", "UNKNOWN").upper()
        print(f"  Statut : {status}")

        if status != "DONE":
            print("  ⏳ Pas prêt.")
            continue

        results_urls = data.get("resultsUrls", [])
        xlsx_url     = find_xlsx_url(results_urls)

        if xlsx_url:
            path = os.path.join(OUTPUT_DIR, f"{road_id}_allday_august2024.xlsx")
            download_xlsx(xlsx_url, path)
        else:
            print(f"  ⚠️  Pas de xlsx trouvé. URLs : {results_urls}")


# ════════════════════════════════════════════════════════
# ÉTAPE 2 — Soumettre jobs horaires
# ════════════════════════════════════════════════════════

def submit_hourly_job(route: dict) -> str | None:
    url = f"{BASE_URL}/routeanalysis/1?key={API_KEY}"
    payload = {
        "jobName": f"Bouznika_{route['road_id']}_Hourly_Aout2024",
        "distanceUnit": "KILOMETERS",
        "routes": [{
            "name":         route["name"],
            "start":        route["start"],
            "end":          route["end"],
            "fullTraversal": False,
            "zoneId":       "Africa/Casablanca",
        }],
        "dateRanges": [{
            "name": "August2024",
            "from": "2024-08-01",
            "to":   "2024-08-31",
        }],
        "timeSets": HOURLY_TIMESETS,
    }

    r    = requests.post(url, json=payload, headers=HEADERS)
    body = r.json()
    job_id = body.get("jobId")

    if r.status_code in (200, 201, 202):
        print(f"  ✅ Job horaire soumis — {route['road_id']} | jobId: {job_id}")
        return job_id
    elif r.status_code == 400 and job_id:
        print(f"  ♻️  Job existant — {route['road_id']} | jobId: {job_id}")
        return job_id
    else:
        print(f"  ❌ Erreur {route['road_id']}: {r.status_code} — {r.text}")
        return None


def submit_all_hourly() -> dict:
    print("\n" + "="*55)
    print("  ÉTAPE 2 — Soumission jobs horaires (24 timeSets)")
    print("="*55)
    job_map = {}
    for route in ROUTES:
        job_id = submit_hourly_job(route)
        if job_id:
            job_map[route["road_id"]] = job_id
        time.sleep(1)
    print(f"\n  Jobs horaires : {job_map}")
    return job_map


# ════════════════════════════════════════════════════════
# ÉTAPE 3 — Attendre + télécharger xlsx horaires
# ════════════════════════════════════════════════════════

def wait_and_download_hourly(job_map: dict, poll_interval: int = 30):
    print("\n" + "="*55)
    print("  ÉTAPE 3 — Attente et téléchargement XLSX horaires")
    print("="*55)

    pending = dict(job_map)

    while pending:
        print()
        for road_id, job_id in list(pending.items()):
            data   = get_job(job_id)
            status = (data.get("status") or data.get("state") or "UNKNOWN").upper()
            print(f"  [{road_id}] jobId={job_id} | Statut: {status}")

            if status in DONE_STATUSES:
                results_urls = data.get("resultsUrls", [])
                xlsx_url     = find_xlsx_url(results_urls)
                if xlsx_url:
                    path = os.path.join(OUTPUT_DIR, f"{road_id}_hourly_august2024.xlsx")
                    download_xlsx(xlsx_url, path)
                else:
                    print(f"  ⚠️  Pas de xlsx pour {road_id}")
                del pending[road_id]

            elif status in FAILED_STATUSES:
                print(f"  ❌ {road_id} échoué")
                del pending[road_id]

        if pending:
            print(f"\n  ⏳ {len(pending)} job(s) en cours — prochain check dans {poll_interval}s...")
            time.sleep(poll_interval)

    print("\n" + "="*55)
    print(f"  ✅ Terminé ! Fichiers dans : {OUTPUT_DIR}/")
    print("="*55)


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*55)
    print("  BOUZNIKA — Données horaires TomTom | Août 2024")
    print("="*55)

    # 1. Télécharger xlsx agrégés (jobs existants)
    download_existing_xlsx()

    # 2. Soumettre jobs horaires
    hourly_job_map = submit_all_hourly()

    # 3. Attendre + télécharger xlsx horaires
    if hourly_job_map:
        wait_and_download_hourly(hourly_job_map, poll_interval=30)