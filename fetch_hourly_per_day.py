import requests
import os
import time
import gzip
import zipfile
import io
import sys

# Ensure stdout uses utf-8 (fixes charmap error)
sys.stdout.reconfigure(encoding='utf-8')

API_KEY    = "U8dmlGS4eUfOgOigAwuemyIftDEmRfKT"
BASE_URL   = "https://api.tomtom.com/traffic/trafficstats"
HEADERS    = {"Content-Type": "application/json"}
OUTPUT_DIR = "tomtom_data_daily"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DONE_STATUSES   = {"DONE", "COMPLETED", "SUCCESS"}
FAILED_STATUSES = {"FAILED", "CANCELLED", "ERROR"}

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

DATE_RANGES_PART1 = [{
    "name": f"Aug_{day:02d}",
    "from": f"2024-08-{day:02d}",
    "to":   f"2024-08-{day:02d}",
} for day in range(1, 16)]

DATE_RANGES_PART2 = [{
    "name": f"Aug_{day:02d}",
    "from": f"2024-08-{day:02d}",
    "to":   f"2024-08-{day:02d}",
} for day in range(16, 32)]


def get_job(job_id: str) -> dict:
    url = f"{BASE_URL}/routeanalysis/1/{job_id}?key={API_KEY}"
    return requests.get(url, headers=HEADERS).json()

def find_xlsx_url(results_urls) -> str | None:
    if isinstance(results_urls, list):
        return next((u for u in results_urls if ".xlsx" in u), None)
    elif isinstance(results_urls, dict):
        return results_urls.get("xlsx")
    return None

def download_xlsx(url: str, path: str):
    r = requests.get(url)
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
    print(f"    [OK] {path}  ({len(content)/1024:.1f} KB)")


def submit_job(route: dict, date_ranges: list, part_name: str) -> str | None:
    url = f"{BASE_URL}/routeanalysis/1?key={API_KEY}"
    payload = {
        "jobName": f"Bouznika_{route['road_id']}_Daily_Aout2024_{part_name}",
        "distanceUnit": "KILOMETERS",
        "routes": [{
            "name":         route["name"],
            "start":        route["start"],
            "end":          route["end"],
            "fullTraversal": False,
            "zoneId":       "Africa/Casablanca",
        }],
        "dateRanges": date_ranges,
        "timeSets": HOURLY_TIMESETS,
    }

    r = requests.post(url, json=payload, headers=HEADERS)
    body = r.json()
    job_id = body.get("jobId")

    if r.status_code in (200, 201, 202):
        print(f"  [OK] Job {part_name} soumis - {route['road_id']} | jobId: {job_id}")
        return job_id
    elif r.status_code == 400 and job_id:
        print(f"  [REUSE] Job {part_name} existant - {route['road_id']} | jobId: {job_id}")
        return job_id
    else:
        print(f"  [ERROR] Erreur {route['road_id']} {part_name}: {r.status_code} - {r.text}")
        return None


if __name__ == "__main__":
    job_map = {}
    
    print("Soumission des jobs...")
    for route in ROUTES:
        jid1 = submit_job(route, DATE_RANGES_PART1, "part1")
        if jid1: job_map[f"{route['road_id']}_part1"] = jid1
        time.sleep(2)
        
        jid2 = submit_job(route, DATE_RANGES_PART2, "part2")
        if jid2: job_map[f"{route['road_id']}_part2"] = jid2
        time.sleep(2)
        
    print("\nAttente des jobs...")
    pending = dict(job_map)
    while pending:
        for key, job_id in list(pending.items()):
            data = get_job(job_id)
            status = (data.get("status") or data.get("state") or "UNKNOWN").upper()
            print(f"  [{key}] jobId={job_id} | Statut: {status}")

            if status in DONE_STATUSES:
                results_urls = data.get("resultsUrls", [])
                xlsx_url = find_xlsx_url(results_urls)
                if xlsx_url:
                    path = os.path.join(OUTPUT_DIR, f"{key}.xlsx")
                    download_xlsx(xlsx_url, path)
                else:
                    print(f"  [WARN] Pas de xlsx pour {key}")
                del pending[key]
            elif status in FAILED_STATUSES:
                print(f"  [FAIL] {key} echoue")
                del pending[key]

        if pending:
            time.sleep(30)

    print("\nTermine ! Fichiers recupere dans tomtom_data_daily/")
