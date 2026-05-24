import requests
import json

API_KEY    = "U8dmlGS4eUfOgOigAwuemyIftDEmRfKT"
BASE_URL   = "https://api.tomtom.com/traffic/trafficstats"
HEADERS    = {"Content-Type": "application/json"}

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

payload = {
    "jobName": "Bouznika_Test_Multiple_Dates",
    "distanceUnit": "KILOMETERS",
    "routes": [{
        "name":         "RN1_Bouznika",
        "start":        {"latitude": 33.790024, "longitude": -7.158405 - 0.001},
        "end":          {"latitude": 33.790024, "longitude": -7.158405 + 0.001},
        "fullTraversal": False,
        "zoneId":       "Africa/Casablanca",
    }],
    "dateRanges": [{
        "name": f"August2024_{day:02d}",
        "from": f"2024-08-{day:02d}",
        "to":   f"2024-08-{day:02d}",
    } for day in range(1, 32)],
    "timeSets": HOURLY_TIMESETS,
}

url = f"{BASE_URL}/routeanalysis/1?key={API_KEY}"
r = requests.post(url, json=payload, headers=HEADERS)
print(r.status_code)
print(r.json())
