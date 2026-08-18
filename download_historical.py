import requests
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

seasons = {
    "2023-24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv",
    "2022-23": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv",
}

for season_name, url in seasons.items():
    filename = f"fpl_historical_{season_name}.csv"
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"{season_name} already exists, skipping")
        continue
    print(f"Downloading {season_name} from {url}...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"  Saved {len(r.text):,} bytes to {path}")