import requests
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_json(url, filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    print(f"Fetching {url}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved {filename} ({len(data) if isinstance(data, list) else 'dict'})")
    return data

def main():
    print("=== Fetching FPL Data ===")

    bootstrap = fetch_json(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        "bootstrap.json"
    )

    fixtures = fetch_json(
        "https://fantasy.premierleague.com/api/fixtures/",
        "fixtures.json"
    )

    elements = bootstrap.get("elements", [])
    teams = bootstrap.get("teams", [])
    element_types = bootstrap.get("element_types", [])
    events = bootstrap.get("events", [])

    print(f"\nElements: {len(elements)}")
    print(f"Teams: {len(teams)}")
    print(f"Fixtures: {len(fixtures)}")
    print(f"Gameweeks: {len(events)}")

    current_gw = None
    for e in events:
        if e.get("is_current"):
            current_gw = e
            break
    if current_gw is None:
        for e in reversed(events):
            if not e.get("finished") and not e.get("is_previous"):
                current_gw = e
                break
    if current_gw is None and events:
        current_gw = events[-1]

    if current_gw:
        print(f"Current/next gameweek: GW{current_gw['id']}")
        print(f"  Deadline: {current_gw.get('deadline_time', 'N/A')}")
        print(f"  Finished: {current_gw.get('finished')}")

    gw_number = current_gw['id'] if current_gw else 1
    print(f"\nFetching per-player summary data for {len(elements)} players...")

    summaries_dir = os.path.join(DATA_DIR, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)

    fetched = 0
    failed = 0
    for elem in elements:
        pid = elem['id']
        summary_path = os.path.join(summaries_dir, f"{pid}.json")
        if os.path.exists(summary_path):
            fetched += 1
            continue

        url = f"https://fantasy.premierleague.com/api/element-summary/{pid}/"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(summary_path, "w") as f:
                    json.dump(r.json(), f)
                fetched += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1

        if (pid) % 50 == 0:
            print(f"  Progress: {pid}/{len(elements)} (fetched: {fetched}, failed: {failed})")
            time.sleep(0.5)

    print(f"\nSummaries: {fetched} fetched, {failed} failed")

    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({
            "current_gw": gw_number,
            "num_elements": len(elements),
            "num_fixtures": len(fixtures),
            "num_teams": len(teams),
        }, f, indent=2)

    print("\nAll data saved to", DATA_DIR)

if __name__ == "__main__":
    main()