import json
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests

from data_pipeline import (
    build_elements_df, build_teams_df, build_fixtures_df,
)
from optimizer import select_best_xi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_json(url, filename):
    path = os.path.join(DATA_DIR, filename)
    print(f"  Fetching {url}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def fetch_latest_data(max_age_hours=12):
    print("\n  Fetching latest data from FPL API...")
    bootstrap = fetch_json(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        "bootstrap.json"
    )
    fixtures = fetch_json(
        "https://fantasy.premierleague.com/api/fixtures/",
        "fixtures.json"
    )
    summaries_dir = os.path.join(DATA_DIR, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)

    elements = bootstrap.get("elements", [])
    fetched = 0
    import time
    now = time.time()
    max_age_sec = max_age_hours * 3600

    for elem in elements:
        pid = elem["id"]
        summary_path = os.path.join(summaries_dir, f"{pid}.json")
        is_fresh = os.path.exists(summary_path) and (now - os.path.getmtime(summary_path) < max_age_sec)
        
        if is_fresh:
            fetched += 1
            continue
        try:
            r = requests.get(
                f"https://fantasy.premierleague.com/api/element-summary/{pid}/",
                timeout=15
            )
            if r.status_code == 200:
                with open(summary_path, "w") as f:
                    json.dump(r.json(), f)
        except Exception:
            pass
        if pid % 100 == 0:
            print(f"    Summaries updated: {pid}/{len(elements)}")

    print(f"  Done. {fetched}/{len(elements)} summaries cached & fresh.")
    return bootstrap, fixtures


def load_historical_data(data_dir):
    frames = []
    for season in ["2022-23", "2023-24"]:
        path = os.path.join(data_dir, f"fpl_historical_{season}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["season"] = season
            frames.append(df)
            print(f"  Loaded {season}: {len(df)} rows")

    if not frames:
        return None

    hist = pd.concat(frames, ignore_index=True)
    print(f"  Total historical rows: {len(hist)}")
    return hist


def filter_injured_players(df, bootstrap):
    df = df.copy()
    elements = bootstrap.get("elements", [])

    injury_map = {}
    for elem in elements:
        pid = elem["id"]
        news = elem.get("news", "")
        status = elem.get("status", "a")

        injury_info = {"news": news, "status": status}

        if not news:
            injury_info["severity"] = 0
        elif "expected back" in news.lower() or "return" in news.lower():
            injury_info["severity"] = 1
        elif "75%" in news or "chance of playing" in news.lower():
            injury_info["severity"] = 2
        elif "unknown" in news.lower() or "out" in news.lower():
            injury_info["severity"] = 3
        else:
            injury_info["severity"] = 1

        injury_map[pid] = injury_info

    df["injury_status"] = df["id"].map(lambda pid: injury_map.get(pid, {}).get("severity", 0))
    df["injury_news"] = df["id"].map(lambda pid: injury_map.get(pid, {}).get("news", ""))

    return df


def compute_xp_predictions(hist_df, elements_df, teams_df, fixtures_df,
                            target_gw, bootstrap):
    """
    Compute xP predictions using trained Machine Learning (LightGBM) model,
    supplemented by current injury/news status from FPL API.
    """
    from predict_ml import predict_points_with_ml

    team_map = dict(zip(teams_df["id"], teams_df["name"]))

    # Generate ML predictions for all players
    df = predict_points_with_ml(
        elements_df, hist_df, fixtures_df, target_gw, bootstrap
    )
    df["team_name"] = df["team"].map(team_map)

    # Map GW1 opponent and difficulty for UI display
    gw1_fixtures = fixtures_df[fixtures_df["event"] == target_gw] if "event" in fixtures_df.columns else pd.DataFrame()
    if not gw1_fixtures.empty:
        team_diff = {}
        team_home = {}
        team_opponent = {}
        for _, row in gw1_fixtures.iterrows():
            team_diff[row["team_h"]] = row["team_h_difficulty"]
            team_diff[row["team_a"]] = row["team_a_difficulty"]
            team_home[row["team_h"]] = 1.0
            team_home[row["team_a"]] = 0.0
            team_opponent[row["team_h"]] = row["team_a"]
            team_opponent[row["team_a"]] = row["team_h"]

        df["gw1_difficulty"] = df["team"].map(team_diff).fillna(3.0)
        df["gw1_home"] = df["team"].map(team_home).fillna(0.5)
        df["gw1_opponent"] = df["team"].map(team_opponent)
    else:
        df["gw1_difficulty"] = 3.0
        df["gw1_home"] = 0.5
        df["gw1_opponent"] = ""

    df["gw1_opponent"] = df["gw1_opponent"].map(team_map).fillna("")

    # Apply injury penalty filter from bootstrap API
    df = filter_injured_players(df, bootstrap)
    injury_penalty = {0: 1.0, 1: 0.9, 2: 0.7, 3: 0.3}
    df["injury_penalty"] = df["injury_status"].map(injury_penalty).fillna(1.0)
    df["predicted_points"] = df["predicted_points"] * df["injury_penalty"]
    df["predicted_points"] = df["predicted_points"].clip(lower=0, upper=30)

    return df


def safe_str(s):
    return s.encode("ascii", errors="replace").decode("ascii")


def main():
    print("=" * 60)
    print("  FPL Gameweek Predictor - ML System")
    print("  New Season 2026-27 | GW1 Prediction")
    print("=" * 60)

    print("\n[1/4] Fetching latest data from FPL API...")
    bootstrap, fixtures_data = fetch_latest_data()

    elements_df = build_elements_df(bootstrap)
    teams_df = build_teams_df(bootstrap)
    fixtures_df = build_fixtures_df(fixtures_data)

    target_gw = 1
    print(f"  Predicting for GW{target_gw} of 2026-27 season")
    print(f"  Players: {len(elements_df)}")
    print(f"  Fixtures: {len(fixtures_df)}")

    print("\n[2/4] Loading historical data (2 seasons)...")
    hist_df = load_historical_data(DATA_DIR)

    print("\n[3/4] Computing xP-based predictions (data-driven)...")
    full_df = compute_xp_predictions(
        hist_df, elements_df, teams_df, fixtures_df, target_gw, bootstrap
    )

    # Show injury news for top predicted players
    injured = full_df[full_df["injury_status"] > 0].nlargest(10, "predicted_points")
    if not injured.empty:
        print(f"\n  Injured/Doubtful Players (top affected):")
        for _, row in injured.iterrows():
            name = safe_str(f"{row.get('first_name', '')} {row.get('second_name', '')}".strip())
            news = safe_str(row.get("injury_news", ""))
            severity = row.get("injury_status", 0)
            sev_labels = {0: "OK", 1: "Minor", 2: "Doubtful", 3: "Out"}
            print(f"    {name:<28} {sev_labels.get(severity, '?'):<8} {news[:60]}")

    top_players = full_df.nlargest(20, "predicted_points")
    print(f"\n  Top 20 Predicted Players for GW{target_gw}:")
    print(f"  {'Rank':<5} {'Name':<28} {'Pos':<4} {'Cost':>5} {'Pred Pts':>9} {'xP':>5} {'Diff':>4} {'Injury':>6}")
    print("  " + "-" * 75)
    for rank, (_, row) in enumerate(top_players.iterrows(), 1):
        name = safe_str(f"{row.get('first_name', '')} {row.get('second_name', '')}".strip())
        xp_val = row.get("base_xp", 0)
        if pd.isna(xp_val):
            xp_val = 0
        diff = row.get("gw1_difficulty", 3.0)
        injury = row.get("injury_status", 0)
        injury_label = {0: "", 1: "Minor", 2: "Doubt", 3: "OUT"}
        print(f"  {rank:<5} {name:<28} {row.get('position', ''):<4} "
              f"{row['now_cost']:>5} "
              f"{row['predicted_points']:>9.1f} {xp_val:>5.2f} {diff:>4.1f} "
              f"{injury_label.get(injury, ''):>6}")

    # Top 11 regardless of price
    top11 = full_df.nlargest(11, "predicted_points")
    print(f"\n  Top 11 Predicted Scorers (regardless of price):")
    print(f"  {'Rank':<5} {'Name':<28} {'Pos':<4} {'Cost':>5} {'Pred Pts':>9} {'xP':>5} {'Diff':>4}")
    print("  " + "-" * 65)
    for rank, (_, row) in enumerate(top11.iterrows(), 1):
        name = safe_str(f"{row.get('first_name', '')} {row.get('second_name', '')}".strip())
        xp_val = row.get("base_xp", 0)
        if pd.isna(xp_val):
            xp_val = 0
        diff = row.get("gw1_difficulty", 3.0)
        print(f"  {rank:<5} {name:<28} {row.get('position', ''):<4} "
              f"{row['now_cost']:>5} "
              f"{row['predicted_points']:>9.1f} {xp_val:>5.2f} {diff:>4.1f}")

    # Save all players xP predictions
    all_players_path = os.path.join(OUTPUT_DIR, f"all_players_gw{target_gw}.csv")
    full_df[["id", "first_name", "second_name", "web_name", "position",
             "team_name", "now_cost", "predicted_points", "base_xp",
             "ep_next", "gw1_difficulty", "gw1_home", "gw1_opponent",
             "injury_status", "injury_news"]].to_csv(
        all_players_path, index=False
    )
    print(f"\n  All players xP saved to: {all_players_path}")

    print(f"\n  Selecting Best XI for GW{target_gw} (max 3 per team)...")
    lineup = select_best_xi(full_df, budget=1000.0, max_per_team=3)

    if lineup and lineup["lineup"]:
        print(f"\n  === BEST XI FOR GW{target_gw} ===")
        print(f"  Total Cost: {lineup['total_cost']} | "
              f"Predicted Points: {lineup['total_predicted_points']:.1f} | "
              f"Budget Remaining: {lineup['budget_remaining']:.1f}")
        print(f"  {'Pos':<4} {'Name':<28} {'Cost':>5} {'Pred Pts':>9} {'xP':>5} {'Diff':>4}")
        print("  " + "-" * 75)
        for player in lineup["lineup"]:
            pname = safe_str(player['name'])
            diff = player.get('fixture_difficulty', 3.0)
            xp_val = player.get('base_xp', 0) or 0
            print(f"  {player['position']:<4} {pname:<28} "
                  f"{player['cost']:>5} "
                  f"{player['predicted_points']:>9.1f} {xp_val:>5.2f} {diff:>4.1f}")

        lineup_path = os.path.join(OUTPUT_DIR, f"lineup_gw{target_gw}.json")
        with open(lineup_path, "w") as f:
            json.dump(lineup, f, indent=2)
        print(f"\n  Lineup saved to: {lineup_path}")

    else:
        print("  ERROR: Could not generate a valid lineup.")
        return None

    predictions_path = os.path.join(OUTPUT_DIR, f"predictions_gw{target_gw}.csv")
    full_df[["id", "position", "team_name",
             "now_cost", "predicted_points", "base_xp",
             "gw1_difficulty", "gw1_home", "gw1_opponent",
             "injury_status", "injury_news"]].to_csv(
        predictions_path, index=False
    )
    print(f"  Full predictions saved to: {predictions_path}")

    print("\n" + "=" * 60)
    print("  Prediction complete!")
    print("=" * 60)

    return lineup, full_df


if __name__ == "__main__":
    main()