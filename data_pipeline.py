import json
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_bootstrap():
    path = os.path.join(DATA_DIR, "bootstrap.json")
    with open(path, "r") as f:
        return json.load(f)


def load_fixtures():
    path = os.path.join(DATA_DIR, "fixtures.json")
    with open(path, "r") as f:
        return json.load(f)


def load_summaries():
    summaries_dir = os.path.join(DATA_DIR, "summaries")
    summaries = {}
    if os.path.exists(summaries_dir):
        for fname in os.listdir(summaries_dir):
            if fname.endswith(".json"):
                pid = int(fname.replace(".json", ""))
                with open(os.path.join(summaries_dir, fname), "r") as f:
                    summaries[pid] = json.load(f)
    return summaries


def build_elements_df(bootstrap):
    elements = bootstrap.get("elements", [])
    df = pd.DataFrame(elements)
    return df


def build_teams_df(bootstrap):
    teams = bootstrap.get("teams", [])
    df = pd.DataFrame(teams)
    return df


def build_fixtures_df(fixtures_data):
    df = pd.DataFrame(fixtures_data)
    return df


def build_player_fixtures(summaries):
    rows = []
    for pid, summary in summaries.items():
        fixtures = summary.get("fixtures", [])
        for f in fixtures:
            f["player_id"] = pid
            rows.append(f)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def build_player_history_past(summaries):
    rows = []
    for pid, summary in summaries.items():
        history_past = summary.get("history_past", [])
        for entry in history_past:
            entry["player_id"] = pid
            rows.append(entry)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def get_upcoming_fixtures(fixtures_df, current_gw):
    upcoming = fixtures_df[fixtures_df["event"] >= current_gw].copy()
    return upcoming


def get_current_gw(bootstrap):
    events = bootstrap.get("events", [])
    for e in events:
        if e.get("is_current"):
            return e["id"]
    for e in reversed(events):
        if not e.get("finished"):
            return e["id"]
    return events[-1]["id"] if events else 1


def prepare_training_data(elements_df, fixtures_df, player_fixtures_df,
                          history_past_df, teams_df, current_gw):
    feature_cols = [
        "id", "element_type", "team", "now_cost", "selected_by_percent",
        "total_points", "points_per_game", "minutes", "goals_scored",
        "assists", "clean_sheets", "goals_conceded", "own_goals",
        "penalties_saved", "penalties_missed", "yellow_cards", "red_cards",
        "saves", "bonus", "bps", "influence", "creativity", "threat",
        "ict_index", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded",
        "starts", "form", "value_form", "value_season",
        "transfers_in", "transfers_out", "transfers_in_event",
        "transfers_out_event", "cost_change_event", "cost_change_start",
        "dreamteam_count",
    ]

    available_cols = [c for c in feature_cols if c in elements_df.columns]
    # Always include name columns for display
    for name_col in ["first_name", "second_name", "web_name"]:
        if name_col in elements_df.columns and name_col not in available_cols:
            available_cols.append(name_col)
    players_df = elements_df[available_cols].copy()

    team_map = dict(zip(teams_df["id"], teams_df["name"]))
    players_df["team_name"] = players_df["team"].map(team_map)

    et_map = {}
    bootstrap = load_bootstrap()
    for et in bootstrap.get("element_types", []):
        et_map[et["id"]] = et["singular_name_short"]
    players_df["position"] = players_df["element_type"].map(et_map)

    # Add upcoming fixture difficulty for each player's team (GW-specific)
    if not player_fixtures_df.empty:
        upcoming_pf = player_fixtures_df[
            player_fixtures_df["event"] == current_gw
        ]
        if not upcoming_pf.empty:
            diff_by_player = upcoming_pf.groupby("player_id").agg(
                upcoming_avg_difficulty=("difficulty", "mean"),
                upcoming_max_difficulty=("difficulty", "max"),
                upcoming_min_difficulty=("difficulty", "min"),
                upcoming_fixtures_count=("difficulty", "count"),
                upcoming_home_count=("is_home", "sum"),
            ).reset_index()
            diff_by_player.rename(columns={"player_id": "id"}, inplace=True)
            players_df = players_df.merge(
                diff_by_player, on="id", how="left"
            )
        else:
            # No GW-specific data, use the first upcoming fixture
            first_upcoming = player_fixtures_df.groupby("player_id").first().reset_index()
            if not first_upcoming.empty:
                diff_by_player = first_upcoming[["player_id", "difficulty", "is_home"]].copy()
                diff_by_player.rename(columns={
                    "player_id": "id",
                    "difficulty": "upcoming_avg_difficulty",
                    "is_home": "upcoming_home_count",
                }, inplace=True)
                diff_by_player["upcoming_max_difficulty"] = diff_by_player["upcoming_avg_difficulty"]
                diff_by_player["upcoming_min_difficulty"] = diff_by_player["upcoming_avg_difficulty"]
                diff_by_player["upcoming_fixtures_count"] = 1
                players_df = players_df.merge(diff_by_player, on="id", how="left")

    # Add historical season averages from history_past
    if not history_past_df.empty:
        hist_agg = history_past_df.groupby("player_id").agg(
            hist_past_avg_points=("total_points", "mean"),
            hist_past_avg_minutes=("minutes", "mean"),
            hist_past_avg_goals=("goals_scored", "mean"),
            hist_past_avg_assists=("assists", "mean"),
            hist_past_avg_clean_sheets=("clean_sheets", "mean"),
            hist_past_avg_bonus=("bonus", "mean"),
            hist_past_total_points=("total_points", "sum"),
            hist_past_seasons=("season_name", "nunique"),
        ).reset_index()
        hist_agg.rename(columns={"player_id": "id"}, inplace=True)
        players_df = players_df.merge(hist_agg, on="id", how="left")

    # Add current season stats from summary fixtures (finished GWs)
    if not player_fixtures_df.empty:
        finished_pf = player_fixtures_df[
            player_fixtures_df["event"] < current_gw
        ]
        if not finished_pf.empty:
            pf_agg = finished_pf.groupby("player_id").agg(
                pf_avg_difficulty=("difficulty", "mean"),
                pf_total_minutes=("minutes", "sum"),
                pf_gws_played=("minutes", lambda x: (x > 0).sum()),
            ).reset_index()
            pf_agg.rename(columns={"player_id": "id"}, inplace=True)
            players_df = players_df.merge(pf_agg, on="id", how="left")

    players_df.fillna(0, inplace=True)
    return players_df


def main():
    bootstrap = load_bootstrap()
    fixtures_data = load_fixtures()
    summaries = load_summaries()

    elements_df = build_elements_df(bootstrap)
    teams_df = build_teams_df(bootstrap)
    fixtures_df = build_fixtures_df(fixtures_data)
    player_fixtures_df = build_player_fixtures(summaries)
    history_past_df = build_player_history_past(summaries)

    current_gw = get_current_gw(bootstrap)
    print(f"Current gameweek: {current_gw}")
    print(f"Players: {len(elements_df)}")
    print(f"Fixtures: {len(fixtures_df)}")
    print(f"Player fixtures in summaries: {len(player_fixtures_df)}")
    print(f"History past records: {len(history_past_df)}")

    training_data = prepare_training_data(
        elements_df, fixtures_df, player_fixtures_df,
        history_past_df, teams_df, current_gw
    )
    print(f"Training data shape: {training_data.shape}")
    print(f"Columns: {list(training_data.columns)}")

    return training_data, elements_df, fixtures_df, teams_df, current_gw


if __name__ == "__main__":
    main()