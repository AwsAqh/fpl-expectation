import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_model_and_metadata():
    """Load trained LightGBM model and feature metadata."""
    model_path = os.path.join(MODELS_DIR, "fpl_lgbm_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Trained ML model or metadata missing in models/ directory. Run train_model.py first.")
        
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    return model, metadata


def build_inference_features(elements_df, hist_df, fixtures_df, target_gw, bootstrap):
    """
    Construct the feature matrix X for active players for upcoming Gameweek `target_gw`.
    Blends historical rolling statistics with current season element stats.
    """
    df = elements_df.copy()
    
    # 1. Position encodings
    et_map = {}
    for et in bootstrap.get("element_types", []):
        name = et["singular_name_short"]
        et_map[et["id"]] = "GKP" if name in ["GK", "GKP"] else name
    df["position"] = df["element_type"].map(et_map).fillna("MID")
    
    pos_clean = {"GKP": "GKP", "GK": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
    df["pos_cat"] = df["position"].map(pos_clean).fillna("MID")
    
    df["is_gk"] = (df["pos_cat"] == "GKP").astype(int)
    df["is_def"] = (df["pos_cat"] == "DEF").astype(int)
    df["is_mid"] = (df["pos_cat"] == "MID").astype(int)
    df["is_fwd"] = (df["pos_cat"] == "FWD").astype(int)
    
    # Cost in millions
    df["cost_millions"] = df["now_cost"] / 10.0
    df["GW"] = float(target_gw)
    
    # 2. Fixture home/away and difficulty match context
    gw_fixtures = fixtures_df[fixtures_df["event"] == target_gw] if "event" in fixtures_df.columns else pd.DataFrame()
    team_home_map = {}
    if not gw_fixtures.empty:
        for _, f in gw_fixtures.iterrows():
            team_home_map[f["team_h"]] = 1
            team_home_map[f["team_a"]] = 0
            
    df["was_home"] = df["team"].map(team_home_map).fillna(1).astype(int)
    
    # 3. Match historical player stats to compute rolling averages
    # Create player lookup dictionary from historical datasets
    hist_stats = {}
    if hist_df is not None and not hist_df.empty:
        # Group by name/element to get last known rolling form
        hist_df_sorted = hist_df.sort_values(by=["season", "GW"])
        for name, group in hist_df_sorted.groupby("name"):
            last_10 = group.tail(10)
            last_5 = group.tail(5)
            last_3 = group.tail(3)
            last_1 = group.tail(1)
            
            hist_stats[name.lower().strip()] = {
                "minutes_lag1": float(last_1["minutes"].values[0]) if len(last_1) > 0 else 0.0,
                "minutes_roll3": float(last_3["minutes"].mean()) if len(last_3) > 0 else 0.0,
                "minutes_roll5": float(last_5["minutes"].mean()) if len(last_5) > 0 else 0.0,
                "minutes_roll10": float(last_10["minutes"].mean()) if len(last_10) > 0 else 0.0,
                "total_points_lag1": float(last_1["total_points"].values[0]) if len(last_1) > 0 else 0.0,
                "total_points_roll3": float(last_3["total_points"].mean()) if len(last_3) > 0 else 0.0,
                "total_points_roll5": float(last_5["total_points"].mean()) if len(last_5) > 0 else 0.0,
                "total_points_roll10": float(last_10["total_points"].mean()) if len(last_10) > 0 else 0.0,
                "expected_goals_lag1": float(last_1["expected_goals"].values[0]) if len(last_1) > 0 and "expected_goals" in last_1 else 0.0,
                "expected_goals_roll3": float(last_3["expected_goals"].mean()) if len(last_3) > 0 and "expected_goals" in last_3 else 0.0,
                "expected_goals_roll5": float(last_5["expected_goals"].mean()) if len(last_5) > 0 and "expected_goals" in last_5 else 0.0,
                "expected_assists_lag1": float(last_1["expected_assists"].values[0]) if len(last_1) > 0 and "expected_assists" in last_1 else 0.0,
                "expected_assists_roll3": float(last_3["expected_assists"].mean()) if len(last_3) > 0 and "expected_assists" in last_3 else 0.0,
                "expected_assists_roll5": float(last_5["expected_assists"].mean()) if len(last_5) > 0 and "expected_assists" in last_5 else 0.0,
                "bps_lag1": float(last_1["bps"].values[0]) if len(last_1) > 0 and "bps" in last_1 else 0.0,
                "bps_roll3": float(last_3["bps"].mean()) if len(last_3) > 0 and "bps" in last_3 else 0.0,
                "bps_roll5": float(last_5["bps"].mean()) if len(last_5) > 0 and "bps" in last_5 else 0.0,
                "ict_index_lag1": float(last_1["ict_index"].values[0]) if len(last_1) > 0 and "ict_index" in last_1 else 0.0,
                "ict_index_roll3": float(last_3["ict_index"].mean()) if len(last_3) > 0 and "ict_index" in last_3 else 0.0,
                "ict_index_roll5": float(last_5["ict_index"].mean()) if len(last_5) > 0 and "ict_index" in last_5 else 0.0,
                "influence_roll3": float(last_3["influence"].mean()) if len(last_3) > 0 and "influence" in last_3 else 0.0,
                "creativity_roll3": float(last_3["creativity"].mean()) if len(last_3) > 0 and "creativity" in last_3 else 0.0,
                "threat_roll3": float(last_3["threat"].mean()) if len(last_3) > 0 and "threat" in last_3 else 0.0,
                "xP_lag1": float(last_1["xP"].values[0]) if len(last_1) > 0 and "xP" in last_1 else 0.0,
                "xP_roll3": float(last_3["xP"].mean()) if len(last_3) > 0 and "xP" in last_3 else 0.0,
                "xP_roll5": float(last_5["xP"].mean()) if len(last_5) > 0 and "xP" in last_5 else 0.0,
            }

    # Initialize feature columns on df
    feature_defaults = {
        "minutes_lag1": 60.0, "minutes_roll3": 60.0, "minutes_roll5": 60.0, "minutes_roll10": 60.0,
        "total_points_lag1": 3.0, "total_points_roll3": 3.0, "total_points_roll5": 3.0, "total_points_roll10": 3.0,
        "expected_goals_lag1": 0.1, "expected_goals_roll3": 0.1, "expected_goals_roll5": 0.1,
        "expected_assists_lag1": 0.1, "expected_assists_roll3": 0.1, "expected_assists_roll5": 0.1,
        "bps_lag1": 10.0, "bps_roll3": 10.0, "bps_roll5": 10.0,
        "ict_index_lag1": 3.0, "ict_index_roll3": 3.0, "ict_index_roll5": 3.0,
        "influence_roll3": 15.0, "creativity_roll3": 15.0, "threat_roll3": 15.0,
        "xP_lag1": 3.0, "xP_roll3": 3.0, "xP_roll5": 3.0,
    }
    
    for col, default_val in feature_defaults.items():
        df[col] = default_val

    # Map matched stats per player
    summaries_dir = os.path.join(DATA_DIR, "summaries")
    for idx, row in df.iterrows():
        pid = row.get("id")
        web_name = row.get("web_name", "").lower().strip()
        first_name = row.get("first_name", "").lower().strip()
        second_name = row.get("second_name", "").lower().strip()
        full_name = f"{first_name} {second_name}".strip()

        # Check if current season match history exists in summary JSON
        summary_file = os.path.join(summaries_dir, f"{pid}.json") if pid else None
        current_season_history = []
        if summary_file and os.path.exists(summary_file):
            try:
                with open(summary_file, "r") as sf:
                    sdata = json.load(sf)
                    current_season_history = sdata.get("history", [])
            except Exception:
                current_season_history = []

        if current_season_history and len(current_season_history) > 0:
            # Player has played in current season! Calculate rolling stats from current season matches
            c_df = pd.DataFrame(current_season_history)
            for c_col in ["minutes", "total_points", "expected_goals", "expected_assists", "bps", "ict_index", "influence", "creativity", "threat"]:
                if c_col in c_df.columns:
                    c_df[c_col] = pd.to_numeric(c_df[c_col], errors="coerce").fillna(0.0)

            last_10 = c_df.tail(10)
            last_5 = c_df.tail(5)
            last_3 = c_df.tail(3)
            last_1 = c_df.tail(1)

            df.at[idx, "minutes_lag1"] = float(last_1["minutes"].values[0]) if len(last_1) > 0 else 0.0
            df.at[idx, "minutes_roll3"] = float(last_3["minutes"].mean()) if len(last_3) > 0 else 0.0
            df.at[idx, "minutes_roll5"] = float(last_5["minutes"].mean()) if len(last_5) > 0 else 0.0
            df.at[idx, "minutes_roll10"] = float(last_10["minutes"].mean()) if len(last_10) > 0 else 0.0

            df.at[idx, "total_points_lag1"] = float(last_1["total_points"].values[0]) if len(last_1) > 0 else 0.0
            df.at[idx, "total_points_roll3"] = float(last_3["total_points"].mean()) if len(last_3) > 0 else 0.0
            df.at[idx, "total_points_roll5"] = float(last_5["total_points"].mean()) if len(last_5) > 0 else 0.0
            df.at[idx, "total_points_roll10"] = float(last_10["total_points"].mean()) if len(last_10) > 0 else 0.0

            if "expected_goals" in c_df.columns:
                df.at[idx, "expected_goals_lag1"] = float(last_1["expected_goals"].values[0]) if len(last_1) > 0 else 0.0
                df.at[idx, "expected_goals_roll3"] = float(last_3["expected_goals"].mean()) if len(last_3) > 0 else 0.0
                df.at[idx, "expected_goals_roll5"] = float(last_5["expected_goals"].mean()) if len(last_5) > 0 else 0.0

            if "expected_assists" in c_df.columns:
                df.at[idx, "expected_assists_lag1"] = float(last_1["expected_assists"].values[0]) if len(last_1) > 0 else 0.0
                df.at[idx, "expected_assists_roll3"] = float(last_3["expected_assists"].mean()) if len(last_3) > 0 else 0.0
                df.at[idx, "expected_assists_roll5"] = float(last_5["expected_assists"].mean()) if len(last_5) > 0 else 0.0

            if "bps" in c_df.columns:
                df.at[idx, "bps_lag1"] = float(last_1["bps"].values[0]) if len(last_1) > 0 else 0.0
                df.at[idx, "bps_roll3"] = float(last_3["bps"].mean()) if len(last_3) > 0 else 0.0
                df.at[idx, "bps_roll5"] = float(last_5["bps"].mean()) if len(last_5) > 0 else 0.0

            if "ict_index" in c_df.columns:
                df.at[idx, "ict_index_lag1"] = float(last_1["ict_index"].values[0]) if len(last_1) > 0 else 0.0
                df.at[idx, "ict_index_roll3"] = float(last_3["ict_index"].mean()) if len(last_3) > 0 else 0.0
                df.at[idx, "ict_index_roll5"] = float(last_5["ict_index"].mean()) if len(last_5) > 0 else 0.0

            if "influence" in c_df.columns:
                df.at[idx, "influence_roll3"] = float(last_3["influence"].mean()) if len(last_3) > 0 else 0.0
            if "creativity" in c_df.columns:
                df.at[idx, "creativity_roll3"] = float(last_3["creativity"].mean()) if len(last_3) > 0 else 0.0
            if "threat" in c_df.columns:
                df.at[idx, "threat_roll3"] = float(last_3["threat"].mean()) if len(last_3) > 0 else 0.0

        else:
            # Fallback to historical past season stats
            matched_stat = None
            for candidate in [web_name, full_name, second_name]:
                if candidate and candidate in hist_stats:
                    matched_stat = hist_stats[candidate]
                    break
                    
            if matched_stat:
                for key, val in matched_stat.items():
                    df.at[idx, key] = val
            else:
                ep_next = pd.to_numeric(row.get("ep_next", 3.0), errors="coerce")
                if pd.isna(ep_next) or ep_next <= 0:
                    ep_next = 3.0
                df.at[idx, "xP_lag1"] = ep_next
                df.at[idx, "xP_roll3"] = ep_next
                df.at[idx, "xP_roll5"] = ep_next

    # Form trend ratios
    df["points_form_ratio"] = np.where(
        df["total_points_roll10"] > 0,
        df["total_points_roll3"] / (df["total_points_roll10"] + 1e-5),
        1.0
    )
    df["minutes_form_ratio"] = np.where(
        df["minutes_roll10"] > 0,
        df["minutes_roll3"] / (df["minutes_roll10"] + 1e-5),
        1.0
    )
    
    df["xg_per_90_roll5"] = np.where(
        df["minutes_roll5"] > 0,
        (df["expected_goals_roll5"] / df["minutes_roll5"]) * 90.0,
        0.0
    )
    df["xa_per_90_roll5"] = np.where(
        df["minutes_roll5"] > 0,
        (df["expected_assists_roll5"] / df["minutes_roll5"]) * 90.0,
        0.0
    )

    return df


def predict_points_with_ml(elements_df, hist_df, fixtures_df, target_gw, bootstrap):
    """
    Run LightGBM ML model predictions for all players.
    Returns elements_df enriched with 'predicted_points' and 'base_xp'.
    """
    model, metadata = load_model_and_metadata()
    feature_cols = metadata["feature_cols"]
    
    df = build_inference_features(elements_df, hist_df, fixtures_df, target_gw, bootstrap)
    
    # Extract feature matrix X matching exact feature columns used during training
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        
    # Generate ML predictions
    predictions = model.predict(X)
    df["predicted_points"] = predictions
    df["base_xp"] = df["xP_roll3"]
    
    return df
