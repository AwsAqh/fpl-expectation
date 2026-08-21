import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_model_and_metadata():
    """Load trained Stacking Ensemble (or LightGBM fallback) and feature metadata."""
    ensemble_path = os.path.join(MODELS_DIR, "fpl_ensemble_models.joblib")
    lgb_path = os.path.join(MODELS_DIR, "fpl_lgbm_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Trained metadata missing in models/ directory. Run train_model.py first.")
        
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    if os.path.exists(ensemble_path):
        model_obj = joblib.load(ensemble_path)
    elif os.path.exists(lgb_path):
        model_obj = joblib.load(lgb_path)
    else:
        raise FileNotFoundError("No trained model found in models/ directory.")
        
    return model_obj, metadata


def build_inference_features(elements_df, hist_df, fixtures_df, target_gw, bootstrap):
    """
    Construct the feature matrix X for active players for upcoming Gameweek `target_gw`.
    Blends historical rolling statistics with current season element stats.
    Includes opponent/fixture features and market signals to match the training pipeline.
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
    team_opponent_map = {}
    if not gw_fixtures.empty:
        for _, f in gw_fixtures.iterrows():
            team_home_map[f["team_h"]] = 1
            team_home_map[f["team_a"]] = 0
            team_opponent_map[f["team_h"]] = f["team_a"]
            team_opponent_map[f["team_a"]] = f["team_h"]
            
    df["was_home"] = df["team"].map(team_home_map).fillna(1).astype(int)
    df["opponent_team_id"] = df["team"].map(team_opponent_map)
    
    # ========== OPPONENT / FIXTURE FEATURES (NEW) ==========
    # Build team strength lookup from bootstrap teams data
    teams_data = bootstrap.get("teams", [])
    teams_df_bootstrap = pd.DataFrame(teams_data)
    
    # Normalize team strengths to match training scale:
    # Training uses rolling goals conceded/scored (roughly 0-3 range per GW)
    # Bootstrap has strength values (roughly 1000-1400 range)
    # We normalize bootstrap strengths to a 0-3 scale to roughly align
    team_strength = {}
    for _, t in teams_df_bootstrap.iterrows():
        tid = t["id"]
        # Compute normalized attack/defence strength (scale ~1000-1400 → ~0-3)
        strength = t.get("strength") or 3
        s_attack_h = t.get("strength_attack_home") or 1200
        s_attack_a = t.get("strength_attack_away") or 1200
        s_def_h = t.get("strength_defence_home") or 1200
        s_def_a = t.get("strength_defence_away") or 1200
        
        # Normalize: higher strength → higher value, range roughly 0.5-3.0
        attack_avg = (s_attack_h + s_attack_a) / 2.0
        def_avg = (s_def_h + s_def_a) / 2.0
        
        team_strength[tid] = {
            "attack_norm": attack_avg / 800.0,   # ~1.2-1.8 range
            "defence_norm": def_avg / 800.0,      # ~1.2-1.8 range
            "overall": strength / 3.0,             # ~1-2 range
        }
    
    # For each player, create opponent & own team features
    opp_attack_list = []
    opp_conceded_list = []
    opp_cs_rate_list = []
    opp_xg_list = []
    opp_xgc_list = []
    own_attack_list = []
    own_defence_list = []
    own_cs_list = []
    
    # Also try to compute from current season summaries if available
    # Aggregate team-level stats from all player summaries for the current season
    summaries_dir = os.path.join(DATA_DIR, "summaries")
    team_season_stats = {}  # team_id -> {goals_scored, goals_conceded, clean_sheets, ...}
    
    for elem in bootstrap.get("elements", []):
        pid = elem["id"]
        team_id = elem["team"]
        summary_file = os.path.join(summaries_dir, f"{pid}.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r") as sf:
                    sdata = json.load(sf)
                    history = sdata.get("history", [])
                    if history:
                        if team_id not in team_season_stats:
                            team_season_stats[team_id] = {"goals": [], "conceded": [], "cs": [], "xg": [], "xgc": []}
                        for h in history[-5:]:  # last 5 GWs
                            team_season_stats[team_id]["goals"].append(float(h.get("goals_scored", 0)))
                            team_season_stats[team_id]["conceded"].append(float(h.get("goals_conceded", 0)))
                            team_season_stats[team_id]["cs"].append(float(h.get("clean_sheets", 0)))
                            team_season_stats[team_id]["xg"].append(float(h.get("expected_goals", 0) or 0))
                            team_season_stats[team_id]["xgc"].append(float(h.get("expected_goals_conceded", 0) or 0))
            except Exception:
                pass
    
    # Compute team averages from current season data
    team_avg_stats = {}
    for tid, stats in team_season_stats.items():
        team_avg_stats[tid] = {
            "avg_goals": np.mean(stats["goals"]) if stats["goals"] else 1.0,
            "avg_conceded": np.mean(stats["conceded"]) if stats["conceded"] else 1.0,
            "avg_cs": np.mean(stats["cs"]) if stats["cs"] else 0.3,
            "avg_xg": np.mean(stats["xg"]) if stats["xg"] else 0.5,
            "avg_xgc": np.mean(stats["xgc"]) if stats["xgc"] else 0.5,
        }
    
    for idx, row in df.iterrows():
        opp_id = row.get("opponent_team_id")
        own_id = row.get("team")
        
        # Opponent features: how strong is the opponent?
        if pd.notna(opp_id) and int(opp_id) in team_avg_stats:
            opp_stats = team_avg_stats[int(opp_id)]
            opp_attack_list.append(opp_stats["avg_goals"])       # opp_attack_strength
            opp_conceded_list.append(opp_stats["avg_conceded"])  # opp_goals_conceded_roll5
            opp_cs_rate_list.append(opp_stats["avg_cs"])         # opp_cs_rate_roll5
            opp_xg_list.append(opp_stats["avg_xg"])              # opp_xg_roll5
            opp_xgc_list.append(opp_stats["avg_xgc"])            # opp_xgc_roll5
        elif pd.notna(opp_id) and int(opp_id) in team_strength:
            # Fallback to bootstrap strength if no season data
            ts = team_strength[int(opp_id)]
            opp_attack_list.append(ts["attack_norm"])
            opp_conceded_list.append(ts["defence_norm"])
            opp_cs_rate_list.append(0.3)
            opp_xg_list.append(ts["attack_norm"] * 0.5)
            opp_xgc_list.append(ts["defence_norm"] * 0.5)
        else:
            opp_attack_list.append(1.0)
            opp_conceded_list.append(1.0)
            opp_cs_rate_list.append(0.3)
            opp_xg_list.append(0.5)
            opp_xgc_list.append(0.5)
        
        # Own team features
        if own_id in team_avg_stats:
            own_stats = team_avg_stats[own_id]
            own_attack_list.append(own_stats["avg_goals"])
            own_defence_list.append(own_stats["avg_conceded"])
            own_cs_list.append(own_stats["avg_cs"])
        elif own_id in team_strength:
            ts = team_strength[own_id]
            own_attack_list.append(ts["attack_norm"])
            own_defence_list.append(ts["defence_norm"])
            own_cs_list.append(0.3)
        else:
            own_attack_list.append(1.0)
            own_defence_list.append(1.0)
            own_cs_list.append(0.3)
    
    df["opp_attack_strength"] = opp_attack_list
    df["opp_goals_conceded_roll5"] = opp_conceded_list
    df["opp_cs_rate_roll5"] = opp_cs_rate_list
    df["opp_xg_roll5"] = opp_xg_list
    df["opp_xgc_roll5"] = opp_xgc_list
    df["own_team_attack_roll5"] = own_attack_list
    df["own_team_defence_roll5"] = own_defence_list
    df["own_team_cs_rate_roll5"] = own_cs_list
    
    # ========== MARKET / CROWD WISDOM FEATURES (NEW) ==========
    df["selected_log"] = np.log1p(pd.to_numeric(df["selected_by_percent"].fillna(0), errors="coerce").fillna(0) * 100000)
    df["transfers_balance_norm"] = (
        pd.to_numeric(df.get("transfers_in_event", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        - pd.to_numeric(df.get("transfers_out_event", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    ) / 10000.0
    
    # 3. Match historical player stats to compute rolling averages
    hist_stats = {}
    if hist_df is not None and not hist_df.empty:
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
    
    # ========== DefCon Features in Inference ==========
    non_offensive_bps = np.maximum(0.0, df.get("bps_roll5", 0.0) - (df.get("goals_scored_roll5", 0.0) * 24.0 + df.get("expected_assists_roll5", 0.0) * 9.0))
    df["defcon_est_roll5"] = non_offensive_bps * 0.65
    
    defcon_bonus = np.zeros(len(df))
    def_mask = df["pos_cat"] == "DEF"
    mid_mask = df["pos_cat"] == "MID"
    gk_mask  = df["pos_cat"] == "GKP"
    
    defcon_bonus[def_mask] = 4.0 * (df.loc[def_mask, "defcon_est_roll5"] / 10.0)
    defcon_bonus[mid_mask] = 2.0 * (df.loc[mid_mask, "defcon_est_roll5"] / 12.0)
    if "saves_roll5" in df.columns:
        defcon_bonus[gk_mask] = 1.0 * (df.loc[gk_mask, "saves_roll5"] / 3.0)
        
    df["defcon_points_bonus"] = defcon_bonus
    
    own_team_xgc = df.get("own_team_xgc_roll5", 1.2)
    df["xcs_prob"] = np.exp(-np.clip(own_team_xgc, 0.0, 5.0))
    
    df["gc_per_90_roll5"] = np.where(
        df["minutes_roll5"] > 0,
        (df.get("goals_conceded_roll5", 0.0) / df["minutes_roll5"]) * 90.0,
        0.0
    )
    df["saves_per_90_roll5"] = np.where(
        df["minutes_roll5"] > 0,
        (df.get("saves_roll5", 0.0) / df["minutes_roll5"]) * 90.0,
        0.0
    )

    return df


def predict_points_with_ml(elements_df, hist_df, fixtures_df, target_gw, bootstrap):
    """
    Run Stacking Ensemble ML model predictions (LightGBM + CatBoost + XGBoost + Ridge) for all players.
    Returns elements_df enriched with 'predicted_points' and 'base_xp'.
    """
    model_obj, metadata = load_model_and_metadata()
    feature_cols = metadata["feature_cols"]
    
    df = build_inference_features(elements_df, hist_df, fixtures_df, target_gw, bootstrap)
    
    # Extract feature matrix X matching exact feature columns used during training
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        
    # Generate predictions using Ensemble (or single model fallback)
    if isinstance(model_obj, dict) and "lgb" in model_obj and "meta" in model_obj:
        p_lgb = model_obj["lgb"].predict(X)
        p_cat = model_obj["cat"].predict(X)
        p_xgb = model_obj["xgb"].predict(X)
        X_meta = np.column_stack([p_lgb, p_cat, p_xgb])
        predictions = model_obj["meta"].predict(X_meta)
    else:
        predictions = model_obj.predict(X)

    df["predicted_points"] = np.round(predictions, 2)
    df["base_xp"] = np.round(df["xP_roll3"], 2)
    
    return df
