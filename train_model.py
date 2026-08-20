import os
import json
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HISTORICAL_FILES = [
    os.path.join(DATA_DIR, "fpl_historical_2022-23.csv"),
    os.path.join(DATA_DIR, "fpl_historical_2023-24.csv")
]


def load_and_preprocess_raw_data():
    """Load historical CSVs and combine them into a single sorted DataFrame."""
    df_list = []
    for filepath in HISTORICAL_FILES:
        if os.path.exists(filepath):
            season = "2022-23" if "2022-23" in filepath else "2023-24"
            df = pd.read_csv(filepath)
            df["season"] = season
            df_list.append(df)
    
    combined = pd.concat(df_list, ignore_index=True)
    
    # Ensure numeric types for core stats
    numeric_cols = [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
        "yellow_cards", "red_cards", "saves", "bonus", "bps", "influence",
        "creativity", "threat", "ict_index", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded", "value", "xP",
        "transfers_in", "transfers_out", "transfers_balance"
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)
            
    # Normalize position string
    pos_map = {"GKP": "GK", "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
    combined["position"] = combined["position"].map(pos_map).fillna("MID")
    
    # Sort chronologically by player and gameweek
    combined.sort_values(by=["season", "element", "GW"], inplace=True)
    return combined


def build_rolling_features(df):
    """
    Compute lag and rolling statistics per player (element) per season to PREVENT DATA LEAKAGE.
    Features for GW t must only depend on performance in GW t-1, t-2, ...
    Also computes opponent-based features from team-level aggregates.
    """
    df = df.copy()
    
    # Map team string names to team_id (int64) per season (FPL uses 1..20 alphabetical IDs)
    team_id_maps = {}
    for season, group in df.groupby("season"):
        teams_sorted = sorted(group["team"].unique())
        team_id_maps[season] = {t: i + 1 for i, t in enumerate(teams_sorted)}
    
    df["team_id"] = df.apply(lambda row: team_id_maps[row["season"]].get(row["team"], 0), axis=1)
    df["opponent_team"] = pd.to_numeric(df["opponent_team"], errors="coerce").fillna(0).astype(int)
    
    # ========== 1. OPPONENT / FIXTURE FEATURES ==========
    # Compute rolling team-level stats from PAST gameweeks only (shifted)
    # so we know how strong/weak a team's attack/defence has been recently.
    
    # Team-level aggregates per season per GW
    team_gw = df.groupby(["season", "team_id", "GW"]).agg(
        team_goals_scored=("goals_scored", "sum"),
        team_goals_conceded=("goals_conceded", "mean"),  # same for all players in team
        team_clean_sheets=("clean_sheets", "max"),  # 1 if any player got CS
        team_xg=("expected_goals", "sum"),
        team_xgc=("expected_goals_conceded", "mean"),
    ).reset_index()
    
    # Rolling averages of team performance (shifted by 1 GW to prevent leakage)
    team_gw.sort_values(["season", "team_id", "GW"], inplace=True)
    team_grouped = team_gw.groupby(["season", "team_id"])
    
    team_gw["team_goals_scored_roll5"] = team_grouped["team_goals_scored"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    team_gw["team_goals_conceded_roll5"] = team_grouped["team_goals_conceded"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    team_gw["team_cs_rate_roll5"] = team_grouped["team_clean_sheets"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    team_gw["team_xg_roll5"] = team_grouped["team_xg"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    team_gw["team_xgc_roll5"] = team_grouped["team_xgc"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    
    # Now create OPPONENT features: for each row, look up the opponent team's stats
    opponent_cols = ["season", "team_id", "GW", 
                     "team_goals_scored_roll5", "team_goals_conceded_roll5",
                     "team_cs_rate_roll5", "team_xg_roll5", "team_xgc_roll5"]
    opponent_stats = team_gw[opponent_cols].copy()
    opponent_stats.rename(columns={
        "team_id": "opponent_team",
        "team_goals_scored_roll5": "opp_attack_strength",      # how many goals opponent scores (danger)
        "team_goals_conceded_roll5": "opp_goals_conceded_roll5", # how many goals opponent lets in (opportunity)
        "team_cs_rate_roll5": "opp_cs_rate_roll5",              # opponent CS rate (bad for attackers)
        "team_xg_roll5": "opp_xg_roll5",                       # opponent xG (danger for defenders)
        "team_xgc_roll5": "opp_xgc_roll5",                     # opponent xGC (opportunity for attackers)
    }, inplace=True)
    
    # Merge opponent stats onto each player-GW row
    df = df.merge(opponent_stats, on=["season", "opponent_team", "GW"], how="left")
    for col in ["opp_attack_strength", "opp_goals_conceded_roll5", "opp_cs_rate_roll5", "opp_xg_roll5", "opp_xgc_roll5"]:
        df[col] = df[col].fillna(0)
    
    # Also merge the PLAYER'S OWN team rolling stats
    own_team_cols = ["season", "team_id", "GW",
                     "team_goals_scored_roll5", "team_goals_conceded_roll5", 
                     "team_cs_rate_roll5", "team_xg_roll5", "team_xgc_roll5"]
    own_team_stats = team_gw[own_team_cols].copy()
    own_team_stats.rename(columns={
        "team_goals_scored_roll5": "own_team_attack_roll5",
        "team_goals_conceded_roll5": "own_team_defence_roll5",
        "team_cs_rate_roll5": "own_team_cs_rate_roll5",
        "team_xg_roll5": "own_team_xg_roll5",
        "team_xgc_roll5": "own_team_xgc_roll5",
    }, inplace=True)
    df = df.merge(own_team_stats, on=["season", "team_id", "GW"], how="left")
    for col in ["own_team_attack_roll5", "own_team_defence_roll5", "own_team_cs_rate_roll5", "own_team_xg_roll5", "own_team_xgc_roll5"]:
        df[col] = df[col].fillna(0)
    
    # ========== 2. PLAYER ROLLING FEATURES (original) ==========
    rolling_metrics = [
        "minutes", "total_points", "expected_goals", "expected_assists",
        "bps", "ict_index", "influence", "creativity", "threat",
        "goals_scored", "assists", "clean_sheets", "bonus", "xP"
    ]
    
    grouped = df.groupby(["season", "element"])
    
    # Shifted values (previous GW performance)
    for col in rolling_metrics:
        df[f"{col}_lag1"] = grouped[col].shift(1).fillna(0)
        
        # Rolling averages (3-GW, 5-GW, 10-GW) on shifted data
        df[f"{col}_roll3"] = grouped[col].shift(1).rolling(3, min_periods=1).mean().fillna(0)
        df[f"{col}_roll5"] = grouped[col].shift(1).rolling(5, min_periods=1).mean().fillna(0)
        df[f"{col}_roll10"] = grouped[col].shift(1).rolling(10, min_periods=1).mean().fillna(0)

    # Form trend features
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
    
    # Expected points per 90 (lagged)
    df["xg_per_90_roll5"] = np.where(
        df["minutes_roll5"] > 0,
        (df["expected_goals_roll5"] / df["minutes_roll5"]) * 90,
        0.0
    )
    df["xa_per_90_roll5"] = np.where(
        df["minutes_roll5"] > 0,
        (df["expected_assists_roll5"] / df["minutes_roll5"]) * 90,
        0.0
    )
    
    # ========== 3. POSITION + CONTEXT FEATURES ==========
    df["is_gk"] = (df["position"] == "GK").astype(int)
    df["is_def"] = (df["position"] == "DEF").astype(int)
    df["is_mid"] = (df["position"] == "MID").astype(int)
    df["is_fwd"] = (df["position"] == "FWD").astype(int)
    
    # Match context features
    df["was_home"] = df["was_home"].astype(int)
    df["cost_millions"] = df["value"] / 10.0
    
    # ========== 4. MARKET / CROWD WISDOM FEATURES ==========
    # Normalize selected to percentage-like scale (log scale since it spans 0 to 9.5M)
    df["selected_log"] = np.log1p(df["selected"].fillna(0))
    df["transfers_balance_norm"] = df["transfers_balance"].fillna(0) / 10000.0  # normalize to reasonable range
    
    return df


def train_and_evaluate():
    """Train GBDT model using historical gameweek data and evaluate performance."""
    print("Loading historical data...")
    raw_df = load_and_preprocess_raw_data()
    print(f"Total raw rows: {len(raw_df)}")
    
    print("Building rolling lag features...")
    df = build_rolling_features(raw_df)
    
    # Filter out rows where player was completely inactive (0 minutes in all past 10 GWs AND 0 min in current GW)
    # to avoid training on thousands of non-playing bench fodder rows.
    active_df = df[(df["minutes_roll5"] > 0) | (df["minutes"] > 0)].copy()
    print(f"Active training rows: {len(active_df)}")
    
    # Define feature set
    feature_cols = [
        # Position & context
        "is_gk", "is_def", "is_mid", "is_fwd",
        "was_home", "cost_millions", "GW",
        # Player rolling form
        "minutes_lag1", "minutes_roll3", "minutes_roll5", "minutes_roll10",
        "total_points_lag1", "total_points_roll3", "total_points_roll5", "total_points_roll10",
        "expected_goals_lag1", "expected_goals_roll3", "expected_goals_roll5",
        "expected_assists_lag1", "expected_assists_roll3", "expected_assists_roll5",
        "bps_lag1", "bps_roll3", "bps_roll5",
        "ict_index_lag1", "ict_index_roll3", "ict_index_roll5",
        "influence_roll3", "creativity_roll3", "threat_roll3",
        "xP_lag1", "xP_roll3", "xP_roll5",
        "points_form_ratio", "minutes_form_ratio",
        "xg_per_90_roll5", "xa_per_90_roll5",
        # Opponent / fixture features (NEW)
        "opp_attack_strength", "opp_goals_conceded_roll5", "opp_cs_rate_roll5",
        "opp_xg_roll5", "opp_xgc_roll5",
        # Own team strength (NEW)
        "own_team_attack_roll5", "own_team_defence_roll5", "own_team_cs_rate_roll5",
        # Market / crowd wisdom (NEW)
        "selected_log", "transfers_balance_norm",
    ]
    
    target_col = "total_points"
    
    # Train / Validation Split based on Season (Train on 2022-23, Test on 2023-24)
    train_mask = active_df["season"] == "2022-23"
    test_mask = active_df["season"] == "2023-24"
    
    X_train = active_df.loc[train_mask, feature_cols]
    y_train = active_df.loc[train_mask, target_col]
    X_test = active_df.loc[test_mask, feature_cols]
    y_test = active_df.loc[test_mask, target_col]
    
    print(f"Training samples (2022-23): {len(X_train)}")
    print(f"Testing samples (2023-24): {len(X_test)}")
    
    # 1. LightGBM Regressor
    lgb_model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    
    y_pred_lgb = lgb_model.predict(X_test)
    lgb_mae = mean_absolute_error(y_test, y_pred_lgb)
    lgb_rmse = root_mean_squared_error(y_test, y_pred_lgb)
    lgb_r2 = r2_score(y_test, y_pred_lgb)
    
    # Baseline: predict the training set mean for every sample
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    # Ranking quality: Spearman correlation
    spearman_corr, spearman_p = spearmanr(y_test, y_pred_lgb)
    
    # Top-N precision: do the top 50 predicted players actually score more?
    test_eval = pd.DataFrame({"actual": y_test.values, "predicted": y_pred_lgb})
    top50_pred = test_eval.nlargest(50, "predicted")
    top50_actual = test_eval.nlargest(50, "actual")
    top50_overlap = len(set(top50_pred.index) & set(top50_actual.index))
    
    print("\n--- LightGBM Model Evaluation (Tested on 2023-24 Season) ---")
    print(f"MAE:  {lgb_mae:.4f}  (baseline mean-predictor: {baseline_mae:.4f}, lift: {(baseline_mae - lgb_mae) / baseline_mae * 100:.1f}%)")
    print(f"RMSE: {lgb_rmse:.4f}")
    print(f"R²:   {lgb_r2:.4f}  (baseline: {baseline_r2:.4f})")
    print(f"Spearman Rank Correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")
    print(f"Top-50 Precision: {top50_overlap}/50 ({top50_overlap/50*100:.0f}%) overlap")
    
    # Train full model on combined dataset (2022-23 + 2023-24) for maximum training coverage
    X_all = active_df[feature_cols]
    y_all = active_df[target_col]
    
    final_model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    final_model.fit(X_all, y_all)
    
    # Save trained model artifact
    model_path = os.path.join(MODELS_DIR, "fpl_lgbm_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    
    joblib.dump(final_model, model_path)
    
    # Feature importances
    importances = dict(zip(feature_cols, [float(x) for x in final_model.feature_importances_]))
    importances_sorted = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    metadata = {
        "model_type": "LightGBMRegressor",
        "mae_2023_24": float(lgb_mae),
        "rmse_2023_24": float(lgb_rmse),
        "r2_2023_24": float(lgb_r2),
        "baseline_mae": float(baseline_mae),
        "spearman_correlation": float(spearman_corr),
        "top50_precision": top50_overlap,
        "feature_cols": feature_cols,
        "feature_importances": importances_sorted
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nModel saved successfully to {model_path}")
    print(f"Metadata saved successfully to {meta_path}")
    print("\nTop 10 Most Important Features:")
    for feature, importance in list(importances_sorted.items())[:10]:
        print(f"  - {feature}: {importance}")
        

if __name__ == "__main__":
    train_and_evaluate()
