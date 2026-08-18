import os
import json
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

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
    """
    df = df.copy()
    
    # Target variable: total_points in GW t
    # Input features: aggregated from past GWs (shift by 1)
    
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
    
    # Position binary encodings
    df["is_gk"] = (df["position"] == "GK").astype(int)
    df["is_def"] = (df["position"] == "DEF").astype(int)
    df["is_mid"] = (df["position"] == "MID").astype(int)
    df["is_fwd"] = (df["position"] == "FWD").astype(int)
    
    # Match context features
    df["was_home"] = df["was_home"].astype(int)
    df["cost_millions"] = df["value"] / 10.0
    
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
        "is_gk", "is_def", "is_mid", "is_fwd",
        "was_home", "cost_millions", "GW",
        "minutes_lag1", "minutes_roll3", "minutes_roll5", "minutes_roll10",
        "total_points_lag1", "total_points_roll3", "total_points_roll5", "total_points_roll10",
        "expected_goals_lag1", "expected_goals_roll3", "expected_goals_roll5",
        "expected_assists_lag1", "expected_assists_roll3", "expected_assists_roll5",
        "bps_lag1", "bps_roll3", "bps_roll5",
        "ict_index_lag1", "ict_index_roll3", "ict_index_roll5",
        "influence_roll3", "creativity_roll3", "threat_roll3",
        "xP_lag1", "xP_roll3", "xP_roll5",
        "points_form_ratio", "minutes_form_ratio",
        "xg_per_90_roll5", "xa_per_90_roll5"
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
    
    print("\n--- LightGBM Model Evaluation (Tested on 2023-24 Season) ---")
    print(f"MAE:  {lgb_mae:.4f}")
    print(f"RMSE: {lgb_rmse:.4f}")
    print(f"R²:   {lgb_r2:.4f}")
    
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
