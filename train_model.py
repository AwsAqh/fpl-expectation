import os
import json
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
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
    team_gw = df.groupby(["season", "team_id", "GW"]).agg(
        team_goals_scored=("goals_scored", "sum"),
        team_goals_conceded=("goals_conceded", "mean"),
        team_clean_sheets=("clean_sheets", "max"),
        team_xg=("expected_goals", "sum"),
        team_xgc=("expected_goals_conceded", "mean"),
    ).reset_index()
    
    team_gw.sort_values(["season", "team_id", "GW"], inplace=True)
    team_grouped = team_gw.groupby(["season", "team_id"])
    
    team_gw["team_goals_scored_roll5"] = team_grouped["team_goals_scored"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    team_gw["team_goals_conceded_roll5"] = team_grouped["team_goals_conceded"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    team_gw["team_cs_rate_roll5"] = team_grouped["team_clean_sheets"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    team_gw["team_xg_roll5"] = team_grouped["team_xg"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    team_gw["team_xgc_roll5"] = team_grouped["team_xgc"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    
    opponent_cols = ["season", "team_id", "GW", 
                     "team_goals_scored_roll5", "team_goals_conceded_roll5",
                     "team_cs_rate_roll5", "team_xg_roll5", "team_xgc_roll5"]
    opponent_stats = team_gw[opponent_cols].copy()
    opponent_stats.rename(columns={
        "team_id": "opponent_team",
        "team_goals_scored_roll5": "opp_attack_strength",
        "team_goals_conceded_roll5": "opp_goals_conceded_roll5",
        "team_cs_rate_roll5": "opp_cs_rate_roll5",
        "team_xg_roll5": "opp_xg_roll5",
        "team_xgc_roll5": "opp_xgc_roll5",
    }, inplace=True)
    
    df = df.merge(opponent_stats, on=["season", "opponent_team", "GW"], how="left")
    for col in ["opp_attack_strength", "opp_goals_conceded_roll5", "opp_cs_rate_roll5", "opp_xg_roll5", "opp_xgc_roll5"]:
        df[col] = df[col].fillna(0)
    
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
    
    # ========== 2. PLAYER ROLLING FEATURES ==========
    rolling_metrics = [
        "minutes", "total_points", "expected_goals", "expected_assists",
        "bps", "ict_index", "influence", "creativity", "threat",
        "goals_scored", "assists", "clean_sheets", "bonus", "xP"
    ]
    
    grouped = df.groupby(["season", "element"])
    
    for col in rolling_metrics:
        df[f"{col}_lag1"] = grouped[col].shift(1).fillna(0)
        df[f"{col}_roll3"] = grouped[col].shift(1).rolling(3, min_periods=1).mean().fillna(0)
        df[f"{col}_roll5"] = grouped[col].shift(1).rolling(5, min_periods=1).mean().fillna(0)
        df[f"{col}_roll10"] = grouped[col].shift(1).rolling(10, min_periods=1).mean().fillna(0)

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
    
    df["was_home"] = df["was_home"].astype(int)
    df["cost_millions"] = df["value"] / 10.0
    
    # ========== 4. MARKET / CROWD WISDOM FEATURES ==========
    df["selected_log"] = np.log1p(df["selected"].fillna(0))
    df["transfers_balance_norm"] = df["transfers_balance"].fillna(0) / 10000.0
    
    return df


def train_and_evaluate():
    """Train Stacking Ensemble models (LGBM + CatBoost + XGBoost + Ridge Meta-Learner)."""
    print("Loading historical data...")
    raw_df = load_and_preprocess_raw_data()
    print(f"Total raw rows: {len(raw_df)}")
    
    print("Building rolling lag features...")
    df = build_rolling_features(raw_df)
    
    active_df = df[(df["minutes_roll5"] > 0) | (df["minutes"] > 0)].copy()
    print(f"Active training rows: {len(active_df)}")
    
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
        "xg_per_90_roll5", "xa_per_90_roll5",
        "opp_attack_strength", "opp_goals_conceded_roll5", "opp_cs_rate_roll5",
        "opp_xg_roll5", "opp_xgc_roll5",
        "own_team_attack_roll5", "own_team_defence_roll5", "own_team_cs_rate_roll5",
        "selected_log", "transfers_balance_norm",
    ]
    
    target_col = "total_points"
    
    train_mask = active_df["season"] == "2022-23"
    test_mask = active_df["season"] == "2023-24"
    
    X_train = active_df.loc[train_mask, feature_cols]
    y_train = active_df.loc[train_mask, target_col]
    X_test = active_df.loc[test_mask, feature_cols]
    y_test = active_df.loc[test_mask, target_col]
    
    print(f"Training samples (2022-23): {len(X_train)}")
    print(f"Testing samples (2023-24): {len(X_test)}")
    
    # 1. Base Model 1: LightGBM
    print("\nTraining LightGBM base model...")
    lgb_model = LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    p_lgb_test = lgb_model.predict(X_test)
    
    # 2. Base Model 2: CatBoost
    print("Training CatBoost base model...")
    cat_model = CatBoostRegressor(
        iterations=300, learning_rate=0.03, depth=6,
        subsample=0.8, random_seed=42, verbose=0
    )
    cat_model.fit(X_train, y_train)
    p_cat_test = cat_model.predict(X_test)
    
    # 3. Base Model 3: XGBoost
    print("Training XGBoost base model...")
    xgb_model = XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    p_xgb_test = xgb_model.predict(X_test)
    
    # Evaluate individual base models
    print("\n--- Base Model Evaluation (Tested on 2023-24 Season) ---")
    models_dict = {
        "LightGBM": p_lgb_test,
        "CatBoost": p_cat_test,
        "XGBoost":  p_xgb_test
    }
    for mname, mpreds in models_dict.items():
        m_r2 = r2_score(y_test, mpreds)
        m_mae = mean_absolute_error(y_test, mpreds)
        m_corr, _ = spearmanr(y_test, mpreds)
        print(f"  {mname:<10} | R²: {m_r2:.4f} | MAE: {m_mae:.4f} | Spearman: {m_corr:.4f}")
    
    # 4. Meta-Learner (Ridge Stacking)
    print("\nFitting Stacking Meta-Learner (Ridge)...")
    X_meta_test = np.column_stack([p_lgb_test, p_cat_test, p_xgb_test])
    
    # Compute Out-of-Fold predictions for train set meta-learner fitting (Simple 5-fold CV on train set)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X_train), 3))
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        
        m_lgb = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
        m_cat = CatBoostRegressor(iterations=300, learning_rate=0.03, depth=6, subsample=0.8, random_seed=42, verbose=0)
        m_xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        
        m_lgb.fit(X_tr, y_tr)
        m_cat.fit(X_tr, y_tr)
        m_xgb.fit(X_tr, y_tr)
        
        oof_preds[val_idx, 0] = m_lgb.predict(X_va)
        oof_preds[val_idx, 1] = m_cat.predict(X_va)
        oof_preds[val_idx, 2] = m_xgb.predict(X_va)
        
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(oof_preds, y_train)
    
    y_pred_ensemble = meta_learner.predict(X_meta_test)
    
    ens_mae = mean_absolute_error(y_test, y_pred_ensemble)
    ens_rmse = root_mean_squared_error(y_test, y_pred_ensemble)
    ens_r2 = r2_score(y_test, y_pred_ensemble)
    ens_spearman, ens_p = spearmanr(y_test, y_pred_ensemble)
    
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    test_eval = pd.DataFrame({"actual": y_test.values, "predicted": y_pred_ensemble})
    top50_pred = test_eval.nlargest(50, "predicted")
    top50_actual = test_eval.nlargest(50, "actual")
    top50_overlap = len(set(top50_pred.index) & set(top50_actual.index))
    
    print("\n" + "=" * 60)
    print("--- STACKING ENSEMBLE EVALUATION (2023-24 Season) ---")
    print(f"MAE:  {ens_mae:.4f}  (baseline: {baseline_mae:.4f}, lift: {(baseline_mae - ens_mae) / baseline_mae * 100:.1f}%)")
    print(f"RMSE: {ens_rmse:.4f}")
    print(f"R²:   {ens_r2:.4f}  (LGBM-only was 0.1885 | relative gain: {(ens_r2 - 0.1885) / 0.1885 * 100:.1f}%)")
    print(f"Spearman Rank Correlation: {ens_spearman:.4f} (p={ens_p:.2e})")
    print(f"Top-50 Precision: {top50_overlap}/50 ({top50_overlap/50*100:.0f}%) overlap")
    print(f"Meta-Learner Weights (LGBM, CatBoost, XGBoost): {meta_learner.coef_}")
    print("=" * 60)
    
    # Train FINAL ENSEMBLE models on combined dataset (2022-23 + 2023-24)
    print("\nTraining final ensemble on full combined historical dataset...")
    X_all = active_df[feature_cols]
    y_all = active_df[target_col]
    
    final_lgb = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    final_cat = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=6, subsample=0.8, random_seed=42, verbose=0)
    final_xgb = XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    
    final_lgb.fit(X_all, y_all)
    final_cat.fit(X_all, y_all)
    final_xgb.fit(X_all, y_all)
    
    # OOF for full dataset meta-learner
    oof_all = np.zeros((len(X_all), 3))
    kf_all = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf_all.split(X_all):
        X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_va = X_all.iloc[val_idx]
        
        m_lgb = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
        m_cat = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=6, subsample=0.8, random_seed=42, verbose=0)
        m_xgb = XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        
        m_lgb.fit(X_tr, y_tr)
        m_cat.fit(X_tr, y_tr)
        m_xgb.fit(X_tr, y_tr)
        
        oof_all[val_idx, 0] = m_lgb.predict(X_va)
        oof_all[val_idx, 1] = m_cat.predict(X_va)
        oof_all[val_idx, 2] = m_xgb.predict(X_va)
        
    final_meta = Ridge(alpha=1.0)
    final_meta.fit(oof_all, y_all)
    
    ensemble_dict = {
        "lgb": final_lgb,
        "cat": final_cat,
        "xgb": final_xgb,
        "meta": final_meta,
        "feature_cols": feature_cols
    }
    
    ensemble_path = os.path.join(MODELS_DIR, "fpl_ensemble_models.joblib")
    lgb_legacy_path = os.path.join(MODELS_DIR, "fpl_lgbm_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    
    joblib.dump(ensemble_dict, ensemble_path)
    joblib.dump(final_lgb, lgb_legacy_path)
    
    importances = dict(zip(feature_cols, [float(x) for x in final_lgb.feature_importances_]))
    importances_sorted = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    metadata = {
        "model_type": "StackingEnsemble (LGBM + CatBoost + XGBoost + Ridge)",
        "mae_2023_24": float(ens_mae),
        "rmse_2023_24": float(ens_rmse),
        "r2_2023_24": float(ens_r2),
        "baseline_mae": float(baseline_mae),
        "spearman_correlation": float(ens_spearman),
        "meta_weights": [float(w) for w in final_meta.coef_],
        "top50_precision": top50_overlap,
        "feature_cols": feature_cols,
        "feature_importances": importances_sorted
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nEnsemble models saved successfully to {ensemble_path}")
    print(f"Metadata saved successfully to {meta_path}")


if __name__ == "__main__":
    train_and_evaluate()
