# FPL Gameweek Predictor

A machine learning system that predicts [Fantasy Premier League](https://fantasy.premierleague.com) (FPL) player points for the next gameweek and picks the best starting XI within the £100m budget.

## How it works

- **Data pipeline** — fetches live player, fixture, and injury data from the official FPL API.
- **Feature engineering** — rolling form, expected goals/assists (xG/xA), fixture difficulty, and injury status.
- **Model** — a LightGBM regressor trained on 2022–23 and 2023–24 gameweek data.
- **Optimizer** — greedy selection of the best XI respecting position limits (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD) and budget.

## Results

Held-out evaluation on the 2023–24 season: **MAE 1.64 pts/GW, RMSE 2.56, R² 0.167**.

Predicting raw weekly points is inherently noisy, so the model is most useful as a *ranking* tool for player selection rather than an exact scorer.

## Quickstart

```bash
pip install -r requirements.txt

python fetch_all_data.py          # download live FPL data
python download_historical.py     # optional: training data (2022-23 / 2023-24)
python train_model.py             # optional: retrain the model
python main.py                    # generate predictions + best XI
python serve.py                   # optional: local web UI at http://localhost:8081
```

## Project structure

```
fpl_predictor/
├── main.py               # end-to-end pipeline
├── fetch_all_data.py     # download live FPL data
├── download_historical.py# download historical training data
├── predict_ml.py         # feature engineering + inference
├── train_model.py        # model training & evaluation
├── optimizer.py          # best-XI selection
├── serve.py / ui.html    # local web dashboard
└── requirements.txt
```

## Data

- FPL API (`fantasy.premierleague.com`) — public, no auth required.
- Historical gameweek data: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).
