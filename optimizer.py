import pandas as pd
import numpy as np
from itertools import combinations


class FPLOptimizer:
    def __init__(self, predictions_df, budget=1000.0, max_per_team=3):
        self.predictions_df = predictions_df.copy()
        # Normalize position strings (GK -> GKP)
        self.predictions_df["position"] = self.predictions_df["position"].replace({"GK": "GKP"})
        self.budget = budget
        self.max_per_team = max_per_team
        self.position_limits = {
            "GKP": (1, 1),
            "DEF": (3, 5),
            "MID": (2, 5),
            "FWD": (1, 3),
        }
        self.total_slots = 11

    def _is_valid_lineup(self, selected):
        if len(selected) != self.total_slots:
            return False

        total_cost = sum(
            self.predictions_df.loc[
                self.predictions_df["id"] == pid, "now_cost"
            ].values[0]
            for pid in selected
        )
        if total_cost > self.budget:
            return False

        pos_counts = {}
        team_counts = {}
        for pid in selected:
            row = self.predictions_df[self.predictions_df["id"] == pid]
            if row.empty:
                return False
            pos = row.iloc[0].get("position", "")
            team = row.iloc[0].get("team_name", "")
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            team_counts[team] = team_counts.get(team, 0) + 1

        for pos, (min_play, max_play) in self.position_limits.items():
            count = pos_counts.get(pos, 0)
            if count < min_play or count > max_play:
                return False

        for team, count in team_counts.items():
            if count > self.max_per_team:
                return False

        return True

    def _lineup_score(self, selected):
        total = 0
        for pid in selected:
            row = self.predictions_df[self.predictions_df["id"] == pid]
            if row.empty:
                return -np.inf
            pred = row.iloc[0].get("predicted_points", 0)
            total += pred
        return total

    def optimize_greedy(self):
        """
        Two-phase greedy optimizer:
        Phase 1: Fill minimum position requirements (1 GKP, 3 DEF, 2 MID, 1 FWD = 7 mandatory)
        Phase 2: Fill remaining 4 flex slots with highest-scoring eligible players
        """
        sorted_players = self.predictions_df.sort_values(
            "predicted_points", ascending=False
        )

        selected = []
        selected_ids = set()
        remaining_budget = self.budget
        team_counts = {}
        pos_counts = {}

        # Phase 1: Fill minimum position requirements
        for pos, (min_play, _) in self.position_limits.items():
            pos_candidates = sorted_players[
                (sorted_players["position"] == pos)
            ].copy()

            filled = 0
            for _, player in pos_candidates.iterrows():
                if filled >= min_play:
                    break

                pid = player["id"]
                cost = player["now_cost"]
                team = player.get("team_name", "")

                if pid in selected_ids:
                    continue
                if cost > remaining_budget:
                    continue
                if team_counts.get(team, 0) >= self.max_per_team:
                    continue

                selected.append(pid)
                selected_ids.add(pid)
                remaining_budget -= cost
                team_counts[team] = team_counts.get(team, 0) + 1
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
                filled += 1

        # Phase 2: Fill remaining flex slots with best available players
        slots_remaining = self.total_slots - len(selected)
        for _, player in sorted_players.iterrows():
            if slots_remaining <= 0:
                break

            pid = player["id"]
            cost = player["now_cost"]
            pos = player.get("position", "")
            team = player.get("team_name", "")

            if pid in selected_ids:
                continue
            if cost > remaining_budget:
                continue
            if team_counts.get(team, 0) >= self.max_per_team:
                continue

            # Check position max limit
            _, max_play = self.position_limits.get(pos, (0, 5))
            if pos_counts.get(pos, 0) >= max_play:
                continue

            selected.append(pid)
            selected_ids.add(pid)
            remaining_budget -= cost
            team_counts[team] = team_counts.get(team, 0) + 1
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            slots_remaining -= 1

        if len(selected) < self.total_slots:
            if not self._fill_remaining_positions(
                selected, selected_ids, remaining_budget, team_counts
            ):
                return None, 0

        score = self._lineup_score(selected)
        return selected, score

    def _fill_remaining_positions(self, selected, selected_ids, remaining_budget, team_counts):
        pos_counts = {}
        for pid in selected:
            row = self.predictions_df[self.predictions_df["id"] == pid]
            if row.empty:
                continue
            ppos = row.iloc[0].get("position", "")
            pos_counts[ppos] = pos_counts.get(ppos, 0) + 1

        for _ in range(self.total_slots - len(selected)):
            best_player = None
            best_score = -1

            for pos, (min_play, max_play) in self.position_limits.items():
                current_count = pos_counts.get(pos, 0)
                if current_count >= max_play:
                    continue

                pos_players = self.predictions_df[
                    (self.predictions_df["position"] == pos)
                    & (~self.predictions_df["id"].isin(selected_ids))
                ]
                if pos_players.empty:
                    continue

                for _, player in pos_players.iterrows():
                    pid = player["id"]
                    cost = player["now_cost"]
                    team = player.get("team_name", "")
                    points = player.get("predicted_points", 0)

                    if cost > remaining_budget:
                        continue
                    if pid in selected_ids:
                        continue
                    if team_counts.get(team, 0) >= self.max_per_team:
                        continue

                    if points > best_score:
                        best_score = points
                        best_player = (pid, cost, team, pos)

            if best_player is None:
                return False

            pid, cost, team, pos = best_player
            selected.append(pid)
            selected_ids.add(pid)
            remaining_budget -= cost
            team_counts[team] = team_counts.get(team, 0) + 1
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        return True

    def get_lineup_details(self, selected_ids):
        if selected_ids is None:
            return None

        lineup = []
        total_cost = 0
        total_predicted = 0

        for pid in selected_ids:
            row = self.predictions_df[self.predictions_df["id"] == pid]
            if row.empty:
                continue
            r = row.iloc[0]
            total_cost += r["now_cost"]
            total_predicted += r.get("predicted_points", 0)
            lineup.append({
                "id": int(pid),
                "name": r.get("web_name") or f"{r.get('first_name', '')} {r.get('second_name', '')}".strip(),
                "web_name": r.get("web_name", ""),
                "full_name": f"{r.get('first_name', '')} {r.get('second_name', '')}".strip(),
                "position": r.get("position", ""),
                "team": r.get("team_name", ""),
                "cost": float(r["now_cost"]),
                "predicted_points": float(r.get("predicted_points", 0)),
                "base_xp": float(r.get("base_xp", 0)),
                "points_per_game": float(r.get("points_per_game", 0) or 0),
                "fixture_difficulty": float(r.get("gw1_difficulty", r.get("upcoming_avg_difficulty", 3.0))),
                "recent_form": float(r.get("form_numeric", 0)),
            })

        return {
            "lineup": lineup,
            "total_cost": int(total_cost),
            "total_predicted_points": float(total_predicted),
            "budget_remaining": float(self.budget - total_cost),
        }


def select_best_xi(predictions_df, budget=1000.0, max_per_team=3):
    optimizer = FPLOptimizer(predictions_df, budget=budget, max_per_team=max_per_team)
    selected, score = optimizer.optimize_greedy()
    details = optimizer.get_lineup_details(selected)
    return details


if __name__ == "__main__":
    print("FPL Optimizer Module")