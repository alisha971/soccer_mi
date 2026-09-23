"""
src/evaluation.py — evaluation helpers for the final models (notebook 06).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.baselines as baselines
import src.metrics as metrics


def fit_non_neural_baselines(features, targets, train_rows, alphas, max_iter=1000, checkpoint_path=None):
    """Fit B1a, B1b, B2, B3 on training rows. Returns a dict of predict functions (rows -> logits)."""
    train_targets = targets[train_rows]
    fitted_lr = baselines.fit_logistic_regression(features, targets, list_categorical(features),
                                                  ["x_norm", "y_norm", "dt", "t_poss"], train_rows, max_iter=max_iter,
                                                  checkpoint_path=checkpoint_path)

    def predict_b1a(rows):
        return baselines.predict_class_prior(train_targets, len(rows))

    def predict_b1b(rows):
        return baselines.predict_zone_prior(features["zone"][train_rows], train_targets, features["zone"][rows],
                                            alphas["B1b zone prior"])

    def predict_b2(rows):
        return baselines.predict_markov(features["prev_action"][train_rows], features["zone"][train_rows], train_targets,
                                        features["prev_action"][rows], features["zone"][rows], alphas["B2 Markov"])

    def predict_b3(rows):
        return baselines.predict_logistic_regression(fitted_lr, features, rows)

    return {"B1a class prior": predict_b1a, "B1b zone prior": predict_b1b,
            "B2 Markov": predict_b2, "B3 logistic regression": predict_b3}, fitted_lr


def list_categorical(features):
    return [name for name in ["player", "team", "role", "competition", "zone", "period", "minute_bin", "score_diff",
                              "poss_len", "possession_flag", "prev_action", "prev_outcome",
                              "prev_end_zone", "same_player_as_prev"] if name in features]


def compare_runs_to_fixed(nll_runs, nll_fixed, match_ids, n_resamples=2000, seed=0):
    """Multi-seed model (nll_runs [n_seeds, n_events]) vs a deterministic baseline (nll_fixed [n_events])."""
    difference = nll_runs.mean(axis=0) - nll_fixed
    interval = metrics.match_bootstrap_mean(difference, match_ids, n_resamples, seed)
    all_seeds_lower = bool((nll_runs.mean(axis=1) < nll_fixed.mean()).all())
    return {"difference": interval["mean"], "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
            "all_seeds_lower": all_seeds_lower,
            "better_beyond_noise": bool(interval["ci_high"] < 0 and all_seeds_lower)}


def stratum_labels(events):
    """Named subgroups used for stratified test evaluation. Each value is a pandas Series of labels."""
    x_metres = events["x_norm"].to_numpy() * 120
    y_metres = events["y_norm"].to_numpy() * 80
    zone_band = np.where(x_metres < 40, "own third", np.where(x_metres < 80, "middle third", "final third"))
    in_box = (x_metres >= 102) & (y_metres >= 18) & (y_metres <= 62)
    zone_band = np.where(in_box, "opponent penalty box", zone_band)

    minute = events["minute"].to_numpy()
    time_band = np.where(events["period"].to_numpy() > 2, "extra time",
                         np.where(minute < 15, "0-15'", np.where(minute < 30, "15-30'", np.where(minute < 45, "30-45'",
                         np.where(minute < 60, "45-60'", np.where(minute < 75, "60-75'", "75'+"))))))

    score = events["score_diff_idx"].to_numpy() - 3
    score_state = np.where(score < 0, "losing", np.where(score == 0, "level", "winning"))

    previous_length = events["prev_poss_len"].to_numpy()
    possession_band = np.where(previous_length == 21, "first event of period",
                               np.where(previous_length <= 2, "early in spell (0-2)",
                               np.where(previous_length <= 9, "mid spell (3-9)", "long spell (10+)")))
    actor_team_changed = np.where(events["team_change"].to_numpy() == 1, "acting team changed", "same acting team")
    coupled = np.where(events["coupled_with_prev"].to_numpy() == 1, "same-moment (coupled) event", "genuine sequence event")
    return {"zone": zone_band, "time": time_band, "score state": score_state,
            "possession spell": possession_band, "turnover of actor team": actor_team_changed, "coupled": coupled}


def frequency_labels(events, train_counts, team_vocabulary):
    count = events["player_id"].map(train_counts).fillna(0).to_numpy()
    player_band = np.where(count < 100, "<100 train events (incl. unseen)",
                           np.where(count < 300, "100-299", np.where(count < 1000, "300-999", ">=1000")))
    known_team = events["team_id"].astype(str).isin(set(team_vocabulary.keys())).to_numpy()
    team_status = np.where(known_team, "team seen in training", "team unseen in training")
    return player_band, team_status
