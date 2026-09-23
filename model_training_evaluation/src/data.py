"""
src/data.py — data preparation for Track A.

This file is written by notebooks 01 and 02. Every function is explained in the notebook
cell that writes it. 
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# The 15 action classes the model predicts (the target vocabulary).
ACTION_NAMES = [
    "PASS_SHORT", "PASS_LONG", "CROSS", "CARRY", "DRIBBLE", "SHOT", "CLEARANCE",
    "RECOVERY", "INTERCEPTION", "DUEL", "FOUL_WON", "FOUL_COMMITTED", "GK_ACTION",
    "LOSS", "OTHER",
]
NUM_ACTIONS = len(ACTION_NAMES)

# Code 15 ("END") appears only in the history feature prev_action_id, on the first event of a
# period, meaning "there is no previous event". It is never a prediction target.
END_ACTION_CODE = 15


def file_sha256(path):
    """Return the SHA-256 fingerprint of a file (reads it in 1 MB blocks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_events(dataset_path):
    """Load the event table in chronological order and add `period_offset`.

    period_offset = 0 for the first event of a (match, period), 1 for the second, and so on.
    """
    events = pd.read_parquet(dataset_path)
    events = events.sort_values(["match_id", "period", "index"]).reset_index(drop=True)
    events["period_offset"] = events.groupby(["match_id", "period"]).cumcount()
    return events

def build_windows(events, context_length, stride, split=None):
    """Build model-ready windows.

    Returns an integer array of shape [n_windows, context_length]. Each row lists the row
    numbers (into `events`) of `context_length` consecutive events from ONE match-period.
    Consecutive windows start `stride` events apart. A window is only formed if it is full,
    so no window crosses a match or period boundary and no padding is needed.
    """
    if split is None:
        selected = events
    else:
        selected = events[events["split"] == split]

    windows = []
    for _, period_events in selected.groupby(["match_id", "period"], sort=True):
        row_numbers = period_events.index.to_numpy()
        n_events = len(row_numbers)
        start = 0
        while start + context_length <= n_events:
            windows.append(row_numbers[start:start + context_length])
            start = start + stride
    return np.array(windows, dtype=np.int64)


def build_evaluation_view(events, split, context_length, stride):
    """Assign every event of a split to exactly ONE (window, position) for evaluation.

    Training windows overlap, so an event appears in up to context_length / stride windows,
    each time with a different amount of history before it. Scoring all copies would count
    the same event several times. Here each event is scored once, in the window where it has
    the MOST preceding history (the earliest window that contains it).

    The regular windows are the training windows. If the last events of a period are not
    covered by any regular window, one extra window ending at the last event is added
    (evaluation only), so that every event is scored.

    Returns a dict with:
      windows       [n_windows, context_length] row numbers
      event_rows    [n_events]  row number of each scored event
      window_index  [n_events]  which window the event is scored in
      position      [n_events]  its position inside that window
    """
    selected = events[events["split"] == split]
    all_windows = []
    event_rows = []
    window_index = []
    positions = []

    for _, period_events in selected.groupby(["match_id", "period"], sort=True):
        row_numbers = period_events.index.to_numpy()
        n_events = len(row_numbers)
        starts = list(range(0, n_events - context_length + 1, stride))
        tail_start = n_events - context_length
        if starts[-1] != tail_start:
            starts.append(tail_start)

        chosen_window = np.full(n_events, -1, dtype=np.int64)
        chosen_position = np.full(n_events, -1, dtype=np.int64)
        for start in starts:
            window_number = len(all_windows)
            all_windows.append(row_numbers[start:start + context_length])
            for position in range(context_length):
                offset = start + position
                # keep the first (earliest) window that contains this event
                if chosen_window[offset] == -1:
                    chosen_window[offset] = window_number
                    chosen_position[offset] = position

        event_rows.extend(row_numbers.tolist())
        window_index.extend(chosen_window.tolist())
        positions.extend(chosen_position.tolist())

    return {
        "windows": np.array(all_windows, dtype=np.int64),
        "event_rows": np.array(event_rows, dtype=np.int64),
        "window_index": np.array(window_index, dtype=np.int64),
        "position": np.array(positions, dtype=np.int64),
    }

def add_pre_action_possession_features(events):
    """Add possession features computed only from events that have already happened.

    team_change          1 if the acting team differs from the previous event's acting team
    prev_poss_len        poss_len of the previous event (21 = no previous event in the period)
    prev_play_pattern_id play pattern of the previous event's possession (0 = none)
    t_poss_prev          log(1 + seconds from the start of the previous event's possession
                         until the current event), clipped at 600 seconds

    Why: StatsBomb possession numbers are assigned after the fact, so the original columns
    (new_possession, poss_len, play_pattern_id, t_poss) can depend on the action at the
    current event. See notebook 01, Section 8.1.
    """
    group = events.groupby(["match_id", "period"])
    is_first = events["period_offset"] == 0

    previous_team = group["team_id"].shift(1)
    team_change = (events["team_id"] != previous_team).astype("int64")
    team_change[is_first] = 0
    events["team_change"] = team_change

    previous_poss_len = group["poss_len"].shift(1)
    events["prev_poss_len"] = previous_poss_len.fillna(21).astype("int64")

    previous_play_pattern = group["play_pattern_id"].shift(1)
    events["prev_play_pattern_id"] = previous_play_pattern.fillna(0).astype("int64")

    possession_start_time = events.groupby(["match_id", "period", "possession"])["t_period_s"].transform("min")
    previous_possession_start = possession_start_time.groupby([events["match_id"], events["period"]]).shift(1)
    elapsed_seconds = events["t_period_s"] - previous_possession_start
    elapsed_seconds = elapsed_seconds.fillna(0.0).clip(lower=0.0, upper=600.0)
    events["t_poss_prev"] = np.log1p(elapsed_seconds).astype("float32")
    return events


def add_coupled_event_flag(events):
    """Flag events recorded at the same moment as the previous event (gap below 0.01 s).

    StatsBomb records some events in pairs with one timestamp (for example Foul Committed
    followed by Foul Won). The second event of such a pair is almost determined by the first,
    so it measures annotation structure rather than a player decision. Metrics are reported
    with and without these events.
    """
    seconds_since_previous = np.expm1(events["dt"].astype("float64"))
    is_first = events["period_offset"] == 0
    events["coupled_with_prev"] = ((seconds_since_previous < 0.01) & ~is_first).astype("int64")
    return events

# ---------------------------------------------------------------------------
# Identity vocabularies and encodings (written by notebook 02)
# ---------------------------------------------------------------------------

def count_train_events_per_player(events):
    """Number of TRAINING events per player. Validation and test are never counted."""
    train_events = events[events["split"] == "train"]
    return train_events["player_id"].value_counts()


def build_player_vocabulary(train_counts, threshold):
    """Map player_id -> index for players with at least `threshold` training events.

    Indices start at 1. Index 0 is reserved for "no dedicated identity" (below threshold or
    unseen in training). Players are ordered by training count (then id) so the mapping is
    deterministic.
    """
    eligible = train_counts[train_counts >= threshold]
    table = pd.DataFrame({"player_id": eligible.index, "train_events": eligible.values})
    table = table.sort_values(["train_events", "player_id"], ascending=[False, True])
    vocabulary = {}
    next_index = 1
    for player_id in table["player_id"]:
        vocabulary[int(player_id)] = next_index
        next_index = next_index + 1
    return vocabulary


def build_train_vocabulary(events, column):
    """Map raw values of `column` seen in TRAIN to indices 1..n. Unseen values will get 0."""
    train_values = sorted(events.loc[events["split"] == "train", column].unique())
    vocabulary = {}
    for position, value in enumerate(train_values):
        vocabulary[int(value)] = position + 1
    return vocabulary


def map_to_index(values, vocabulary):
    """Apply a vocabulary; anything not in it becomes 0."""
    mapped = values.map(vocabulary)
    return mapped.fillna(0).astype("int64").to_numpy()


def shuffle_player_band(events, player_index, train_counts, band_low, band_high, seed,
                        cell_columns=("split", "position_id")):
    """T100-shuffled control.

    For players with band_low <= training events < band_high, the player index of their
    events is randomly permuted among all band events that share the same `cell_columns`
    (default: split and role). Players outside the band are untouched. The number of
    dedicated vectors and the number of events using each vector stay exactly the same;
    only the link between a vector and the specific player who acted is destroyed.
    """
    band_players = train_counts[(train_counts >= band_low) & (train_counts < band_high)].index
    in_band = events["player_id"].isin(band_players).to_numpy()
    shuffled_index = player_index.copy()
    rng = np.random.default_rng(seed)
    band_events = events[in_band]
    for _, cell in band_events.groupby(list(cell_columns), sort=True):
        rows = cell.index.to_numpy()
        values_in_cell = shuffled_index[rows]
        shuffled_index[rows] = rng.permutation(values_in_cell)
    return shuffled_index


def build_encoding(events, name, threshold, shuffle_band=None, shuffle_seed=None,
                   shuffle_cell=("split", "position_id")):
    """Build a complete identity encoding. Returns (encoding dict, per-event player index)."""
    train_counts = count_train_events_per_player(events)
    player_vocabulary = build_player_vocabulary(train_counts, threshold)
    player_index = map_to_index(events["player_id"], player_vocabulary)
    if shuffle_band is not None:
        player_index = shuffle_player_band(events, player_index, train_counts,
                                           shuffle_band[0], shuffle_band[1], shuffle_seed, shuffle_cell)
    encoding = {
        "name": name,
        "threshold": threshold,
        "shuffle_band": shuffle_band,
        "shuffle_seed": shuffle_seed,
        "shuffle_cell": list(shuffle_cell) if shuffle_band is not None else None,
        "n_events": int(len(events)),
        "player_vocabulary": {str(key): value for key, value in player_vocabulary.items()},
        "team_vocabulary": {str(key): value for key, value in build_train_vocabulary(events, "team_id").items()},
        "role_vocabulary": {str(key): value for key, value in build_train_vocabulary(events, "position_id").items()},
        "competition_vocabulary": {str(key): value for key, value in build_train_vocabulary(events, "competition_id").items()},
    }
    return encoding, player_index


def save_encoding(folder, encoding, player_index):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "encoding.json", "w") as handle:
        json.dump(encoding, handle, indent=1)
    np.save(folder / "player_index.npy", player_index)


def load_encoding(folder):
    folder = Path(folder)
    with open(folder / "encoding.json") as handle:
        encoding = json.load(handle)
    encoding["player_index"] = np.load(folder / "player_index.npy")
    return encoding


# ---------------------------------------------------------------------------
# Model input features
# ---------------------------------------------------------------------------

# The 15 categorical inputs, in a fixed order.
CATEGORICAL_FEATURES = [
    "player", "team", "role", "competition",                           # identity
    "zone",                                                            # space
    "period", "minute_bin",                                            # time
    "score_diff", "poss_len", "possession_flag",       # game state
    "prev_action", "prev_outcome", "prev_end_zone", "same_player_as_prev",  # history
]
# The 4 continuous inputs.
CONTINUOUS_FEATURES = ["x_norm", "y_norm", "dt", "t_poss"]

# Which source column provides each possession feature (see notebook 01, Section 8.1).
POSSESSION_SOURCE_COLUMNS = {
    "pre_action": {"play_pattern": "prev_play_pattern_id", "poss_len": "prev_poss_len",
                   "possession_flag": "team_change", "t_poss": "t_poss_prev"},
    "original":   {"play_pattern": "play_pattern_id", "poss_len": "poss_len",
                   "possession_flag": "new_possession", "t_poss": "t_poss"},
}


def encode_features(events, encoding, possession_features="pre_action"):
    """Turn the event table into model inputs.

    Returns (features, vocabulary_sizes):
      features          dict name -> numpy array with one value per event
      vocabulary_sizes  dict categorical name -> number of rows its embedding table needs
    """
    source = POSSESSION_SOURCE_COLUMNS[possession_features]
    features = {}

    # identity (index 0 = no dedicated identity / unseen in training)
    features["player"] = encoding["player_index"].astype("int64")
    team_vocabulary = {int(key): value for key, value in encoding["team_vocabulary"].items()}
    role_vocabulary = {int(key): value for key, value in encoding["role_vocabulary"].items()}
    competition_vocabulary = {int(key): value for key, value in encoding["competition_vocabulary"].items()}
    features["team"] = map_to_index(events["team_id"], team_vocabulary)
    features["role"] = map_to_index(events["position_id"], role_vocabulary)
    features["competition"] = map_to_index(events["competition_id"], competition_vocabulary)

    # context
    features["zone"] = events["zone"].to_numpy().astype("int64")
    features["period"] = events["period"].to_numpy().astype("int64")
    features["minute_bin"] = events["minute_bin"].to_numpy().astype("int64")
    features["score_diff"] = events["score_diff_idx"].to_numpy().astype("int64")
    features["play_pattern"] = events[source["play_pattern"]].to_numpy().astype("int64")
    features["poss_len"] = events[source["poss_len"]].to_numpy().astype("int64")
    features["possession_flag"] = events[source["possession_flag"]].to_numpy().astype("int64")
    features["prev_action"] = events["prev_action_id"].to_numpy().astype("int64")
    features["prev_outcome"] = events["prev_outcome"].to_numpy().astype("int64")
    features["prev_end_zone"] = events["prev_end_zone"].to_numpy().astype("int64")
    features["same_player_as_prev"] = events["same_player_as_prev"].to_numpy().astype("int64")

    features["x_norm"] = events["x_norm"].to_numpy().astype("float32")
    features["y_norm"] = events["y_norm"].to_numpy().astype("float32")
    features["dt"] = events["dt"].to_numpy().astype("float32")
    features["t_poss"] = events[source["t_poss"]].to_numpy().astype("float32")

    vocabulary_sizes = {
        "player": len(encoding["player_vocabulary"]) + 1,
        "team": len(encoding["team_vocabulary"]) + 1,
        "role": len(encoding["role_vocabulary"]) + 1,
        "competition": len(encoding["competition_vocabulary"]) + 1,
        "zone": 96,
        "period": 5,            # codes 1-4 (0 unused)
        "minute_bin": 24,
        "score_diff": 7,
        "play_pattern": 10,     # codes 1-9, 0 = no previous event
        "poss_len": 22,         # 0-20, 21 = no previous event
        "possession_flag": 2,
        "prev_action": 16,      # 15 action classes + END
        "prev_outcome": 3,
        "prev_end_zone": 96,
        "same_player_as_prev": 2,
    }
    return features, vocabulary_sizes
