"""
mi_analysis/src/conflict.py — reference predictors and conflict-set construction (MB01).

Two small classifiers ("reference predictors") are trained to label events from two disjoint
views of the input: who is acting (identity) and the situation they act in (context). A third,
player-only identity variant is trained alongside the full identity model for later player-level
analysis, but does not take part in the conflict definition itself. Reference predictors are
deliberately simple (embeddings + a shallow head) and are never the model under study, so the
conflict definition below cannot be circular with respect to it.

Architecture and training: 64-dim embeddings, Adam (lr 3e-3), batch 8192, 25 epochs, unweighted
cross-entropy (so each model's predicted probabilities reflect true class frequencies, not a
re-weighted objective), keeping the epoch with the lowest validation NLL, then one temperature per
model fitted on validation NLL.

The context reference reads only PRE-ACTION columns -- computed from the event BEFORE the one
being predicted (`team_change`, `prev_poss_len`, `t_poss_prev`, ...). It never reads a column
that StatsBomb assigns only once the action is known (after-the-action possession bookkeeping, or
the play pattern), because those would partly leak the answer. Identity reads the frozen `*_ix`
vocabulary indices events_prepared.parquet already carries (0 = no dedicated index for that id).
"""
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

NUM_ACTIONS = 15
EMBEDDING_DIM = 64
LEARNING_RATE = 3e-3
BATCH_SIZE = 8192

IDENTITY_COLUMNS = {"player": "player_ix", "team": "team_ix", "role": "role_ix", "competition": "comp_ix"}
CONTEXT_CATEGORICAL_COLUMNS = {
    "zone": "zone", "period": "period", "minute_bin": "minute_bin", "score_diff": "score_diff_idx",
    "poss_len": "prev_poss_len", "possession_flag": "team_change", "prev_action": "prev_action_id",
    "prev_outcome": "prev_outcome", "prev_end_zone": "prev_end_zone", "same_player_as_prev": "same_player_as_prev",
}
CONTEXT_CONTINUOUS_COLUMNS = {"x_norm": "x_norm", "y_norm": "y_norm", "dt": "dt", "t_poss": "t_poss_prev"}
AFTER_ACTION_COLUMNS = {"new_possession": "team_change", "poss_len": "prev_poss_len",
                        "play_pattern_id": None, "t_poss": "t_poss_prev"}

IDENTITY_INPUTS = list(IDENTITY_COLUMNS)
CONTEXT_CATEGORICAL = list(CONTEXT_CATEGORICAL_COLUMNS)
CONTEXT_CONTINUOUS = list(CONTEXT_CONTINUOUS_COLUMNS)

TAU_PERCENTILES = [50, 75, 80, 85, 90, 95]
TAU_RATE_BAND = (0.10, 0.25)
TAU_TOP_PAIR_LIMIT = 0.45


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vocabulary_sizes_from(events):
    """Embedding-table sizes read straight off the data (max index + 1), no encoding file needed."""
    sizes = {}
    for name, column in {**IDENTITY_COLUMNS, **CONTEXT_CATEGORICAL_COLUMNS}.items():
        sizes[name] = int(events[column].max()) + 1
    return sizes


class IdFull(nn.Module):
    """Identity reference: player + team + role + competition embeddings, summed, one linear layer."""

    kind = "identity"

    def __init__(self, sizes):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        for name in IDENTITY_INPUTS:
            self.embeddings[name] = nn.Embedding(sizes[name], EMBEDDING_DIM)
        self.head = nn.Linear(EMBEDDING_DIM, NUM_ACTIONS)

    def forward(self, categorical, continuous):
        total = None
        for column, name in enumerate(IDENTITY_INPUTS):
            vector = self.embeddings[name](categorical[:, column])
            total = vector if total is None else total + vector
        return self.head(total)


class IdPlayer(nn.Module):
    """Player-only identity reference."""

    kind = "player"

    def __init__(self, sizes):
        super().__init__()
        self.embedding = nn.Embedding(sizes["player"], EMBEDDING_DIM)
        self.head = nn.Linear(EMBEDDING_DIM, NUM_ACTIONS)

    def forward(self, categorical, continuous):
        return self.head(self.embedding(categorical[:, 0]))


class Ctx(nn.Module):
    """Context reference on pre-action inputs only (no identity, no after-action possession fields)."""

    kind = "context"

    def __init__(self, sizes):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        for name in CONTEXT_CATEGORICAL:
            self.embeddings[name] = nn.Embedding(sizes[name], EMBEDDING_DIM)
        self.continuous_projection = nn.Linear(len(CONTEXT_CONTINUOUS), EMBEDDING_DIM)
        self.head = nn.Sequential(nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM), nn.ReLU(),
                                  nn.Linear(EMBEDDING_DIM, NUM_ACTIONS))

    def forward(self, categorical, continuous):
        total = self.continuous_projection(continuous)
        for column, name in enumerate(CONTEXT_CATEGORICAL):
            total = total + self.embeddings[name](categorical[:, column])
        return self.head(total)


def input_columns(kind):
    """The events_prepared.parquet columns each reference reads (asserted against leakage in MB01)."""
    if kind == "identity":
        return [IDENTITY_COLUMNS[n] for n in IDENTITY_INPUTS], []
    if kind == "player":
        return [IDENTITY_COLUMNS["player"]], []
    return ([CONTEXT_CATEGORICAL_COLUMNS[n] for n in CONTEXT_CATEGORICAL],
            [CONTEXT_CONTINUOUS_COLUMNS[n] for n in CONTEXT_CONTINUOUS])


def reference_tensors(events, rows, kind):
    """(categorical LongTensor [n, k], continuous FloatTensor [n, m]) for the given event rows."""
    categorical_columns, continuous_columns = input_columns(kind)
    part = events.iloc[rows]
    categorical = np.stack([part[c].to_numpy() for c in categorical_columns], axis=1).astype(np.int64)
    if continuous_columns:
        continuous = np.stack([part[c].to_numpy() for c in continuous_columns], axis=1).astype(np.float32)
    else:
        continuous = np.zeros((len(rows), 0), dtype=np.float32)
    return torch.as_tensor(categorical), torch.as_tensor(continuous)


def train_reference(model, events, targets, train_rows, val_rows, epochs, seed, device, log=print):
    """Adam, unweighted cross-entropy, keep the best validation-NLL epoch."""
    set_seed(seed)
    kind = model.kind
    train_categorical, train_continuous = reference_tensors(events, train_rows, kind)
    val_categorical, val_continuous = reference_tensors(events, val_rows, kind)
    train_targets = torch.as_tensor(targets[train_rows].astype(np.int64))
    val_targets = torch.as_tensor(targets[val_rows].astype(np.int64))
    dataset = torch.utils.data.TensorDataset(train_categorical, train_continuous, train_targets)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    best_nll = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for categorical, continuous, y in loader:
            categorical, continuous, y = categorical.to(device), continuous.to(device), y.to(device)
            optimiser.zero_grad()
            loss = loss_function(model(categorical, continuous), y)
            loss.backward()
            optimiser.step()
            loss_sum += loss.item() * len(y)
            count += len(y)
        val_logits = predict_logits(model, val_categorical, val_continuous, device)
        val_nll = float(nn.functional.cross_entropy(torch.as_tensor(val_logits), val_targets))
        history.append({"epoch": epoch + 1, "train_nll": loss_sum / count, "val_nll": val_nll})
        log(f"  {type(model).__name__} seed {seed} epoch {epoch + 1:2d}: train {loss_sum / count:.4f}  val {val_nll:.4f}")
        if val_nll < best_nll:
            best_nll = val_nll
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def predict_logits(model, categorical, continuous, device, batch_size=65536):
    model.eval()
    outputs = []
    for start in range(0, len(categorical), batch_size):
        outputs.append(model(categorical[start:start + batch_size].to(device),
                             continuous[start:start + batch_size].to(device)).cpu().numpy())
    return np.concatenate(outputs)


def logits_for_rows(model, events, rows, device):
    categorical, continuous = reference_tensors(events, rows, model.kind)
    return predict_logits(model, categorical, continuous, device)


def fit_temperature(logits, targets):
    """One scalar T minimising validation cross-entropy of logits / T."""
    logits_tensor = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    targets_tensor = torch.as_tensor(np.asarray(targets, dtype=np.int64))

    def nll_at(temperature):
        return nn.functional.cross_entropy(logits_tensor / temperature, targets_tensor).item()

    result = minimize_scalar(nll_at, bounds=(0.05, 10), method="bounded", options={"xatol": 1e-3})
    return float(result.x), nll_at(1.0), nll_at(float(result.x))


def softmax(logits, temperature=1.0):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def ece(probabilities, targets, bins=15):
    """Expected calibration error, bins open on the left."""
    confidence = probabilities.max(1)
    predicted = probabilities.argmax(1)
    total = 0.0
    for b in range(bins):
        in_bin = (confidence > b / bins) & (confidence <= (b + 1) / bins)
        if in_bin.sum():
            total += in_bin.mean() * abs(confidence[in_bin].mean() - (predicted[in_bin] == targets[in_bin]).mean())
    return float(total)


def reference_report(probabilities, targets):
    predicted = probabilities.argmax(1)
    top3 = np.argsort(-probabilities, axis=1)[:, :3]
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, None)).mean()
    recall = {}
    for action in range(NUM_ACTIONS):
        members = targets == action
        recall[action] = float((predicted[members] == action).mean()) if members.sum() else None
    return {"accuracy": float((predicted == targets).mean()),
            "top3_accuracy": float((top3 == targets[:, None]).any(1).mean()),
            "nll": float(nll), "ece": ece(probabilities, targets), "recall_per_class": recall}


def jsd(p, q):
    """Jensen-Shannon divergence in nats (0 .. ln 2), computed directly (not scipy's sqrt-distance form)."""
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m + 1e-12), 1) + np.sum(q * np.log(q / m + 1e-12), 1))


def select_tau(p_identity, p_context):
    """Choose tau on VALIDATION distributions only.

    Among rows whose argmaxes differ, take JSD percentiles 50/75/80/85/90/95; choose the first
    whose conflict rate lies in [0.10, 0.25] and whose most common (a_id, a_ctx) pair holds
    < 45% of conflicts; otherwise the 80th percentile. Returns (tau, percentile, sweep_rows, rate).
    """
    a_id = p_identity.argmax(1)
    a_ctx = p_context.argmax(1)
    divergence = jsd(p_identity, p_context)
    disagree = a_id != a_ctx
    thresholds = np.percentile(divergence[disagree], TAU_PERCENTILES)
    rows = []
    for percentile, threshold in zip(TAU_PERCENTILES, thresholds):
        conflict = disagree & (divergence > threshold)
        rate = conflict.sum() / len(conflict)
        if conflict.sum():
            pairs = {}
            for i, c in zip(a_id[conflict], a_ctx[conflict]):
                pairs[(int(i), int(c))] = pairs.get((int(i), int(c)), 0) + 1
            top_share = max(pairs.values()) / conflict.sum()
        else:
            top_share = 1.0
        rows.append({"percentile": percentile, "tau": float(threshold), "n_conflicts": int(conflict.sum()),
                     "conflict_rate": round(float(rate), 4), "top_pair_share": round(float(top_share), 3)})
    tau = None
    chosen = None
    for row in rows:
        if TAU_RATE_BAND[0] <= row["conflict_rate"] <= TAU_RATE_BAND[1] and row["top_pair_share"] < TAU_TOP_PAIR_LIMIT:
            tau = row["tau"]
            chosen = row["percentile"]
            break
    if tau is None:
        tau = rows[2]["tau"]
        chosen = rows[2]["percentile"]
    return tau, chosen, rows, float(disagree.mean())


def conflict_mask(p_identity, p_context, tau):
    """conflict = (argmax(P_identity) != argmax(P_context)) AND (JSD(P_identity, P_context) > tau)."""
    a_id = p_identity.argmax(1)
    a_ctx = p_context.argmax(1)
    divergence = jsd(p_identity, p_context)
    return (a_id != a_ctx) & (divergence > tau), a_id, a_ctx, divergence
