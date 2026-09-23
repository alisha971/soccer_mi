"""
src/model.py — custom decoder-only Transformer for next-action prediction (notebook 03).

Every step is written with explicit tensor operations so the computation can be read and
verified. Shapes use: B = batch size, T = positions (context length), D = d_model,
H = number of heads, E = head dimension (D / H), F = d_ff, C = number of action classes.
"""
import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

IDENTITY_FEATURES = ["player", "team", "role", "competition"]

# Categorical inputs belonging to each factor group.
FACTOR_GROUPS = {
    "player": ["player"],
    "team": ["team"],
    "role": ["role"],
    "competition": ["competition"],
    "space": ["zone"],
    "time": ["period", "minute_bin"],
    "state": ["score_diff", "poss_len", "possession_flag"],
    "history": ["prev_action", "prev_outcome", "prev_end_zone", "same_player_as_prev"],
}
# Continuous inputs belonging to each factor group (projected by a bias-free linear layer).
CONTINUOUS_GROUPS = {
    "space": ["x_norm", "y_norm"],
    "time": ["dt", "t_poss"],
}
ALL_GROUPS = ["player", "team", "role", "competition", "space", "time", "state", "history"]


@dataclass
class ModelConfig:
    vocabulary_sizes: dict
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    context_length: int = 32
    n_actions: int = 15
    use_groups: list = field(default_factory=lambda: list(ALL_GROUPS))
    use_position: bool = True
    init_std: float = 0.02

    def to_dict(self):
        return asdict(self)


class FactorizedEventEmbedding(nn.Module):
    """Input vector of each event = sum of one vector per factor group + a position vector."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.groups = list(config.use_groups)
        self.embeddings = nn.ModuleDict()
        for group in self.groups:
            for feature in FACTOR_GROUPS[group]:
                table_size = config.vocabulary_sizes[feature]
                if feature in IDENTITY_FEATURES:
                    # index 0 = "no dedicated identity": a fixed zero vector, never trained
                    self.embeddings[feature] = nn.Embedding(table_size, config.d_model, padding_idx=0)
                else:
                    self.embeddings[feature] = nn.Embedding(table_size, config.d_model)
        self.continuous_projections = nn.ModuleDict()
        for group, names in CONTINUOUS_GROUPS.items():
            if group in self.groups:
                self.continuous_projections[group] = nn.Linear(len(names), config.d_model, bias=False)
        if config.use_position:
            self.position_embedding = nn.Embedding(config.context_length, config.d_model)

    def forward(self, inputs):
        """inputs: dict name -> tensor [B, T]. Returns (total [B, T, D], factor_vectors dict)."""
        factor_vectors = {}
        for group in self.groups:
            group_vector = None
            for feature in FACTOR_GROUPS[group]:
                vector = self.embeddings[feature](inputs[feature])            # [B, T, D]
                if group_vector is None:
                    group_vector = vector
                else:
                    group_vector = group_vector + vector
            if group in self.continuous_projections:
                values = torch.stack([inputs[name] for name in CONTINUOUS_GROUPS[group]], dim=-1)   # [B, T, 2]
                group_vector = group_vector + self.continuous_projections[group](values)
            factor_vectors[group] = group_vector

        any_input = inputs["zone"]
        batch_size = any_input.shape[0]
        n_positions = any_input.shape[1]
        if self.config.use_position:
            positions = torch.arange(n_positions, device=any_input.device)
            position_vectors = self.position_embedding(positions)             # [T, D]
            factor_vectors["position"] = position_vectors.unsqueeze(0).expand(batch_size, n_positions, -1)

        total = None
        for name in factor_vectors:
            if total is None:
                total = factor_vectors[name]
            else:
                total = total + factor_vectors[name]
        return total, factor_vectors

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)
        self.W_O = nn.Linear(config.d_model, config.d_model)
        # causal_mask[i, j] is True when position i may look at position j (j <= i)
        causal_mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def split_heads(self, tensor):
        """[B, T, D] -> [B, H, T, E]"""
        batch_size, n_positions, _ = tensor.shape
        tensor = tensor.view(batch_size, n_positions, self.n_heads, self.d_head)
        return tensor.transpose(1, 2)

    def forward(self, x, return_probabilities=False):
        batch_size, n_positions, d_model = x.shape

        queries = self.split_heads(self.W_Q(x))                      # [B, H, T, E]
        keys = self.split_heads(self.W_K(x))                         # [B, H, T, E]
        values = self.split_heads(self.W_V(x))                       # [B, H, T, E]

        scores = torch.matmul(queries, keys.transpose(-2, -1))       # [B, H, T, T]
        scores = scores / math.sqrt(self.d_head)

        mask = self.causal_mask[:n_positions, :n_positions]
        scores = scores.masked_fill(~mask, float("-inf"))            # future positions -> -inf

        probabilities = torch.softmax(scores, dim=-1)                # rows sum to 1
        head_outputs = torch.matmul(probabilities, values)           # [B, H, T, E]

        head_outputs = head_outputs.transpose(1, 2)                  # [B, T, H, E]
        concatenated = head_outputs.reshape(batch_size, n_positions, d_model)
        output = self.W_O(concatenated)                              # [B, T, D]

        if return_probabilities:
            return output, probabilities
        return output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layer = nn.Linear(config.d_model, config.d_ff)     # D -> F
        self.output_layer = nn.Linear(config.d_ff, config.d_model)    # F -> D

    def forward(self, x):
        hidden = self.input_layer(x)
        hidden = F.gelu(hidden)
        return self.output_layer(hidden)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: residual += Attention(LN(residual)); residual += MLP(LN(residual))."""

    def __init__(self, config):
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, residual, internals=None):
        attention_input = self.ln_attention(residual)
        if internals is not None:
            attention_output, probabilities = self.attention(attention_input, return_probabilities=True)
        else:
            attention_output = self.attention(attention_input)
        residual_mid = residual + attention_output

        mlp_input = self.ln_mlp(residual_mid)
        mlp_output = self.mlp(mlp_input)
        residual_post = residual_mid + mlp_output

        if internals is not None:
            internals.append({
                "resid_pre": residual, "attention_input": attention_input,
                "attention_probabilities": probabilities, "attention_output": attention_output,
                "resid_mid": residual_mid, "mlp_input": mlp_input, "mlp_output": mlp_output,
                "resid_post": residual_post,
            })
        return residual_post


class EventTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = FactorizedEventEmbedding(config)
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            self.blocks.append(TransformerBlock(config))
        self.ln_final = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.n_actions)    # D -> C
        self.initialise_weights()

    def initialise_weights(self):
        std = self.config.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
        # layers that write into the residual stream get a smaller scale
        residual_std = std / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.W_O.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.output_layer.weight, mean=0.0, std=residual_std)

    def forward(self, inputs, return_internals=False):
        """inputs: dict name -> [B, T]. Returns logits [B, T, C] (and internals if requested)."""
        residual, factor_vectors = self.embedding(inputs)
        internals = [] if return_internals else None
        for block in self.blocks:
            residual = block(residual, internals)
        final = self.ln_final(residual)
        logits = self.unembed(final)
        if return_internals:
            return logits, {"embedding": residual if self.config.n_layers == 0 else internals[0]["resid_pre"],
                            "factor_vectors": factor_vectors, "blocks": internals,
                            "final_residual": residual, "final_normalised": final}
        return logits


def compute_loss(logits, targets, valid_mask=None):
    """Mean cross-entropy over valid positions. logits [B, T, C], targets [B, T]."""
    n_classes = logits.shape[-1]
    flat_logits = logits.reshape(-1, n_classes)
    flat_targets = targets.reshape(-1)
    per_position = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    if valid_mask is None:
        return per_position.mean()
    flat_mask = valid_mask.reshape(-1).float()
    return (per_position * flat_mask).sum() / flat_mask.sum()


def count_parameters(model):
    """Parameter counts by component. Fixed zero rows (identity index 0) are reported separately."""
    counts = {}
    counts["embeddings (tables + projections + positions)"] = sum(p.numel() for p in model.embedding.parameters())
    counts["transformer blocks"] = sum(p.numel() for p in model.blocks.parameters())
    counts["final LayerNorm"] = sum(p.numel() for p in model.ln_final.parameters())
    counts["unembedding"] = sum(p.numel() for p in model.unembed.parameters())
    total = sum(p.numel() for p in model.parameters())
    fixed_zero = 0
    for module in model.embedding.embeddings.values():
        if module.padding_idx is not None:
            fixed_zero += module.embedding_dim
    counts["total"] = total
    counts["fixed zero rows (never trained)"] = fixed_zero
    counts["trainable"] = total - fixed_zero
    return counts
