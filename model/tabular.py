# Variant of TransTab + SAINT

import math
import torch
import torch.nn as nn
import torch.nn.init as nn_init

# input -> no pd.DataFrame, just raw tensors


class FeatureEmbedding(nn.Module):
    """Embed fixed-width numerical feature tensors (radiomics) into per-feature token embeddings.

    Adapts the TransTab numerical embedding pipeline for tensor-only input:
    - Replaces BERT-tokenized column names with a learnable per-column index embedding
      (TransTabWordEmbedding role, without tokenizer dependency)
    - Applies TransTabNumEmbedding scaling: emb = col_emb * value + bias
    - Projects through an align_layer (TransTabFeatureProcessor role)

    Output shape: (B, num_features, hidden_dim) — one token per radiomics feature.
    """

    def __init__(self,
        num_features: int,
        hidden_dim: int = 128,
        hidden_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-5,
        device: str = "cuda:0",
    ):
        super().__init__()
        # Learnable per-column embedding (replaces BERT tokenization of column names)
        self.col_embedding = nn.Embedding(num_features, hidden_dim)
        nn_init.kaiming_normal_(self.col_embedding.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Additive bias after value scaling (from TransTabNumEmbedding)
        self.num_bias = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn_init.uniform_(self.num_bias, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))

        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.register_buffer('col_indices', torch.arange(num_features))
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_features) — raw numerical feature tensor
        Returns:
            (B, num_features, hidden_dim) — per-feature token embeddings
        """
        col_emb = self.col_embedding(self.col_indices)              # (F, D)
        col_emb = self.norm(col_emb)
        col_emb = self.dropout(col_emb)
        col_emb = col_emb.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, F, D)

        # TransTabNumEmbedding: scale column embedding by feature value, add bias
        feat_emb = col_emb * x.unsqueeze(-1).float() + self.num_bias  # (B, F, D)
        feat_emb = self.align_layer(feat_emb)
        return feat_emb


class ColumnAttention(nn.Module):
    pass 


class RowAttention(nn.Module):
    pass


class SummaryTableModel(nn.Module): 
    pass 


