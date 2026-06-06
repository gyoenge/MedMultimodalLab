import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxNeighborAttention(nn.Module):
    """Teacher branch: aggregates UNI embeddings of anchor + neighbors via MHA.

    Produces Z^T — the context-aware teacher embedding for the anchor.
    UNI ViT-L is pre-extracted and frozen; only this module is trained.

    Input batch layout: [anchor | n_neighbors spots | ...]
    Only the first (1 + n_neighbors) rows are used; globals are ignored.

    Args:
        uni_dim:    UNI embedding dimension (1024 for ViT-L).
        num_heads:  Attention heads.
        zs_dim:     Output dimension — must equal 3 * proj_dim to match Z^S.
        dropout:    Attention dropout.
    """

    def __init__(
        self,
        uni_dim: int = 1024,
        num_heads: int = 8,
        zs_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            uni_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(uni_dim)
        self.proj = nn.Sequential(
            nn.Linear(uni_dim, uni_dim // 2),
            nn.GELU(),
            nn.Linear(uni_dim // 2, zs_dim),
        )

    def forward(
        self,
        uni_emb: torch.Tensor,  # (B_total, uni_dim)
        n_neighbors: int,
    ) -> torch.Tensor:          # (zs_dim,) — Z^T for the anchor
        seq = uni_emb[: 1 + n_neighbors].unsqueeze(0)  # (1, 1+n, uni_dim)
        h, _ = self.attn(seq, seq, seq)
        h = self.norm(seq + h)          # (1, 1+n, uni_dim)
        anchor_h = h[0, 0]              # (uni_dim,)
        return F.normalize(self.proj(anchor_h), dim=-1)  # (zs_dim,)
