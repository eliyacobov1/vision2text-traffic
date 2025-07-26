# Cross-modal fusion layers
import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Single cross-attention + feed-forward block."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        attn_out, attn_w = self.attn(q, context, context, need_weights=True)
        self.last_attn = attn_w  # (B, heads, Q, K)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        return x + ff_out


class CrossModalFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads) for _ in range(depth)
        ])

    def forward(self, txt_tokens: torch.Tensor, img_tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            txt_tokens = layer(txt_tokens, img_tokens)
        return txt_tokens

    def get_last_attention(self) -> torch.Tensor | None:
        if self.layers:
            return self.layers[-1].last_attn
        return None
