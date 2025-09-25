import torch
import torch.nn as nn

from . import normalizers


# -------------------------------
# Basic MLP (kept from your code)
# -------------------------------
class FCNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 128,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        assert n_layers >= 2, "n_layers must be >= 2 (input->...->output)"
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
            + [nn.Linear(hidden_dim, output_dim)]
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc_layers[0](x))
        for i in range(1, self.n_layers - 1):
            x = self.dropout(x)
            x = self.act(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)  # logits
        return x


class EMAFCNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 128,
        dropout_rate: float = 0.5,
        eta: float = 0.99,
    ) -> None:
        super().__init__()
        self.normalization_layer = normalizers.EMANet(
            input_dim=input_dim,
            n_type="minmax",
            use_running_stats=True,
            momentum=eta,
            eps=1e-6,
        )
        self.model = FCNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization_layer(x)
        return self.model(x)


# -------------------------------
# 1) Residual MLP for tabular
# -------------------------------
class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h  # residual


class ResFCNet(nn.Module):
    """
    Residual MLP with Pre-LN, GELU, dropout. Good robust baseline for tabular.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        assert n_layers >= 2
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        num_blocks = max(0, n_layers - 2)  # keep same convention as FCNet
        self.blocks = nn.Sequential(
            *[_ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)  # logits


class EMAResFCNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        eta: float = 0.99,
    ) -> None:
        super().__init__()
        self.normalization_layer = normalizers.EMANet(
            input_dim=input_dim,
            n_type="minmax",
            use_running_stats=True,
            momentum=eta,
            eps=1e-6,
        )
        self.model = ResFCNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalization_layer(x))


# -------------------------------
# 2) Gated MLP (GEGLU) + SE
# -------------------------------
class _SE(nn.Module):
    """Squeeze-and-Excitation over feature channels (hidden dim)."""

    def __init__(self, dim: int, r: int = 8) -> None:
        super().__init__()
        hidden = max(1, dim // r)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D), global squeeze over batch is not meaningful; use channel-wise gate
        w = self.net(x)
        return x * w


class _GEGLU(nn.Module):
    """GEGLU: y = (Wv x) * GELU(Wg x)"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        v, g = torch.chunk(h, 2, dim=-1)
        return v * nn.functional.gelu(g)


class _GatedBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, use_se: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gelu = _GEGLU(dim)
        self.fc = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.use_se = use_se
        self.se = _SE(dim) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.gelu(h)
        h = self.fc(h)
        h = self.drop(h)
        h = self.se(h)
        return x + h


class GLUFCNet(nn.Module):
    """
    Gated MLP with GEGLU blocks + optional Squeeze-and-Excitation.
    Tends to work well on medium-high dimensional tabular data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        assert n_layers >= 2
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        num_blocks = max(0, n_layers - 2)
        self.blocks = nn.Sequential(
            *[_GatedBlock(hidden_dim, dropout_rate, use_se) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class EMAGLUFCNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
        eta: float = 0.99,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.normalization_layer = normalizers.EMANet(
            input_dim=input_dim,
            n_type="minmax",
            use_running_stats=True,
            momentum=eta,
            eps=1e-6,
        )
        self.model = GLUFCNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            use_se=use_se,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalization_layer(x))


# -------------------------------
# 3) FT-Transformer (numerical-only)
#    Per-feature tokenization + Transformer encoder + CLS pooling
# -------------------------------
class _FeatureTokenizer(nn.Module):
    """
    Tokenizes each scalar feature x_j into a d_token vector via an independent linear map.
    Adds a learnable [CLS] token as the first token.
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        # weight: (n_features, d_token), bias: (n_features, d_token)
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, F) -> (N, F, d_token)
        # per-feature linear: token_j = x[:, j][:, None] * W[j] + b[j]
        tokens = x.unsqueeze(-1) * self.weight + self.bias  # broadcast over batch
        # prepend [CLS]
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (N, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)  # (N, 1+F, d_token)
        return tokens


class FTTransformer(nn.Module):
    """
    FT-Transformer for *numerical* tabular data.
    Reference-style idea: tokenize each feature to a vector, add [CLS], run TransformerEncoder, use [CLS] for head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_token: int = 192,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_mult: int = 4,
        dropout_rate: float = 0.1,
        prenorm: bool = True,
    ) -> None:
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"
        self.tokenizer = _FeatureTokenizer(n_features=input_dim, d_token=d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_token,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=prenorm,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tokenize -> (N, 1+F, d_token)
        t = self.tokenizer(x)
        # Transformer
        h = self.encoder(t)  # (N, 1+F, d_token)
        cls = h[:, 0, :]  # (N, d_token)
        return self.head(cls)  # logits


class EMAFTTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_token: int = 192,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_mult: int = 4,
        dropout_rate: float = 0.1,
        eta: float = 0.99,
        prenorm: bool = True,
    ) -> None:
        super().__init__()
        self.normalization_layer = normalizers.EMANet(
            input_dim=input_dim,
            n_type="minmax",
            use_running_stats=True,
            momentum=eta,
            eps=1e-6,
        )
        self.model = FTTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            dropout_rate=dropout_rate,
            prenorm=prenorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalization_layer(x))
