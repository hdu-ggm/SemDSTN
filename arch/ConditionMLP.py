import torch
from torch import nn

class ConditionalMLP(nn.Module):
    """Conditional MLP with meta_axis support (T, S, ST)."""

    def __init__(self, input_dim, hidden_dim, embed_dim=None, meta_axis=None, dropout=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.meta_axis = meta_axis

        if meta_axis is not None:
            assert embed_dim is not None, "embed_dim must be set when using meta_axis"
            if meta_axis == "ST":
                self.scale_gen = nn.Linear(embed_dim, hidden_dim)
                self.bias_gen = nn.Linear(embed_dim, hidden_dim)
                self.st_fc1 = nn.Conv2d(input_dim, self.hidden_dim, kernel_size=1, bias=True)
                self.st_fc2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=True)
            else:
                # 权重池：embedding_dim → 动态生成 (in_dim, hidden_dim)
                self.weights_pool1 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(embed_dim, input_dim, hidden_dim)))
                self.bias_pool1 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(embed_dim, hidden_dim)))
                self.weights_pool2 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(embed_dim, hidden_dim, hidden_dim)))
                self.bias_pool2 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(embed_dim, hidden_dim)))
        else:
            # 固定权重
            self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=True)
            self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, embeddings=None):
        """
        Args:
            x: (B, D, N, 1)
            embeddings: depends on meta_axis
                - T: (B, E)
                - S: (N, E)
                - ST: (B, N, E)
        """

        if self.meta_axis is None:
            # 常规 MLP
            hidden = self.fc2(self.drop(self.act(self.fc1(x))))
        else:
            if self.meta_axis == "T":
                # batch 特定
                W1 = torch.einsum("be,eih->bih", embeddings, self.weights_pool1)
                b1 = torch.einsum("be,eh->bh", embeddings, self.bias_pool1)
                W2 = torch.einsum("be,eho->bho", embeddings, self.weights_pool2)
                b2 = torch.einsum("be,eo->bo", embeddings, self.bias_pool2)

                hidden = torch.einsum("binp,bih->bhnp", x, W1) + b1[:, :, None, None]
                hidden = self.act(hidden)
                hidden = self.drop(hidden)
                hidden = torch.einsum("bhnp,bho->bonp", hidden, W2) + b2[:, :, None, None]  # B D N 1

            elif self.meta_axis == "S":
                # 节点特定
                W1 = torch.einsum("ne,eih->nih", embeddings, self.weights_pool1)
                b1 = torch.einsum("ne,eh->nh", embeddings, self.bias_pool1)
                W2 = torch.einsum("ne,eho->nho", embeddings, self.weights_pool2)
                b2 = torch.einsum("ne,eo->no", embeddings, self.bias_pool2)

                hidden = torch.einsum("binp,nih->bnhp", x, W1) + b1[None, :, :,None]
                hidden = self.act(hidden)
                hidden = self.drop(hidden)
                hidden = torch.einsum("bnhp,nho->bnop", hidden, W2) + b2[None, :, :, None]
                hidden = hidden.permute(0, 2, 1, 3)  # (B D N 1)

            elif self.meta_axis == "ST":
                # --- FiLM 风格时空联合 ---
                # 先做共享 MLP
                hidden = self.st_fc2(self.drop(self.act(self.st_fc1(x)))) # (B, H, N, 1)

                # 从 embedding 生成 FiLM 参数
                gamma = self.scale_gen(embeddings)  # (B, N, H)
                beta = self.bias_gen(embeddings)  # (B, N, H)

                # reshape 对齐 (B, H, N, 1)
                gamma = gamma.permute(0, 2, 1).unsqueeze(-1)
                beta = beta.permute(0, 2, 1).unsqueeze(-1)

                # FiLM 调制
                hidden = gamma * hidden + beta
            else:
                raise ValueError(f"Unsupported meta_axis: {self.meta_axis}")

        # 残差 + LayerNorm
        hidden = self.norm((hidden + x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return hidden




