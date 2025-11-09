
import numpy as np
from math import ceil

import torch
from openpyxl.styles.builtins import output
from torch import nn
from baselines.Demo.arch.DSTN import DSTEncoder


class Main(nn.Module):
    #  ,**model_args
    def __init__(self, **model_args):
        super(Main, self).__init__()
        self.node_size = model_args["node_size"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.output_len = model_args["output_len"]
        self.td_size = model_args["td_size"]
        self.dw_size = model_args["dw_size"]
        self.d_td = model_args["d_td"]
        self.d_dw = model_args["d_dw"]
        self.d_d = model_args["d_d"]
        self.d_spa = model_args["d_spa"]

        self.if_time_in_day = model_args["if_time_in_day"]
        self.if_day_in_week = model_args["if_day_in_week"]
        self.if_spatial = model_args["if_spatial"]
        self.num_layer = model_args["num_layer"]

        # temporal embeddings
        self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
        nn.init.xavier_uniform_(self.td_codebook)

        self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
        nn.init.xavier_uniform_(self.dw_codebook)

        # spatial embeddings
        # if self.if_spatial:
        #     self.spa_codebook = nn.Parameter(torch.empty(self.node_size, self.d_spa))
        #     nn.init.xavier_uniform_(self.spa_codebook)
        self.embedding = np.load("/home/ggm/Code/BasicTS/datasets/GBA/node_embeddings_st.npy")
        self.embedding = torch.tensor(self.embedding, dtype=torch.float32)
        self.spa_codebook = nn.Parameter(self.embedding, requires_grad=True)

        # Encoder
        self.patch_encoder = DSTEncoder(self.td_size, self.td_codebook, self.dw_codebook, self.spa_codebook,
                                          self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          self.input_dim, self.input_len, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        # Residual
        self.residual = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        Args:   history_data (torch.Tensor): history data with shape [B, L, N, C]
        Returns:    torch.Tensor: prediction wit shape [B, L, N, C]
        """
        # prepare data
        input_data = history_data[..., range(self.input_dim)]
        patch_predict = self.patch_encoder(input_data)  # B T N 1

        # Residual
        output = patch_predict + self.residual(input_data[...,0].unsqueeze(-1))

        return output



