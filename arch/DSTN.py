from math import ceil

import torch
from einops import rearrange
from torch import nn
from torch.version import cuda

from baselines.LSTNN.arch.mlp import MultiLayerPerceptron
from baselines.Demo.arch.ConditionMLP import ConditionalMLP


class DSTEncoder(nn.Module):
    def __init__(self, td_size, td_codebook, dw_codebook, spa_codebook, if_time_in_day, if_day_in_week, if_spatial,
                 input_dim, input_len, d_d, d_td, d_dw, d_spa, output_len, num_layer):
        super(DSTEncoder, self).__init__()
        self.td_codebook = td_codebook
        self.dw_codebook = dw_codebook
        self.spa_codebook = spa_codebook
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.output_len = output_len
        self.td_size = td_size
        self.num_layer = num_layer

        self.data_embedding_layer = nn.Conv2d(in_channels=input_dim*input_len, out_channels=d_d, kernel_size=(1, 1), bias=True)
        self.hidden_dim = d_d + d_dw*int(self.if_day_in_week)*2 + d_td*int(self.if_time_in_day)*2

        self.spatial_encoder = nn.Sequential(
            *[ConditionalMLP(d_d+d_spa*int(self.if_spatial), d_d+d_spa*int(self.if_spatial), embed_dim=d_spa, meta_axis="S") for _ in range(num_layer)])

        self.temporal_encoder = nn.Sequential(
            *[ConditionalMLP(self.hidden_dim + d_spa * int(self.if_spatial), self.hidden_dim + d_spa * int(self.if_spatial), embed_dim=d_td + d_dw, meta_axis="T") for _ in range(num_layer)])  # +d_spa*int(self.if_spatial)

        self.data_encoder = nn.Sequential(
            *[ConditionalMLP(d_d, d_d) for _ in range(num_layer)]
        )

        self.st_encoder = nn.Sequential(
            *[ConditionalMLP(self.hidden_dim + d_spa * int(self.if_spatial), self.hidden_dim + d_spa * int(self.if_spatial), embed_dim=d_spa+d_td+d_dw, meta_axis="ST") for _ in range(num_layer)])

        self.projection1 = nn.Conv2d(in_channels=(self.hidden_dim+d_spa*int(self.if_spatial))+d_td+d_dw, out_channels=output_len, kernel_size=(1, 1), bias=True)
        self.text_proj = nn.Linear(384  , d_spa)  # 384 -> 128


    def forward(self, patch_input):
        # B L N C
        batch_size, _, _, _ = patch_input.shape

        # Temporal Embedding
        day_in_week_data = patch_input[..., 2]  # B L N
        day_in_week_start_emb = self.dw_codebook[(day_in_week_data[:, 0, :]).type(torch.LongTensor)]  # B N D
        day_in_week_end_emb = self.dw_codebook[(day_in_week_data[:, -1, :]).type(torch.LongTensor)]  # B N D
        future_day_in_week_emb = day_in_week_end_emb.permute(0, 2, 1).unsqueeze(-1)


        time_in_day_data = patch_input[..., 1]  # B L N
        time_in_day_start_emb = self.td_codebook[(time_in_day_data[:, 0, :] * self.td_size).type(torch.LongTensor)]  # 查询每一个Patch的第一个（当前时间点）的time-day-index 0-287  B N D
        time_in_day_end_emb = self.td_codebook[(time_in_day_data[:, -1, :] * self.td_size).type(torch.LongTensor)]  # 查询每一个Patch的最后一个（当前时间点）的time-day-index 0-287  B N D
        # 查询未来数据的最后一个时间点的time-day-index B D N 1
        future_time_in_day_emb = self.td_codebook[((time_in_day_data[:, -1, :] * self.td_size + self.output_len) % self.td_size).type(torch.LongTensor)].permute(0, 2, 1).unsqueeze(-1)


        # Spatial Embedding
        text_emb = self.text_proj(self.spa_codebook)

        spatial_emb = text_emb.unsqueeze(0).expand(batch_size, -1, -1)  # B N D
        # spatial_emb = self.spa_codebook.unsqueeze(0).expand(batch_size, -1, -1)  # B N D

        # time series embedding
        data_emb = self.data_embedding_layer(torch.concat((patch_input[..., 0], patch_input[..., 1], patch_input[..., 2]), dim=1).unsqueeze(-1))  # B d_d N 1
        for i in range(self.num_layer):
            data_emb = self.data_encoder[i](data_emb)

        # spatial encoding
        hidden = torch.concat((data_emb, spatial_emb.unsqueeze(-1).permute(0, 2, 1, 3)), dim=1)
        for i in range(self.num_layer):
            hidden = self.spatial_encoder[i](hidden, text_emb)  # B D N 1

        # temporal encoding
        hidden = torch.concat(
            (time_in_day_start_emb, day_in_week_start_emb, hidden.squeeze(-1).permute(0, 2, 1), time_in_day_end_emb, day_in_week_end_emb),
            dim=-1).permute(0, 2, 1).unsqueeze(-1)  # B D N 1
        tem_emb = torch.concat((time_in_day_end_emb[:, 0, :], day_in_week_end_emb[:, 0, :]), dim=-1)
        for i in range(self.num_layer):
            hidden = self.temporal_encoder[i](hidden, tem_emb)   # B D N 1

        # ST_encoder
        for i in range(self.num_layer):
            hidden = self.st_encoder[i](hidden, torch.concat((spatial_emb, time_in_day_end_emb, day_in_week_end_emb),dim=-1))

        hidden = torch.concat((hidden, future_time_in_day_emb, future_day_in_week_emb), dim=1)
        predict = self.projection1(hidden)

        return predict

