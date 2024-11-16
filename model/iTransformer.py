import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.local_global import Seasonal_Prediction, series_decomp_multi


# 原版
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# class Model(nn.Module):
#     """
#     Paper link: https://arxiv.org/abs/2310.06625
#     """
#
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.c_out = configs.c_out
#         self.output_attention = configs.output_attention
#         self.use_norm = configs.use_norm
#         # Embedding
#         self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         self.class_strategy = configs.class_strategy
#         # self.conv1d = nn.Conv1d(in_channels=3, out_channels=21, kernel_size=3, padding=1)
#         self.rnn = nn.GRU(input_size=4, hidden_size=1, batch_first=True)
#         # Encoder-only architecture
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#
#         self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
#
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             # Normalization from Non-stationary Transformer
#             means = x_enc.mean(1, keepdim=True).detach()
#             x_enc = x_enc - means
#             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x_enc /= stdev
#
#         _, _, N = x_enc.shape  # B L N
#         # B: batch_size;    E: d_model;
#         # L: seq_len;       S: pred_len;
#         # N: number of variate (tokens), can also includes covariates
#
#         # Embedding
#         # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens
#
#         # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
#         # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#
#         # B N E -> B N S -> B S N
#         dec_out = self.projector(enc_out).permute(0, 2, 1)  # filter the covariates
#
#         dec_out, _ = self.rnn(dec_out)
#
#         if self.use_norm:
#             # De-Normalization from Non-stationary Transformer
#             dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#             dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#
#         return dec_out
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # [B, L, D]

# 变量趋势分离 + itransformer + 丢弃特征回收（不包括trend）
# class Model(nn.Module):
#     """
#     Paper link: https://arxiv.org/abs/2310.06625
#     """
#
#     def __init__(self, configs, device):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.decomp_kernel = configs.decomp_kernel
#         self.mode = configs.mode
#         self.c_out = configs.c_out
#         self.output_attention = configs.output_attention
#         self.use_norm = configs.use_norm
#         self.decomp_multi = series_decomp_multi(configs.decomp_kernel)
#         # Embedding
#         self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         self.class_strategy = configs.class_strategy
#         # self.conv1d = nn.Conv1d(in_channels=3, out_channels=21, kernel_size=3, padding=1)
#         self.rnn = nn.GRU(input_size=5, hidden_size=21, batch_first=True)
#         self.rnn2 = nn.GRU(input_size=21, hidden_size=512, batch_first=True)
#         self.rnn1 = nn.GRU(input_size=configs.d_model, hidden_size=configs.pred_len, batch_first=True)
#         # Encoder-only architecture
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#
#         self.conv_trans = Seasonal_Prediction(embedding_size=configs.d_model, n_heads=configs.n_heads, dropout=configs.dropout,
#                                               d_layers=configs.d_layers, decomp_kernel=configs.decomp_kernel, c_out=configs.d_model,
#                                               conv_kernel=configs.conv_kernel,
#                                               isometric_kernel=configs.isometric_kernel, device=device)
#
#         self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
#
#         self.regression = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
#         self.regression.weight = nn.Parameter((1 / configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]), requires_grad=True)
#
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             # Normalization from Non-stationary Transformer
#             means = x_enc.mean(1, keepdim=True).detach()
#             x_enc = x_enc - means
#             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x_enc /= stdev
#
#         _, _, N = x_enc.shape  # B L N
#
#         seasonal_init_enc, trend = self.decomp_multi(x_enc)
#         trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
#
#         # Embedding
#         # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
#         enc_out = self.enc_embedding(seasonal_init_enc, x_mark_enc, self.rnn)
#
#         enc_out, _ = self.encoder(enc_out, attn_mask=None)
#         dec_out = self.projector(enc_out).permute(0, 2, 1)
#
#         dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
#
#         if self.use_norm:
#             # De-Normalization from Non-stationary Transformer
#             dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#             dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#
#         return dec_out
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # [B, L, D]

# 变量趋势分离 + itransformer + 丢弃特征回收
# class Model(nn.Module):
#     """
#     Paper link: https://arxiv.org/abs/2310.06625
#     """
#
#     def __init__(self, configs, device):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.c_out = configs.c_out
#         self.output_attention = configs.output_attention
#         self.use_norm = configs.use_norm
#         # Embedding
#         self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         self.decomp_multi = series_decomp_multi(configs.decomp_kernel)
#         self.class_strategy = configs.class_strategy
#         # self.conv1d = nn.Conv1d(in_channels=3, out_channels=21, kernel_size=3, padding=1)
#         self.rnn = nn.GRU(input_size=5, hidden_size=21, batch_first=True)
#         # Encoder-only architecture
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#
#         self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
#
#         self.regression = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
#         self.regression.weight = nn.Parameter((1 / configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]), requires_grad=True)
#
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             # Normalization from Non-stationary Transformer
#             means = x_enc.mean(1, keepdim=True).detach()
#             x_enc = x_enc - means
#             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x_enc /= stdev
#
#         _, _, N = x_enc.shape  # B L N
#
#         seasonal_init_enc, trend = self.decomp_multi(x_enc)
#         # trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
#
#         # Embedding
#         # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
#         seasonal_enc_out = self.enc_embedding(seasonal_init_enc, x_mark_enc)
#         trend_enc_out = self.enc_embedding(trend, x_mark_enc)
#         enc_out = seasonal_enc_out + trend_enc_out
#         enc_out, _ = self.encoder(enc_out, attn_mask=None)
#         # B N E -> B N S -> B S N
#         dec_out = self.projector(enc_out).permute(0, 2, 1)  # filter the covariates
#
#         # 处理丢弃的变量（N:），即从N到最后的部分
#         discarded_data = dec_out[:, :, N:]  # 获取被丢弃的变量
#
#         dec_out = dec_out[:, :, :N]
#         mapped_data, _ = self.rnn(discarded_data)
#
#         dec_out = dec_out + mapped_data
#
#         if self.use_norm:
#             # De-Normalization from Non-stationary Transformer
#             dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#             dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#
#         return dec_out
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# 变量趋势分离 + itransformer + 丢弃特征回收
# class Model(nn.Module):
#     """
#     Paper link: https://arxiv.org/abs/2310.06625
#     """
#
#     def __init__(self, configs, device):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.decomp_kernel = configs.decomp_kernel
#         self.mode = configs.mode
#         self.c_out = configs.c_out
#         self.output_attention = configs.output_attention
#         self.use_norm = configs.use_norm
#         self.decomp_multi = series_decomp_multi(configs.decomp_kernel)
#         # Embedding
#         self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         self.class_strategy = configs.class_strategy
#         self.rnn = nn.GRU(input_size=5, hidden_size=21, batch_first=True)
#         # Encoder-only architecture
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#
#         self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
#
#         self.regression = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
#         self.regression.weight = nn.Parameter((1 / configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]), requires_grad=True)
#
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             # Normalization from Non-stationary Transformer
#             means = x_enc.mean(1, keepdim=True).detach()
#             x_enc = x_enc - means
#             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x_enc /= stdev
#
#         _, _, N = x_enc.shape  # B L N
#
#         seasonal_init_enc, trend = self.decomp_multi(x_enc)
#         trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
#
#         # Embedding
#         # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
#         enc_out = self.enc_embedding(seasonal_init_enc, x_mark_enc)
#
#         enc_out, _ = self.encoder(enc_out, attn_mask=None)
#         # B N E -> B N S -> B S N
#         dec_out = self.projector(enc_out).permute(0, 2, 1)  # filter the covariates
#
#         # 处理丢弃的变量（N:），即从N到最后的部分
#         discarded_data = dec_out[:, :, N:]  # 获取被丢弃的变量
#
#         dec_out = dec_out[:, :, :N]
#         mapped_data, _ = self.rnn(discarded_data)
#
#         dec_out = dec_out + mapped_data + trend
#
#         if self.use_norm:
#             # De-Normalization from Non-stationary Transformer
#             dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#             dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#
#         return dec_out
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # [B, L, D]