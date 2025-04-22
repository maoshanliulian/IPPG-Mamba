
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
from sklearn.ensemble import RandomForestRegressor
from wtconv.wtconv1d import WTConv1d

@dataclass
class ModelArgs:

    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: int | str = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):

        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':

            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()

        self.args = args


        self.embedding = nn.Embedding(args.vocab_size, args.d_model)

        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])

        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

        self.lm_head.weight = self.embedding.weight

        self.ff=nn.Sequential(
            nn.Dropout(0.3),

        )
        self.ffa = nn.Sequential(
            nn.Dropout(0.3),

        )
        self.ffc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64000, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        self.fc = nn.Sequential(

            nn.Flatten(),
            nn.Linear(4, 1),

        )
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ELU(),

            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, input_ids):


        def calculate_hrv_metrics_and_append(ppg_signals, sampling_rate=30):
            def detect_peaks(ppg_signal):  # 0.2  0.5 0
                if isinstance(ppg_signal, torch.Tensor):
                    ppg_signal = ppg_signal.cpu().numpy()  #1
                diff_signal = np.diff(ppg_signal)
                mean = np.mean(ppg_signal)
                peaks = []
                for i in range(1, len(diff_signal) - 1):
                    if diff_signal[i - 1] > 0 and diff_signal[i + 1] < 0 and ppg_signal[i] > mean:
                        if len(peaks) == 0 or (i - peaks[-1]) > 0.2 * sampling_rate:
                            peaks.append(i)
                return np.array(peaks)


            updated_signal = []
            for signal_idx in range(ppg_signals.shape[0]):

                ppg_signal = ppg_signals[signal_idx]


                peaks = detect_peaks(ppg_signal)
                rr_intervals = np.diff(peaks) / sampling_rate


                if len(rr_intervals) < 2:
                    hrv_values = [None]*7
                else:

                    sdnn = np.std(rr_intervals)
                    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
                    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)
                    pnn50 = (nn50 / len(rr_intervals)) * 100


                    lomb_scargle = LombScargle(np.arange(len(rr_intervals)), rr_intervals)
                    frequencies, power = lomb_scargle.autopower()


                    lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
                    hf_band = (frequencies >= 0.15) & (frequencies < 0.4)
                    lf_power = np.trapz(power[lf_band], frequencies[lf_band])
                    hf_power = np.trapz(power[hf_band], frequencies[hf_band])
                    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

                    hrv_values = [nn50,sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio]

                    hrv_values = torch.tensor(hrv_values, dtype=torch.float32)

                updated_signal.append(hrv_values)


            updated_signal = np.array(updated_signal, dtype=np.float32)
            updated_signal_tensor = torch.tensor(updated_signal, dtype=torch.float32)




            return updated_signal

        ##################

        ppg_HRV= calculate_hrv_metrics_and_append(input_ids)
        input_ids = input_ids.long()

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        x=self.ff(x)
        logits = self.lm_head(x)
        logits  = self.ffa(logits)




        logits = self.global_avg_pool(logits)


        logits = logits.squeeze(-1)

        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        ppg_HRV_tensor = torch.tensor(ppg_HRV, dtype=torch.float32)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logits_tensor = logits_tensor.to(device)
        ppg_HRV_tensor = ppg_HRV_tensor.to(device)

        concatenated_tensor = torch.cat((logits_tensor, ppg_HRV_tensor), dim=1)
        output =self.network(concatenated_tensor)
        return output


    @staticmethod
    def from_pretrained(pretrained_model_name: str):

        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))


        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model

########################

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()

        self.args = args

        self.mixer = MambaBlock(args)

        self.norm = RMSNorm(args.d_model)


    def forward(self, x):


        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 保存模型参数
        self.args = args
        # 输入线性变换层
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.model = nn.Sequential(
            WTConv1d(in_channels=args.d_inner, out_channels=args.d_inner, kernel_size=3, wt_levels=3),
            nn.BatchNorm1d(args.d_inner),
            nn.MaxPool1d(4, stride=1, padding=2),
            WTConv1d(in_channels=args.d_inner, out_channels=args.d_inner, kernel_size=5, wt_levels=3),
            nn.BatchNorm1d(args.d_inner),
            nn.MaxPool1d(2, stride=1, padding=1),#
            nn.Dropout(0.3),

        )

        self.gru = nn.LSTM(
            input_size=args.d_inner,
            hidden_size=args.d_inner,
            num_layers=1,
            batch_first=True
        )

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)


        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)


        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)

        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(args.d_inner))

        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)


    def forward(self, x):


        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)

        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b d_in l -> b l d_in')

        x = self.conv1d(x)


        x = x[:, :, :l]


        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output


    def ssm(self, x):


        (d_in, n) = self.A_log.shape


        A = -torch.exp(self.A_log.float())  # shape (d_in, n)

        D = self.D.float()


        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)


        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)

        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)
        return y


    def selective_scan(self, u, delta, A, B, C, D):

        # 获取输入u的维度
        (b, l, d_in) = u.shape
        # 获取矩阵A的列数
        n = A.shape[1]  #  A: shape (d_in, n)

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        # delta、B和u,这个计算和原始论文不同
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')


        # 执行选择性扫描,初始化状态x为零
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        # 初始化输出列表ys
        ys = []
        for i in range(l):
            # 更新状态x
            # deltaA:((b,l,d_in, n)
            # deltaB_u:( b,l,d_in,n)
            # x:
            x = deltaA[:, i] * x + deltaB_u[:, i]
            # 计算输出y
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            # 将输出y添加到列表ys中
            ys.append(y)
        # 将列表ys堆叠成张量y
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        # 将输入u乘以D并加到输出y上
        y = y + u * D

        return y


class RMSNorm(nn.Module):

    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):

        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
