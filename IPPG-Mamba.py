"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
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
from wtconv.wtconv2d import WTConv1d

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
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """

        def calculate_hrv_metrics_and_append(ppg_signals, sampling_rate=30):
            def detect_peaks(ppg_signal, threshold=0):#0.2  0.5 0
                if isinstance(ppg_signal, torch.Tensor):
                    ppg_signal = ppg_signal.cpu().numpy()  # 移动到 CPU 并转换为 NumPy 数组
                diff_signal = np.diff(ppg_signal)
                peaks = []
                for i in range(1, len(diff_signal) - 1):
                    if diff_signal[i - 1] > 0 and diff_signal[i + 1] < 0 and ppg_signal[i] > threshold:
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
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
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
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
             x (Tensor): 输入张量，形状为(batch_size, sequence_length, hidden_size)
        Returns:
            output: shape (b, l, d)
            输出张量，形状与输入相同
        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """

        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
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
        """MambaBlock的前向传播函数，与Mamba论文图3 Section 3.4相同.

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """

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
        """运行状态空间模型，参考Mamba论文 Section 3.2和注释[2]:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """

        (d_in, n) = self.A_log.shape


        A = -torch.exp(self.A_log.float())  # shape (d_in, n)

        D = self.D.float()


        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)


        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)

        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)
        return y


    def selective_scan(self, u, delta, A, B, C, D):
        """执行选择性扫描算法，参考Mamba论文[1] Section 2和注释[2]. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        经典的离散状态空间公式:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
       除了B和C (以及step size delta用于离散化) 与输入x(t)相关.

        参数:
            u: shape (b, l, d_in)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        过程概述：

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        # 获取输入u的维度
        (b, l, d_in) = u.shape
        # 获取矩阵A的列数
        n = A.shape[1]  #  A: shape (d_in, n)

        # 离散化连续参数(A, B)
        # - A 使用 zero-order hold (ZOH) 离散化 (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is 使用一个简化的Euler discretization而不是ZOH.根据作者的讨论:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"

        # 计算离散化的A
        # 将delta和A进行点乘，将A沿着delta的最后一个维度进行广播，然后执行逐元素乘法
        # A:(d_in, n),delta:(b, l, d_in)
        # A广播拓展->(b,l,d_in, n)，deltaA对应原论文中的A_bar
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        # delta、B和u,这个计算和原始论文不同
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
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
    """
    初始化RMSNorm模块，该模块实现了基于均方根的归一化操作。

    参数:
    d_model (int): 模型的特征维度。
    eps (float, 可选): 为了避免除以零，添加到分母中的一个小的常数。
    """
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps# 保存输入的eps值，用于数值稳定性。
        # 创建一个可训练的权重参数，初始值为全1，维度与输入特征维度d_model相同。
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        """
                计算输入x的均方根值，用于后续的归一化操作。
                x.pow(2) 计算x中每个元素的平方。
                mean(-1, keepdim=True) 对x的最后一个维度（特征维度）进行平方和求平均，保持维度以便进行广播操作。
                torch.rsqrt 对求得的平均值取倒数和平方根，得到每个特征的均方根值的逆。
                + self.eps 添加一个小的常数eps以保持数值稳定性，防止除以零的情况发生。
                x * ... * self.weight 将输入x与计算得到的归一化因子和可训练的权重相乘，得到最终的归一化输出。
                """
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
