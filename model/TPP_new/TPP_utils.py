import torch
import math
import torch.nn as nn
import math
import torch
from torch import nn

class ScaledSoftplus(nn.Module):
    '''
    Use different beta for mark-specific intensities
    '''
    def __init__(self, num_marks, threshold=20.):
        super(ScaledSoftplus, self).__init__()
        self.threshold = threshold
        self.log_beta = nn.Parameter(torch.zeros(num_marks), requires_grad=True)  # [num_marks]

    def forward(self, x):
        '''
        :param x: [..., num_marks]
        '''
        beta = self.log_beta.exp()
        beta_x = beta * x
        return torch.where(
            beta_x <= self.threshold,
            torch.log1p(beta_x.clamp(max=math.log(1e5)).exp()) / beta,
            x,  # if above threshold, then the transform is effectively linear
        )

class TimePositionalEncoding(nn.Module):
    """Temporal encoding in THP, ICML 2020
    """

    def __init__(self, d_model, device='cpu'):
        super().__init__()
        i = torch.arange(0, d_model, 1, device=device)
        div_term = (2 * (i // 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        """Compute time positional encoding defined in Equation (2) in THP model.

        Args:
            x (tensor): time_seqs, [batch_size, seq_len]

        Returns:
            temporal encoding vector, [batch_size, seq_len, model_dim]

        """
        result = x.unsqueeze(-1) * self.div_term
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            x = self.self_attn(x, x, x, mask)
            if self.feed_forward is not None:
                return self.feed_forward(x)
            else:
                return x
            
class SublayerConnection(nn.Module):
    # used for residual connection
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear

        if output_linear:
            self.linears = nn.ModuleList(
                [nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask, output_weight=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            if output_weight:
                return self.linears[-1](x), attn_weight
            else:
                return self.linears[-1](x)
        else:
            if output_weight:
                return x, attn_weight
            else:
                return x
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # small change here -- we use "1" for masked element
        scores = scores.masked_fill(mask > 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class EventSampler(nn.Module):
    """Event Sequence Sampler based on thinning algorithm, which corresponds to Algorithm 2 of
    The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process,
    https://arxiv.org/abs/1612.09328.

    The implementation uses code from https://github.com/yangalan123/anhp-andtt/blob/master/anhp/esm/thinning.py.
    """

    def __init__(self, num_sample, num_exp, over_sample_rate, num_samples_boundary, dtime_max, patience_counter,
                 device):
        """Initialize the event sampler.

        Args:
            num_sample (int): number of sampled next event times via thinning algo for computing predictions.
            num_exp (int): number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
            over_sample_rate (float): multiplier for the intensity up bound.
            num_samples_boundary (int): number of sampled event times to compute the boundary of the intensity.
            dtime_max (float): max value of delta times in sampling
            patience_counter (int): the maximum iteration used in adaptive thinning.
            device (torch.device): torch device index to select.
        """
        super(EventSampler, self).__init__()
        self.num_sample = num_sample
        self.num_exp = num_exp
        self.over_sample_rate = over_sample_rate
        self.num_samples_boundary = num_samples_boundary
        self.dtime_max = dtime_max
        self.patience_counter = patience_counter
        self.device = device

    def compute_intensity_upper_bound(self, time_seqs, absolute_time_seqs, time_delta_seqs, poi_seqs, user_seqs, poi_cat_seqs, \
                                        attention_mask, seq_mask, intensity_fn, compute_last_step_only):
        """Compute the upper bound of intensity at each event timestamp.

        Args:
            time_seq (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            intensity_fn (fn): a function that computes the intensity.
            compute_last_step_only (bool): wheter to compute the last time step pnly.

        Returns:
            tensor: [batch_size, seq_len]
        """
        batch_size, seq_len = time_seqs.size()

        # [1, 1, num_samples_boundary]
        time_for_bound_sampled = torch.linspace(start=0.0,
                                                end=1.0,
                                                steps=self.num_samples_boundary,
                                                device=self.device)[None, None, :]

        # [batch_size, seq_len, num_samples_boundary]
        dtime_for_bound_sampled = time_delta_seqs[:, :, None] * time_for_bound_sampled

        # [batch_size, seq_len, num_samples_boundary, event_num]
        intensities_for_bound = intensity_fn(time_seqs=time_seqs,
                                                    absolute_time_seqs=absolute_time_seqs,
                                                    poi_seqs=poi_seqs,
                                                    user_seq=user_seqs,
                                                    poi_cat_seq=poi_cat_seqs,
                                                    attention_mask=attention_mask,
                                                    seq_mask=seq_mask,
                                                    sample_dtimes=dtime_for_bound_sampled, 
                                                    max_steps=seq_len, 
                                                    compute_last_step_only=compute_last_step_only)

        # [batch_size, seq_len]
        bounds = intensities_for_bound.sum(dim=-1).max(dim=-1)[0] * self.over_sample_rate

        return bounds

    def sample_exp_distribution(self, sample_rate):
        """Sample an exponential distribution.

        Args:
            sample_rate (tensor): [batch_size, seq_len], intensity rate.

        Returns:
            tensor: [batch_size, seq_len, num_exp], exp numbers at each event timestamp.
        """

        batch_size, seq_len = sample_rate.size()

        # For fast approximation, we reuse the rnd for all samples
        # [batch_size, seq_len, num_exp]
        exp_numbers = torch.empty(size=[batch_size, seq_len, self.num_exp],
                                  dtype=torch.float32,
                                  device=self.device)

        # [batch_size, seq_len, num_exp]
        # exp_numbers.exponential_(1.0)
        exp_numbers.exponential_(1.0)

        # [batch_size, seq_len, num_exp]
        # exp_numbers = torch.tile(exp_numbers, [1, 1, self.num_sample, 1])

        # [batch_size, seq_len, num_exp]
        # div by sample_rate is equivalent to exp(sample_rate),
        # see https://en.wikipedia.org/wiki/Exponential_distribution
        exp_numbers = exp_numbers / sample_rate[:, :, None]

        return exp_numbers

    def sample_uniform_distribution(self, intensity_upper_bound):
        """Sample an uniform distribution

        Args:
            intensity_upper_bound (tensor): upper bound intensity computed in the previous step.

        Returns:
            tensor: [batch_size, seq_len, num_sample, num_exp]
        """
        batch_size, seq_len = intensity_upper_bound.size()

        unif_numbers = torch.empty(size=[batch_size, seq_len, self.num_sample, self.num_exp],
                                   dtype=torch.float32,
                                   device=self.device)
        unif_numbers.uniform_(0.0, 1.0)

        return unif_numbers

    def sample_accept(self, unif_numbers, sample_rate, total_intensities, exp_numbers):
        """Do the sample-accept process.

        For the accumulated exp (delta) samples drawn for each event timestamp, find (from left to right) the first
        that makes the criterion < 1 and accept it as the sampled next-event time. If all exp samples are rejected 
        (criterion >= 1), then we set the sampled next-event time dtime_max.

        Args:
            unif_numbers (tensor): [batch_size, max_len, num_sample, num_exp], sampled uniform random number.
            sample_rate (tensor): [batch_size, max_len], sample rate (intensity).
            total_intensities (tensor): [batch_size, seq_len, num_sample, num_exp]
            exp_numbers (tensor): [batch_size, seq_len, num_sample, num_exp]: sampled exp numbers (delta in Algorithm 2).

        Returns:
            result (tensor): [batch_size, seq_len, num_sample], sampled next-event times.
        """

        # [batch_size, max_len, num_sample, num_exp]
        criterion = unif_numbers * sample_rate[:, :, None, None] / total_intensities
        
        # [batch_size, max_len, num_sample, num_exp]
        masked_crit_less_than_1 = torch.where(criterion<1,1,0)
        
        # [batch_size, max_len, num_sample]
        non_accepted_filter = (1-masked_crit_less_than_1).all(dim=3)
        
        # [batch_size, max_len, num_sample]
        first_accepted_indexer = masked_crit_less_than_1.argmax(dim=3)
        
        # [batch_size, max_len, num_sample,1]
        # indexer must be unsqueezed to 4D to match the number of dimensions of exp_numbers
        result_non_accepted_unfiltered = torch.gather(exp_numbers, 3, first_accepted_indexer.unsqueeze(3))
        
        # [batch_size, max_len, num_sample,1]
        result = torch.where(non_accepted_filter.unsqueeze(3), torch.tensor(self.dtime_max).to(self.device), result_non_accepted_unfiltered)
        
        # [batch_size, max_len, num_sample]
        result = result.squeeze(dim=-1)
        
        return result

    def draw_next_time_one_step(self, time_seqs, absolute_time_seqs, time_delta_seqs, poi_seqs, user_seqs, poi_cat_seqs, \
                                attention_mask, seq_mask, intensity_fn, compute_last_step_only=False):
        """Compute next event time based on Thinning algorithm.

        Args:
            time_seq (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            dtime_boundary (tensor): [batch_size, seq_len], dtime upper bound.
            intensity_fn (fn): a function to compute the intensity.
            compute_last_step_only (bool, optional): whether to compute last event timestep only. Defaults to False.

        Returns:
            tuple: next event time prediction and weight.
        """
        """
        # 1. compute the upper bound of the intensity at each timestamp
        intensity_upper_bound = self.compute_intensity_upper_bound(time_seqs=time_seqs,
                                                                    absolute_time_seqs=absolute_time_seqs,
                                                                    time_delta_seqs=time_delta_seqs,
                                                                    poi_seqs=poi_seqs,
                                                                    user_seqs=user_seqs,
                                                                    poi_cat_seqs=poi_cat_seqs,
                                                                    attention_mask=attention_mask,
                                                                    seq_mask=seq_mask,
                                                                   intensity_fn=intensity_fn,
                                                                   compute_last_step_only=compute_last_step_only)

        # 2. draw exp distribution with intensity = intensity_upper_bound
        # we apply fast approximation, i.e., re-use exp sample times for computation
        # [batch_size, seq_len, num_exp]
        exp_numbers = self.sample_exp_distribution(intensity_upper_bound)
        exp_numbers = torch.clamp(torch.cumsum(exp_numbers, dim=-1), max=self.dtime_max)
        
        # 3. compute intensity at sampled times from exp distribution
        # [batch_size, seq_len, num_exp, event_num]
        intensities_at_sampled_times = intensity_fn(time_seqs=time_seqs,
                                                    absolute_time_seqs=absolute_time_seqs,
                                                    poi_seqs=poi_seqs,
                                                    user_seq=user_seqs,
                                                    poi_cat_seq=poi_cat_seqs,
                                                    attention_mask=attention_mask,
                                                    seq_mask=seq_mask,
                                                    sample_dtimes=exp_numbers, 
                                                    compute_last_step_only=compute_last_step_only)

        # [batch_size, seq_len, num_exp]
        total_intensities = intensities_at_sampled_times.sum(dim=-1)

        # add one dim of num_sample: re-use the intensity for samples for prediction
        # [batch_size, seq_len, num_sample, num_exp]
        total_intensities = torch.tile(total_intensities[:, :, None, :], [1, 1, self.num_sample, 1])
        
        # [batch_size, seq_len, num_sample, num_exp]
        exp_numbers = torch.tile(exp_numbers[:, :, None, :], [1, 1, self.num_sample, 1])
        
        # 4. draw uniform distribution
        # [batch_size, seq_len, num_sample, num_exp]
        unif_numbers = self.sample_uniform_distribution(intensity_upper_bound)

        # 5. find out accepted intensities
        # [batch_size, seq_len, num_sample]
        res = self.sample_accept(unif_numbers, intensity_upper_bound, total_intensities, exp_numbers)

        # [batch_size, seq_len, num_sample]
        weights = torch.ones_like(res)/res.shape[2]
        """
        time_steps = 24
        # 数值积分计算期望时间
        t_grid = torch.linspace(0, self.dtime_max, time_steps).to(self.device)  # 500个采样点
        t_grid = t_grid.view(1, 1, -1).expand(time_seqs.size(0), time_seqs.size(1), -1)  # [B, L, 500]
        
        lambda_t_poi = intensity_fn(time_seqs=time_seqs,
                                absolute_time_seqs=absolute_time_seqs,
                                poi_seqs=poi_seqs,
                                user_seq=user_seqs,
                                poi_cat_seq=poi_cat_seqs,
                                attention_mask=attention_mask,
                                seq_mask=seq_mask,
                                sample_dtimes=t_grid,
                                compute_last_step_only=compute_last_step_only)  # [B, L, 500]
        
        lambda_t = lambda_t_poi.sum(dim=-1)  
        
        Lambda_cumsum = torch.cumsum(lambda_t * (24/time_steps), dim=-1)
        survival = torch.exp(-Lambda_cumsum)
        pdf = lambda_t * survival
        
        accepted_dtimes = torch.sum(t_grid * pdf * (24/time_steps), dim=-1).unsqueeze(dim=-1)
        weights = torch.ones_like(accepted_dtimes)

        # add a upper bound here in case it explodes, e.g., in ODE models
        return accepted_dtimes.clamp(max=24), weights, lambda_t, pdf, lambda_t_poi


def calculate_distances(longitude, latitude):
    # 将经度和纬度转换为弧度
    longitude_rad = torch.deg2rad(longitude)
    latitude_rad = torch.deg2rad(latitude)

    # 计算相邻点之间的经度差和纬度差
    delta_lon = longitude_rad[1:] - longitude_rad[:-1]
    delta_lat = latitude_rad[1:] - latitude_rad[:-1]

    # 使用 Haversine 公式计算相邻点之间的距离
    R = 6371  # 地球半径，单位为千米
    a = torch.sin(delta_lat / 2)**2 + torch.cos(latitude_rad[:-1]) * torch.cos(latitude_rad[1:]) * torch.sin(delta_lon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distances = R * c

    return distances

# 在时间位置编码中加入周期信息
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, period=24):
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # shape: [period, d_model]

    def forward(self, t):
        phase = t % self.pe.size(0)
        return self.pe[phase]

class FourierPeriodicEncoder(nn.Module):
    def __init__(self, time_dim, device, periods=[7, 24, 12, 1], harmonics=3, learnable_weights=True):
        """
        基于傅里叶级数的周期性时间编码器
        
        Args:
            time_dim (int): 输出编码维度
            periods (list): 周期列表，如[7, 24]表示周和小时周期
            harmonics (int): 每个周期的谐波数量
            learnable_weights (bool): 是否使用可学习的谐波权重
        """
        super().__init__()
        self.time_dim = time_dim
        self.periods = periods
        self.harmonics = harmonics
        self.learnable_weights = learnable_weights
        self.device = device
        
        # 计算傅里叶基函数的总数
        # 每个周期：harmonics个sin + harmonics个cos = 2*harmonics
        self.fourier_dim = len(periods) * 2 * harmonics
        
        # 可学习的谐波权重和相位
        if learnable_weights:
            # 为每个周期的每个谐波创建可学习权重
            self.harmonic_weights = nn.ParameterList([
                nn.Parameter(torch.ones(harmonics)).to(self.device) for _ in periods
            ])
            self.phase_shifts = nn.ParameterList([
                nn.Parameter(torch.zeros(harmonics)).to(self.device) for _ in periods
            ])
        
        # 线性映射层：从傅里叶特征映射到目标维度
        self.fourier_projection = nn.Sequential(
            nn.Linear(self.fourier_dim, time_dim * 2),
            nn.ReLU(),
            nn.Linear(time_dim * 2, time_dim),
            nn.LayerNorm(time_dim)
        )
        
        # 可选的残差连接
        if self.fourier_dim != time_dim:
            self.residual_proj = nn.Linear(self.fourier_dim, time_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def generate_fourier_features(self, time_values, period, harmonic_idx):
        """
        为指定周期生成傅里叶基函数特征
        
        Args:
            time_values: (batch_size, seq_len) 时间值
            period: 周期长度
            harmonic_idx: 当前周期在periods中的索引
        Returns:
            fourier_features: (batch_size, seq_len, 2*harmonics)
        """
        batch_size, seq_len = time_values.shape
        
        # 归一化时间到[0, 2π]
        normalized_time = 2 * math.pi * time_values.float() / period
        
        features = []
        
        for h in range(1, self.harmonics + 1):  # 谐波从1开始
            # 使用可学习权重和相位
            weight = self.harmonic_weights[harmonic_idx][h-1]
            phase = self.phase_shifts[harmonic_idx][h-1]
            
            sin_component = weight * torch.sin(h * normalized_time + phase)
            cos_component = weight * torch.cos(h * normalized_time + phase)
            
            features.extend([sin_component.unsqueeze(-1), cos_component.unsqueeze(-1)])
        
        return torch.cat(features, dim=-1)  # (batch_size, seq_len, 2*harmonics)
    
    def forward(self, time_dict):
        """
        Args:
            time_dict: 字典，包含不同周期的时间值
                      例如: {'day': tensor, 'hour': tensor}
        Returns:
            encoded: (batch_size, seq_len, time_dim)
        """
        batch_size, seq_len = list(time_dict.values())[0].shape
        
        all_fourier_features = []
        
        # 为每个周期生成傅里叶特征
        for period_idx, period in enumerate(self.periods):
            if period == 7 and 'dayofweek' in time_dict:
                features = self.generate_fourier_features(
                    time_dict['dayofweek'], period, period_idx
                )
            elif period == 24 and 'hour' in time_dict:
                features = self.generate_fourier_features(
                    time_dict['hour'], period, period_idx
                )
            elif period == 12 and 'month' in time_dict:
                features = self.generate_fourier_features(
                    time_dict['month'], period, period_idx
                )
            elif period == 1 and 'dayofmonth' in time_dict:
                features = self.generate_fourier_features(
                    time_dict['dayofmonth'], period, period_idx
                )
            else:
                # 如果没有对应的时间值，创建零特征
                features = torch.zeros(batch_size, seq_len, 2 * self.harmonics,
                                     device=list(time_dict.values())[0].device)
            
            all_fourier_features.append(features)
        
        # 拼接所有傅里叶特征
        fourier_concat = torch.cat(all_fourier_features, dim=-1)  # (batch_size, seq_len, fourier_dim)
        
        # 通过线性映射层
        projected_features = self.fourier_projection(fourier_concat)
        
        # 残差连接
        residual = self.residual_proj(fourier_concat)
        if residual.shape[-1] == projected_features.shape[-1]:
            final_encoding = projected_features + 0.1 * residual  # 加权残差
        else:
            final_encoding = projected_features
        
        return final_encoding
