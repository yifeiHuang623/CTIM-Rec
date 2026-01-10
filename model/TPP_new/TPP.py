import torch.nn as nn
import torch
import torch.nn.functional as F
from .TPP_utils import TimePositionalEncoding, ScaledSoftplus, EventSampler, FourierPeriodicEncoder
from argparse import Namespace
from datetime import datetime
import calendar
import numpy as np
import math
from torch.nn.init import xavier_uniform_
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

class TPP(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.hidden_size = args.hidden_size
        self.device = args.device
        self.layer_temporal_encoding = TimePositionalEncoding(d_model=self.hidden_size, device=args.device)
        
        self.num_pois = args.num_pois  
        
        self.n_head = args.num_heads
        self.dropout = args.dropout_rate
        self.n_layers = args.num_layers
        
        self.ngrams = 6
        self.quadkey_len = 25
        self.user_embed_dim = 128
        
        self.loss_integral_num_sample_per_step = args.loss_integral_num_sample_per_step
        self.eps = torch.finfo(torch.float32).eps
        self.use_mc_samples = True
        self.gen_config = Namespace(**{
            "num_sample": 1,
            "num_exp": 500,
            "over_sample_rate": 5,
            "patience_counter": 5,
            "num_samples_boundary": 5,
            "dtime_max": 5
        })
        
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        # self.stack_layers = nn.ModuleList(
        #     [EncoderLayer(
        #         self.hidden_size,
        #         MultiHeadAttention(self.n_head, self.hidden_size, self.hidden_size, self.dropout,
        #                            output_linear=False),
        #         use_residual=False,
        #         feed_forward=self.feed_forward,
        #         dropout=self.dropout
        #     ) for _ in range(self.n_layers)])
        
        ninp = self.hidden_size + self.hidden_size + self.hidden_size
        # self.enc_layer = TransformerEncoderLayer(ninp, self.n_head, ninp, 0.5, batch_first=True)
        # self.encoder = TransformerEncoder(self.enc_layer, self.n_layers)
        
        self.encoder = nn.LSTM(
            input_size=ninp,          # 输入维度
            hidden_size=ninp,         # 隐藏层维度（可与 ninp 相同）
            num_layers=self.n_layers, # LSTM 层数
            dropout=0.5,              # dropout，与原来保持一致
            batch_first=True          # 与 Transformer 一致，(batch, seq, feature)
        )
        
        # self.encoder = TemporalTransformerEncoder(d_model=ninp, nhead=self.n_head, num_layers=self.n_layers, dim_feedforward=ninp, dropout=0.5)
        
        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_pois], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_pois], device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)
        
        # convert hidden vectors into event-type-sized vector
        self.layer_intensity_hidden = nn.Linear(ninp, self.num_pois)
        self.softplus = ScaledSoftplus(self.num_pois)   # learnable mark-specific beta
        
        self.event_sampler = EventSampler(num_sample=self.gen_config.num_sample,
                                              num_exp=self.gen_config.num_exp,
                                              over_sample_rate=self.gen_config.over_sample_rate,
                                              patience_counter=self.gen_config.patience_counter,
                                              num_samples_boundary=self.gen_config.num_samples_boundary,
                                              dtime_max=self.gen_config.dtime_max,
                                              device=self.device)
        
        self.time_period_encoder = FourierPeriodicEncoder(time_dim=self.hidden_size, device=self.device)
        
        self.layer_poi_emb = Embedding(self.num_pois + 1,  self.hidden_size)
        self.user_embed_model = Embedding(args.num_users, self.user_embed_dim)
        self.poi_type_embed_model = Embedding(args.num_poi_types, self.hidden_size)
        
        self.concat_layer = nn.Linear(self.hidden_size + self.hidden_size + self.hidden_size, self.hidden_size)
        self.pos_encoder = PositionalEncoding(ninp, 0.5)
        
        self.layer_user_hidden = nn.Linear(self.user_embed_dim, self.num_pois)
        self.factor_distance_decay = nn.Parameter(torch.empty([1, self.num_pois], device=self.device))
        nn.init.xavier_normal_(self.factor_distance_decay)
        
        self.time_projector = nn.Linear(self.hidden_size, self.num_pois)
        
        self.alpha = nn.Parameter(torch.empty([1, self.num_pois], device=self.device)) 
        self.beta = nn.Parameter(torch.empty([1, self.num_pois], device=self.device))   
        self.gamma = nn.Parameter(torch.empty([1, self.num_pois], device=self.device))  
        
        t_projector_dim = 8
        self.t_projector = nn.Sequential(
                nn.Linear(1, 8),  # 从1维到8维
                nn.ReLU(),
                nn.Linear(8, t_projector_dim))  # 
        self.gate_net = nn.Sequential(
            nn.Linear(t_projector_dim, 8),  # 合并所有特征
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出在0-1之间，作为调制因子
        )
        self.offset_net = nn.Sequential(
            nn.Linear(t_projector_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        self.distance_matrix = args.distance_matrix

    def handle_sequence(self, batch_data):
        poi_id, time, time_delta = batch_data['POI_id'], batch_data['timestamps'], batch_data['time_delta']
        user_id = batch_data['user_id'].unsqueeze(dim=1).expand(poi_id.shape[0], poi_id.shape[1])
        
        seq_len = batch_data['mask']
        
        y_poi_id, y_time, y_time_delta = batch_data['y_POI_id']['POI_id'], batch_data['y_POI_id']['timestamps'], batch_data['y_POI_id']['time_delta']
        
        B, L = batch_data['POI_id'].size()
        y_poi_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_time_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_time_delta_seq = torch.full((B, L), 0, dtype=torch.float, device=self.device)
        
        for i in range(B):
            end = seq_len[i].item()
            y_poi_seq[i, :end] = torch.cat((poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1)
            y_time_seq[i, :end] = torch.cat((time[i, 1:end], y_time[i].unsqueeze(dim=-1)), dim=-1)
            y_time_delta_seq[i, :end] = torch.cat((time_delta[i, 1:end], y_time_delta[i].unsqueeze(dim=-1)), dim=-1)
        
        batch_data['user_id'] = user_id
        batch_data['y_POI_id']['POI_id'] = y_poi_seq
        batch_data['y_POI_id']['timestamps'] = y_time_seq
        batch_data['y_POI_id']['time_delta'] = y_time_delta_seq
        
        return batch_data
    
    '''
    def calculate_poi_loss(self, batch_data):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        
        batch_data = self.handle_sequence(batch_data)
        
        u_id, poi_seqs, poi_cat_seqs, time_seqs, time_delta, lat, lon, seq_len = batch_data['user_id'], batch_data['POI_id'], batch_data['POI_catid'], batch_data['timestamps'], \
                                                                     batch_data['time_delta'], batch_data['latitude'], batch_data['longitude'], batch_data['mask']
        
        y_poi_seqs, y_time_seqs, y_time_delta = batch_data['y_POI_id']['POI_id'], batch_data['y_POI_id']['timestamps'], batch_data['y_POI_id']['time_delta']
        
        attention_mask = torch.triu(torch.ones((poi_seqs.shape[1], poi_seqs.shape[1]), dtype=torch.bool)).to(self.device)
        indices = torch.arange(poi_seqs.shape[1]).unsqueeze(0).expand(poi_seqs.shape[0], poi_seqs.shape[1]).to(self.device)
        invalid_positions = (indices >= seq_len.unsqueeze(1)).to(torch.bool)
        attention_mask = attention_mask | invalid_positions.unsqueeze(1).expand(-1, poi_seqs.shape[1], -1)
        
        user_emb = self.user_embed_model(u_id)
        
        time_norm_seqs = (time_seqs - time_seqs[:, 0:1]) / 3600
        enc_out = self.forward(time_seqs=time_norm_seqs, poi_seqs=poi_seqs, attention_mask=attention_mask, loc_seq=(lat, lon), poi_cat_seq=poi_cat_seqs, seq_mask=invalid_positions)

        period_time_embed = self.get_time_features(time_seq=y_time_seqs)

        factor_intensity_base = self.factor_intensity_base[None, ...] 
        factor_intensity_decay = self.factor_distance_decay[None, ...]
        intensity_states = self.layer_intensity_hidden(enc_out)  + factor_intensity_base + self.layer_user_hidden(user_emb) 
            # self.time_projector(period_time_embed)
        
        # [batch_size, seq_len, num_pois]
        lambda_at_event = self.softplus(intensity_states)
        
        # First, calculate the sum intensities of all pois [batch_size, seq_len, 1]
        lambda_at_event_sum = lambda_at_event.sum(dim=-1) + self.eps
        lambda_at_event_sum = lambda_at_event_sum[..., None]
        # Second, add an epsilon to every marked intensity for stability [batch_size, seq_len, num_pois]
        lambda_at_event = lambda_at_event + self.eps
        
        log_marked_event_lambdas = (lambda_at_event / lambda_at_event_sum).log()

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=y_poi_seqs,
            ignore_index=0,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )
        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]

        # compute loss to minimize
        loss = - event_ll.sum()
        return loss / num_events
    '''
    
    def calculate_both_loss(self, batch_data):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        
        batch_data = self.handle_sequence(batch_data)
        
        u_id, poi_seqs, poi_cat_seqs, time_seqs, time_delta, lat, lon, seq_len = batch_data['user_id'], batch_data['POI_id'], batch_data['POI_catid'], batch_data['timestamps'], \
                                                                     batch_data['time_delta'], batch_data['latitude'], batch_data['longitude'], batch_data['mask']
        
        y_poi_seqs, y_time_seqs, y_time_delta = batch_data['y_POI_id']['POI_id'], batch_data['y_POI_id']['timestamps'], batch_data['y_POI_id']['time_delta']
        
        # attention_mask = torch.triu(torch.ones((poi_seqs.shape[1], poi_seqs.shape[1]), dtype=torch.bool)).to(self.device)
        attention_mask = self._generate_square_mask_(poi_seqs.shape[1], self.device)
        indices = torch.arange(poi_seqs.shape[1]).unsqueeze(0).expand(poi_seqs.shape[0], poi_seqs.shape[1]).to(self.device)
        invalid_positions = (indices >= seq_len.unsqueeze(1)).to(torch.bool)
        # attention_mask = attention_mask | invalid_positions.unsqueeze(1).expand(-1, poi_seqs.shape[1], -1)
        
        user_emb = self.user_embed_model(u_id)
        
        time_norm_seqs = (time_seqs - time_seqs[:, 0:1]) / 3600.0
        enc_out = self.forward(time_seqs=time_norm_seqs, poi_seqs=poi_seqs, attention_mask=attention_mask, loc_seq=(lat, lon), poi_cat_seq=poi_cat_seqs, seq_mask=invalid_positions)

        period_time_embed = self.get_time_features(time_seq=y_time_seqs)
        
        # batch_size, seq_len, 1
        poi_seqs_np = poi_seqs.detach().cpu().numpy()
        y_poi_seqs_np = y_poi_seqs.detach().cpu().numpy()
        distance_decay = self.distance_matrix[poi_seqs_np, y_poi_seqs_np]
        distance_decay = torch.from_numpy(distance_decay).to(self.device).float()[..., None]
        
        # convert to day, [0, 1] 
        # batch_size, seq_len, 1
        time_delta_decay = y_time_delta[..., None] / 24.0
        
        # batch_size, seq_len, 1
        alpha = self.alpha[None, ...]
        # batch_size, seq_len, time_dim
        time_delta_decay = self.t_projector(time_delta_decay)
        gate = self.gate_net(time_delta_decay)
        offset = self.offset_net(time_delta_decay)
        # [batch_size, seq_len, 1]
        base_spatial_decay = - torch.log(1 + distance_decay)
        # batch_size, seq_len, 1
        time_distance_factor = base_spatial_decay * gate + offset
        
        factor_intensity_base = self.factor_intensity_base[None, ...]
        # intensity_states = self.layer_intensity_hidden(enc_out) + factor_intensity_base + self.layer_user_hidden(user_emb) \
        #     + self.time_projector(period_time_embed)
        intensity_states = factor_intensity_base + self.layer_user_hidden(user_emb) \
            + time_distance_factor \
            + self.time_projector(period_time_embed) \
            + self.layer_intensity_hidden(enc_out) \
            
        # [batch_size, seq_len, num_pois]
        lambda_at_event = self.softplus(intensity_states)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample dtimes
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seq=y_time_delta)

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out,
                                                             sample_dtimes=sample_dtimes,
                                                             user_emb=user_emb,
                                                             time_seqs=time_seqs,
                                                             poi_seqs=poi_seqs)
        lambda_t_sample = self.softplus(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=y_time_delta,
                                                                        seq_mask=~invalid_positions,
                                                                        type_seq=y_poi_seqs)

        # compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        
        return loss / num_events
    
    @staticmethod
    def _generate_square_mask_(sz, device):
        mask = (torch.triu(torch.ones(sz, sz).to(device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def predict(self, batch_data):
        """One-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            poi_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """

        batch_data = self.handle_sequence(batch_data)
        
        u_id, poi_seqs, poi_cat_seqs, time_seqs, time_delta, lat, lon, seq_len = batch_data['user_id'], batch_data['POI_id'], batch_data['POI_catid'], batch_data['timestamps'], \
                                                                     batch_data['time_delta'], batch_data['latitude'], batch_data['longitude'], batch_data['mask']
        
        y_time_delta = batch_data['y_POI_id']['time_delta']
        
        time_norm_seqs = (time_seqs - time_seqs[:, 0:1]) / 3600.0
        # attention_mask = torch.triu(torch.ones((poi_seqs.shape[1], poi_seqs.shape[1]), dtype=torch.bool)).to(self.device)
        attention_mask = self._generate_square_mask_(poi_seqs.shape[1], self.device)
        indices = torch.arange(poi_seqs.shape[1]).unsqueeze(0).expand(poi_seqs.shape[0], poi_seqs.shape[1]).to(self.device)
        invalid_positions = (indices >= seq_len.unsqueeze(1)).to(torch.bool)
        # attention_mask = attention_mask | invalid_positions.unsqueeze(1).expand(-1, poi_seqs.shape[1], -1)
        
        # use the time label as the accepted time, and weights are all ones
        accepted_dtimes = y_time_delta.unsqueeze(dim=-1)
        weights = torch.ones_like(accepted_dtimes)

        # [batch_size, seq_len]
        dtime_boundary = torch.max(time_delta * self.event_sampler.dtime_max, time_delta + self.event_sampler.dtime_max)

        # [batch_size, seq_len, num_sample]
        # accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seq,
        #                                                                       time_delta_seq,
        #                                                                       event_seq,
        #                                                                       dtime_boundary,
        #                                                                       self.compute_intensities_at_sample_times,
        #                                                                       compute_last_step_only=False)  # make it explicit

        # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = self.compute_intensities_at_sample_times(time_seqs=time_norm_seqs,
                                                                        absolute_time_seqs=time_seqs,
                                                                        loc_seq=(lat, lon),
                                                                        poi_seqs=poi_seqs,
                                                                        user_seq=u_id,
                                                                        poi_cat_seq=poi_cat_seqs,
                                                                        sample_dtimes=accepted_dtimes, 
                                                                        attention_mask=attention_mask,
                                                                        seq_mask=invalid_positions)
        
        # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
        # Each of the last dimension is a categorical distribution over all marks.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)

        # 3. Compute weighted sum of distributions and then take argmax.
        # [batch_size, seq_len, num_marks]
        intensities_weighted = torch.einsum('...s,...sm->...m', weights, intensities_normalized)

        # [batch_size, seq_len]
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)  # compute the expected next event time
        return intensities_weighted, dtimes_pred

    def forward(self, time_seqs, poi_seqs, attention_mask, loc_seq, poi_cat_seq, seq_mask):
        # 计算时间间隔矩阵: [batch_size, seq_len, seq_len]
        time_diffs = torch.abs(time_seqs.unsqueeze(2) - time_seqs.unsqueeze(1))  # 绝对时间差
        # 可选：对时间间隔进行归一化，避免数值问题
        # time_diffs = time_diffs / (time_diffs.max(dim=-1, keepdim=True)[0] + 1e-5)

        # 生成嵌入
        tem_enc = self.layer_temporal_encoding(time_seqs)  # [batch_size, seq_len, time_embed_dim]
        poi_cat_emb = self.poi_type_embed_model(poi_cat_seq)  # [batch_size, seq_len, d_model]
        poi_emb = self.layer_poi_emb(poi_seqs)  # [batch_size, seq_len, d_model]

        # 拼接所有特征
        enc_output = torch.cat((poi_emb, poi_cat_emb, tem_enc), dim=-1)  # [batch_size, seq_len, d_model * 3]
        # enc_output = self.concat_layer(enc_output)  # 投影到d_model

        # 缩放嵌入（可选）
        enc_output = enc_output * math.sqrt(enc_output.size(1))
        enc_output = self.pos_encoder(enc_output)

        # 通过时间感知Transformer编码器
        enc_output, (h_n, c_n) = self.encoder(enc_output)
        # enc_output = self.encoder(enc_output, time_diffs, src_mask=attention_mask, src_key_padding_mask=seq_mask)
        # enc_output = self.encoder(enc_output, mask=attention_mask, src_key_padding_mask=seq_mask)

        return enc_output

    def compute_states_at_sample_times(self, event_states, sample_dtimes, user_emb, time_seqs, poi_seqs):
        """Compute the hidden states at sampled times.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]
        
        # [batch_size, seq_len, num_samples, time_dim]
        absolute_sample_times = (time_seqs[..., None] + sample_dtimes * 3600).reshape(-1, sample_dtimes.shape[-1])
        period_time_embed = self.get_time_features(absolute_sample_times).reshape(event_states.shape[0], event_states.shape[1], sample_dtimes.shape[-1], -1)
        
        sample_dtimes = sample_dtimes[..., None]

        # [1, 1, 1, num_pois]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]

        # 索引行 
        poi_seqs_np = poi_seqs.detach().cpu().numpy()
        distance_decay = self.distance_matrix[poi_seqs_np]
        # [batch_size, seq_len, 1, num_pois]
        distance_decay = torch.from_numpy(distance_decay).to(self.device).float()[:, :, None, :]
        # [batch_size, seq_len, num_samples, 1]
        time_delta_decay = sample_dtimes / 24.0
        
        alpha = self.alpha[None, None, ...]
        time_delta_decay = self.t_projector(time_delta_decay)
        gate = self.gate_net(time_delta_decay)
        offset = self.offset_net(time_delta_decay)
        # [batch_size, seq_len, 1, num_pois]
        base_spatial_decay = - torch.log(1 + distance_decay)
        # batch_size, seq_len, 1
        time_distance_factor = base_spatial_decay * gate + offset

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_samples, num_pois]
        # intensity_states = self.layer_intensity_hidden(event_states) + factor_intensity_base + self.layer_user_hidden(user_emb)[:, :, None, :] \
        #     + self.time_projector(period_time_embed) 
        intensity_states = factor_intensity_base + self.layer_user_hidden(user_emb)[:, :, None, :] \
            + time_distance_factor \
            + self.time_projector(period_time_embed) \
            + self.layer_intensity_hidden(event_states) \

        return intensity_states

    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            absolute_time_seqs, 
                                            loc_seq, 
                                            poi_seqs,
                                            user_seq, 
                                            poi_cat_seq,
                                            sample_dtimes,
                                            attention_mask,
                                            seq_mask,
                                            **kwargs):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            poi_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_pois], intensity at all sampled times.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        # [batch_size, seq_len, num_samples]
        enc_out = self.forward(time_seqs=time_seqs, poi_seqs=poi_seqs, attention_mask=attention_mask, poi_cat_seq=poi_cat_seq, loc_seq=loc_seq, seq_mask=seq_mask)    

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes, user_emb=self.user_embed_model(user_seq), time_seqs=absolute_time_seqs,
                                                             poi_seqs=poi_seqs)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_pois]
            lambdas = self.softplus(encoder_output)
        
        return lambdas


    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(start=0.0,
                                              end=1.0,
                                              steps=self.loss_integral_num_sample_per_step,
                                              device=self.device)[None, None, :]

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes
    
    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_pois], unmasked intensity at
            (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_pois],
            intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            type_seq (tensor): [batch_size, seq_len], sequence of mark ids, with padded events having a mark of self.pad_token_id

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
        """

        # TODO First, add an epsilon to every marked intensity for stability
        lambda_at_event = lambda_at_event + self.eps
        lambdas_loss_samples = lambdas_loss_samples + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        # Compute event LL - [batch_size, seq_len
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seq,
            ignore_index=0,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # Compute non-event LL [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)
        if self.use_mc_samples:
            non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        else: # Use trapezoid rule
            non_event_ll = 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events
    
    def get_time_features(self, time_seq):

        timestamps_np = time_seq.cpu().numpy()

        datetimes = np.array([[datetime.fromtimestamp(ts) for ts in seq] for seq in timestamps_np])
        hours = np.array([[dt.hour for dt in time_seq] for time_seq in datetimes])
        dayofweeks = np.array([[dt.weekday() for dt in time_seq] for time_seq in datetimes])
        months = np.array(object=[[dt.month for dt in time_seq] for time_seq in datetimes])
        dayofmonths = np.array([[dt.day / calendar.monthrange(dt.year, dt.month)[1] for dt in time_seq] for time_seq in datetimes])

        hour_seqs = torch.from_numpy(hours).long().to(self.device)
        dayofweek_seqs = torch.from_numpy(dayofweeks).long().to(self.device)
        dayofmonth_seqs = torch.from_numpy(dayofmonths).float().to(self.device)
        month_seqs = torch.from_numpy(months).long().to(self.device)

        # hour_seqs = []
        # dayofweek_seqs = []
        # dayofmonth_seqs = []
        # month_seqs = []

        # start_idx = 0
        # for batch_idx in range(batch_size):
        #     valid_count = seq_len[batch_idx]
        #     end_idx = start_idx + valid_count
            
        #     hour_seqs.append(hours_tensor[start_idx:end_idx])
        #     dayofweek_seqs.append(dayofweeks_tensor[start_idx:end_idx])
        #     dayofmonth_seqs.append(dayofmonths_tensor[start_idx:end_idx])
        #     month_seqs.append(months_tensor[start_idx:end_idx])
        #     start_idx = end_idx

        # # 6. 批量padding
        # hour_seqs = torch.nn.utils.rnn.pad_sequence(hour_seqs, batch_first=True, padding_value=0)
        # dayofweek_seqs = torch.nn.utils.rnn.pad_sequence(dayofweek_seqs, batch_first=True, padding_value=0)
        # dayofmonth_seqs = torch.nn.utils.rnn.pad_sequence(dayofmonth_seqs, batch_first=True, padding_value=0)
        # month_seqs = torch.nn.utils.rnn.pad_sequence(month_seqs, batch_first=True, padding_value=0)
        
        time_features = self.time_period_encoder({
            "hour": hour_seqs, 
            "dayofweek": dayofweek_seqs
        })
        return time_features
    
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        '''Embeds a given Variable.
        Args:
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
        '''
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(
            inputs, self.lookup_table,
            self.padding_idx, None, 2, False, False)  # copied from torch.nn.modules.sparse.py

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TemporalAttention(nn.Module):
    """
    时间感知的自注意力机制，将时间间隔作为偏差加入注意力得分。
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # 时间衰减参数：可学习的指数衰减系数
        self.beta = nn.Parameter(torch.tensor(1.0))  # 缩放参数
        self.gamma = nn.Parameter(torch.tensor(1.0))  # 衰减率参数
        
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_linear.weight)
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)

    def forward(self, q, k, v, time_diffs, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)

        # 线性变换并分头
        q = self.q_linear(q).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # 计算内容注意力得分
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 计算时间衰减项：使用对数衰减避免数值爆炸，可调整
        # time_diffs形状: [batch_size, seq_len, seq_len]
        # time_decay = self.beta * torch.tanh(self.gamma * time_diffs)
        time_decay = torch.exp(-self.gamma * time_diffs)
        time_decay = time_decay.unsqueeze(1)  # 添加头维度 [batch_size, 1, seq_len, seq_len]

        # 将时间衰减项加到注意力得分上
        attn_scores = attn_scores + time_decay

        # 应用掩码（如需要）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(output)
        return output

class TemporalTransformerEncoderLayer(nn.Module):
    """
    自定义Transformer编码器层，集成时间感知注意力。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TemporalTransformerEncoderLayer, self).__init__()
        self.self_attn = TemporalAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, time_diffs, src_mask=None, src_key_padding_mask=None):
        # 时间感知自注意力
        src2 = self.self_attn(src, src, src, time_diffs, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TemporalTransformerEncoder(nn.Module):
    """
    自定义Transformer编码器，由多个TemporalTransformerEncoderLayer组成。
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TemporalTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TemporalTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, time_diffs, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, time_diffs, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output