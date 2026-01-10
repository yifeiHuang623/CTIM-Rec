from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

@register_view("STAN_preview")
def STAN_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for STAN that prepares the dataset for training.
    """
    logger.info("Applying STAN_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    
    raw_df['time_encode'] = pd.to_datetime(raw_df['timestamps']).apply(lambda x: x.hour + x.weekday() * 24 + 1)
    
    poi_df = raw_df.drop_duplicates(subset=['POI_id']).reset_index(drop=True).sort_values('POI_id')
    lon = np.radians(poi_df['longitude'].values)
    lat = np.radians(poi_df['latitude'].values)
    
    R = 6371.0
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist_matrix = 2 * R * np.arcsin(np.sqrt(a))
    dist_matrix_pad = np.zeros((num_pois + 1, num_pois + 1))
    dist_matrix_pad[1:, 1:] = dist_matrix

    view_value['spatial_matrix'] = dist_matrix_pad * 10

    return raw_df, view_value
    
@register_view("STAN_post_view")
def STAN_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from tqdm import tqdm
    from model.STAN.STAN_utils import cal_adj_time, cal_candidate_time
    """
    A post view for STAN that prepares the dataset for training.
    """
    logger.info("Applying STAN_post_view to dataset")
    
    max_time = 0
    
    new_raw_df = []
    
    for seq_data in tqdm(raw_df):
        traj_temporal_mat, max_time = cal_adj_time(seq_data['timestamps'], max_time)
        candiate_temporal_vec , max_time = cal_candidate_time(np.concatenate([seq_data['timestamps'], np.array([seq_data['y_POI_id']['timestamps']])],
                                                                                        axis=-1), max_time)
        seq_len = len(seq_data['POI_id'])
        valid_len = seq_data['mask']
        seq_user = np.zeros_like(seq_data['POI_id'])
        seq_user[:valid_len] = np.array([seq_data['user_id']] * valid_len) 
        
        traj = np.stack([seq_user, seq_data['POI_id'], seq_data['time_encode']], axis=-1)
        
        # seq_data['traj_temporal_mat'] = traj_temporal_mat
        # seq_data['candidate_temporal_vec'] = candiate_temporal_vec[valid_len-1]
        # seq_data['traj'] = traj
        valid_seq = seq_data['mask']
        
        max_time, min_time = 0, 10000
        for i in range(valid_seq):
            new_data = {}
            new_data['y_POI_id'] = {}
            
            mask = np.zeros((seq_len, 3), np.int32)
            mask[:i+1, :] = 1
            mask_traj = traj * mask
            mask = np.zeros((seq_len, seq_len))
            mask[:i+1, :i+1] = 1
            mask_traj_temporal_mat = traj_temporal_mat * mask
                
            new_data['traj'] = mask_traj
            new_data['traj_temporal_mat'] = mask_traj_temporal_mat
            new_data['candiate_temporal_vec'] = candiate_temporal_vec[i]
            new_data['mask'] = i + 1
            
            if i != (valid_len - 1):
                new_data['target'] = traj[i+1][1] - 1
            else:
                new_data['target'] = seq_data['y_POI_id']['POI_id'] - 1
                
            max_time = max(max_time, np.max(new_data['traj_temporal_mat']))
            min_time = min(min_time, np.min(new_data['traj_temporal_mat']))
                
            new_raw_df.append(new_data)

    raw_df = new_raw_df
    # convert each L-length trajectory into L trajctories
    view_value['max_len'] = len(raw_df[0]['traj'])    
    view_value['ex'] = [np.max(view_value['spatial_matrix'].max()), np.min(view_value['spatial_matrix']), max_time, 0]
    
    return raw_df, view_value