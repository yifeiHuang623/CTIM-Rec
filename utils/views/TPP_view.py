from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

@register_view("TPP_preview")
def TPP_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for TPP that prepares the dataset for training.
    """
    logger.info("Applying TPP_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    num_poi_types = raw_df['POI_catid'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    view_value['num_poi_types'] = num_poi_types + 1
    
    # 1. 按 POI_id 去重，保留唯一坐标
    poi_df = raw_df.drop_duplicates(subset=['POI_id']).reset_index(drop=True).sort_values('POI_id')

    # 2. 经纬度转弧度
    lon = np.radians(poi_df['longitude'].values)
    lat = np.radians(poi_df['latitude'].values)

    # 3. 广播方式计算所有 POI 间距离（Haversine公式）
    R = 6371.0  # 地球半径 km
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist_matrix = 2 * R * np.arcsin(np.sqrt(a))
    
    # normalization
    dist_matrix = np.clip(dist_matrix, 0, 1000)
    # dist_matrix = (dist_matrix - np.min(dist_matrix, axis=-1)) / (np.max(dist_matrix, axis=-1) - np.min(dist_matrix, axis=-1) + 1e-8)
    dist_matrix_pad = np.zeros((num_pois + 1, num_pois + 1))
    dist_matrix_pad[1:, 1:] = dist_matrix
    view_value['distance_matrix'] = dist_matrix_pad
    
    return raw_df, view_value

@register_view("TPP_post_view")
def TPP_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for TPP.
    """
    logger.info("Applying TPP_post_view to dataset")

    for seq_data in raw_df:
        
        # mask掉最后k个数据
        k = 0
        mask_pos = int(seq_data['mask'])  # 确保是 Python 整数
        new_len = max(mask_pos - k, 0)    # 新的有效长度
        seq_data['mask'] = new_len
        # 遮住最后 k 个有效元素
        if mask_pos > 0:
            start = max(new_len, 0)
            seq_data['POI_id'][start:mask_pos] = 0
            seq_data['timestamps'][start:mask_pos] = 0
        
        time_delta = np.zeros_like(seq_data['timestamps'], dtype=np.float32)
        seq_data_length = seq_data['mask']
        for i in range(1, seq_data_length):
            # convert seconds to hours
            time_delta[i] = (seq_data['timestamps'][i] - seq_data['timestamps'][i-1]) / 3600.0
        seq_data['time_delta'] = time_delta
        seq_data['y_POI_id']['time_delta'] = (seq_data['y_POI_id']['timestamps'] - seq_data['timestamps'][seq_data_length - 1]) / 3600.0
    
    return raw_df, view_value
