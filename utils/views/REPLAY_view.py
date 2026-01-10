from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

@register_view("REPLAY_preview")
def REPLAY_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for REPLAY that prepares the dataset for training.
    """
    logger.info("Applying REPLAY_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    num_poi_types = raw_df['POI_catid'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    view_value['num_poi_types'] = num_poi_types + 1
    
    
    raw_df['time_slot'] = (pd.to_datetime(raw_df['timestamps'], unit='s').dt.weekday * 24 +
                       pd.to_datetime(raw_df['timestamps'], unit='s').dt.hour)
    
    return raw_df, view_value

@register_view("REPLAY_post_view")
def REPLAY_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for REPLAY.
    """
    logger.info("Applying REPLAY_post_view to dataset")
    
    for seq_data in raw_df:
        
        # mask掉最后k个数据
        k = 2
        mask_pos = int(seq_data['mask'])  # 确保是 Python 整数
        new_len = max(mask_pos - k, 0)    # 新的有效长度
        seq_data['mask'] = new_len
        # 遮住最后 k 个有效元素
        if mask_pos > 0:
            start = max(new_len, 0)
            seq_data['POI_id'][start:mask_pos] = 0
            seq_data['timestamps'][start:mask_pos] = 0
        
    
    return raw_df, view_value
