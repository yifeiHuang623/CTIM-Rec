import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

@register_view("RMTPP_preview")
def RMTPP_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for RMTPP that prepares the dataset for training.
    """
    logger.info("Applying RMTPP_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    num_poi_types = raw_df['POI_catid'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    view_value['num_poi_types'] = num_poi_types + 1
    
    return raw_df, view_value

@register_view("RMTPP_post_view")
def RMTPP_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for RMTPP.
    """
    logger.info("Applying RMTPP_post_view to dataset")

    for seq_data in raw_df:
        time_delta = np.zeros_like(seq_data['timestamps'], dtype=np.float32)
        seq_data_length = seq_data['mask']
        for i in range(1, seq_data_length):
            # convert seconds to hours
            time_delta[i] = (seq_data['timestamps'][i] - seq_data['timestamps'][i-1]) / 3600.0
        seq_data['time_delta'] = time_delta
        seq_data['y_POI_id']['time_delta'] = (seq_data['y_POI_id']['timestamps'] - seq_data['timestamps'][seq_data_length - 1]) / 3600.0
    
    return raw_df, view_value
