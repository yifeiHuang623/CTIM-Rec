from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

@register_view("TAPT_preview")
def TAPT_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for TAPT that prepares the dataset for training.
    """
    logger.info("Applying TAPT_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    
    return raw_df, view_value

@register_view("TAPT_post_view")
def TAPT_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from tqdm import tqdm
    from model.TAPT.TAPT_utils import timestamp_sequence_to_datetime_sequence_batch0
    """
    A post view for TAPT that prepares the dataset for training.
    """
    logger.info("Applying TAPT_post_view to dataset")
    
    for seq_data in tqdm(raw_df):
        time_seqs = seq_data['timestamps']
        
        len_seq = seq_data['mask'] + 1
        total_time_seqs = np.zeros(len(time_seqs) + 1)
        total_time_seqs[:len_seq - 1] = time_seqs[:len_seq - 1]
        total_time_seqs[len_seq - 1] = seq_data['y_POI_id']['timestamps']
        
        datetime_sequence = timestamp_sequence_to_datetime_sequence_batch0(total_time_seqs)
        
        sequence_data = []
        for dt0 in datetime_sequence:
            if dt0 == None:
                sequence_data.append([0, 0, 0])
            else:
                sequence_data.append([dt0.hour, dt0.minute, dt0.second])
        sequence_data = np.array(sequence_data)

        seq_data["convert_time"] = np.zeros((len(time_seqs), 3),  dtype=np.int32)
        seq_data["convert_time"][:len_seq - 1] = sequence_data[:len_seq - 1]

        seq_data["y_POI_id"]["convert_time"] = np.zeros((len(time_seqs), 3), dtype=np.int32)
        seq_data["y_POI_id"]["convert_time"][:len_seq - 1] = sequence_data[1: len_seq]
        
        time_delta = np.zeros_like(seq_data['timestamps'], dtype=np.float32)
        seq_data_length = seq_data['mask']
        for i in range(1, seq_data_length):
            # convert seconds to hours
            time_delta[i] = (seq_data['timestamps'][i] - seq_data['timestamps'][i-1]) / 3600.0
        seq_data['time_delta'] = time_delta
        seq_data['y_POI_id']['time_delta'] = (seq_data['y_POI_id']['timestamps'] - seq_data['timestamps'][seq_data_length - 1]) / 3600.0
    
    return raw_df, view_value