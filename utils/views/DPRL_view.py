from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

@register_view("DPRL_preview")
def DPRL_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from model.DPRL.DPRL_utils import build_region_id
    """
    A preprocessing view for DPRL that prepares the dataset for training.
    """
    logger.info("Applying DPRL_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    
    poi2region = build_region_id(raw_df, num_clusters=3000)
    raw_df['region'] = raw_df['POI_id'].map(poi2region)
    raw_df['region'] = pd.factorize(raw_df['region'])[0] + 1 
    view_value['num_regions'] = raw_df['region'].max() + 1
    raw_df['time_slot'] = (pd.to_datetime(raw_df['timestamps'], unit='s').dt.weekday * 24 +
                       pd.to_datetime(raw_df['timestamps'], unit='s').dt.hour)
    view_value['num_time_slots'] = raw_df['time_slot'].nunique() + 1
    
    return raw_df, view_value

@register_view("DPRL_post_view")
def DPRL_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for DPRL.
    """
    logger.info("Applying DPRL_post_view to dataset")
    
    return raw_df, view_value