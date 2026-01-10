from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

@register_view("MTNet_preview")
def MTNet_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from model.MTNet.MTNet_utils import build_region_id
    """
    A preprocessing view for MTNet that prepares the dataset for training.
    """
    logger.info("Applying MTNet_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    num_poi_cats = raw_df['POI_catid'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    view_value['num_poi_cats'] = num_poi_cats + 1
    
    region_list = build_region_id(raw_df, num_clusters=300)
    raw_df['region_id'] = region_list
    view_value['num_regions'] = len(set(region_list)) + 1
    
    return raw_df, view_value

@register_view("MTNet_post_view")
def MTNet_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for MTNet.
    """
    logger.info("Applying MTNet_post_view to dataset")
    
    for idx in range(len(raw_df)):
        raw_df[idx]['trajectory_id'] = idx + 1
    
    return raw_df, view_value