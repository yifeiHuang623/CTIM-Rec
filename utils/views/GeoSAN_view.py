from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

@register_view("GeoSAN_preview")
def GeoSAN_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from model.GeoSAN.GeoSAN_utils import build_region_id, get_visited_locs_times, LocQuerySystem, KNNSampler, QuadkeyField
    """
    A preprocessing view for GeoSAN that prepares the dataset for training.
    """
    logger.info("Applying GeoSAN_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    num_poi_cats = raw_df['POI_catid'].nunique()
    
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    view_value['num_poi_cats'] = num_poi_cats + 1
    
    view_value['region_id_map'], QUADKEY = build_region_id(raw_df['POI_id'], raw_df['latitude'], raw_df['longitude'])
    view_value['num_time'] = raw_df['time_id'].nunique() + 1
    
    user_visited_locs, user_visited_times = get_visited_locs_times(raw_df)
    loc_query_sys = LocQuerySystem()
    loc_query_sys.build_tree(raw_df['POI_id'], raw_df['latitude'], raw_df['longitude'])
    view_value['sampler'] = KNNSampler(
            query_sys=loc_query_sys,
            user_visited_locs=user_visited_locs,
            user_visited_times=user_visited_times
        )
    
    global_quadkeys = QuadkeyField()
    global_quadkeys.build_vocab(QUADKEY)
    view_value['QUADKEY'] = global_quadkeys
    view_value['nquadkey'] = len(global_quadkeys.vocab)

    return raw_df, view_value
    
@register_view("GeoSAN_post_view")
def GeoSAN_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from tqdm import tqdm
    """
    A post view for GeoSAN that prepares the dataset for training.
    """
    logger.info("Applying GeoSAN_post_view to dataset")
    
    global_quadkeys = view_value['QUADKEY']
    
    # not necessary
    for seq_data in tqdm(raw_df):
        region = [view_value['region_id_map'].get(poi_id, None) for poi_id in seq_data['POI_id']] 
        r = global_quadkeys.numericalize(list(region))  # (L, LEN_QUADKEY) 
        seq_data['region_id'] = r
        
    return raw_df, view_value