from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

@register_view("TPG_preview")
def TPG_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from model.TPG.TPG_utils import build_region_id, get_visited_locs_times, LocQuerySystem, KNNSampler, QuadkeyField
    """
    A preprocessing view for TPG that prepares the dataset for training.
    """
    logger.info("Applying TPG_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    num_poi_cats = raw_df['POI_catid'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    view_value['num_poi_cats'] = num_poi_cats + 1
    
    raw_df['time_id'] = raw_df['timestamps'].apply(lambda x: datetime.fromtimestamp(x).weekday() * 24 + datetime.fromtimestamp(x).hour + 1)
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
    
@register_view("TPG_post_view")
def TPG_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from tqdm import tqdm
    """
    A post view for TPG that prepares the dataset for training.
    """
    logger.info("Applying TPG_post_view to dataset")
    
    global_quadkeys = view_value['QUADKEY']
    sampler = view_value['sampler']
    
    length = 9
    
    for seq_data in tqdm(raw_df):
        
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
        
        region = []
        for _ in range(length):
            region.append([])
        region_seq = [view_value['region_id_map'].get(poi_id, None) for poi_id in seq_data['POI_id']]
        for idx in range(length):
            r = [r_[idx] for r_ in region_seq]
            r = tuple(r)
            r = global_quadkeys.numericalize(list(r))  # (L, LEN_QUADKEY) 
            region[idx].append(r)
        # TODO length, seq_len, LEN_QUADKEY -> seq_len, length, LEN_QUADKEY
        seq_data['region_id'] = np.concatenate(region, axis=0).transpose(1, 0, 2)
        
        # neg, probs, times_neg, _ = sampler(seq_data, k=100, user=seq_data['user_id'])
        # # seq_len, 1+num_neg
        # y_poi_id = np.zeros_like(seq_data['POI_id'])
        # y_poi_id[:seq_data['mask']-1] = seq_data['POI_id'][1: seq_data['mask']]
        # y_poi_id[seq_data['mask']-1] = seq_data['y_POI_id']['POI_id']
        
        # y_time_id = np.zeros_like(seq_data['time_id'])
        # y_time_id[:seq_data['mask']-1] = seq_data['time_id'][1:seq_data['mask']]
        # y_time_id[seq_data['mask']-1] = seq_data['y_POI_id']['time_id']
        
        # trg_seq = np.concatenate((np.expand_dims(y_poi_id, axis=-1), neg), axis=-1)
        # trg_time = np.concatenate((np.expand_dims(y_time_id, axis=-1), times_neg), axis=-1)
        
        # trg_regs = []
        # for _ in range(length):
        #     trg_regs.append([]) 
            
        # for trg_seq_idx in range(len(trg_seq)):
        #     regs = []
        #     for _ in range(length):
        #         regs.append([])
                
        #     for loc in trg_seq[trg_seq_idx]:
        #         query_region = view_value['region_id_map'].get(loc, None)
        #         for idx in range(length):
        #             regs[idx].append(query_region[idx])
                    
        #     for idx in range(length):
        #         trg_regs[idx].append(global_quadkeys.numericalize(regs[idx]))
        # # length, seq_len, 1+num_neg, LEN_QUADKEY -> seq_len, length, (1+num_neg), LEN_QUADKEY
        # trg_regs = np.stack(trg_regs, axis=0).transpose(1, 0, 2, 3)
    
        # # seq_len, 1+num_neg
        # seq_data['target_seqs'] = trg_seq
        # seq_data['time_query'] = trg_time
        # seq_data['target_seqs_probs'] = probs
        
        # # seq_len, length, 1+num_neg, LEN_QUADKEY
        # seq_data['target_seqs_region_id'] = trg_regs
        
    return raw_df, view_value