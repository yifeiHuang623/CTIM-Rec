import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import gensim
from gensim import models
import datetime
from tqdm import tqdm

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

def datetime_to_features(timestamp):
    """Convert timestamp to weekday and hour features"""
    dt = datetime.datetime.fromtimestamp(int(timestamp) // 1000)
    weekday = dt.weekday()
    hour = dt.hour
    return weekday, hour

@register_view("MCLP_preview")
def MCLP_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    A preprocessing view for MCLP that prepares the dataset for training.
    This corresponds to the preprocess_data function in the original code.
    """
    logger.info("Applying MCLP_preview to dataset")
    
    # Get unique users and POIs
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    num_poi_types = raw_df['POI_catid'].nunique()
    
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    view_value['num_poi_types'] = num_poi_types + 1
    
    # Group by user to build individual transition matrices and occurrence patterns
    logger.info("Building time transition matrices and occurrence patterns")
    
    trans_time_individual = [np.ones((24,24))]
    occur_time_individual = np.zeros(shape=(num_users + 1, 24))
    user_loc_matrix = np.zeros((num_users + 1, num_pois + 1))  # for LDA
    
    # Sort by user and timestamp
    raw_df = raw_df.sort_values(['user_id', 'timestamps']).reset_index(drop=True)
    
    for user_idx in tqdm(range(1, num_users + 1), desc='Preprocessing users'):
        user_data = raw_df[raw_df['user_id'] == user_idx].reset_index(drop=True)
        
        # Initialize time transition matrix for this user
        trans_matrix_time = np.ones((24, 24))
        
        for i in range(len(user_data) - 1):
            loc_idx = user_data.loc[i, 'POI_id']
            timestamp = user_data.loc[i, 'timestamps']
            next_loc_idx = user_data.loc[i + 1, 'POI_id']
            next_timestamp = user_data.loc[i + 1, 'timestamps']
            
            weekday, hour = datetime_to_features(timestamp)
            next_weekday, next_hour = datetime_to_features(next_timestamp)
            
            # Update transition matrix
            trans_matrix_time[hour, next_hour] += 1
            
            # Update occurrence pattern
            occur_time_individual[user_idx][hour] += 1
            
            # Update user-location matrix for LDA
            user_loc_matrix[user_idx, loc_idx] += 1
            
            # Handle last item
            if i == len(user_data) - 2:
                occur_time_individual[user_idx][next_hour] += 1
                user_loc_matrix[user_idx, next_loc_idx] += 1
        
        # Normalize transition matrix
        time_row_sums = trans_matrix_time.sum(axis=1)
        trans_matrix_time = trans_matrix_time / time_row_sums[:, np.newaxis]
        trans_time_individual.append(trans_matrix_time)
    
    trans_time_individual = np.array(trans_time_individual).astype(np.float32)
    view_value['prob_matrix_time_individual'] = trans_time_individual
    view_value['occur_time_individual'] = occur_time_individual
    
    # LDA topic modeling (optional, controlled by topic_num parameter)
    topic_num = 450
    if topic_num > 0:
        logger.info(f'Generating LDA topic distribution with {topic_num} topics')
        
        num_users_lda, num_locations = user_loc_matrix.shape
        dictionary = gensim.corpora.Dictionary([[str(i)] for i in range(num_locations)])
        corpus = []
        
        for user in user_loc_matrix:
            user_doc = [str(loc) for loc, count in enumerate(user) for _ in range(int(count))]
            corpus.append(dictionary.doc2bow(user_doc))
        
        lda = models.LdaModel(corpus, num_topics=topic_num, random_state=42)
        user_topics = np.zeros((num_users_lda, topic_num), dtype=np.float32)
        
        for i, user in enumerate(user_loc_matrix):
            user_doc = [str(loc) for loc, count in enumerate(user) for _ in range(int(count))]
            for item in lda[dictionary.doc2bow(user_doc)]:
                j = item[0]
                prob = item[1]
                user_topics[i, j] = prob
        
        view_value['user_topic_loc'] = user_topics
    
    return raw_df, view_value

@register_view("MCLP_post_view")
def MCLP_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[list, dict]:
    """
    A preprocessing view for MCLP that generates sequence data.
    This corresponds to the generate_data and load_npy_file functions.
    """
    logger.info("Applying MCLP_post_view to dataset")
    
    for seq_data in tqdm(raw_df, desc='Processing sequences'):
        occur_time_user = view_value['occur_time_individual'][seq_data['user_id']]
        seq_data['prob_matrix_time_individual'] = view_value['prob_matrix_time_individual'][seq_data['user_id']]
        seq_data['occur_time_individual'] = occur_time_user
        seq_data['user_topic_loc'] = view_value['user_topic_loc'][seq_data['user_id']]
            
        # Extract hour features and masks
        hour_x = []
        hour_mask = []
        
        for ts in seq_data['timestamps']:
            weekday, hour = datetime_to_features(ts)
            hour_x.append(hour)
            
            # Create mask for hours where user has never appeared
            mask = np.zeros(24, dtype=np.int32)
            mask[occur_time_user == 0] = 1
            hour_mask.append(mask)
        
        _, time_slot_y = datetime_to_features(seq_data['y_POI_id']['timestamps'])
            
        seq_data['hour'] = np.array(hour_x, dtype=np.int32)
        seq_data['hour_mask'] = np.array(hour_mask, dtype=np.int32)
        seq_data['timeslot_y'] = time_slot_y
    
    return raw_df, view_value