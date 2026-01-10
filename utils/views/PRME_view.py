from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset

from utils.register import register_view
from utils.logger import get_logger
logger = get_logger(__name__)

@register_view("PRME_count")
def PRME_count(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for PRME that prepares the dataset for training.
    """
    logger.info("Applying PRME_count to dataset")
    
    # count the number of unique users and items
    n_user = raw_df['user_id'].nunique()
    n_item = raw_df['POI_id'].nunique()
    view_value['n_user'] = n_user
    view_value['n_item'] = n_item

    return raw_df, view_value