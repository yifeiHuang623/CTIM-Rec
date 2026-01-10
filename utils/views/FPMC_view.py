import pandas as pd
from torch.utils.data import Dataset

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)


@register_view("FPMC_count")
def FPMC_count(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for FPMC that prepares the dataset for training.
    """
    logger.info("Applying FPMC_count to dataset")

    # count the number of unique users and items
    n_user = raw_df["user_id"].nunique()
    n_item = raw_df["POI_id"].nunique()
    view_value["n_user"] = n_user + 1
    view_value["n_item"] = n_item + 1

    return raw_df, view_value
