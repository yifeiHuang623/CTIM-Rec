from utils.register import register_view
from utils.logger import get_logger
logger = get_logger(__name__)
import pandas as pd
from torch.utils.data import Dataset

@register_view("CLSPRec_preprocess")
def preprocess(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    num_user = len(raw_df["user_id"].unique())
    num_poi = len(raw_df["POI_id"].unique())
    num_cat = len(raw_df["POI_catid"].unique())
    view_value["num_user"] = num_user + 1
    view_value["num_poi"] = num_poi + 1
    view_value["num_cat"] = num_cat + 1
    view_value["raw_df"] = raw_df
    return raw_df, view_value
