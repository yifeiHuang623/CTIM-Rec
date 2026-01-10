from utils.register import register_view
from utils.logger import get_logger
import pandas as pd
from torch.utils.data import Dataset

logger = get_logger(__name__)


@register_view("GETNext_view")
def GETNext_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from model.GETNext.utils import get_norm_time
    num_user = len(raw_df["user_id"].unique())
    num_poi = len(raw_df["POI_id"].unique())
    num_cat = len(raw_df["POI_catid"].unique())
    raw_df["norm_time"] = raw_df["timestamps"].apply(get_norm_time)
    view_value["num_users"] = num_user + 1
    print(num_user)
    print(num_poi)
    print(len(raw_df))
    exit()
    view_value["num_pois"] = num_poi + 1
    view_value["num_cats"] = num_cat + 1
    return raw_df, view_value
