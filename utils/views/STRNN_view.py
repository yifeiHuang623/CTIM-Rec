import pandas as pd
from torch.utils.data import Dataset
from typing import List

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)


@register_view("STRNN_preview")
def STRNN_preview(
    raw_df: pd.DataFrame, view_value: dict = None
) -> tuple[Dataset, dict]:
    """
    A preprocessing view for STRNN that prepares the dataset for training.
    """
    logger.info("Applying STRNN_preview to dataset")

    # count the number of unique users and items
    n_user = raw_df["user_id"].nunique()
    user_list = raw_df["user_id"].unique().tolist()
    raw_df["user_id"] = raw_df["user_id"].apply(lambda x: user_list.index(x))
    n_item = raw_df["POI_id"].nunique()
    item_list = raw_df["POI_id"].unique().tolist()
    raw_df["POI_id"] = raw_df["POI_id"].apply(lambda x: item_list.index(x))
    view_value["uid_size"] = n_user
    view_value["loc_size"] = n_item

    time_start, time_end = raw_df["timestamps"].min(), raw_df["timestamps"].max()
    view_value["tim_size"] = time_end - time_start

    return raw_df, view_value


def _dis_cul(lan1:float, lon1:float, lan2:float, lon2:float) -> float:
    """
    Calculate the distance between two geographical points.
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Radius of the Earth in kilometers
    dlon = radians(lon2 - lon1)
    dlat = radians(lan2 - lan1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lan1)) * cos(radians(lan2)) * sin(dlon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in kilometers


@register_view("STRNN_postview")
def STRNN_postview(raw_df: List[dict], view_value: dict = None) -> pd.DataFrame:
    """
    A postprocessing view for STRNN that returns the processed DataFrame.
    """
    logger.info("Applying STRNN_postview to dataset")
    max_dis = 0
    for i in range(len(raw_df)):
        raw_df[i]["current_dis"] = []
        for j in range(len(raw_df[i]["POI_id"])):
            dis = _dis_cul(
                raw_df[i]["latitude"][j],
                raw_df[i]["longitude"][j],
                raw_df[i]["y_POI_id"]["latitude"],
                raw_df[i]["y_POI_id"]["longitude"],
            )
            raw_df[i]["current_dis"].append(dis)

            if raw_df[i]["current_dis"][j] > max_dis:
                max_dis = raw_df[i]["current_dis"][j]
    view_value["distance_upper"] = max_dis
    return raw_df, view_value
