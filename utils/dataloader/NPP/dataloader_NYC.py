from pathlib import Path
from typing import Any

import pandas as pd

from utils.register import register_dataloader
from utils.logger import get_logger
from utils.dataloader.NPP.dataloader_base import BaseDataset, BaseDataLoader
from utils.exargs import ConfigResolver
from utils.register import VIEW_REGISTRY

logger = get_logger(__name__)
dataset_name = 'NYC'
root_path = Path(__file__).resolve().parent.parent.parent.parent
args = ConfigResolver(f'{root_path}/data/{dataset_name}/{dataset_name}.yaml').parse()


def pre_process_func() -> pd.DataFrame:
    df = pd.read_csv(f'{root_path}/data/{dataset_name}/{dataset_name}.csv')
    df = df[["UTCTimeOffsetEpoch","trajectory_id","POI_id","user_id","latitude", "longitude", "POI_catid"]]
    df["POI_id"] = pd.factorize(df["POI_id"])[0] + 1
    df["user_id"] = pd.factorize(df["user_id"])[0] + 1
    df["POI_catid"] = pd.factorize(df["POI_catid"])[0] + 1
    df["timestamps"] = pd.to_datetime(df["UTCTimeOffsetEpoch"], unit="s", errors="coerce").view("int64") // 10**9
    df = df.drop(columns=["UTCTimeOffsetEpoch"])
    df["user_id"] = df["user_id"].astype('int64')
    df["POI_id"] = df["POI_id"].astype('int64')
    df["POI_catid"] = df["POI_catid"].astype('int64')
    df = df.sort_values("timestamps")
    return df

class MyDataset(BaseDataset):
    def __init__(self, pre_views=None, post_views=None) -> None:
        super().__init__(
            preprocess_func=pre_process_func,
            sequence_length=args.get("sequence_length", 30),
            n_jobs=args.get("n_jobs", 1),
            pre_views=pre_views,
            post_views=post_views,
        )

@register_dataloader(name=dataset_name)
class MyDataLoader(BaseDataLoader):
    def __init__(self, model_args=None,pre_views=None, post_views=None) -> None:
        super().__init__(
            MyDataset=MyDataset,
            dataset_name=dataset_name,
            logger=logger,
            args=args,
            model_args=model_args,
            pre_views=pre_views,
            post_views=post_views
        )

if __name__ == '__main__':
    dict_args = dict()
    d = MyDataLoader(model_args=dict_args)
    train_dataloader = d.train_dataloader
    for i, batch in enumerate(train_dataloader):
        print(batch)
        exit(0)