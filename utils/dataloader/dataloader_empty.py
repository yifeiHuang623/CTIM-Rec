# from pathlib import Path
# from typing import Any

# import pandas as pd
# from torch.utils.data import DataLoader, default_collate

# from utils.register import register_dataloader
# from utils.logger import get_logger
# from utils.dataloader.NPP.dataloader_base import BaseDataset, flex_collate
# from utils.exargs import ConfigResolver

# logger = get_logger(__name__)
# dataset_name = ''
# root_path = Path(__file__).resolve().parent.parent.parent.parent
# args = ConfigResolver(f'').parse()


# def pre_process_func() -> pd.DataFrame:
#     df = 
#     return df

# class MyDataset(BaseDataset):
#     def __init__(self) -> None:
#         super().__init__(
#             preprocess_func=pre_process_func,
#             sequence_length=,
#             n_jobs=args.get("n_jobs", 1)
#         )

# # @register_dataloader(name=dataset_name)
# class MyDataLoader(DataLoader):
#     def __init__(self) -> None:
#         super().__init__(
#             dataset=MyDataset(),
#             batch_size=args.get("batch_size", 32),
#             collate_fn=flex_collate
#         )