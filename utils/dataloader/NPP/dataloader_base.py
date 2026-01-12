from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Callable
import numpy as np
import pandas as pd

from utils.register import VIEW_REGISTRY

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import OrderedDict 

def post_process_func(
    df: pd.DataFrame,
    sequence_length: int,
    n_jobs: int = 1,
    backend: str = "loky"
) -> List[Dict[str, Any]]:
    df = df.sort_values(by='timestamps')
    group_key = ['user_id', 'trajectory_id'] if 'trajectory_id' in df.columns else ['user_id']
    groupby_user = df.groupby(group_key)
    data_list = []
    for index, data in tqdm(groupby_user):
        user_id = index[0] if isinstance(index, tuple) else np.int64(index)
        start_idx = 0
        data = data.sort_values(by='timestamps')
        cols = data.columns
    
        time_diff = np.diff(data['timestamps'].values, prepend=data['timestamps'].values[0])
        time_diff[0] = 0
        breakpoints = (time_diff > 86400)
        break_indices = np.where(breakpoints)[0]
        break_indices = np.concatenate(([0], break_indices, [len(data)])).astype(int)
        
        for i in range(len(break_indices) - 1):
            start_idx = break_indices[i]
            end_idx = break_indices[i + 1] - 1
            length = end_idx - start_idx + 1
            
            if length <= 2:
                continue
            
            slice_end = min(end_idx, start_idx + sequence_length)
            real_len  = slice_end - start_idx
            pad_num   = sequence_length - real_len

            sample = {
                'user_id': user_id,
                'mask'   : real_len,
                'trajectory_id': len(data_list), 
                'y_POI_id': data.iloc[slice_end].to_dict()
            }
            sample['y_POI_id']['trajectory_id'] = sample['trajectory_id']

            for col in cols:
                if col in sample:
                    continue
                seq = data[col].iloc[start_idx:slice_end].to_numpy()

                pad_val = 0 if np.issubdtype(seq.dtype, np.number) else None
                if pad_num: 
                    seq = np.pad(seq, (0, pad_num), constant_values=pad_val)

                sample[col]        = seq
            data_list.append(sample)
            
    return data_list


def flex_collate(batch):
    all_keys = set().union(*batch)
    collated = {}
    for k in all_keys:
        values = [b.get(k, None) for b in batch]

        if all(isinstance(v, torch.Tensor) for v in values):
            collated[k] = torch.stack(values)

        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            collated[k] = torch.tensor(values)

        elif all(isinstance(v, dict) for v in values if v is not None):

            sub_batch = [{**(v or {})} for v in values]
            collated[k] = flex_collate(sub_batch)
        else:
            
            collated[k] = values

    return collated

def _maybe_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return torch.tensor(x, dtype=torch.long)
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
        return torch.as_tensor(x)
    return x   

class BaseDataset(Dataset):
    def __init__(
        self,
        preprocess_func: Callable[..., pd.DataFrame],
        sequence_length: int,
        n_jobs: int = 1,
        pre_views: List[str] = None,
        post_views: List[str] = None,
    ) -> None:
        super().__init__()
        raw_df = preprocess_func()
        view_value = {}
        if pre_views:
            for view in pre_views:
                if view not in VIEW_REGISTRY:
                    raise ValueError(f"View '{view}' not registered")
                raw_df, view_value = VIEW_REGISTRY[view](raw_df, view_value)
  
        raw_df = post_process_func(
            raw_df, sequence_length=sequence_length, n_jobs=n_jobs
        )
        if post_views:
            for view in post_views:
                if view not in VIEW_REGISTRY:
                    raise ValueError(f"View '{view}' not registered")
                raw_df, view_value = VIEW_REGISTRY[view](raw_df, view_value)
        self.samples = raw_df
        self.view_value = view_value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ordered = OrderedDict()
        for k, v in s.items():
            ordered[k] = _maybe_tensor(v)
        return ordered
    
class BaseDataLoader():
    def __init__(self, MyDataset, dataset_name, logger, args=None, model_args=None,pre_views=None, post_views=None) -> None:
        dataset = MyDataset(pre_views=pre_views, post_views=post_views)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        torch.manual_seed(args.get("seed", 42))
        np.random.seed(args.get("seed", 42))
        train_dataset, test_dataset, val_dataset = random_split(dataset, lengths=[train_size,val_size,test_size])

        train_batch_size = model_args.get("train_batch_size", model_args.get("batch_size", 32))
        val_batch_size = model_args.get("val_batch_size", model_args.get("batch_size", 32))
        test_batch_size = model_args.get("test_batch_size", model_args.get("batch_size", 32))

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            collate_fn=flex_collate)
        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            collate_fn=flex_collate)
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size,
            collate_fn=flex_collate)
        
        self.view_value = dataset.view_value 
        logger.info("DataLoader initialized with dataset: %s", dataset_name)
        logger.info("Train Batch size: %d, Total Batches: %d",
                        train_batch_size,
                        len(self.train_dataloader.dataset) // train_batch_size)
        logger.info("Validation Batch size: %d, Total Batches: %d",
                        val_batch_size,
                        len(self.val_dataloader.dataset) // val_batch_size)
        logger.info("Test Batch size: %d, Total Batches: %d",
                        test_batch_size,
                        len(self.test_dataloader.dataset) // test_batch_size)
        logger.info("Sequence length: %d", args.get("sequence_length", 30))
        logger.info("Number of jobs for preprocessing: %d", args.get("n_jobs", 1))