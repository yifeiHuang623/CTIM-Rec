import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from utils.register import register_view
from utils.logger import get_logger
from utils.exargs import ConfigResolver

model_args = ConfigResolver("./model/DeepMove/DeepMove.yaml").parse()

logger = get_logger(__name__)


def _lower_bound(x, lower):
    """
    Find the lower bound index of x in a sorted array.
    """
    if x < lower[0]:
        return 0
    elif x > lower[-1]:
        return len(lower) - 1
    else:
        for i in range(len(lower) - 1):
            if lower[i] <= x < lower[i + 1]:
                return i
    return len(lower) - 1


import numpy as np
import pandas as pd
import torch

# 可选：你的日志器
import logging
logger = logging.getLogger(__name__)

# 假设 model_args 在全局可见，包含 {"time_interval": int}
# model_args = {"time_interval": 3600}

def _to_int_tensor(series: pd.Series) -> tuple[torch.Tensor, bool]:
    """
    将 Series 转成整数张量。
    返回: (tensor(int64, CPU), already_numeric)
    - 如果是数值型，直接转为 int64 numpy 再变成 tensor（仍在 CPU）。
    - 如果是对象/字符串，使用 pandas.factorize 按首次出现次序得到 codes（CPU 上完成）。
    """
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
        arr = series.to_numpy(copy=False)
        if arr.dtype != np.int64:
            arr = arr.astype(np.int64, copy=False)
        return torch.from_numpy(arr), True
    else:
        codes, uniques = pd.factorize(series, sort=False)  # 稳定、按首次出现
        return torch.from_numpy(codes.astype(np.int64, copy=False)), False

@torch.no_grad()
def _factorize_on_gpu(x_cpu: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
    """
    在 GPU 上将任意整数向量 x 映射到 [0, n_unique) 的稳定编码（按“首次出现次序”）。
    返回: (codes(int64, GPU), n_unique)
    说明：
    - torch.unique 默认按排序返回唯一值；我们用 scatter_reduce_ 求每个唯一值首次出现位置，
      再根据“首次出现位置”的升序来重排，实现稳定编码。
    """
    x = x_cpu.to(device, non_blocking=True)
    n = x.numel()
    # unique with inverse (unique 值排序，但 inverse 给出每个元素对应的 unique 索引)
    uniques_sorted, inverse = torch.unique(x, return_inverse=True)
    n_unique = uniques_sorted.numel()

    # 计算每个 unique 的首次出现位置
    pos = torch.arange(n, device=device, dtype=torch.int64)
    first_pos = torch.full((n_unique,), n, device=device, dtype=torch.int64)
    # 对 same index 取最小位置（首次出现）
    first_pos.scatter_reduce_(0, inverse, pos, reduce="amin")

    # “首次出现位置”越小，说明越早出现；据此得到 old_unique_idx -> stable_rank 的映射
    order = torch.argsort(first_pos)                      # [rank] -> old_unique_idx
    rank = torch.empty_like(order)                        # [old_unique_idx] -> rank
    rank[order] = torch.arange(n_unique, device=device)   # 反向映射
    codes = rank[inverse]                                 # 元素映射到稳定 rank

    return codes, int(n_unique)

def _to_int32_if_possible(t: torch.Tensor) -> torch.Tensor:
    # 显存敏感：优先转为 int32（若 n_unique 太大需保留 int64）
    if t.dtype != torch.int64:
        t = t.to(torch.int64)
    if t.numel() == 0:
        return t.to(torch.int32)
    max_code = torch.max(t)
    if max_code <= torch.iinfo(torch.int32).max:
        return t.to(torch.int32)
    return t  # 退回 int64

@torch.no_grad()
@register_view("DeepMove_preview")
def DeepMove_preview(
    raw_df: pd.DataFrame, view_value: dict, device: str = "cuda"
) -> tuple[pd.DataFrame, dict]:
    """
    显存友好的 GPU 版 DeepMove 预处理（≤~1GB）
    - user_id / POI_id：按首次出现次序进行稳定离散化（与 pandas.unique + index 语义一致）
    - timestamps：按 model_args['time_interval'] 进行区间下取整（等价于 _lower_bound 到 time_range）
    """
    assert "user_id" in raw_df.columns and "POI_id" in raw_df.columns and "timestamps" in raw_df.columns, \
        "raw_df 必须包含 'user_id', 'POI_id', 'timestamps' 列"

    if view_value is None:
        view_value = {}

    use_cuda = torch.cuda.is_available() and (device.startswith("cuda") or device == "auto")
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # ========= 1) user_id 稳定映射 =========
    user_cpu, user_was_numeric = _to_int_tensor(raw_df["user_id"])
    codes_user_gpu, n_user = _factorize_on_gpu(user_cpu, device) if use_cuda else (None, None)
    if not use_cuda:
        # CPU 退化路径：直接用 pandas.factorize 的结果（已稳定）
        codes_user_gpu = user_cpu.to(device)  # factorize 已经给了 codes
        n_user = int(codes_user_gpu.max().item() + 1) if codes_user_gpu.numel() > 0 else 0

    codes_user_gpu = _to_int32_if_possible(codes_user_gpu)
    raw_df["user_id"] = codes_user_gpu.cpu().numpy()
    del user_cpu, codes_user_gpu
    if use_cuda:
        torch.cuda.empty_cache()

    # ========= 2) POI_id 稳定映射 =========
    poi_cpu, poi_was_numeric = _to_int_tensor(raw_df["POI_id"])
    codes_poi_gpu, n_item = _factorize_on_gpu(poi_cpu, device) if use_cuda else (None, None)
    if not use_cuda:
        codes_poi_gpu = poi_cpu.to(device)
        n_item = int(codes_poi_gpu.max().item() + 1) if codes_poi_gpu.numel() > 0 else 0

    codes_poi_gpu = _to_int32_if_possible(codes_poi_gpu)
    raw_df["POI_id"] = codes_poi_gpu.cpu().numpy()
    del poi_cpu, codes_poi_gpu
    if use_cuda:
        torch.cuda.empty_cache()

    # ========= 3) timestamps 区间下取整离散化 =========
    # 注意：不构造 time_range，节省显存；离散值等价于 _lower_bound 到 time_range 的索引
    ts_np = raw_df["timestamps"].to_numpy(copy=False)
    if not np.issubdtype(ts_np.dtype, np.integer):
        ts_np = ts_np.astype(np.int64, copy=False)
    ts_gpu = torch.from_numpy(ts_np).to(device)

    time_interval = int(model_args["time_interval"])
    time_start = int(ts_gpu.min().item()) if ts_gpu.numel() else 0
    time_end = int(ts_gpu.max().item()) if ts_gpu.numel() else -1

    # (x - start)//interval 即为所在区间索引（lower_bound 的结果）
    if ts_gpu.numel():
        bucket = torch.div(ts_gpu - time_start, time_interval, rounding_mode="floor")
        bucket = bucket.clamp_min(0).to(torch.int32)  # 索引用 int32
        raw_df["timestamps"] = bucket.cpu().numpy()
        del bucket
    else:
        raw_df["timestamps"] = ts_gpu.cpu().numpy()

    if use_cuda:
        torch.cuda.empty_cache()

    # ========= 4) 统计信息（与原逻辑对齐） =========
    # 原代码 tim_size = len(time_range) + 1, 其中 len(time_range) = floor((end - start)/interval) + 1
    if time_end >= time_start:
        n_bins = (time_end - time_start) // time_interval + 1
    else:
        n_bins = 0
    tim_size = n_bins + 1

    # 日志与返回值（按原逻辑）
    logger.info(f"time end is {time_end}")
    # 重新计算离散后的范围（与原代码一致，虽然仅用于日志/校验）
    new_time_start = int(raw_df["timestamps"].min()) if len(raw_df) else 0
    new_time_end = int(raw_df["timestamps"].max()) if len(raw_df) else 0

    view_value.update(
        {
            "tim_size": int(tim_size),
            "uid_size": int(n_user),
            "loc_size": int(n_item),
            "loc_pad": 0,
            "tim_pad": 0,
            # 可选暴露，便于下游做反归一化或校验
            "time_start_raw": time_start,
            "time_end_raw": time_end,
            "time_interval": time_interval,
            "time_start_bucket": new_time_start,
            "time_end_bucket": new_time_end,
        }
    )

    return raw_df, view_value



import numpy as np
from typing import List, Tuple, Dict, Any
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 假设外部给出:
# model_args = {"neg_num": int, "min_seq_len": int}
# view_value["loc_size"] 已在前序阶段设置

def _auto_batch_size(n_rows: int, k: int, bytes_budget: int = 200 * 1024**2) -> int:
    """
    估算一个安全的 batch_size，粗略按 ~6 份工作内存预留（候选/排序/掩码等），dtype=int32。
    默认以内存预算 ~200MB 计算，可自行调大/调小。
    """
    bytes_per_el = 4  # int32
    work_factor = 6
    if k <= 0:
        return min(n_rows, 1_000_000)
    b = max(1024, int(bytes_budget // (bytes_per_el * k * work_factor)))
    return int(min(n_rows, max(1024, b)))

def _skip_map(samples: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    将采样自 [0, loc_size-2] 的整数映射到 [0, loc_size-1] 且跳过 pos:
    y = x + 1_{x >= pos}
    samples: [B, K]  (值域: [0, loc_size-2])
    pos:     [B]     (正类 id)
    """
    return samples + (samples >= pos.unsqueeze(1)).to(samples.dtype)

def _rowwise_unique_fix(negs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    检测每行是否存在重复值，返回(need_fix_mask_per_el, has_dup_per_row)
    negs: [B, K] (已在最终值域内)
    """
    B, K = negs.shape
    sorted_vals, sort_idx = torch.sort(negs, dim=1)
    dup_pairs = (sorted_vals[:, 1:] == sorted_vals[:, :-1])
    # 仅将“右侧重复”的位置标为 True（保留每个值的首次出现）
    dup_right = torch.zeros((B, K), dtype=torch.bool, device=negs.device)
    dup_right[:, 1:] = dup_pairs
    # 将重复标记映射回原顺序
    need_fix = torch.zeros_like(dup_right)
    need_fix.scatter_(1, sort_idx, dup_right)
    has_dup_row = dup_pairs.any(dim=1)
    return need_fix, has_dup_row

@torch.no_grad()
def _sample_negatives_gpu(pos_all: torch.Tensor, k: int, loc_size: int,
                          device: torch.device, batch_size: int) -> torch.Tensor:
    """
    逐批为 pos_all（长度 N）生成每行 k 个“无放回且不含正类”的负样本（int32）。
    返回张量形状 [N, K]。
    """
    assert loc_size >= 2, "loc_size 必须≥2，才能进行负采样。"
    N = pos_all.numel()
    if batch_size is None:
        batch_size = _auto_batch_size(N, k)

    out_list = []
    rng = torch.Generator(device=device)

    for s in tqdm(range(0, N, batch_size), desc="Negative Sampling (GPU)", leave=False):
        e = min(N, s + batch_size)
        pos = pos_all[s:e].to(device, non_blocking=True)              # [B]
        B = pos.shape[0]

        # 首次候选：在 [0, loc_size-2] 上无约束采样，然后跳过正类映射到最终值域
        negs = torch.randint(low=0, high=loc_size - 1, size=(B, k), device=device, generator=rng, dtype=torch.int32)
        negs = _skip_map(negs, pos)                                   # [B, K] in [0, loc_size-1]\{pos}

        # 去重修复：仅对重复元素位置重采，迭代到行内唯一
        MAX_ITERS = 8
        for _ in range(MAX_ITERS):
            need_fix, has_dup_row = _rowwise_unique_fix(negs)
            num_fix = int(need_fix.sum().item())
            if num_fix == 0:
                break
            # 对需要重采的元素位点，重新在 [0, loc_size-2] 采样并映射
            # 每个位置需要其所在行的 pos 值，用 repeat_interleave 构造
            per_row_counts = need_fix.sum(dim=1).to(torch.int64)                      # [B]
            pos_rep = pos.repeat_interleave(per_row_counts) if num_fix > 0 else pos   # [num_fix]
            fresh = torch.randint(0, loc_size - 1, size=(num_fix,), device=device, generator=rng, dtype=torch.int32)
            fresh = fresh + (fresh >= pos_rep).to(fresh.dtype)
            negs[need_fix] = fresh

        # 少数极端行：若仍存在重复，使用每行 randperm 精确修复
        need_fix, has_dup_row = _rowwise_unique_fix(negs)
        bad_rows = torch.nonzero(has_dup_row, as_tuple=False).flatten()
        for r in bad_rows.tolist():
            perm = torch.randperm(loc_size - 1, device=device, generator=rng)[:k].to(torch.int32)
            perm = perm + (perm >= pos[r]).to(torch.int32)
            negs[r] = perm

        out_list.append(negs.cpu())

        # 显存友好清理
        del pos, negs
        torch.cuda.empty_cache()

    return torch.vstack(out_list) if len(out_list) > 1 else out_list[0]

def _sample_negatives_cpu(pos_np: np.ndarray, k: int, loc_size: int) -> np.ndarray:
    """
    CPU/numpy 退化路径：使用与 GPU 同步的“跳过正类 + 去重修复”策略，避免逐行 Python 循环。
    """
    N = pos_np.shape[0]
    rng = np.random.default_rng()
    # 初始采样（在 [0, loc_size-2]）
    negs = rng.integers(0, loc_size - 1, size=(N, k), dtype=np.int32)
    negs = negs + (negs >= pos_np[:, None]).astype(np.int32)  # 跳过正类

    # 迭代修复重复
    MAX_ITERS = 8
    for _ in range(MAX_ITERS):
        sorted_idx = np.argsort(negs, axis=1)
        sorted_vals = np.take_along_axis(negs, sorted_idx, axis=1)
        dup_pairs = (sorted_vals[:, 1:] == sorted_vals[:, :-1])
        dup_right = np.zeros_like(sorted_vals, dtype=bool)
        dup_right[:, 1:] = dup_pairs
        need_fix = np.zeros_like(dup_right, dtype=bool)
        # 反向映射回原序
        row_idx = np.arange(N)[:, None]
        need_fix[row_idx, sorted_idx] = dup_right
        if not need_fix.any():
            break
        num_fix = int(need_fix.sum())
        # 重采并映射
        fresh = rng.integers(0, loc_size - 1, size=(num_fix,), dtype=np.int32)
        # 为每个要修复元素取对应行的 pos
        per_row_counts = need_fix.sum(axis=1)
        pos_rep = np.repeat(pos_np, per_row_counts)
        fresh = fresh + (fresh >= pos_rep).astype(np.int32)
        negs[need_fix] = fresh

    # 极端行：仍重复则逐行 randperm 精确修复（通常很少）
    sorted_vals = np.sort(negs, axis=1)
    has_dup_row = (sorted_vals[:, 1:] == sorted_vals[:, :-1]).any(axis=1)
    for r in np.where(has_dup_row)[0]:
        perm = np.random.permutation(loc_size - 1)[:k].astype(np.int32)
        perm = perm + (perm >= pos_np[r]).astype(np.int32)
        negs[r] = perm

    return negs

@register_view("DeepMove_postview")
def DeepMove_post(raw_df: List[dict], view_value: dict) -> Tuple[List[dict], dict]:
    """
    优化后的 DeepMove Post-View：
    - 仅对满足 mask>=min_seq_len 的样本做负采样
    - GPU 并行 & 批处理；CPU 提供等价退化路径
    - 负样本“无放回”，且不包含正类
    """
    assert view_value is not None and "loc_size" in view_value, "view_value['loc_size'] 缺失"
    loc_size = int(view_value["loc_size"])
    assert "neg_num" in model_args and "min_seq_len" in model_args, "model_args 缺少 neg_num/min_seq_len"
    k = int(model_args["neg_num"])
    min_len = int(model_args["min_seq_len"])

    logger.info("Applying DeepMove_post to dataset")

    if len(raw_df) == 0 or k <= 0:
        # 没有数据或不需要负样本：仅过滤
        new_data = [ex for ex in raw_df if ex.get("mask", 0) >= min_len]
        return new_data, view_value

    # ---------- 1) 预先过滤，减少采样规模 ----------
    masks = np.fromiter((ex.get("mask", 0) for ex in raw_df), dtype=np.int32, count=len(raw_df))
    keep_idx = np.flatnonzero(masks >= min_len)
    if keep_idx.size == 0:
        return [], view_value

    # 正类 y
    pos_np = np.fromiter(
        (ex["y_POI_id"]["POI_id"] for ex in (raw_df[i] for i in keep_idx)),
        dtype=np.int64,
        count=keep_idx.size,
    )

    # ---------- 2) 负采样（优先 GPU） ----------
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        pos_t = torch.from_numpy(pos_np.astype(np.int64))
        device = torch.device("cuda")
        negs = _sample_negatives_gpu(pos_t, k=k, loc_size=loc_size, device=device, batch_size=model_args["batch_size"])  # [M, K]
        negs_np = negs.cpu().numpy()
        del pos_t, negs
        torch.cuda.empty_cache()
    else:
        negs_np = _sample_negatives_cpu(pos_np.astype(np.int64, copy=False), k=k, loc_size=loc_size)

    # ---------- 3) 回填结果并构造 new_data ----------
    new_data: List[dict] = []
    for j, i in enumerate(keep_idx):
        ex = raw_df[i]
        ex["neg_loc"] = negs_np[j].astype(np.int32, copy=False)
        new_data.append(ex)

    return new_data, view_value

