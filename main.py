# traj_lib/main.py
import argparse, importlib, json, sys
from utils.logger   import get_logger, set_model_name, set_dataset_name, set_log_file_name
import warnings
import numpy as np
warnings.filterwarnings("ignore")
# ---------- CLI ----------
def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="model dir name under traj_lib/model/")
    p.add_argument("--dataset", default="all", help="'all' or comma list of datasets")
    p.add_argument("--metrics", default="all", help="'all' or comma list of metrics")
    p.add_argument("--cfg",     default=None,  help="extra YAML config path")
    p.add_argument("--task",    default='NPP', help="task name, e.g., NPP, Recoverym2m, etc.")
    return p.parse_args()
args = parse_cli()

set_model_name(args.model)  # 设置日志文件名
set_dataset_name(args.dataset)  # 设置数据集名称
set_log_file_name()  # 设置日志文件名
log = get_logger('main')

from pathlib import Path
from utils.register import DATALOADER_REGISTRY, EVAL_REGISTRY, VIEW_REGISTRY
from utils.exargs   import ConfigResolver
import utils.dataloader   # 触发注册
import utils.eval         # 触发注册
import utils.views        # 触发注册

utils.dataloader.register_all(task=args.task)  # 注册所有数据加载器


# ---------- Config ----------
def load_cfg(dataset, cfg_path):
    if cfg_path:
        return ConfigResolver(cfg_path).parse()
    default = Path(__file__).resolve().parent / "data" / dataset / f"{dataset}.yaml"
    if default.exists():
        return ConfigResolver(str(default)).parse()
    return {}  # 没有 YAML 就返回空 dict

# ---------- 主流程 ----------
def main():

    # 数据集列表
    datasets = (
        list(DATALOADER_REGISTRY)
        if args.dataset.lower() == "all"
        else [d.strip() for d in args.dataset.split(",")]
    )

    # 评测指标列表
    metric_keys = (
        list(EVAL_REGISTRY)
        if args.metrics.lower() == "all"
        else [m.strip() for m in args.metrics.split(",")]
    )

    # 动态 import 模型
    mod_path = f"model.{args.model}.main"
    try:
        model_mod = importlib.import_module(mod_path)
    except ModuleNotFoundError:
        log.error("Cannot import model module %s", mod_path)
        sys.exit(1)

    if not hasattr(model_mod, "inference"):
        log.error("%s must expose inference()", mod_path)
        sys.exit(1)
    if not hasattr(model_mod, "train"):
        log.warning("%s does not expose train(), only inference() will be used", mod_path)

    model_args = ConfigResolver(f"model/{args.model}/{args.model}.yaml").parse()

    # 逐数据集评测
    for ds in datasets:
        if hasattr(model_mod, "pre_views") and model_mod.pre_views:
            log.info("Using pre-views: %s", model_mod.pre_views)
        else:
            log.info("No pre-views defined for model %s", args.model)
        if hasattr(model_mod, "post_views") and model_mod.post_views:
            log.info("Using post-views: %s", model_mod.post_views)
        else:
            log.info("No post-views defined for model %s", args.model)
        dataloader = DATALOADER_REGISTRY[ds](
            model_args=model_args,
            pre_views=model_mod.pre_views if hasattr(model_mod, "pre_views") else None,
            post_views=model_mod.post_views if hasattr(model_mod, "post_views") else None
        )
        cfg = load_cfg(ds, args.cfg)

        if hasattr(model_mod, "train"):
            model_mod.train(dataloader, ds, **cfg)
        preds, gts = model_mod.inference(dataloader.test_dataloader, **cfg)

        scores = {}
        for m in metric_keys:
            if type(preds) == dict:
                poi = preds["poi"]
                poi_gts = gts["poi"]
            else:
                poi = preds
                poi_gts = gts
            if m not in EVAL_REGISTRY:
                log.error("Metric '%s' not registered; skip", m)
                continue
            score = EVAL_REGISTRY[m](poi, poi_gts)
            scores[m] = score
            log.info("[%s] %-12s : %.6f", ds, m, score)
            
        if type(preds) == dict:
            log.info("Time prediction results:")
            for m in ["MSE", "MAE"]:
                time = preds["time"]
                time_gts = gts["time"]
                if m == "MSE":
                    # MSE
                    score = np.mean((time_gts - time) ** 2)
                else:
                    # MAE
                    score = np.mean(np.abs(time_gts - time))
                scores[m] = score
                log.info("[%s] %-12s : %.6f", ds, m, score)
            # time = preds["time"]
            # time_gts = gts["time"]
            # diff = np.abs(time - time_gts)
            # for k in [1, 3, 6]:
            #     accuracy = np.mean(diff <= k)
            #     log.info("Time Accuracy @%d : %.6f", k, accuracy)
        # for m in metric_keys:
        #     score = EVAL_REGISTRY[m](preds, gts)
        #     scores[m] = score
        #     log.info("[%s] %-12s : %.6f", ds, m, score)

        # 保存 JSON
        if scores:
            out = {"model": args.model, "dataset": ds, "scores": scores}
            print(out)
            out_dir = Path(__file__).resolve().parent / "outputs"
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"{args.model}_{ds}.json"
            out_file.write_text(json.dumps(out, indent=2))
            log.info("scores saved to %s", out_file)

if __name__ == "__main__":
    main()
