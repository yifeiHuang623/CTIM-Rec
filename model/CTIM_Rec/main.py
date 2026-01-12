import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import shutil
import os

from types import SimpleNamespace

from utils.logger import get_logger
from tqdm import tqdm
from utils.register import EVAL_REGISTRY

from .CTIM_Rec import CTIM_Rec
from utils.EarlyStopping import EarlyStopping
torch.set_num_threads(1)

log = get_logger(__name__)

pre_views = ['CTIM_Rec_preview']
post_views = ['CTIM_Rec_post_view']

model = None

def train_model(model: CTIM_Rec, train_dataloader: DataLoader, valid_dataloader: DataLoader, metric_keys, args):
    model = model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=args.save_model_folder, save_model_name=f"{args.save_model_name}", logger=log)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( opt, T_max=args.num_epochs, eta_min=1e-6)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        train_data_loader_tqdm = tqdm(train_dataloader, ncols=120)
        for batch_idx, batch_data in enumerate(train_data_loader_tqdm):
            batch_data_device = {k: v.to(args.device, non_blocking=True) for k, v in batch_data.items() if 'y_POI_id' not in k}
            batch_data_device['y_POI_id'] = {k: v.to(args.device, non_blocking=True) for k, v in batch_data['y_POI_id'].items()}
            loss = model.calculate_both_loss(batch_data_device)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        
            total_loss += loss.item()

            train_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
        
        log.info(f'Epoch: {epoch + 1}, train loss: {total_loss / len(train_dataloader)}')
        lr_scheduler.step()
        y_predict, y_truth = inference(valid_dataloader)
        scores = {}
        for m in metric_keys:
            if type(y_predict) == dict:
                poi = y_predict["poi"]
                poi_y_truth = y_truth["poi"]
            score = EVAL_REGISTRY[m](poi, poi_y_truth)
            scores[m] = score
            log.info("%-12s : %.6f", m, score)
            
        if type(y_predict) == dict:
            for m in ["MSE", "MAE"]:
                time = y_predict["time"]
                time_y_truth = y_truth["time"]
                if m == "MSE":
                    score = np.mean((time_y_truth - time) ** 2)
                else:
                    score = np.mean(np.abs(time_y_truth - time))
                scores[m] = score
                log.info("%-12s : %.6f", m, score)
            
        val_metric_indicator = []
        for metric_name, value in scores.items():
            if metric_name not in ["MSE", "MAE"]:
                val_metric_indicator.append((metric_name, value, True))
            
        early_stop = early_stopping.step(val_metric_indicator, model)

        if early_stop:
            break
    
    early_stopping.load_checkpoint(model)

def train(dataloader: dict[DataLoader], dataset_name, **kw):
    global model
    with open('./model/CTIM_Rec/CTIM_Rec.yaml', 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
    args.update(dataloader.view_value)
    args = SimpleNamespace(**args)
    log.info('=============args==============')
    log.info(args)
    
    train_dataloader, val_dataloader = dataloader.train_dataloader, dataloader.val_dataloader
    
    model = CTIM_Rec(args)
    
    metric_keys = (list(EVAL_REGISTRY))
    
    args.save_model_name = f'{args.model_name}'
    save_model_folder = f"./saved_models/{dataset_name}/{args.save_model_name}/"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    args.save_model_folder = save_model_folder
    
    train_model(model, train_dataloader, val_dataloader, metric_keys, args)

def inference(dataloader: DataLoader, **kv):
    global model
    log.info('=============begin valid/test==============')
    
    if kv.get('model_para', False):   
        with open('./model/CTIM_Rec/CTIM_Rec.yaml', 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        args.update(kv.get('view_value', {}))
        args = SimpleNamespace(**args)
        log.info('=============args==============')
        log.info(args)
        model = CTIM_Rec(args).to(args.device)
        model.load_state_dict(kv['model_para'])
    
    model.eval()
    y_predict_list = []
    
    with torch.no_grad():
       y_predict_list, y_truth_list = [], []
       y_predict_time_list, y_truth_time_list = [], []
       
       for batch_data in tqdm(dataloader):
            batch_data_device = {k: v.to(model.device, non_blocking=True) for k, v in batch_data.items() if 'y_POI_id' not in k}
            batch_data_device['y_POI_id'] = {k: v.to(model.device, non_blocking=True) for k, v in batch_data['y_POI_id'].items()}

            y_predict, y_predict_time = model.predict(batch_data_device)
            batch_indices = torch.arange(batch_data_device['mask'].shape[0])
            y_predict = y_predict[batch_indices, batch_data_device['mask'] - 1] 
            y_predict_time = y_predict_time[batch_indices, batch_data['mask'] - 1]
            
            y_predict_list.extend(y_predict.detach().cpu().numpy().reshape(-1, y_predict.shape[-1]).tolist())
            y_truth_list.extend(batch_data['y_POI_id']['POI_id'].detach().cpu().numpy().tolist())
            
            y_predict_time_list.extend(y_predict_time.detach().cpu().numpy().reshape(-1).tolist())
            y_truth_time_list.extend(batch_data['y_POI_id']['time_delta'].detach().cpu().numpy().tolist())
            
    y_predict = np.array(y_predict_list)
    y_truth = np.array(y_truth_list)
    y_predict_time = np.array(y_predict_time_list)
    y_truth_time = np.array(y_truth_time_list)
    
    return {'poi': y_predict, 'time': y_predict_time}, {'poi': y_truth, 'time': y_truth_time}