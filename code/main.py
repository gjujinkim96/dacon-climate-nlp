import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

import wandb

from setting import Setting
from main_loop import run_trial
from model import get_model, get_tokenizer
from data_provider import get_dataloader, get_test
from criterion import get_criterion
from optimizer import get_optimizer
from lr_scheduler import get_lr_scheduler
from logging_stuff import log_init

def main(setting):
    log_init(setting)

    device = torch.device(setting.device_pref if torch.cuda.is_available() else 'cpu')

    tokenizer = get_tokenizer(setting.model_name)
    model = get_model(setting).to(device)
    if setting.model_type == 'mixed' and setting.freeze_emb:
        model.setup_cache()

    train_dl, val_dl = get_dataloader(setting, tokenizer)

    if setting.use_weight_in_loss:
        y = train_dl.dataset.df.label.tolist() + list(range(46)) # 최소 1번은 나오게
        class_weight = compute_class_weight('balanced', classes=np.arange(46), y=y)
        class_weight = torch.tensor(class_weight).float()
    else:
        class_weight = None
    criterion = get_criterion(setting, device, class_weight=class_weight)
    optimizer = get_optimizer(setting, model)
    lr_scheduler = get_lr_scheduler(setting, optimizer, train_dl)
    
    print(device)
    model = run_trial(setting, model, train_dl, val_dl, criterion, optimizer, lr_scheduler)
   
    result = {
        'model': model,
        'tokenizer': tokenizer,
        'criterion': criterion,

        'train_dl': train_dl,
        'val_dl': val_dl,
    }

    return result

