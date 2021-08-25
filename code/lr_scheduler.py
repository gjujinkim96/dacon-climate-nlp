from transformers import get_cosine_schedule_with_warmup
from torch.optim.swa_utils import SWALR

def get_lr_scheduler(setting, optimizer, train_dl):
    total_steps = setting.epochs * len(train_dl)

    if setting.warmup_step is None:
        warmup_steps = int(total_steps * setting.warmup_ratio)
    else:
        warmup_steps = setting.warmup_step

    if setting.lr_scheduler == 'get_cosine_schedule_with_warmup':
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps,
            total_steps)
    else:
        raise NotImplementedError(f'No lr_scheduler support for {setting.lr_scheduler}')