import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report

from logging_stuff import Logger
from metric import get_metrics, get_zero_metrics, show_classification_report

from tqdm.auto import tqdm, trange

def get_multi_sample_loss(criterion, output, y):
    total_loss = []
    for i in range(output.size(1)):
        total_loss.append(criterion(output[:, i, :], y))

    return torch.stack(total_loss, dim=-1).mean(dim=-1)

def get_loss_by_setting(setting, criterion, output, y):
    if setting.multi_sample_dropout_n > 1:
        return get_multi_sample_loss(criterion, output, y)
    else:
        return criterion(output, y)

def get_loss(setting, criterion, output, y):
    if setting.model_type in ['gate', 'mixed_gate']:
        main_criterion, gate_critierion = criterion

        main_output = output['logits'][y != 0]
        main_y = y[y != 0]
        main_y -= 1
        if setting.gate_44:
            index_45 = main_y == 44
            main_y[index_45] = 0
            one_hot_y = F.one_hot(main_y, num_classes=44).to(main_output.dtype)
            one_hot_y[index_45] = 0
            main_loss = gate_critierion(main_output, one_hot_y)
        else:
            main_loss = main_criterion(main_output, main_y)

        gate_loss = gate_critierion(output['gate'], (y == 0).to(output['gate'].dtype))
        return main_loss + gate_loss
    else:
        if setting.use_extra_label:
            big_y = y[:, 0]
            middle_y = y[:, 1]
            y = y[:, 2]

            big_loss = get_loss_by_setting(setting, criterion, output['big_logits'], big_y)
            middle_loss = get_loss_by_setting(setting, criterion, output['middle_logits'], middle_y)
            ret_dict = {
                'middle_loss': middle_loss,
                'big_loss': big_loss,
            }

            if setting.el_single_output:
                small_loss = get_loss_by_setting(setting, criterion, output['small_logits'], y)
                last_loss = get_loss_by_setting(setting, criterion, output['logits'], y)
                loss = small_loss + middle_loss + big_loss + last_loss
                ret_dict.update({
                    'loss': loss,
                    'small_loss': small_loss,  
                    'last_loss': last_loss,
                })
            else:
                small_loss = get_loss_by_setting(setting, criterion, output['logits'], y)
                loss = small_loss + middle_loss + big_loss
                ret_dict.update({
                    'loss': loss,
                    'small_loss': small_loss,
                })
        else:
            loss = get_loss_by_setting(setting, criterion, output['logits'], y)
            ret_dict = {
                'loss': loss,
            }

        return ret_dict

def get_pred_by_setting(setting, output):
    if setting.multi_sample_dropout_n > 1:
        return output.detach().mean(dim=1).cpu().argmax(dim=-1)
    else:
        return output.detach().cpu().argmax(dim=-1)

def get_pred(setting, output):
    if setting.model_type in ['gate', 'mixed_gate']:
        is_gate = torch.sigmoid(output['gate'].detach().cpu().float()) >= setting.gate_th
        if setting.gate_44:
            val, idx = torch.max(torch.sigmoid(output['logits'].detach().cpu().float()), dim=-1)
            idx = idx + 1
            idx[val < setting.gate_44_th] = 45
            logits = idx
        else:
            logits = output['logits'].detach().cpu().argmax(dim=-1).float() + 1

        logits[is_gate] = 0
        return logits
    else:
        pred = get_pred_by_setting(setting, output['logits'])
        ret_dict = {
            'pred': pred,
        }

        if setting.use_extra_label:
            big_pred = get_pred_by_setting(setting, output['big_logits'])
            middle_pred = get_pred_by_setting(setting, output['middle_logits'])

            ret_dict.update({
                'middle_pred': middle_pred,
                'big_pred': big_pred,
            })
            
            if setting.el_single_output:
                small_pred = get_pred_by_setting(setting, output['small_logits'])
                ret_dict.update({
                    'small_pred': small_pred,
                })

    return ret_dict

def detach(setting, output, y):
    if setting.model_type == 'gate':
        output['gate'] = output['gate'].detach().cpu().float()
    output['logits'] = output['logits'].detach().cpu().float()

    y = y.detach().cpu()

    return output, y


def recursive_device(x, device):
    if isinstance(x, dict):
        return {k: recursive_device(ele, device) for k, ele in x.items()}
    elif isinstance(x, list):
        return [recursive_device(ele, device) for ele in x]
    elif torch.is_tensor(x):
        return x.to(device)


def run_epoch(
    epoch,
    setting,
    model,
    dl, 
    criterion,
    optimizer=None,
    lr_scheduler=None,
    scaler=None,
    should_log=False,
):
    is_train = optimizer is not None
    device = next(model.parameters()).device
    model.train(mode=is_train)
    torch.set_grad_enabled(is_train)

    if not is_train:
        all_preds = []
        all_ys = []

    pbar = tqdm(enumerate(dl), total=len(dl))
    pbar.set_description(f"{'Train' if is_train else 'Valid'} Loss: NaN")

    logger = Logger(should_log=should_log)
    for i, (x, y) in pbar:
        x = recursive_device(x, device)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=setting.use_amp):
            # TODO: model(**x)로 
            # THEN: model + dataset 수정하기 + 아마 model, dataset 파일 구조 바꾸기
            output = model(**x)
            loss_dict = get_loss(setting, criterion, output, y)
            loss = loss_dict['loss']

        output, y = detach(setting, output, y)
        # log stuff
        pbar.set_description(f"{'Train' if is_train else 'Val'} Loss: {loss.item():.4f}")

        logger.reset(epoch + i/len(dl))
        logger.add_log_dict(loss_dict, prefix='train' if is_train else 'val')
        if is_train:
            pred_dict = get_pred(setting, output)
            metrics = get_metrics(setting, y, pred_dict)
            logger.add_log_dict(metrics, prefix='train' if is_train else 'val')

            logger.add_log_item('lr', lr_scheduler.get_last_lr()[0])
            optimizer.zero_grad()

            if scaler is None:
                loss.backward()

                if setting.max_norm is not None:
                    clip_grad_norm_(model.parameters(), setting.max_norm)
                optimizer.step()
                lr_scheduler.step()
            else:
                scaler.scale(loss).backward()

                if setting.max_norm is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), setting.max_norm)
                scaler.step(optimizer)

                old_scale = scaler.get_scale()
                scaler.update()
                new_scale = scaler.get_scale()

                if old_scale == new_scale:
                    lr_scheduler.step()
        
            logger.log()
        else:
            pred_dict = get_pred(setting, output)
            all_preds.append(pred_dict)
            all_ys.append(y)

        logger.log()

    if is_train:
        return
    else:
        if epoch == 0: # wandb 다른 그래프랑 정렬되게
            logger.reset(epoch)
            metrics = get_zero_metrics(setting)
            logger.add_log_dict(metrics, prefix= 'val')
            logger.log()

        logger.reset(epoch+1)
    
        pred = {k: torch.cat([pd[k] for pd in all_preds]) for k in all_preds[0]}
        y = torch.cat(all_ys)

        metrics = get_metrics(setting, y, pred)
        logger.add_log_dict(metrics, prefix= 'val')
        logger.log()

        show_classification_report(setting, y, pred)

        return metrics, pred, y

def run_trial(
    setting,
    model, 
    train_dl, 
    test_dl, 
    criterion, 
    optimizer,
    lr_scheduler,
):
    if setting.use_swa:
        swa_model = AveragedModel(model)

        if setting.swa_annel_steps is None:
            total_steps = (setting.epochs - setting.swa_start_epoch) * len(train_dl)
            anneal_steps = int(setting.swa_annel_steps_ratio * total_steps)
        else:
            anneal_steps = setting.swa_annel_steps

        swa_scheduler = SWALR(optimizer, swa_lr=setting.swa_lr, 
                anneal_strategy="cos", anneal_epochs=anneal_steps)

    if setting.use_amp:
        scaler= GradScaler()
    else:
        scaler = None

    for epoch in trange(setting.epochs):
        if setting.use_swa and setting.swa_start_epoch <= epoch + 1:
            lr_scheduler = swa_scheduler

        run_epoch(epoch, setting, model, train_dl, criterion, optimizer, lr_scheduler,
            scaler=scaler, should_log=True)

        if setting.use_swa:
            swa_model.update_parameters(model)

        if test_dl is not None:
            run_epoch(epoch, setting, model, test_dl, criterion, should_log=True)
    
    if setting.use_swa:
        update_bn(train_dl, swa_model)
        model = swa_model

        if test_dl is not None:
            run_epoch(epoch+1, setting, model, test_dl, criterion, should_log=True)

    return model