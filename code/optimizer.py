from torch.optim import SGD, AdamW
from madgrad import MADGRAD

def get_top_down_parameters(setting, backbone):
    parameters = [
            {
                'params': x.parameters(), 
                'lr': setting.lr * setting.llrd_decay**(i+1),
            } for i, x in enumerate(backbone.encoder.layer)
        ]

    parameters.append({
        'params': backbone.embeddings.parameters(),
        'lr': setting.lr * setting.llrd_decay**(len(backbone.encoder.layer) + 1),
    })

    return parameters

def get_optimizer(setting, model):
    if setting.use_llrd:
        parameters = get_top_down_parameters(setting, model.backbone)

        if setting.mixed_two_models:
            parameters += get_top_down_parameters(setting, model.backbone2)
    else:
        parameters = model.parameters()
    
    if setting.optimizer == 'SGD':
        return SGD(parameters, lr=setting.lr, momentum=0.9)
    elif setting.optimizer == 'AdamW':
        return AdamW(parameters, setting.lr)
    elif setting.optimizer == 'MADGRAD':
        return MADGRAD(parameters, setting.lr,
             weight_decay=setting.madgrad_weight_decay)
    else:
        raise NotImplementedError(f'No optimizer support for {setting.optimizer}')