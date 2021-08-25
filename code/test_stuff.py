from data_provider import get_test
from main_loop import get_pred, recursive_device

import pandas as pd
import torch
from torch.cuda.amp import autocast

from tqdm.auto import tqdm

def run_test(
    setting,
    model,
    dl
):
    device = next(model.parameters()).device
    model.train(mode=False)
    torch.set_grad_enabled(False)

    predictions = []
    pbar = tqdm(enumerate(dl), total=len(dl))
    pbar.set_description(f"Test")
    for i, (x, _) in pbar:
        x = recursive_device(x, device)

        with autocast(enabled=setting.use_amp):
            output = model(**x)

        pred_dict = get_pred(setting, output)
        pred = pred_dict['pred']
        predictions.append(pred)
                
    return torch.cat(predictions).numpy()

def test_main(setting, result):
    model = result['model']
    tokenizer = result['tokenizer']

    test_df, test_dl = get_test(setting, tokenizer)

    preds = run_test(setting, model, test_dl)
    answer_df = pd.DataFrame.from_dict({'index': test_df['index'], 'label': preds})

    return answer_df
