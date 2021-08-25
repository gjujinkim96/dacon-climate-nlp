import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import DataFrameDataset, DataFrameTwoDataset, DataFrameBigDataset

# code from https://pytorch.org/docs/stable/notes/randomness.html Dataloader section
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_labels_mapping(setting):
    labels_mapping = pd.read_csv(setting.labels_mapping)
    labels_mapping['소분류'] = labels_mapping['소분류'].str[4:]
    return labels_mapping

def get_dataset(setting, tokenizer, train_df, test_df=None, is_test=False):
    dataset_dict = {
        'default': DataFrameDataset,
        'long': DataFrameDataset,
        'relation': DataFrameDataset,
        'gate': DataFrameDataset,
        'mixed': DataFrameTwoDataset, 
        'mixed_gate': DataFrameTwoDataset,
        'big_mixed': DataFrameBigDataset,
    }

    if setting.use_extra_label:
        extra_label_df = pd.read_csv(setting.extra_label_file)
    else:
        extra_label_df = None

    train_ds = dataset_dict[setting.model_type](setting, train_df, tokenizer, is_test=is_test, 
    extra_label_df=extra_label_df)
    
    if test_df is None:
        return train_ds, None
    else:
        test_ds = dataset_dict[setting.model_type](setting, test_df, tokenizer, is_test=is_test,
        extra_label_df=extra_label_df)
        return train_ds, test_ds

def get_dataloader(setting, tokenizer, checkpoint=None):
    if checkpoint is None:
        if setting.line_stuff:
            df = pd.read_csv(setting.line_file, lineterminator='\n')
        else:
            df = pd.read_csv(setting.data_file)

        df = df.fillna('없음')

        if setting.no_empty_label:
            df = df[df.label != 0]

        if setting.test_ratio == 0:
            train_df = df
            test_df = None
        else:
            train_df, test_df = train_test_split(df, 
                test_size=setting.test_ratio, stratify=df['label'], random_state=setting.seed)
    else:
        train_df, test_df = given_df
    
    if setting.small_data:
        train_df = train_df.iloc[:100]
        if test_df is not None:
            test_df = test_df.iloc[:100]

    train_ds, test_ds = get_dataset(setting, tokenizer, train_df, test_df)

    g = torch.Generator()
    g.manual_seed(setting.seed)

    train_dl = DataLoader(train_ds, batch_size=setting.batch_size, collate_fn=train_ds.collate_fn,
            shuffle=True, drop_last=True, num_workers=setting.num_workers, 
            worker_init_fn=seed_worker, generator=g, pin_memory=True)

    if test_ds is not None:
        test_dl = DataLoader(test_ds, batch_size=setting.batch_size, collate_fn=test_ds.collate_fn,
                shuffle=False, drop_last=True, num_workers=setting.num_workers, pin_memory=True)
    else:
        test_dl = None

    return train_dl, test_dl

def get_test(setting, tokenizer):
    df = pd.read_csv(setting.test_file)
    df = df.fillna('없음')

    ds, _ = get_dataset(setting, tokenizer, df, is_test=True)

    dl = DataLoader(ds, batch_size=setting.batch_size, collate_fn=ds.collate_fn,
            shuffle=False, drop_last=False, num_workers=setting.num_workers,
            pin_memory=True)

    return df, dl