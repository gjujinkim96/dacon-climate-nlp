import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import itertools

col_key_kor_to_eng = {
    '사업명': 'b_name',
    '사업_부처명': 'bb_name',
    '내역사업명': 'h_name',
    '과제명': 't_name',
    '요약문_한글키워드': 'k_keyword', 
    '요약문_영문키워드': 'e_keyword',
    '요약문_연구목표': 'goal',
    '요약문_연구내용': 'content',
    '요약문_기대효과': 'effect',
    'goal_keywords': 'goal_keywords',
    'content_keywords': 'content_keywords',
    'effect_keywords': 'effect_keywords',
}

def pad_ids(ids, pad_id):
    lens = [len(x) for x in ids]
    max_len = max(lens)

    return [x + [pad_id]*(max_len - l) for x, l in zip(ids, lens)]

class DataFrameDataset(Dataset):
    def __init__(self, setting, df, tokenizer, is_test=False, is_long=False, extra_label_df=None):
        super().__init__()

        self.df = df
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.use_extra_label = setting.use_extra_label
        self.extra_label_df = extra_label_df

        if setting.clean_dirty:
            a = ['*', '&quot;', '?', '-', '●', '&lt;', '&gt;', '○', 'ㅇ', ',',
                        '.', '<br>', '？', '■', '·', '①', '②', '③', '④', '⑤', '⑥', '▷', '[', ']',
                        '(1)', '(2)', '(3)', '(4)', '1)', '2)', '3)', '4)', '□', '§', '◆', '#',
                        '※', '▶', '◦', '⦁', '★', '◎', '▣', ' o ', '(', ')'
                        ]
            b = ['.', '?', '？', '●', '○', 'ㅇ', '■', '▷', '※', '▶', '◦', '⦁', '★', '◎',
                                '▣', '□', '§', '◆', '#', '◇', '-', '<br>', '<BR>',
                                '·', '∘',  ' o ', ',', '<', '>', '[', ']', '(', ')', '◾', '❍', '❏',
                                '•', '．', ':']

            b.extend(f'{n})' for n in range(10))
            b.extend(f'({n})' for n in range(10))
            b.extend(f'{"i"*n})' for n in range(1, 10))
            b.extend('⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨'.split())
            b.extend(chr(ord('❶') + n) for n in range(9))

            delete = set(a + b)
            delete = sorted(delete, key=lambda x: -len(x))
            for k in delete:
                self.df[setting.single_col_data] = \
                    self.df[setting.single_col_data].str.replace(k, ' ', regex=False)

        if is_long:
            tok = tokenizer(self.df[setting.single_col_data].tolist(),
                return_attention_mask=False, return_token_type_ids=False)
        else:
            tok = tokenizer(self.df[setting.single_col_data].tolist(), truncation=True,
             max_length=setting.tokenize_max_seq, return_attention_mask=False, return_token_type_ids=False)
        sz = len(self.df)
        key_val = [{k:v[i] for k, v in tok.items()} for i in range(sz)]

        if self.is_test:
            self.cache = list(zip(key_val, df['index']))
        else:
            self.cache = list(zip(key_val, df['label']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return self.cache[i]

    def collate_fn(self, batch):
        input_ids = torch.tensor(pad_ids([b[0]['input_ids'] for b in batch], self.tokenizer.pad_token_id))
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        token_type_ids = torch.zeros_like(input_ids) # roberta does not use token type ids

        if self.is_test:
            label = None
        else:
            if self.use_extra_label:
                a = np.array([b[1] for b in batch])
                aa = pd.DataFrame.from_dict({'label':a}).merge(self.extra_label_df, how='left')
                label = aa[['big_label', 'middle_label', 'label']].to_numpy()

                label = torch.tensor(label)
            else:
                label = torch.tensor([b[1] for b in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }, label

class DataFrameTwoDataset(Dataset):
    def __init__(self, setting, df, tokenizer, is_test=False, extra_label_df=None):
        super().__init__()

        self.df = df
        self.extra_label_df = extra_label_df
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.use_extra_label = setting.use_extra_label

        self.col_keys_to_use = setting.mixed_col_keys_to_use

        specials = set(['요약문_연구목표', '요약문_연구내용', '요약문_기대효과'])

        if setting.clean_dirty:
            a = ['*', '&quot;', '?', '-', '●', '&lt;', '&gt;', '○', 'ㅇ', ',',
                        '.', '<br>', '？', '■', '·', '①', '②', '③', '④', '⑤', '⑥', '▷', '[', ']',
                        '(1)', '(2)', '(3)', '(4)', '1)', '2)', '3)', '4)', '□', '§', '◆', '#',
                        '※', '▶', '◦', '⦁', '★', '◎', '▣', ' o ', '(', ')'
                        ]
            b = ['.', '?', '？', '●', '○', 'ㅇ', '■', '▷', '※', '▶', '◦', '⦁', '★', '◎',
                                '▣', '□', '§', '◆', '#', '◇', '-', '<br>', '<BR>',
                                '·', '∘',  ' o ', ',', '<', '>', '[', ']', '(', ')', '◾', '❍', '❏',
                                '•', '．', ':']

            b.extend(f'{n})' for n in range(10))
            b.extend(f'({n})' for n in range(10))
            b.extend(f'{"i"*n})' for n in range(1, 10))
            b.extend('⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨'.split())
            b.extend(chr(ord('❶') + n) for n in range(9))

            delete = set(a + b)
            delete = sorted(delete, key=lambda x: -len(x))

        self.cache = [{} for _ in range(len(self.df))]
        for key in self.col_keys_to_use:
            eng_key = col_key_kor_to_eng[key]
            if key in specials:
                if setting.clean_dirty:
                    for k in delete:
                        self.df[key] = self.df[key].str.replace(k, ' ', regex=False)

                tok = tokenizer(self.df[key].tolist(), truncation=True, max_length=setting.special_tms,
                    return_attention_mask=False, return_token_type_ids=False)
            else:
                tok = tokenizer(self.df[key].tolist(), truncation=True, max_length=setting.tokenize_max_seq,
                    return_attention_mask=False, return_token_type_ids=False)
            
            for tok_key, tok_value in tok.items():
                for i in range(len(self.df)):
                    aa = tok_value[i]
                    self.cache[i][eng_key] = aa
        
        for i in range(len(self.df)):
            if is_test:
                label = df['index'].iloc[i]
            else:
                label = df['label'].iloc[i]

            self.cache[i]['y'] = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return self.cache[i]

    def collate_fn(self, batch):
        collated = []
        for key in self.col_keys_to_use:
            eng_key = col_key_kor_to_eng[key]
            input_ids = torch.tensor(pad_ids([b[eng_key] for b in batch], self.tokenizer.pad_token_id))

            collated.append(input_ids)

        if self.is_test:
            label = None
        else:
            if self.use_extra_label:
                a = np.array([b['y'] for b in batch])
                aa = pd.DataFrame.from_dict({'label':a}).merge(self.extra_label_df, how='left')
                label = aa[['big_label', 'middle_label', 'label']].to_numpy()

                label = torch.tensor(label)
            else:
                label = torch.tensor([b['y'] for b in batch])

        return {
            'inputs': collated
        }, label

class DataFrameBigDataset(Dataset):
    def __init__(self, setting, df, tokenizer, is_test=False, extra_label_df=None):
        super().__init__()

        self.df = df
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.use_extra_label = setting.use_extra_label
        self.extra_label_df = extra_label_df

        self.col_keys_to_use = setting.mixed_col_keys_to_use

        tok_tmp = {}
        for key in self.col_keys_to_use:
            eng_key = col_key_kor_to_eng[key]
            tok = tokenizer(self.df[key].tolist(), truncation=True, max_length=setting.tokenize_max_seq,
                return_attention_mask=False, return_token_type_ids=False)
            tok_tmp[eng_key] = tok

        self.cache = []
        for i in range(len(self.df)):
            a = [tok_tmp[k][i].ids for k in tok_tmp]

            if not setting.big_mixed_use_each:
                a = [a[0]] + [v[1:] for v in a[1:]]

            self.cache.append(
                {
                    'input': list(itertools.chain.from_iterable(a)),
                }
            )
        
        for i in range(len(self.df)):
            if is_test:
                label = df['index'].iloc[i]
            else:
                label = df['label'].iloc[i]

            self.cache[i]['y'] = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return self.cache[i]

    def collate_fn(self, batch):
        input_ids = torch.tensor(pad_ids([b['input'] for b in batch], self.tokenizer.pad_token_id))

        if self.is_test:
            label = None
        else:
            if self.use_extra_label:
                a = np.array([b['y'] for b in batch])
                aa = pd.DataFrame.from_dict({'label':a}).merge(self.extra_label_df, how='left')
                label = aa[['big_label', 'middle_label', 'label']].to_numpy()

                label = torch.tensor(label)
            else:
                label = torch.tensor([b['y'] for b in batch])

        return {
            'input_ids': input_ids,
        }, label