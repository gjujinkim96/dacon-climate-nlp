import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification

from data_provider import get_labels_mapping


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def get_model(setting):
    model_dict = {
        'default': SingleModel,
        'mixed': MixedModel,
        'long': LongSingleModel,
        'relation': SingleModelByRelation,
        'gate': SingleModelGate,
        'mixed_gate': MixedModelGate,
        'big_mixed': BigMixedModel,
    }

    config = AutoConfig.from_pretrained(setting.model_name, num_labels=46)
    return model_dict[setting.model_type](setting, config)

class PoolerLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        return x

class PoolerLayer2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, cleanup=True):
        if cleanup:
            x = x['last_hidden_state']
            x = x[:, 0, :]

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        return x

class CertainLayerPooler(nn.Module):
    def __init__(self, hidden_size, which=-2):
        super().__init__()
        self.which = which
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, res):
        x = res['hidden_states'][self.which]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        return x

class PoolingLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, res, attn_mask, is_mean=True):
        attn_mask = attn_mask.unsqueeze(-1)
        x = res['last_hidden_state']

        if is_mean:
            x_sum = (x * attn_mask).sum(dim=1)
            attn_sum = attn_mask.sum(dim=1).to(x_sum.dtype)
            attn_sum = torch.clamp(attn_sum, min=1e-6)
            x = x_sum / attn_sum
        else:
            x = x.masked_fill(attn_mask == False, -6e4)
            x = torch.max(x, dim=1)[0]

        x = self.dropout(x)

        x = self.dense(x)
        x = torch.tanh(x)
        return x

class LastLayerPooler(nn.Module):
    def __init__(self, hidden_size, n_layers=4, is_sum=False):
        super().__init__()
        self.n_layers = n_layers
        self.is_sum = is_sum

        if self.is_sum:
            input_size = hidden_size
        else:
            input_size = hidden_size * self.n_layers

        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, res):
        hidden_states = res['hidden_states']
        last_layers = hidden_states[-self.n_layers:]

        if self.is_sum:
            x = torch.stack(last_layers, dim=-1).sum(dim=-1)
        else:
            x = torch.cat(last_layers, dim=-1)
            
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        return x
        

class RobertaClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_start_token=True):
        super().__init__()
        self.dense = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(in_dim, out_dim)
        self.use_start_token = use_start_token

    def forward(self, features, **kwargs):
        if self.use_start_token:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            x = features

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class DropWithLinear(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, multi_samples=1):
        super().__init__()

        self.use_multi_sample_dropout = multi_samples > 1
        if self.use_multi_sample_dropout:
            self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(multi_samples)])
        else:
            self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.use_multi_sample_dropout:
            outputs = []
            for dropout_layer in self.dropout:
                outputs.append(self.output(dropout_layer(x)))
            return torch.stack(outputs, dim=1)
        else:
            return self.output(self.dropout(x))

def freeze_word_embedding(bert_embedding):
    bert_embedding.word_embeddings.weight.requires_grad = False
    bert_embedding.position_embeddings.weight.requires_grad = False
    bert_embedding.token_type_embeddings.weight.requires_grad = False

class SingleModel(nn.Module):
    def __init__(self, setting, config):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(setting.model_name)

        self.head = PoolerLayer2(self.config.hidden_size)
        self.ln = nn.LayerNorm(self.config.hidden_size, elementwise_affine=False)

        self.use_extra_label = setting.use_extra_label
        self.el_single_output = setting.el_single_output

        if self.use_extra_label:
            self.big_output = DropWithLinear(self.config.hidden_size, 4)
            self.middle_output = DropWithLinear(self.config.hidden_size, 15)
        
        self.output = DropWithLinear(self.config.hidden_size, 46)

        if self.el_single_output:
            self.last_output = DropWithLinear(4+15+46, 46)

    def forward(self, input_ids, attention_mask, token_type_ids):
        res = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        res = self.head(res)
        res = self.ln(res)
        
        output = self.output(res)

        ret_dict = {'logits': output}
        if self.use_extra_label:
            big_output = self.big_output(res)
            middle_output = self.middle_output(res)
            ret_dict['big_logits'] = big_output
            ret_dict['middle_logits'] = middle_output

            if self.el_single_output:
                last_res = torch.cat([big_output, middle_output, output], dim=-1)
                last_output = self.last_output(last_res)
                ret_dict['small_logits'] = output
                ret_dict['logits'] = last_output
        
        return ret_dict

class SingleModelByRelation(nn.Module):
    def __init__(self, setting, config):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(setting.model_name)

        lm = get_labels_mapping(setting)
        tokenizer = get_tokenizer(setting.model_name)
        self.tok_lm = tokenizer(lm['소분류'].tolist(), padding=True, return_tensors='pt')
        self.qs = self.tok_lm['input_ids'].size(1)
        self.cache_query = None

        # self.qs = 1

        self.attn = nn.MultiheadAttention(768, 8, batch_first=True,
            dropout=0.1)

        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.qs * 768, 768),
            nn.ReLU(),
        )

        self.layernorm = nn.LayerNorm([self.qs, 768])
        
        self.output = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 1),
        )

    def setup_cache(self):
        device = next(self.parameters()).device

        tok_input_ids = self.tok_lm['input_ids'].to(device)
        tok_attention_mask = self.tok_lm['attention_mask'].to(device)
        tok_token_type = self.tok_lm['token_type_ids'].to(device)

        self.cache_query = self.backbone(
            input_ids=tok_input_ids,
            attention_mask=tok_attention_mask,
            token_type_ids=tok_token_type,
        ) # 46 QS D

        self.cache_query = self.cache_query['last_hidden_state']
        self.tok_attention_mask = tok_attention_mask

    def forward(self, input_ids, attention_mask, token_type_ids):
        res = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        device = next(self.parameters()).device
        res = res['last_hidden_state'] # B S D
        b, s, d = res.shape
        
        res = res.repeat_interleave(46, dim=0) # B*46 S D
        key_mask = attention_mask.repeat_interleave(46, dim=0).eq(0) # B*46 S

        if self.cache_query is None:
            tok_input_ids = self.tok_lm['input_ids'].to(device)
            tok_attention_mask = self.tok_lm['attention_mask'].to(device)
            tok_token_type = self.tok_lm['token_type_ids'].to(device)

            query = self.backbone(
                input_ids=tok_input_ids,
                attention_mask=tok_attention_mask,
                token_type_ids=tok_token_type,
            ) # 46 QS D

            query = query['last_hidden_state']
        else:
            query = self.cache_query
            tok_attention_mask = self.tok_attention_mask

        query = query.repeat(b, 1, 1) # B*46 QS D

        # B*46 QS D
        res, _ = self.attn(query, res, res)
        res = self.layernorm(res)

        res = res.reshape(b, 46, -1, d) # B 46 QS D
        res = torch.mean(res, dim=2) # B 46 D
        
        output = self.output(res).squeeze(-1) # B 46
    
        return {
            'logits': output
        }

class SingleModelGate(nn.Module):
    def __init__(self, setting, config):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(setting.model_name)

        self.head = PoolerLayer2(768)
        # self.head = PoolingLayer()

        self.gate = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 1),
            nn.Flatten(0),
        )

        if setting.gate_44:
            self.output = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(768, 44),
            )
        else:
            self.output = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(768, 45),
            )

    def forward(self, input_ids, attention_mask, token_type_ids):
        res = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        res = self.head(res)

        gate_output = self.gate(res)
        # res = self.head(res, attention_mask, is_mean=False)
        output = self.output(res)
        return {
            'logits': output,
            'gate': gate_output,
        }

class AttentionLayer(nn.Module):
    def __init__(self, hidden_shape):
        super().__init__()

        self.w1 = nn.Linear(hidden_shape, hidden_shape)
        self.w2 = nn.Linear(hidden_shape, hidden_shape)

    def forward(self, outputs, context):
        # output = B CH_LEN 2*D
        # context = B 2*D 1
        uit = torch.tanh(self.w1(outputs))
        cit = self.w2(context).unsqueeze(-1)

        score = torch.matmul(uit, cit) # B CH_LEN 1
        attn = torch.softmax(score, dim=1)

        return torch.sum(outputs * attn, dim=1)


class LongSingleModel(nn.Module):
    def __init__(self, setting, config):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(setting.model_name)

        self.head = PoolerLayer2(self.config.hidden_size)
        # self.head = PoolingLayer()
        self.output = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2 * self.config.hidden_size, 46),
        )

        self.gru = nn.GRU(input_size=self.config.hidden_size, hidden_size=self.config.hidden_size,
                 num_layers=1, dropout=0.1, bidirectional=True, batch_first=True)

        self.attn = AttentionLayer(2 * self.config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids, max_len=200):
        b, s = input_ids.shape
        chunk_len = (s + max_len - 1) // max_len  # 최대 chunk 갯수
        chunk_size = (s + chunk_len -1) // chunk_len # 각각의 chunk 최대한 비 균일하게 나누기 + 

        chunk_results = []
        for i in range(chunk_len):
            it_id = input_ids[:, i*chunk_size:(i+1)*chunk_size]
            at_mask = attention_mask[:, i*chunk_size:(i+1)*chunk_size]
            tt_id = token_type_ids[:, i*chunk_size:(i+1)*chunk_size]

            chunk_result = self.backbone(
                input_ids=it_id,
                attention_mask=at_mask,
                token_type_ids=tt_id,
                output_hidden_states=True,
            )
            chunk_result = self.head(chunk_result)
            chunk_results.append(chunk_result)

        chunks = torch.stack(chunk_results, dim=1) # B CH_LEN D
        
        # gru_output = B CH_LEN 2*D
        # gru_hidden = 2*1, B, D
        gru_output, gru_hidden = self.gru(chunks) 
        context_emb = torch.reshape(gru_hidden.permute(1, 0, 2), (b, -1)) # B 2*D

        doc_emb = self.attn(gru_output, context_emb) # B 2*D

        output = self.output(doc_emb)

        return {
            'logits': output
        }

class MixedModel(nn.Module):
    def __init__(self, setting, config):
        super().__init__()
        self.config = config
        self.features_size = len(setting.mixed_col_keys_to_use)

        tokenizer = get_tokenizer(setting.model_name)
        self.pad_token_id = tokenizer.pad_token_id

        self.backbone = AutoModel.from_pretrained(setting.model_name)
        self.mixed_two_models = setting.mixed_two_models
        self.mixed_second_model_index = setting.mixed_second_model_index

        if self.mixed_two_models:
            self.backbone2 = AutoModel.from_pretrained(setting.model_name)

        if setting.freezing_layer_num >= 0:
            freeze_word_embedding(self.backbone.embeddings)

            for i in range(setting.freezing_layer_num):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False

        self.ln = nn.LayerNorm(self.config.hidden_size, elementwise_affine=False)

        self.heads = nn.ModuleList(
            [PoolerLayer2(self.config.hidden_size) \
                    for _ in range(self.features_size)]
        )

        self.use_extra_label = setting.use_extra_label

        in_dim = self.config.hidden_size * self.features_size
        if self.use_extra_label:
            self.big_output = DropWithLinear(in_dim, 4,
                dropout=setting.output_dropout_p, multi_samples=setting.multi_sample_dropout_n)
            self.middle_output = DropWithLinear(in_dim, 15,
                dropout=setting.output_dropout_p, multi_samples=setting.multi_sample_dropout_n)
        
        self.output = DropWithLinear(in_dim, 46, 
            dropout=setting.output_dropout_p, multi_samples=setting.multi_sample_dropout_n)

    def forward(self, inputs):
        tmp = []
        for idx, input_ids in enumerate(inputs):
            # to(device) 덜 쓰면 더 빠를까봐 테스트
            attention_mask = input_ids.ne(self.pad_token_id)
            token_type_ids = torch.zeros_like(input_ids) # roberta does not use token type ids
                                                        # and currently just one sentence
            
            if self.mixed_two_models and idx in self.mixed_second_model_index:
                res = self.backbone2(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids,
                    output_hidden_states=True,
                )
            else:
                res = self.backbone(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids,
                    output_hidden_states=True,
                )

            res = self.heads[idx](res, cleanup=True)
            res = self.ln(res)
            tmp.append(res)

        tmp = torch.cat(tmp, dim=-1) # B, 768*6

        output = self.output(tmp)

        ret_dict = {'logits': output}
        if self.use_extra_label:
            big_output = self.big_output(tmp)
            middle_output = self.middle_output(tmp)
            ret_dict['big_logits'] = big_output
            ret_dict['middle_logits'] = middle_output
        
        return ret_dict

class BigMixedModel(nn.Module):
    def __init__(self, setting, config):
        super().__init__()
        self.setting = setting
        self.config = config

        tokenizer = get_tokenizer(setting.model_name)
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id

        self.backbone = AutoModel.from_pretrained(setting.model_name)

        if setting.freezing_layer_num >= 0:
            freeze_word_embedding(self.backbone.embeddings)

            for i in range(setting.freezing_layer_num):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False

        self.ln = nn.LayerNorm(self.config.hidden_size, elementwise_affine=False)

        self.features_size = len(setting.mixed_col_keys_to_use)
        self.head = PoolerLayer2(self.config.hidden_size)
        
        if setting.big_mixed_use_each:
            input_size =self.config.hidden_size * self.features_size
        else:
            input_size =self.config.hidden_size

        self.use_extra_label = setting.use_extra_label

        if self.use_extra_label:
            self.big_output = DropWithLinear(input_size, 4,
                dropout=setting.output_dropout_p, multi_samples=setting.multi_sample_dropout_n)
            self.middle_output = DropWithLinear(input_size, 15,
                dropout=setting.output_dropout_p, multi_samples=setting.multi_sample_dropout_n)
  
        self.output = DropWithLinear(input_size, 46,
            dropout=setting.output_dropout_p, multi_samples=setting.multi_sample_dropout_n)

    def forward(self, input_ids):
        b, s = input_ids.shape

        attention_mask = input_ids.ne(self.pad_token_id)
        token_type_ids = torch.zeros_like(input_ids)
        
        res = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        if self.setting.big_mixed_use_each:
            bos_places = input_ids == self.bos_token_id

            res = res['last_hidden_state']
            res = res[bos_places].reshape(b, self.features_size, -1)

            res = self.head(res, cleanup=False)
        else:
            res = self.head(res)
        
        res = self.ln(res)
        res = res.reshape(b, -1)

        output = self.output(res)

        ret_dict = {'logits': output}
        if self.use_extra_label:
            big_output = self.big_output(res)
            middle_output = self.middle_output(res)
            ret_dict['big_logits'] = big_output
            ret_dict['middle_logits'] = middle_output
            
        return ret_dict

class MixedModelGate(nn.Module):
    def __init__(self, setting, config):
        super().__init__()
        self.config = config
        self.features_size = len(setting.mixed_col_keys_to_use)

        tokenizer = get_tokenizer(setting.model_name)
        self.pad_token_id = tokenizer.pad_token_id
        self.backbone = AutoModel.from_pretrained(setting.model_name)

        if setting.freezing_layer_num >= 0:
            freeze_word_embedding(self.backbone.embeddings)

            for i in range(setting.freezing_layer_num):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False


        self.heads = nn.ModuleList(
            [PoolerLayer2(self.config.hidden_size) \
                    for _ in range(self.features_size)]
        )

        self.gate = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size * self.features_size, 1),
            nn.Flatten(0),
        )

        if setting.gate_44:
            self.output = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size * self.features_size, 44),
            )
        else:
            self.output = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size * self.features_size, 45),
            )

    def forward(self, inputs):
        tmp = []
        for idx, input_ids in enumerate(inputs):
            # to(device) 덜 쓰면 더 빠를까봐 테스트
            attention_mask = input_ids.ne(self.pad_token_id)
            token_type_ids = torch.zeros_like(input_ids) # roberta does not use token type ids
                                                        # and currently just one sentence
            
            res = self.backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids,
                output_hidden_states=True,
            )

            res = self.heads[idx](res)
            tmp.append(res)

        tmp = torch.cat(tmp, dim=-1) # B, 768*6

        gate_output = self.gate(tmp)
        # res = self.head(res, attention_mask, is_mean=False)
        output = self.output(tmp)
        return {
            'logits': output,
            'gate': gate_output,
        }
