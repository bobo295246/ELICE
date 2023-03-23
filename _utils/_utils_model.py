# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
from transformers import (
   BertTokenizer,
   BertTokenizerFast,
   AlbertModel,
   AutoModelForMaskedLM
)


#-------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f'[INFO] the device {device} is ready for pytorch objects computing')

#-------------------------------------------------------------------------#
## model class ##
class ngram_span():
    def __init__(self, n_Num, corpus):
        super(ngram_span, self).__init__()
        
        from gensim.models import Phrases
        corpus_ngram = [i for i in corpus]
        self.n_gram_model = []
        for i in range(n_Num):
            n_gram = Phrases(corpus_ngram, min_count=2, threshold=1)
            
            # apply the trained model to a sentence
            n_gram_Sent = []
            for ws_sent in corpus_ngram:
                n_gram_Sent.append([j.replace('_', '') for j in n_gram[ws_sent]])
                
            corpus_ngram = [j for j in n_gram_Sent]
            self.n_gram_model.append(n_gram)
    
    def word_segmentation(self, sent_str):
        sent_rc = [sent_str][0]
        for ngram in self.n_gram_model:
            rc = ngram[sent_rc]
            sent_rc = ([j.replace('_', '') for j in rc])
        return sent_rc

class Elice_Backbone(nn.Module):
    def __init__(self, config=None):
        super(Elice_Backbone, self).__init__()
        
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.NLU_Model = AutoModelForMaskedLM.from_pretrained('ckiplab/albert-tiny-chinese')
        
    def forward(self, obj_content, limit_fragment, padding='longest', Masked_LM=False):
        # max_length: output shape = [content size, 512, 312]
        # longest: output shape = [content size, sequence length, 312]
        if Masked_LM:
            MLM_inputs = self.tokenizer(obj_content, max_length=self.NLU_Model.config.max_position_embeddings, 
                                        return_tensors='pt', padding=padding, truncation=True).to(device)
            model_output = self.NLU_Model(**MLM_inputs, output_hidden_states=True)
            
            del MLM_inputs
            torch.cuda.empty_cache()
            return model_output[0]

        else:
            MLM_inputs = self.tokenizer(obj_content[:limit_fragment], max_length=self.NLU_Model.config.max_position_embeddings, 
                                        return_tensors='pt', padding=padding, truncation=True).to(device)
            
            model_output = self.NLU_Model(**MLM_inputs, output_hidden_states=True)
            hidden_state = model_output[1][-1]
            del MLM_inputs, model_output
            torch.cuda.empty_cache()
            return hidden_state

class Elice_Aggregator(nn.Module):
    def __init__(self):
        super(Elice_Aggregator, self).__init__()
        
    def forward(self, fusion_state):
        # content-level averaging
        fusion_state = fusion_state.mean(0, keepdim=True)
        # sentence-level averaging
        fusion_state = fusion_state.mean(1, keepdim=True)
        return fusion_state

class ELICE_Encoder(nn.Module):
    def __init__(self, config, embedding_dim, layers=3):
        super(ELICE_Encoder, self).__init__()
        
        self.Dense = nn.Linear(config.hidden_size, embedding_dim, bias=True)
        # nn.SELU(True), nn.GELU(), nn.ReLU(True), nn.Tanh()
        self.activate = nn.SELU(True)
        
        self.hidden = nn.ModuleList()
        for k in range(layers):
            linear_layer = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.hidden.append(linear_layer)

    def forward(self, hidden_state):
        hidden_state = self.Dense(hidden_state)
        hidden_state = self.activate(hidden_state)
        
        for linear_layer in self.hidden:
            hidden_state = linear_layer(hidden_state)
            hidden_state = self.activate(hidden_state)
        return hidden_state

class ELICE21_Encoder(nn.Module):
    def __init__(self, config, embedding_dim, In_layers=3, Out_layers=3):
        super(ELICE21_Encoder, self).__init__()
        
        self.Dropout = nn.Dropout(0.10)
        # nn.SELU(True), nn.GELU(), nn.ReLU(True), nn.Tanh()
        self.Activ = nn.ReLU(True)
        
        self.Norm_In = nn.LayerNorm(config.hidden_size)
        self.Norm_Out = nn.LayerNorm(embedding_dim)
        
        self.Hidden_In = nn.ModuleList()
        for k in range(In_layers):
            linear_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.Hidden_In.append(linear_layer)
        
        self.Project = nn.Linear(config.hidden_size, embedding_dim, bias=True)
                
        self.Hidden_Out = nn.ModuleList()
        for k in range(Out_layers):
            linear_layer = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.Hidden_Out.append(linear_layer)

    def forward(self, hidden_state):
        cache_state = []
        for linear_layer in self.Hidden_In:
            cache_state.append(hidden_state)
            
            hidden_state = self.Dropout(hidden_state)
            hidden_state = linear_layer(hidden_state)
            hidden_state = self.Activ(hidden_state)
            
            for cache in cache_state:
                hidden_state = hidden_state + cache
            hidden_state = self.Norm_In(hidden_state)
        
        fusion_state = self.Dropout(hidden_state)
        fusion_state = self.Project(fusion_state)
        fusion_state = self.Activ(fusion_state)
        
        cache_state = []
        for linear_layer in self.Hidden_Out:
            cache_state.append(fusion_state)
            
            fusion_state = self.Dropout(fusion_state)
            fusion_state = linear_layer(fusion_state)
            fusion_state = self.Activ(fusion_state)
            
            for cache in cache_state:
                fusion_state = fusion_state + cache
            fusion_state = self.Norm_Out(fusion_state)
        return fusion_state

#-------------------------------------------------------------------------#
def Whitening(embeddings):
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings

def random_samlping(object_, ratio):
    ranidx = [j for j in range(len(object_))]
    random.shuffle(ranidx)
    ranidx = ranidx[0:int(len(ranidx)*ratio)]
    return ranidx

def span_masking_generator(sent_str, tokenizer, ngram_model, ratio):
    aa = tokenizer.encode(sent_str, add_special_tokens=False)
    bb = tokenizer.decode(aa)
    
    cc = str()
    for i in bb.split(' '):
        cc += i if len(i) == 1 else ' '+ i
    cc = cc.split(' ')
    
    dd = []
    for i in cc:
        dd += ngram_model.word_segmentation(i) if len(i) > 4 else [i]
    
    ee = tokenizer(dd, add_special_tokens=False)['input_ids']
    ranidx = random_samlping(ee, ratio)
    
    token_i, token_o = [], []
    for idx in range(len(ee)):
        if idx in ranidx:
            token_i += [tokenizer.mask_token_id for i in ee[idx]]
            token_o += ee[idx]
        else:
            token_i += ee[idx]
            token_o += [tokenizer.pad_token_id for i in ee[idx]]
    
    #token_i = [tokenizer.cls_token_id] + token_i + [tokenizer.sep_token_id]
    #token_o = [tokenizer.cls_token_id] + token_o + [tokenizer.sep_token_id]
    
    sent_input = tokenizer.decode(token_i)
    sent_target = tokenizer.decode(token_o)
    return sent_input, sent_target

def list_batchify(object_list, batch_size, shuffle=False):
    Data_ = [i for i in object_list]
    if shuffle == True:
        random.shuffle(Data_)
    
    num_ = 1
    batch_, Batch_Data = [], []
    for object_ in Data_:        
        batch_ += [object_]
        
        if num_ % batch_size == 0:
            Batch_Data.append(batch_)
            batch_ = []
        num_ += 1
    
    if len(batch_) != 0:
        Batch_Data.append(batch_)
        del batch_
    return Batch_Data

def param_amount(torch_model):
    print('[INFO] print amount of model parameters')
    
    pytorch_total_params = sum(p.numel() for p in torch_model.parameters())
    print('    all parameters\n\t\t', pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    print('    trainable parameters\n\t\t', pytorch_total_params)
    print()

def save_pretrained_model(pretrained_model, model_pt_filePATH):
    torch.save(pretrained_model.state_dict(), model_pt_filePATH)
    return None