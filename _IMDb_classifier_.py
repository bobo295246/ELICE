# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 21:40:36 2021

@author: Jask

"""

from _utils._utils_process import *
from _utils._utils_model import *
from _utils._utils_encoder import *
import torch
import torch.nn as nn
from draw import *
from transformers import (
   BertTokenizer,
   BertTokenizerFast,
   AlbertModel,
   AutoModelForMaskedLM,
   AlbertTokenizer,
   AlbertForSequenceClassification,
   AlbertForMultipleChoice,
   AlbertForQuestionAnswering
)

# %%
# 設定專案與文本資料夾位置
projectPath = 'D:\\jask\\_aaai22_codedata'
IMDbPath = projectPath + '\\_corpus\\EngNLU\\IMDb\\aclImdb'
inf = 0

# path of trained weights
modelPath = projectPath + '\\_saved_models'

# downstream task
downstream = 'IMDb'
version = 1
export_model = f'glue_{downstream}_v{version}'
import_model = f'glue_{downstream}_v{version}'

# ablation setting
meta_mode = 'base'  # samll, base, large, full

# %%
#-------------------------------------------------------------------------#
## the corresponding configuration ##
if version == 1:
    _config_pretrain_ = 'albert-base-v2'
    tokenizer = AlbertTokenizer.from_pretrained(_config_pretrain_)
    
    embedding_dim = 128
    In_layers, Out_layers = 3, 3

# %%
## import dependency ##
import math
import torch
import torch.nn as nn
from transformers import (
   BertTokenizer,
   BertTokenizerFast,
   AlbertModel,
   AutoModelForMaskedLM,
   AlbertTokenizer,
   AlbertForSequenceClassification,
   AlbertForMultipleChoice,
   AlbertForQuestionAnswering
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f'[INFO] the device {device} is ready for pytorch objects computing')

class Elice_Backbone(nn.Module):
    def __init__(self, config=None):
        super(Elice_Backbone, self).__init__()
        
        self.NLU_Model = AutoModelForMaskedLM.from_pretrained(_config_pretrain_)
        self.Dropout = nn.Dropout(0.1)
        
    def forward(self, obj_content, limit_fragment, padding='longest', Masked_LM=False):
        # max_length: output shape = [content size, 512, 312]
        # longest: output shape = [content size, sequence length, 312]
        if Masked_LM:
            MLM_inputs = tokenizer(obj_content, max_length=self.NLU_Model.config.max_position_embeddings, 
                                   return_tensors='pt', padding=padding, truncation=True).to(device)
            model_output = self.NLU_Model(**MLM_inputs, output_hidden_states=True)
            
            del MLM_inputs
            torch.cuda.empty_cache()
            return model_output[0]

        else:
            MLM_inputs = tokenizer(obj_content[:limit_fragment], max_length=self.NLU_Model.config.max_position_embeddings, 
                                   return_tensors='pt', padding=padding, truncation=True).to(device)
            
            model_output = self.NLU_Model(**MLM_inputs, output_hidden_states=True)
            hidden_state = model_output[1][-1]
            del MLM_inputs, model_output
            torch.cuda.empty_cache()
            return hidden_state

class Elice_Aggregator(nn.Module):
    def __init__(self):
        super(Elice_Aggregator, self).__init__()
        self.Dropout = nn.Dropout(0.10)
        
    def forward(self, fusion_state):
        fusion_state = self.Dropout(fusion_state)
        
        # content-level averaging
        fusion_state = fusion_state.mean(0, keepdim=True)
        # sentence-level averaging
        fusion_state = fusion_state.mean(1, keepdim=True)
        return fusion_state

class ELICE_Encoder(nn.Module):
    def __init__(self, config, embedding_dim, In_layers=3, Out_layers=3):
        super(ELICE_Encoder, self).__init__()
        
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

class Sequence_Classifier(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(Sequence_Classifier, self).__init__()
        
        self.Dropout = nn.Dropout(0.05)
        self.Dense = nn.Linear(dim_input, dim_output, bias=True)
        
        initRange = 0.005
        self.Dense.bias.data.zero_()
        self.Dense.weight.data.uniform_(-initRange, initRange)

    def forward(self, hidden_state):
        #hidden_state = self.Dropout(hidden_state)
        hidden_state = self.Dense(hidden_state)
        return hidden_state

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

# %%
#-------------------------------------------------------------------------#
## import data ##
batchs = 10

DataRootPath = Path(IMDbPath + '\\train\\pos')
TxtFileName = find_txt(DataRootPath)
IMDb_Src_trpos, IMDb_txt_trpos = import_txt(TxtFileName)

DataRootPath = Path(IMDbPath + '\\train\\neg')
TxtFileName = find_txt(DataRootPath)
IMDb_Src_trneg, IMDb_txt_trneg = import_txt(TxtFileName)

DataRootPath = Path(IMDbPath + '\\test\\pos')
TxtFileName = find_txt(DataRootPath)
IMDb_Src_tepos, IMDb_txt_tepos = import_txt(TxtFileName)

DataRootPath = Path(IMDbPath + '\\test\\neg')
TxtFileName = find_txt(DataRootPath)
IMDb_Src_teneg, IMDb_txt_teneg = import_txt(TxtFileName)

# %%
tune_data_tr = []
for i in IMDb_txt_trpos:
    tune_data_tr.append([i, '1'])
for i in IMDb_txt_trneg:
    tune_data_tr.append([i, '0'])
random.shuffle(tune_data_tr)

tune_data_eva = []
for i in IMDb_txt_tepos:
    tune_data_eva.append([i, '1'])
for i in IMDb_txt_teneg:
    tune_data_eva.append([i, '0'])
random.shuffle(tune_data_eva)

#-------------------------------------------------------------------------#
## training configuration ##
if meta_mode == 'samll':
    output_MaxLen = 64
    limit_fragment = 64
elif meta_mode == 'base':
    output_MaxLen = 128
    limit_fragment = 32
elif meta_mode == 'large':
    output_MaxLen = 256
    limit_fragment = 16
elif meta_mode == 'full':
    output_MaxLen = 512
    limit_fragment = 8

datastep = 30000  # for limiting the maximal amount of training steps

sampling = min(int(len(tune_data_tr)), 30000)
epochs = int(datastep/(sampling/batchs))
print(epochs)

#-------------------------------------------------------------------------#
## instantiate model ##
elice_backbone = Elice_Backbone().to(device)
param_amount(elice_backbone)

config = elice_backbone.NLU_Model.config

elice_aggregator = Elice_Aggregator().to(device)
param_amount(elice_aggregator)

elice_encoder = ELICE_Encoder(config, embedding_dim, In_layers=In_layers, Out_layers=Out_layers).to(device)
param_amount(elice_encoder)

num_class = 2
seq_classifier = Sequence_Classifier(embedding_dim, num_class).to(device)
param_amount(seq_classifier)

#-------------------------------------------------------------------------#
## loss & optimizer ##
optimizer_tune = torch.optim.Adam([  # v1
            {'params': elice_encoder.parameters(), 'lr': 1e-4},
            {'params': seq_classifier.parameters(), 'lr': 1e-4}
            ])
optimizer_tune = torch.optim.Adam([  # v2
            {'params': elice_backbone.parameters(), 'lr': 5e-6},
            {'params': elice_aggregator.parameters(), 'lr': 5e-6},
            {'params': elice_encoder.parameters(), 'lr': 5e-6},
            {'params': seq_classifier.parameters(), 'lr': 5e-6}
            ])
scheduler_tune = torch.optim.lr_scheduler.StepLR(optimizer_tune, int(epochs/4), gamma=0.75)
#-------------------------------------------------------------------------#

best_eval_loss = float('inf')

# %%
print('[INFO] start for ELICE Pretraining')
data_step = [datastep][0]
l_elice,l_task,l_acc = [],[],[]
error_array = []
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    Batch_Data = list_batchify(random.sample(tune_data_tr, k=sampling), batchs, shuffle=True)
    
    num_ = 0
    Pred_match = 0
    Loss_elice, Loss_task, Loss_all = 0, 0, 0
    
    for training_pair in Batch_Data:
        if data_step <= 0:
            print(f'[INFO] the training reaches the maximal datastep ({datastep})')
            break
        
        data_step -= 1
        num_ += 1
        
        #-------------------------------------------------------------------------#
        ## model process
        #-------------------------------------------------------------------------#
        elice_backbone.train()
        elice_aggregator.train()
        elice_encoder.train()
        seq_classifier.train()
        optimizer_tune.zero_grad()
        
        label_class = []
        Anchor_Tensor = []
        for obj_ in training_pair:
            label_class.append(int(obj_[1]))
            
            obj_content = fragment_content(converter.convert(obj_[0]), output_MaxLen, split_sent=False)
            
            anchor_tensor_ = elice_backbone(obj_content, limit_fragment, padding='longest', Masked_LM=False)
            anchor_tensor_ = elice_aggregator(anchor_tensor_)
            Anchor_Tensor.append(anchor_tensor_)

        anchor_tensor = torch.cat(Anchor_Tensor, 0)
        del anchor_tensor_, Anchor_Tensor
        torch.cuda.empty_cache()
        
        anchor_elice = elice_encoder(anchor_tensor)
        class_logit = seq_classifier(anchor_elice)
        
        #-------------------------------------------------------------------------#
        ## NT-Xent loss
        # dropout noiser
        noise_ratio = 0.60
        noiser = nn.Dropout(p=noise_ratio, inplace=False)
        positive_elice = noiser(anchor_elice)
        
        cos_simi = nn.CosineSimilarity(dim=-1)
        loss_fct = nn.CrossEntropyLoss()
        temperature = 0.05
        
        cos_sim = cos_simi(anchor_elice, positive_elice.transpose(0, 1).contiguous()) / temperature
        label_ = torch.arange(cos_sim.size(0)).long().to(device)
        
        loss_1 = loss_fct(cos_sim, label_)
        Loss_elice += loss_1
        
        #-------------------------------------------------------------------------#
        ## task loss
        label_class = torch.tensor(label_class).unsqueeze(1).to(device)
        Pred_match += int((class_logit.argmax(-1) == label_class).sum())
        
        loss_2 = loss_fct(class_logit.transpose(1, 2).contiguous(), label_class)
        Loss_task += loss_2
        
        #-------------------------------------------------------------------------#
        Loss_all = loss_1 + loss_2
        Loss_all.backward()
        torch.nn.utils.clip_grad_norm_(elice_backbone.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(elice_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(seq_classifier.parameters(), 0.5)
        optimizer_tune.step()
        
        #-------------------------------------------------------------------------#
        ## print metrics
        loss_elice_ = round(float(Loss_elice/num_), 5)
        loss_task_ = round(float(Loss_task/num_), 5)
        accuracy_ = round(float(Pred_match/(num_*batchs)), 5)

        print(f'\r| remaining step {data_step}/{datastep} | step {num_}/{len(Batch_Data)} | loss_elice {loss_elice_} | loss_task {loss_task_} | acc. {accuracy_}', end='')
    
    scheduler_tune.step()
    #-------------------------------------------------------------------------#
    #-------------------------------------------------------------------------#
    if epoch % max(int((epochs/10)/4), 1) == 0:
        elice_backbone.eval()
        elice_aggregator.eval()
        elice_encoder.eval()
        seq_classifier.eval()
        
        evalPred_match = 0
        evalLoss_elice, evalLoss_task, evalLoss_all = 0, 0, 0
        eval_start_time = time.time()
        with torch.no_grad():
            for obj_ in tune_data_eva:
                #-------------------------------------------------------------------------#
                ## model process
                #-------------------------------------------------------------------------#
                obj_content = fragment_content(converter.convert(obj_[0]), output_MaxLen, split_sent=False)
                
                anchor_tensor_ = elice_backbone(obj_content, limit_fragment, padding='longest', Masked_LM=False)
                anchor_tensor_ = elice_aggregator(anchor_tensor_)
                anchor_elice = elice_encoder(anchor_tensor_)
                class_logit = seq_classifier(anchor_elice)
                
                #-------------------------------------------------------------------------#
                ## NT-Xent loss
                # dropout noiser
                positive_elice = noiser(anchor_elice)
                
                cos_sim = cos_simi(anchor_elice, positive_elice.transpose(0, 1).contiguous()) / temperature
                label_ = torch.arange(cos_sim.size(0)).long().to(device)
                
                evalLoss_elice += loss_fct(cos_sim, label_)
                
                #-------------------------------------------------------------------------#
                ## task loss
                label_class = torch.tensor([int(obj_[1])]).unsqueeze(1).to(device)
                
                evalPred_match += int((class_logit.argmax(-1) == int(obj_[1])).sum())
                evalLoss_task += loss_fct(class_logit.transpose(1, 2).contiguous(), label_class)
        print("inf_time:",round(float((time.time() - eval_start_time) / len(tune_data_eva)), 5))

        #-------------------------------------------------------------------------#
        ## print metrics
        evalloss_elice_ = round(float(evalLoss_elice/len(tune_data_eva)),5)
        evalloss_task_ = round(float(evalLoss_task/len(tune_data_eva)),5)
        evalaccuracy_ = round(float(evalPred_match/len(tune_data_eva)),5)
        
        accum_eval_loss = 1/evalaccuracy_  # based on accuracy
        if accum_eval_loss < best_eval_loss:
            best_eval_loss = accum_eval_loss

            #-------------------------------------------------------------------------#
            ## save model checkpoint ##
            model_pt_filePATH = modelPath + '\\Elice_' + export_model + '_Backbone.pt'
            save_pretrained_model(elice_backbone, model_pt_filePATH)
            
            model_pt_filePATH = modelPath + '\\Elice_' + export_model + '_Encoder.pt'
            save_pretrained_model(elice_encoder, model_pt_filePATH)
            
            model_pt_filePATH = modelPath + '\\Elice_' + export_model + '_Decoder.pt'
            save_pretrained_model(seq_classifier, model_pt_filePATH)

        print(f"""
            {'-' * 65}
            | end of epoch {epoch}
            | time: {(time.time() - epoch_start_time)} sec
            |
            | Training Metrics:
            |    loss_elice {loss_elice_}
            |    loss_task  {loss_task_}
            |    acc.       {accuracy_}
            |            
            | Evaluation Metrics:
            |    loss_elice {evalloss_elice_}  (return 0 because of one sample in the batch)
            |    loss_task  {evalloss_task_}
            |    acc.       {evalaccuracy_}
            |
            |    current loss {accum_eval_loss}
            |    best loss    {best_eval_loss}
            {'-' * 65}
              """, end='')
        l_elice.append(loss_elice_)
        l_task.append(evalloss_task_)
        l_acc.append(accuracy_)
save_loss(l_elice, l_task, l_acc, "imdb_base")

time.sleep(0.25)
print()
print(f'[INFO] the best acc. {round(100/best_eval_loss, 3)} %')


