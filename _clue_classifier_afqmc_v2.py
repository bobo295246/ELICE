# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:49:41 2021

@author: Jask

"""

from _utils._utils_process import *
from _utils._utils_model import *
from _utils._utils_encoder import *
import argparse


parser.add_argument('--version', type=int, default=6,
                    help='模型版本')
args = parser.parse_args()

# %%
# 設定專案與文本資料夾位置
inf = 0
cluePath = '_corpus\\CLUE'

# path of trained weights
modelPath ='_saved_models'

# downstream task
downstream = 'afqmc'  # afqmc, iflytek, ocnli, tnews
version = args.version
export_model = f'clue_{downstream}_v{version}'
import_model = f'clue_{downstream}_v{version}'

# pre-trained setting
_config_pretrain_ = 'ckiplab/albert-tiny-chinese'  # ckiplab/albert-base-chinese, ckiplab/albert-tiny-chinese

# %%
# -------------------------------------------------------------------------#
## the corresponding configuration ##
if version == 2:
    In_layers, Out_layers = 3, 3

elif version == 3:
    In_layers, Out_layers = 8, 8

elif version == 4:
    _config_pretrain_ = 'clue/albert_chinese_tiny'
    embedding_dim = 256
    In_layers, Out_layers = 12, 12

elif version == 5:
    _config_pretrain_ = 'voidful/albert_chinese_base'
    embedding_dim = 256
    In_layers, Out_layers = 5, 5

elif version == 6:
    _config_pretrain_ = 'voidful/albert_chinese_base'
    embedding_dim = 256
    In_layers, Out_layers = 3, 3

# import CJK tokenizer
if 'ckiplab' in _config_pretrain_:
    converter = opencc.OpenCC('s2t.json')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
else:
    converter = opencc.OpenCC('t2s.json')
    tokenizer = BertTokenizer.from_pretrained(_config_pretrain_)

# path of downstream task
if downstream == 'afqmc':
    cluePath_task = cluePath + '\\afqmc_public'
    pred_example = cluePath + '\\clue_submit_examples\\afqmc_predict.json'

    Path_label = None
    Path_train = cluePath_task + '\\train.json'
    Path_dev = cluePath_task + '\\dev.json'
    Path_test = cluePath_task + '\\test.json'

    Path_pred = cluePath + '\\_clue_pred\\afqmc_predict.json'

elif downstream == 'iflytek':
    cluePath_task = cluePath + '\\iflytek_public'
    pred_example = cluePath + '\\clue_submit_examples\\iflytek_predict.json'

    Path_label = cluePath_task + '\\labels.json'
    Path_train = cluePath_task + '\\train.json'
    Path_dev = cluePath_task + '\\dev.json'
    Path_test = cluePath_task + '\\test.json'

    Path_pred = cluePath + '\\_clue_pred\\iflytek_predict.json'

elif downstream == 'ocnli':
    cluePath_task = cluePath + '\\ocnli_public'
    pred_example = cluePath + '\\clue_submit_examples\\ocnli_50k_predict.json'

    Path_label = None
    Path_train = cluePath_task + '\\train.50k.json'
    Path_dev = cluePath_task + '\\dev.json'
    Path_test = cluePath_task + '\\test.json'

    Path_pred = cluePath + '\\_clue_pred\\ocnli_50k_predict.json'

elif downstream == 'tnews':
    cluePath_task = cluePath + '\\tnews_public'
    pred_example = cluePath + '\\clue_submit_examples\\tnews_predict.json'

    Path_label = cluePath_task + '\\labels.json'
    Path_train = cluePath_task + '\\train.json'
    Path_dev = cluePath_task + '\\dev.json'
    Path_test = cluePath_task + '\\test.json'

    Path_pred = cluePath + '\\_clue_pred\\tnews_predict.json'

else:
    print('[ERROR] the downstream must be one of the tasks (afqmc, iflytek, ocnli, tnews)')
    print('    reference: https://github.com/CLUEbenchmark/CLUE')


# %%
def clue_evaluate(dev_jsObj):
    elice_backbone.eval()
    elice_aggregator.eval()
    elice_encoder.eval()
    seq_classifier.eval()

    time.sleep(0.25)
    pred_match = 0
    with torch.no_grad():
        for obj_ in tqdm.tqdm(dev_jsObj):
            # sent 1 & sent 2 #----------------------------#

            obj_content = obj_['sentence1'] + '[SEP]' + obj_['sentence2']
            obj_content = fragment_content(converter.convert(obj_content), output_MaxLen, split_sent=False)

            anchor_tensor_ = elice_backbone(obj_content, limit_fragment, padding='longest', Masked_LM=False)
            anchor_tensor_ = elice_aggregator(anchor_tensor_)
            anchor_elice = elice_encoder(anchor_tensor_)
            class_logit = seq_classifier(anchor_elice)

            # -------------------------------------------------------------------------#
            class_logit = class_logit.squeeze(1)
            label_class = torch.tensor([int(obj_['label'])]).to(device)

            pred_match += int((class_logit.argmax(-1) == int(obj_['label'])).sum())

    time.sleep(0.25)
    print(f'[INFO] classification accuracy: {pred_match}/{len(dev_jsObj)} = {round(pred_match / len(dev_jsObj), 5)}')
    return None


def clue_inference(test_jsObj):
    elice_backbone.eval()
    elice_aggregator.eval()
    elice_encoder.eval()
    seq_classifier.eval()

    time.sleep(0.25)
    pred_jsObj = []
    with torch.no_grad():
        for obj_ in tqdm.tqdm(test_jsObj):
            # sent 1 & sent 2 #----------------------------#
            obj_content = obj_['sentence1'] + '[SEP]' + obj_['sentence2']
            obj_content = fragment_content(converter.convert(obj_content), output_MaxLen, split_sent=False)

            anchor_tensor_ = elice_backbone(obj_content, limit_fragment, padding='longest', Masked_LM=False)
            anchor_tensor_ = elice_aggregator(anchor_tensor_)
            anchor_elice = elice_encoder(anchor_tensor_)
            class_logit = seq_classifier(anchor_elice)

            # -------------------------------------------------------------------------#
            pred_jsObj.append({'id': obj_['id'], 'label': str(class_logit.argmax(-1).item())})

    time.sleep(0.25)
    return pred_jsObj


def import_clue_json(jsonPath):
    jsObj = []
    with open(jsonPath, 'r', encoding='utf8') as outfile:
        strObj = outfile.read().split('\n')

    del_EmptySentence(strObj)
    for i in strObj:
        jsObj.append(json.loads(i))
    return jsObj


def export_clue_pred(jsObj, jsonPath):
    with open(jsonPath, 'w', encoding='utf8') as outfile:
        for i in jsObj:
            obj_js = json.dumps(i, ensure_ascii=False)
            outfile.write(obj_js + '\n')
    return None


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
            MLM_inputs = tokenizer(obj_content[:limit_fragment],
                                   max_length=self.NLU_Model.config.max_position_embeddings,
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
        # hidden_state = self.Dropout(hidden_state)
        hidden_state = self.Dense(hidden_state)
        return hidden_state


# %%
# -------------------------------------------------------------------------#
## import data ##
batchs = 8

tune_data_tr = import_clue_json(Path_train)
tune_data_eva = import_clue_json(Path_dev)

# -------------------------------------------------------------------------#
## training configuration ##
output_MaxLen = 512  # basic: 505
limit_fragment = 8  # for limiting the maximal amount of fragments of large-scale document
datastep = 50000  # for limiting the maximal amount of training steps

sampling = max(int(len(tune_data_tr)), 16000)
print(sampling)
epochs = int(datastep / (sampling / batchs))
print(epochs)

# -------------------------------------------------------------------------#
## instantiate model ##
elice_backbone = Elice_Backbone().to(device)
# model_pt_filePATH = modelPath + '\\Elice_clue_afqmc_v4pre_Backbone.pt'
# elice_backbone.load_state_dict(torch.load(model_pt_filePATH))
param_amount(elice_backbone)

config = elice_backbone.NLU_Model.config

elice_aggregator = Elice_Aggregator().to(device)
param_amount(elice_aggregator)

elice_encoder = ELICE_Encoder(config, embedding_dim, In_layers=In_layers, Out_layers=Out_layers).to(device)
param_amount(elice_encoder)

num_class = 2
seq_classifier = Sequence_Classifier(embedding_dim, num_class).to(device)

param_amount(seq_classifier)

# -------------------------------------------------------------------------#
## loss & optimizer ##
optimizer_tune = torch.optim.Adam([  # v1
    {'params': elice_encoder.parameters(), 'lr': 1e-4},
    {'params': seq_classifier.parameters(), 'lr': 1e-4}
])
optimizer_tune = torch.optim.Adam([  # v2
    {'params': elice_backbone.parameters(), 'lr': 5e-6},
    {'params': elice_aggregator.parameters(), 'lr': 1e-4},
    {'params': elice_encoder.parameters(), 'lr': 1e-4},  # , 'weight_decay': 0.005
    {'params': seq_classifier.parameters(), 'lr': 1e-4}
])
scheduler_tune = torch.optim.lr_scheduler.StepLR(optimizer_tune, int(epochs / 5), gamma=0.75)
# -------------------------------------------------------------------------#

best_eval_loss = float('inf')

# %%
print('[INFO] start for ELICE Pretraining')
data_step = [datastep][0]
error_array = []
l_elice, l_task, l_acc = [], [], []
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

        # -------------------------------------------------------------------------#
        ## model process
        # -------------------------------------------------------------------------#
        elice_backbone.train()
        elice_aggregator.train()
        elice_encoder.train()
        seq_classifier.train()
        optimizer_tune.zero_grad()

        label_class = []
        Anchor_Tensor = []
        for obj_ in training_pair:
            label_class.append(int(obj_['label']))

            obj_content = obj_['sentence1'] + '[SEP]' + obj_[
                'sentence2']  # if random.uniform(0,1) > 0.5 else obj_['sentence2'] + '[SEP]' + obj_['sentence1']
            obj_content = fragment_content(converter.convert(obj_content), output_MaxLen, split_sent=False)

            anchor_tensor_ = elice_backbone(obj_content, limit_fragment, padding='longest', Masked_LM=False)
            anchor_tensor_ = elice_aggregator(anchor_tensor_)
            Anchor_Tensor.append(anchor_tensor_)

        anchor_tensor = torch.cat(Anchor_Tensor, 0)
        del anchor_tensor_, Anchor_Tensor
        torch.cuda.empty_cache()

        anchor_elice = elice_encoder(anchor_tensor)
        class_logit = seq_classifier(anchor_elice)

        # -------------------------------------------------------------------------#
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

        # -------------------------------------------------------------------------#
        ## task loss
        class_logit = class_logit.squeeze(1)
        label_class = torch.tensor(label_class).to(device)
        Pred_match += int((class_logit.argmax(-1) == label_class).sum())

        loss_2 = loss_fct(class_logit, label_class)
        Loss_task += loss_2

        # -------------------------------------------------------------------------#
        Loss_all = loss_1 + loss_2
        Loss_all.backward()
        torch.nn.utils.clip_grad_norm_(elice_backbone.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(elice_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(seq_classifier.parameters(), 0.5)
        optimizer_tune.step()

        # -------------------------------------------------------------------------#
        ## print metrics
        loss_elice_ = round(float(Loss_elice / num_), 5)
        loss_task_ = round(float(Loss_task / num_), 5)
        accuracy_ = round(float(Pred_match / (num_ * batchs)), 5)

        print(
            f'\r| remaining step {data_step}/{datastep} | step {num_}/{len(Batch_Data)} | loss_elice {loss_elice_} | loss_task {loss_task_} | acc. {accuracy_}',
            end='')

    scheduler_tune.step()

    # -------------------------------------------------------------------------#
    # -------------------------------------------------------------------------#
    if epoch % max(int((epochs / 10) / 4), 1) == 0:
        elice_backbone.eval()
        elice_aggregator.eval()
        elice_encoder.eval()
        seq_classifier.eval()

        evalPred_match = 0
        evalLoss_elice, evalLoss_task, evalLoss_all = 0, 0, 0
        eval_start_time = time.time()

        with torch.no_grad():
            for obj_ in tune_data_eva:
                # -------------------------------------------------------------------------#
                ## model process
                # -------------------------------------------------------------------------#
                obj_content = obj_['sentence1'] + '[SEP]' + obj_['sentence2']
                obj_content = fragment_content(converter.convert(obj_content), output_MaxLen, split_sent=False)

                anchor_tensor_ = elice_backbone(obj_content, limit_fragment, padding='longest', Masked_LM=False)
                anchor_tensor_ = elice_aggregator(anchor_tensor_)
                anchor_elice = elice_encoder(anchor_tensor_)
                class_logit = seq_classifier(anchor_elice)

                # -------------------------------------------------------------------------#
                ## NT-Xent loss
                # dropout noiser
                positive_elice = noiser(anchor_elice)

                cos_sim = cos_simi(anchor_elice, positive_elice.transpose(0, 1).contiguous()) / temperature
                label_ = torch.arange(cos_sim.size(0)).long().to(device)

                evalLoss_elice += loss_fct(cos_sim, label_)

                # -------------------------------------------------------------------------#
                ## task loss
                class_logit = class_logit.squeeze(1)
                label_class = torch.tensor([int(obj_['label'])]).to(device)

                evalPred_match += int((class_logit.argmax(-1) == int(obj_['label'])).sum())
                evalLoss_task += loss_fct(class_logit, label_class)

        print("inf_time:",round(float((time.time() - eval_start_time) / len(tune_data_eva)), 5))

        # -------------------------------------------------------------------------#
        ## print metrics
        evalloss_elice_ = round(float(evalLoss_elice / len(tune_data_eva)), 5)
        evalloss_task_ = round(float(evalLoss_task / len(tune_data_eva)), 5)
        evalaccuracy_ = round(float(evalPred_match / len(tune_data_eva)), 5)

        accum_eval_loss = 1 / evalaccuracy_  # based on accuracy
        if accum_eval_loss < best_eval_loss:
            best_eval_loss = accum_eval_loss

            # -------------------------------------------------------------------------#
            ## save model checkpoint
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


time.sleep(0.25)
print()
print(f'[INFO] the best acc. {round(100 / best_eval_loss, 3)} %')

# %%
# -------------------------------------------------------------------------#
## instantiate model ##
elice_backbone = Elice_Backbone().to(device)
model_pt_filePATH = modelPath + '\\Elice_' + import_model + '_Backbone.pt'
elice_backbone.load_state_dict(torch.load(model_pt_filePATH))
param_amount(elice_backbone)

config = elice_backbone.NLU_Model.config

elice_aggregator = Elice_Aggregator().to(device)
param_amount(elice_aggregator)

# embedding_dim = 256
elice_encoder = ELICE_Encoder(config, embedding_dim, In_layers=In_layers, Out_layers=Out_layers).to(device)
model_pt_filePATH = modelPath + '\\Elice_' + import_model + '_Encoder.pt'
elice_encoder.load_state_dict(torch.load(model_pt_filePATH))
param_amount(elice_encoder)

# num_class = 3
seq_classifier = Sequence_Classifier(embedding_dim, num_class).to(device)
model_pt_filePATH = modelPath + '\\Elice_' + import_model + '_Decoder.pt'
seq_classifier.load_state_dict(torch.load(model_pt_filePATH))
param_amount(seq_classifier)

# -------------------------------------------------------------------------#
## experimental results ##
clue_evaluate(tune_data_eva)

# %%
# -------------------------------------------------------------------------#
## export clue prediction ##
tune_data_te = import_clue_json(Path_test)
pred_jsObj = clue_inference(tune_data_te)
export_clue_pred(pred_jsObj, Path_pred)
