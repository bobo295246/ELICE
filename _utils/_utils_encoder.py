# -*- coding: utf-8 -*-
import gc
import numba as nb
import tqdm
import joblib
import json
import numpy as np
import time
import torch
import os
import re
import opencc


#-------------------------------------------------------------------------#
converter = opencc.OpenCC('s2t.json')
converter.convert('汉字')  # 漢字

#-------------------------------------------------------------------------#
## punctuation RegEx (regular expression) ##
# pattern: 
RegEx_FwSpace = re.compile('[\u3000]')
RegEx_mFwSpace = re.compile('[\u3000]{2,}')
RegEx_HwSpace = re.compile('[\u0020]')
RegEx_mHwSpace = re.compile('[\u0020]{2,}')
# pattern: Fw and Hw stop, comma
RegEx_Stop = re.compile('[\u3002\uFF61]')
RegEx_Comma = re.compile('[\uFF0C\u002C]')
RegEx_mComma = re.compile('[\uFF0C]+')
# pattern: general punctuation, non-word-character, except HwSpace, FwStop, FwComma, \u25CB (white circle)
RegEx_GenPunc = re.compile('[^\u3002\uFF0C\u0020\u25CB\u4E00-\u9FA5\u0041-\u005A\u0061-\u007A\uFF21-\uFF3A\uFF41-\uFF5A\u0030-\u0039\uFF10-\uFF19]')
# abnormal pattern
RegEx_ErrorSet_0 = re.compile('([\uFF0C]nbsp[\uFF0C])')

#-------------------------------------------------------------------------#
def del_EmptySentence(Content_SentList):
    useless_row = []
    for i in range(len(Content_SentList)):
        tempSent = re.sub(RegEx_HwSpace, '', Content_SentList[i])
        tempSent = re.sub(RegEx_Comma, '', tempSent)
        if len(tempSent) == 0:
            useless_row.append(i)
    
    useless_row.reverse()
    i = 0
    while i < len(useless_row):
        del Content_SentList[useless_row[i]]
        i += 1    
    return(0)

def fragment_content(raw_content, output_MaxLen, split_sent=False):
        text = raw_content.strip()
        text = text.replace('\ufeff', '')  # UTF-8 BOM
        text = text.replace('─', '')  # table notation
        text = text.replace('┼', '')  # table notation
        text = text.replace('│', '')  # table notation
        text = text.replace('├', '')  # table notation
        text = text.replace('┤', '')  # table notation
        text = text.replace('_', '')  # table notation
        text = text.replace('\u3000', '')
        text = text.replace('\xa0', '')
        text = text.replace('\n', '')
        text = re.sub(RegEx_mHwSpace, '', text)
        
        if split_sent:
            text = text.split('。')  # split content by stop notation
            del_EmptySentence(text)
            text = [i + '。' for i in text]
        else:
            text = [text]  # only constrain on sequence length
        
        text_ = []
        idx = []
        for j in range(len(text)):            
            if len(text[j]) > output_MaxLen:
                idx.append(j)
                SoS, EoS = 0, [output_MaxLen][0]
                for k in range((len(text[j]) // output_MaxLen)+1):
                    text_.append(text[j][SoS:EoS])
                    SoS, EoS = [EoS][0], (EoS + output_MaxLen)
            else:
                text_.append(text[j])
        
        del_EmptySentence(text_)
        return text_

def ELICE_DocEncoder(document_object, output_MaxLen, limit_fragment, module_dict):
    if len(module_dict) == 3:
        encoder = True
    else:
        encoder = False
    
    elice_backbone = module_dict['elice_backbone']
    elice_aggregator = module_dict['elice_aggregator']
    
    elice_backbone.eval()
    elice_aggregator.eval()
    
    obj_title = document_object['Title']
    obj_raw_content = document_object['Content']
    
    obj_content = fragment_content(obj_raw_content, output_MaxLen)
    obj_content = [obj_title] + obj_content
    #print(obj_content)
    
    base_tensor_ = elice_backbone(obj_content, limit_fragment, padding='longest', Masked_LM=False)
    fusion_state = elice_aggregator(base_tensor_)

    if encoder:
        elice_encoder = module_dict['elice_encoder']
        elice_encoder.eval()
        fusion_state = elice_encoder(fusion_state)
    return fusion_state

def embedding_process(document_list, dump_filePATH, output_MaxLen, limit_fragment, module_dict):
    #print('[CAUTION] have to define output_MaxLen, limit_fragment, elice_module firstly')
    
    time.sleep(0.25)
    num_ = 0
    for doc in tqdm.tqdm(document_list):
        # convert torch.tensor into numpy.array
        ELICE_obj = {
            'Title': doc['Title'],
            'Content': doc['Content'],
            'Elice_Vec': ELICE_DocEncoder(doc, output_MaxLen, limit_fragment, module_dict).squeeze(0).squeeze(0).cpu().detach().numpy()
            }
        filename =  '\\Elice-' + str(num_) + '.pkl'
        joblib.dump(ELICE_obj, dump_filePATH + filename, compress=0)
        
        num_ += 1
        
        del ELICE_obj
        gc.collect()
        torch.cuda.empty_cache()
    return None

def import_ELICEpkl(DataRootPath):
    pklNameList = os.listdir(DataRootPath)
    
    time.sleep(0.25)
    print('[INFO] start to import articles')
    time.sleep(0.25)
    
    pklAmount = 0
    ELICE_ = []
    for i in tqdm.tqdm(pklNameList):
        eachPklPath = DataRootPath / i
        
        pkl_ = joblib.load(eachPklPath, mmap_mode=None)
        ELICE_.append(pkl_)
        pklAmount += 1
        
    print('    successfully import ' + str(pklAmount) + ' pkl')
    print()
    return ELICE_

@nb.jit(nopython=True, parallel=False)
def np_cos_simi(a, b):
    return (a @ b.T) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-10)

@nb.jit(nopython=True, parallel=False)
def np_l2_simi(a, b):
    return 1 / (np.linalg.norm(a-b) + 1e-10)

def similarity_ranking(base_ELICE, reference_ELICE, ranking=20, evaluate_output=True):
    time.sleep(0.25)
    output_table = []
    for base_ in tqdm.tqdm(base_ELICE):
        score_ = []
        score_table = []
        for refer_ in reference_ELICE:
            score_.append(np_cos_simi(base_['Elice_Vec'], refer_['Elice_Vec']))
            #score_.append(np_l2_simi(base_['Elice_Vec'], refer_['Elice_Vec']))
            
            if evaluate_output:
                score_table.append({
                    'Title': refer_['Title'],
                    'Score': float(score_[-1])
                    })
            else:
                score_table.append([refer_['Title'], float(score_[-1])])
        
        # obtain the top-N ranking results
        rank = 0
        ranking_table = []
        while rank < ranking and rank <= len(score_table):
            idx_ = score_.index(max(score_))
            ranking_table.append(score_table[idx_])
            
            del score_[idx_], score_table[idx_]
            rank += 1
        
        if evaluate_output:
            output_table.append({
                    'Base': base_['Title'],
                    'Ref': ranking_table
                    })
        else:
            output_table.append({base_['Title']: {'albert': ranking_table}})
    return output_table
