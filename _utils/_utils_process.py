# -*- coding: utf-8 -*-
from path import Path
import os
import time
import tqdm


#-------------------------------------------------------------------------#
def find_txt(DataNodePath):
    DataPath = os.listdir(DataNodePath)
    if '.txt' in DataPath[0]:
        txt_filename = [DataNodePath / i for i in DataPath]
        return txt_filename
    else:
        txt_filename = []
        for folder_ in DataPath:
            DataNextPath = DataNodePath / folder_
            txt_filename += find_txt(DataNextPath)
        return txt_filename

def import_txt(TxtFilePath_List):
    time.sleep(0.25)
    
    TxtAmount = 0
    TxtSourceList, TxtContentList = [], []
    for eachTxtPath in tqdm.tqdm(TxtFilePath_List):
        try:
            with open(eachTxtPath, 'r', encoding='utf-8') as file:
                txt = file.read()
                if len(txt) < 10:
                    continue
                TxtContentList.append(txt)
            
            TxtSourceList.append(str(eachTxtPath))
            TxtAmount += 1
            
        except:
            with open(eachTxtPath, 'r', encoding='ANSI') as file:
                txt = file.read()
                if len(txt) < 10:
                    continue
                TxtContentList.append(txt)
                
            TxtSourceList.append(str(eachTxtPath))
            TxtAmount += 1
    
    time.sleep(0.25)
    print('    successfully import ' + str(TxtAmount) + ' articles')
    print()
    return TxtSourceList, TxtContentList

