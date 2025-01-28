import numpy as np
import torch
import random
from transformers import BertTokenizer,AutoTokenizer
import transformers
import logging
import CONFIG
import os
import sklearn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import model_selection
from datetime import datetime
from copy import deepcopy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import json
import argparse
import ysy_util
from ysy_util import ClassDataset,read_data,create_logger,Test_ClassDataset
from model import Prompt_Model

logger = None
pad_id = 0
PAD = '[PAD]'

def answer_top_k(inputs, ques_embed, start, end, tokenizer):
    input_ids = inputs['input_ids'].squeeze()
    start=start.squeeze()
    end=end.squeeze()
    l=input_ids.size(0)
    ql=ques_embed.size(1)
    input_ids = input_ids.tolist()  
    start = start.tolist()  
    end = end.tolist()
    logit_ls={}
    for i in range(l):
        for j in range(l):
            if i <= j and i >= 2+ql and j < l-1:
                if input_ids[j] == 0:
                    break
                else:
                    logit_ls[start[i]+end[j]] = [start[i], end[j], i,j,tokenizer.decode(input_ids[i:j+1]).replace(" ","")]
    result = sorted(logit_ls.items(),key=lambda x:x[0],reverse=True)
    if len(result) > 8:
        result=result[:8]
    return result

def test(model, test_loader, device, cfg, tokenizer):
    model.eval()
    logger.info("starting Testing.")
    result = {}
    start_time = datetime.now()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            inputs = sample[0]
            ques_embed = sample[1].to(device)
            #topic_embed=sample[2].to(device)
            #info_embed = sample[3].to(device)
            #start = sample[4].to(device)
            #end = sample[5].to(device)
            outputs = model.mlm_test_step(inputs, ques_embed, device)
            answer_texts = answer_top_k(inputs,ques_embed,outputs.start_logits, outputs.end_logits,tokenizer)
            result[batch_idx]=answer_texts
            logger.info("batch {}/{}".format(batch_idx + 1, test_loader.__len__()))
            #print("id:{},reslut:{}".format(batch_idx+1,result[batch_idx]))
    json_str = json.dumps(result,indent=4,ensure_ascii=False)
    with open('result.json', 'w',encoding='utf-8') as json_file:
        logger.info("Test results saved to result.json.")
        json_file.write(json_str)
    finish_time = datetime.now()
    logger.info("time for test{}".format(finish_time - start_time))

def main():
    global logger
    cfg = CONFIG.CONFIG()
    device = ysy_util.device_info(cfg.device)
    print(device)
    logger = create_logger(cfg.log_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    model = Prompt_Model(cfg.pretrained_model_path)
    model.to(device)
    # last_epoch_model_path = os.path.join(cfg.saved_model_path, 'model_epoch{}'.format(cfg.epochs))
    # model.load_state_dict(torch.load(last_epoch_model_path+'/Prompt_Model.pth', map_location='cuda:0'))
    model.load_state_dict(torch.load('D:\\vscode-projects\\medical\\Prompt_MRCModel.pth', map_location='cuda:0'))
    num_parameters = ysy_util.model_paramters_num(model)
    logger.info("number of model parameters:{}".format(num_parameters))
    #test_set = read_data(cfg.test_data_path)
    test_set = read_data("D:\\vscode-projects\\medical\\dataset\\zh\\answer.txt")
    test_dataset = Test_ClassDataset(test_set,tokenizer)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = cfg.num_workers)
    test(model,test_loader, device, cfg, tokenizer)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
if __name__ == "__main__":
    print("Hello,Test")
    main()
