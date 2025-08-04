import numpy as np
import torch
import random
from transformers import BertTokenizer,AutoTokenizer
import transformers
import logging
import CONFIG
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import json
import argparse
import ysy_util
from ysy_util import Test_ClassDataset,read_data,create_logger
from model import Prompt_Model

logger = None


def Test_Seqcls(model,test_loader,device,cfg):
    total_steps = int(train_dataset.__len__() * cfg.epochs / (cfg.batch_size*cfg.gradient_accumulation))
    logger.info("We will process {} steps.".format(total_steps))
    logger.info("Starting Testing.")
    rank={}
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        total_accuracy=0
        total_batch=0
        model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                inputs=sample[0]
                labels=sample[1]
                vali_outputs=model.scl_test_step(inputs,device)
                batch_accuracy=compute_accuracy_yn(vali_outputs.logits,labels)
                # batch_accuracy=compute_micro_f1(vali_outputs.logits,labels)
                # batch_accuracy=compute_pearson(vali_outputs.logits,labels)
                total_accuracy+=batch_accuracy
                total_batch+=1
                logger.info("batch {}/{}".format(batch_idx + 1, vali_loader.__len__()))
        overal_accuracy=total_accuracy/total_batch
        epoch_id='epoch{}'.format(epoch+1)
        rank[epoch_id]=overal_accuracy
        logger.info("epoch {} validated,accuracy:{}.".format(epoch + 1,overal_accuracy))
        logger.info("epoch {} validated finished.".format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info("time for epoch{} : {}".format(epoch + 1,epoch_finish_time - epoch_start_time))
    logger.info("Test Finished ")
    average_accuracy = sum(rank.values()) / len(rank)
    logger.info("Average Accuracy: {}".format(average_accuracy))
    print(rank)

def Test_MultiChoice(model,test_loader,device,cfg):
    total_steps = int(train_dataset.__len__() * cfg.epochs / (cfg.batch_size*cfg.gradient_accumulation))
    logger.info("We will process {} steps.".format(total_steps))
    logger.info("Starting Testing.")
    rank={}
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        total_accuracy=0
        total_batch=0
        model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                inputs=sample[0]
                labels=sample[1].squeeze(1)
                vali_outputs=model.mlc_test_step(inputs,device)
                batch_accuracy=compute_accuracy_mlc(vali_outputs.logits,labels)
                total_accuracy+=batch_accuracy
                total_batch+=1
                logger.info("batch {}/{}".format(batch_idx + 1, vali_loader.__len__()))
        overal_accuracy=total_accuracy/total_batch
        epoch_id='epoch{}'.format(epoch+1)
        rank[epoch_id]=overal_accuracy
        logger.info("epoch {} ,accuracy:{}.".format(epoch + 1,overal_accuracy))
        logger.info("epoch {} test finished.".format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info("time for epoch{} : {}".format(epoch + 1,epoch_finish_time - epoch_start_time))
    logger.info("Test Finished ")
    average_accuracy = sum(rank.values()) / len(rank)
    logger.info("Average Accuracy: {}".format(average_accuracy))
    print(rank)

def main():
    global logger
    cfg = CONFIG.CONFIG(6)
    device = ysy_util.device_info(0)
    logger = create_logger(cfg.log_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    model = Prompt_Model(cfg.pretrained_model_path,3)
    model.load_state_dict(torch.load('/home/chenzhichen/projects/medical/save_model/best_prompt1_ori_MLC_model/Prompt_MlcModel.pth'),map_location='cuda:0')
    model.to(device)
    num_parameters = ysy_util.model_paramters_num(model)
    logger.info("number of model parameters:{}".format(num_parameters))
    test_set = read_json(cfg.test_data_path)
    test_dataset = Test_ClassDataset(test_set,tokenizer)
    test_loader = DataLoader(test_dataset, cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
    Test_MultiChoice(model,test_loader, device, cfg)
    # Test_Seqcls(model,test_loader, device, cfg)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
if __name__ == "__main__":
    print("Hello,Test")
    main()
