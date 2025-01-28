import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizer,AutoTokenizer,BertForMultipleChoice
import transformers
import logging
import CONFIG
import os
from torch.utils.data import DataLoader
from sklearn import model_selection
from datetime import datetime
from copy import deepcopy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import ysy_util
from ysy_util import ClassDataset,read_txt,create_logger,device_info,read_json,read_jsonl
from model import Prompt_Model
import evaluate
logger = None

def compute_accuracy_qa(pred_start, pred_end, true_start, true_end):
    true_start=true_start.squeeze(1)
    true_end=true_end.squeeze(1)
    pred_start_idx = pred_start.argmax(dim=1)#tensor([3, 3, 0]) 
    pred_end_idx = pred_end.argmax(dim=1)
    # correct_start = pred_start_idx == torch.tensor(true_start)#tensor([true,true,false])
    # correct_end = pred_end_idx == torch.tensor(true_end)
    correct_start = pred_start_idx == true_start
    correct_end = pred_end_idx == true_end
    correct = correct_start & correct_end
    accuracy = correct.float().mean().item()
    return accuracy

def compute_accuracy_mlc(pred_logits,true_logits):
    pred_logits = pred_logits.to('cpu')
    true_logits = true_logits.to('cpu')
    pred_logits = pred_logits.argmax(dim=1)#tensor([3, 3, 0]) 
    correct = pred_logits == true_logits
    accuracy = correct.float().mean().item()
    return accuracy

def compute_accuracy_yn(pred_logits,true_logits):
    pred_logits = pred_logits.to('cpu')
    true_logits = true_logits.to('cpu')
    true_logits=true_logits.squeeze(1)
    pred_logits=F.softmax(pred_logits,dim=1)
    pred_logits = pred_logits.argmax(dim=1)#tensor([3, 3, 0]) 
    correct = pred_logits == true_logits
    accuracy = correct.float().mean().item()
    return accuracy

def evaluate_accuracy(pred_start, pred_end, true_start, true_end):
    all_preds = []
    all_trues = []
    accuracy_metric=evaluate.load('accuracy')
    true_start=true_start.squeeze(1)
    true_end=true_end.squeeze(1)
    pred_start_idx = pred_start.argmax(dim=1)#tensor([3, 3, 0]) 
    pred_end_idx = pred_end.argmax(dim=1)
    all_preds.extend(zip(pred_start_idx.cpu().numpy(),pred_end_idx.cpu().numpy()))
    all_trues.extend(zip(true_start.cpu().numpy(),true_end.cpu().numpy()))
    accuracy=accuracy_metric.compute(predictions=all_preds,references=all_trues)
    return accuracy['accuracy']

def train_QA(model,train_dataset,train_loader,vali_loader,device,cfg):
    model.train()
    total_steps = int(train_dataset.__len__() * cfg.epochs / cfg.batch_size)
    logger.info("We will process {} steps.".format(total_steps))
    no_decay = ['bias', 'LayerNorm.weight']
    #weight_decay check?
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    # ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    logger.info("starting training.")
    rank={}
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        total_accuracy=0
        total_batch=0
        optimizer.zero_grad()
        for batch_idx, sample in enumerate(train_loader):
            inputs = sample[0]
            #ques_embed = sample[1].to(device)
            topic_embed=sample[2].to(device)    
            #info_embed = sample[3].to(device)
            start = sample[4].to(device)#tensor([[3],[2],[1],[5],[6]])
            end = sample[5].to(device)
            outputs= model.mlm_train_step(inputs, topic_embed, start, end, device)
            # start_position=start.squeeze(1).to(device)
            # end_position=end.squeeze(1).to(device)
            # ignore_idx = outputs['start_logits'].size(1)#[2.0, 0.5, 0.3, 3.0, 0.2]
            # start_position=start_position.clamp_(0, ignore_idx)
            # end_position=end_position.clamp_(0, ignore_idx)
            # loss_fct =CrossEntropyLoss(ignore_index=ignore_idx)
            # start_loss=loss_fct(outputs['start_logits'],start_position)
            # end_loss=loss_fct(outputs['end_logits'],end_position)
            # loss=(start_loss+end_loss)/2.0
            loss=outputs.loss
            loss=loss/4
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), cfg.max_grad_norm)
            if (batch_idx + 1) % 4 == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            logger.info("batch {}/{} of epoch {}/{}, loss {}".format(batch_idx + 1, train_loader.__len__(),epoch + 1, cfg.epochs, loss.item()))
        logger.info("epoch {} train finished.".format(epoch + 1))
        logger.info("epoch {} starting validating.".format(epoch + 1))
        model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(vali_loader):
                inputs = sample[0]
                #ques_embed = sample[1].to(device)
                topic_embed=sample[2].to(device)
                #info_embed = sample[3].to(device)
                start = sample[4].to(device)
                end = sample[5].to(device)
                vali_outputs=model.mlm_test_step(inputs, topic_embed, device)
                vali_start=vali_outputs.start_logits
                vali_end=vali_outputs.end_logits
                batch_accuracy=compute_accuracy_qa(vali_start,vali_end,start,end)
                total_accuracy+=batch_accuracy
                total_batch+=1
                logger.info("batch {}/{}".format(batch_idx + 1, vali_loader.__len__()))
        overal_accuracy=total_accuracy/total_batch
        epoch_id='epoch{}'.format(epoch+1)
        rank[epoch_id]=overal_accuracy
        logger.info("epoch {} ,accuracy:{}.".format(epoch + 1,overal_accuracy))
        logger.info("epoch {} validated finished.".format(epoch + 1))
        model_path = os.path.join(cfg.saved_model_path, 'model_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if cfg.save_mode:
            # qa_outputs_weight = model.model.qa_outputs.weight
            # qa_outputs_bias = model.model.qa_outputs.bias
            torch.save(model.state_dict(), model_path + '/Prompt_MRCModel.pth')
            # torch.save({'qa_outputs.weight':qa_outputs_weight,'qa_outputs.bias':qa_outputs_bias},model_path+'/qa_outputs.pth')
        epoch_finish_time = datetime.now()
        logger.info("time for epoch{} : {}".format(epoch + 1,epoch_finish_time - epoch_start_time))
    logger.info("Train Finished ")
    print(rank)

def train_MultiChoice(model,train_dataset,train_loader,vali_loader,device,cfg):
    model.train()
    total_steps = int(train_dataset.__len__() * cfg.epochs / cfg.batch_size)
    logger.info("We will process {} steps.".format(total_steps))
    no_decay = ['bias', 'LayerNorm.weight']
    #weight_decay check?
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.pre_model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
    #     {'params': [p for n, p in model.pre_model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    # ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    logger.info("starting training.")
    rank={}
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        total_accuracy=0
        total_batch=0
        optimizer.zero_grad()
        for batch_idx, sample in enumerate(train_loader):
            inputs=sample[0]
            labels=sample[1].squeeze(1)
            outputs= model.mlc_train_step(inputs,device)
            #loss=outputs.loss
            print(outputs.logits.shape)
            pred_logits=outputs.logits.to('cpu')
            true_logits=labels.to('cpu')
            loss_fct =CrossEntropyLoss()
            loss=loss_fct(pred_logits,true_logits)
            loss=loss/4
            loss=loss.to('cuda')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), cfg.max_grad_norm)
            #torch.nn.utils.clip_grad_norm_(model.pre_model.parameters(), cfg.max_grad_norm)
            if (batch_idx + 1) % 4 == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            logger.info("batch {}/{} of epoch {}/{}, loss {}".format(batch_idx + 1, train_loader.__len__(),epoch + 1, cfg.epochs, loss.item()))
        logger.info("epoch {} train finished.".format(epoch + 1))
        logger.info("epoch {} starting validating.".format(epoch + 1))
        model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(vali_loader):
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
        logger.info("epoch {} validated finished.".format(epoch + 1))
        model_path = os.path.join(cfg.saved_model_path, 'model_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if cfg.save_mode:
            # qa_outputs_weight = model.model.qa_outputs.weight
            # qa_outputs_bias = model.model.qa_outputs.bias
            torch.save(model.state_dict(), model_path + '/Prompt_MRCModel.pth')
            # torch.save({'qa_outputs.weight':qa_outputs_weight,'qa_outputs.bias':qa_outputs_bias},model_path+'/qa_outputs.pth')
        epoch_finish_time = datetime.now()
        logger.info("time for epoch{} : {}".format(epoch + 1,epoch_finish_time - epoch_start_time))
    logger.info("Train Finished ")
    print(rank)

def train_seqcls(model,train_dataset,train_loader,vali_loader,device,cfg):
    model.train()
    total_steps = int(train_dataset.__len__() * cfg.epochs / cfg.batch_size)
    logger.info("We will process {} steps.".format(total_steps))
    no_decay = ['bias', 'LayerNorm.weight']
    #weight_decay check?
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    logger.info("starting training.")
    rank={}
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        total_accuracy=0
        total_batch=0
        optimizer.zero_grad()
        for batch_idx, sample in enumerate(train_loader):
            inputs=sample[0]
            labels=sample[1]
            outputs= model.scl_train_step(inputs,labels,device)
            loss=outputs.loss
            # pred_logits=outputs.logits.to('cpu')
            # true_logits=labels.to('cpu')
            # loss_fct =CrossEntropyLoss()
            # loss=loss_fct(pred_logits,true_logits)
            # loss=loss/4
            # loss=loss.to('cuda')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), cfg.max_grad_norm)
            if (batch_idx + 1) % 4 == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            logger.info("batch {}/{} of epoch {}/{}, loss {}".format(batch_idx + 1, train_loader.__len__(),epoch + 1, cfg.epochs, loss.item()))
        logger.info("epoch {} train finished.".format(epoch + 1))
        logger.info("epoch {} starting validating.".format(epoch + 1))
        model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(vali_loader):
                inputs=sample[0]
                labels=sample[1]
                vali_outputs=model.scl_test_step(inputs,device)
                batch_accuracy=compute_accuracy_yn(vali_outputs.logits,labels)
                total_accuracy+=batch_accuracy
                total_batch+=1
                logger.info("batch {}/{}".format(batch_idx + 1, vali_loader.__len__()))
        overal_accuracy=total_accuracy/total_batch
        epoch_id='epoch{}'.format(epoch+1)
        rank[epoch_id]=overal_accuracy
        logger.info("epoch {} validated,accuracy:{}.".format(epoch + 1,overal_accuracy))
        logger.info("epoch {} validated finished.".format(epoch + 1))
        model_path = os.path.join(cfg.saved_model_path, 'model_epoch{}'.format(epoch + 1))
        # if not os.path.exists(model_path):
        #     os.mkdir(model_path)
        # qa_outputs_weight = model.model.qa_outputs.weight
        # qa_outputs_bias = model.model.qa_outputs.bias
        torch.save(model.state_dict(), model_path + '/Prompt_Model.pth')
        # torch.save({'qa_outputs.weight':qa_outputs_weight,'qa_outputs.bias':qa_outputs_bias},model_path+'/qa_outputs.pth')
        epoch_finish_time = datetime.now()
        logger.info("time for epoch{} : {}".format(epoch + 1,epoch_finish_time - epoch_start_time))
    logger.info("Train Finished ")
    print(rank)

def main():
    global logger
    cfg = CONFIG.CONFIG()
    device = ysy_util.device_info(cfg.device)
    print(device)
    logger = create_logger(cfg.log_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    model = Prompt_Model(cfg.pretrained_model_path)
    model.to(device)
    #model.load_state_dict(torch.load('D:\\vscode-projects\\medical\\Prompt_MRCModel.pth', map_location='cuda:0'))
    # num_parameters = ysy_util.model_paramters_num(model.pre_model)
    # logger.info("number of model parameters:{}".format(num_parameters))
    print(model.model.print_trainable_parameters())
    train_set = read_json(cfg.train_data_path)
    train_dataset = ClassDataset(train_set,tokenizer)
    train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.num_workers)
    vali_set=read_json(cfg.dev_data_path)
    vali_dataset=ClassDataset(vali_set,tokenizer)
    vali_loader=DataLoader(vali_dataset,batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.num_workers)
    train_seqcls(model, train_dataset, train_loader, vali_loader,device, cfg)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
if __name__ == "__main__":
    print("Hello,Train")
    main()
