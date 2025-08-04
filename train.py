import numpy as np
import torch.nn as nn
import torch
from torch.optim import AdamW,SGD
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer,AutoTokenizer
import CONFIG
import os
from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import ysy_util
from ysy_util import Train_ClassDataset,create_logger,read_json,compute_accuracy_yn,compute_accuracy_qa,compute_accuracy_mlc,get_custom_linear_schedule_with_warmup,model_load_dict,get_custom_schedule_with_warmup,compute_micro_f1,compute_pearson,set_random_seed,compute_accuracy_mlc2,compute_micro_f12
from model import Prompt_Model
logger = None

def train_QA(model,train_dataset,train_loader,vali_loader,device,cfg,task_type):                                                                                                                                                                                                       
    total_steps = int(train_dataset.__len__() * cfg.epochs / cfg.batch_size)
    logger.info("We will process {} steps.".format(total_steps))
    no_decay = ['bias', 'LayerNorm.weight']
    #weight_decay check?=0.0001
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
    #     {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    # ]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=cfg.lr)
    #optimizer = AdamW(params=optimizer_grouped_parameters, lr=cfg.lr,betas=(0.9, 0.999),eps=1e-8,amsgrad=False)
    #optimizer = SGD(params=optimizer_grouped_parameters,lr=cfg.lr,momentum=0.9,dampening=0.0,nesterov=True)
    #scheduler = get_custom_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    scheduler= get_custom_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,num_decay_steps=cfg.num_decay_steps)
    logger.info("starting training.")
    rank={}
    best_accuracy = 0.0  
    best_model_path = None  
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        total_accuracy=0
        total_batch=0
        optimizer.zero_grad()
        model.train()
        for batch_idx, sample in enumerate(train_loader):
            inputs=sample[0]
            start=sample[1]
            end=sample[2]
            outputs= model.qa_train_step(inputs,start,end,device)
            # start_position=start.squeeze(1).to(device)
            # end_position=end.squeeze(1).to(device)
            # ignore_idx = outputs['start_logits'].size(1)
            # start_position=start_position.clamp_(0, ignore_idx)
            # end_position=end_position.clamp_(0, ignore_idx)
            # loss_fct =CrossEntropyLoss(ignore_index=ignore_idx)
            # start_loss=loss_fct(outputs['start_logits'],start_position)
            # end_loss=loss_fct(outputs['end_logits'],end_position)
            # loss=(start_loss+end_loss)/2.0
            loss=outputs.loss
            loss=loss/cfg.gradient_accumulation
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.model.parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if (batch_idx + 1) % cfg.gradient_accumulation == 0:
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
                start=sample[1]
                end=sample[2]
                vali_outputs=model.qa_test_step(inputs,device)
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
        if task_type==0:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_QA_model')
        elif task_type==1:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_origin_QA_model')
        elif task_type==2:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_pv2_QA_model')
        elif task_type==3:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prefix_QA_model')
        elif task_type==4:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_ori_pv2_QA_model')
        elif task_type==5:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_ori_prefix_QA_model')
        if overal_accuracy > best_accuracy:
           best_accuracy = overal_accuracy
           model_file_path = best_model_path + '/Prompt_QAModel.pth'
           qa_outputsfile_path=best_model_path+'/qa_outputs.pth'
           if os.path.exists(model_file_path):
                os.remove(model_file_path)
           if os.path.exists(qa_outputsfile_path):
                os.remove(qa_outputsfile_path)
           if not os.path.exists(best_model_path):
               os.mkdir(best_model_path)
           #torch.save(model.model.state_dict(), best_model_path + '/Prompt_QAModel.pth')
           torch.save(model.state_dict(), best_model_path + '/Prompt_QAModel.pth')
        #    qa_outputs_weight = model.model.qa_outputs.weight
        #    qa_outputs_bias = model.model.qa_outputs.bias
        #    torch.save({'qa_outputs.weight':qa_outputs_weight,'qa_outputs.bias':qa_outputs_bias},best_model_path+'/qa_outputs.pth')
           logger.info("New best model saved with accuracy: {}".format(best_accuracy))
        epoch_finish_time = datetime.now()
        logger.info("time for epoch{} : {}".format(epoch + 1,epoch_finish_time - epoch_start_time))
    logger.info("Train Finished ")
    logger.info("Best Accuracy: {}".format(best_accuracy))
    logger.info("Train Rank:{}".format(rank))
    print(rank)

def train_MultiChoice(model,train_dataset,train_loader,vali_loader,device,cfg,task_type):
    total_steps = int(train_dataset.__len__() * cfg.epochs / (cfg.batch_size*cfg.gradient_accumulation))
    logger.info("We will process {} steps.".format(total_steps))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.1},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    # optimizer_grouped_parameters = [
    # {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
    #  'weight_decay': 0.001},
    # {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    #  'weight_decay': 0.0}
    # ]
    # optimizer_grouped_parameters = []
    # no_decay_params = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad]
    # decay_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad]

    # optimizer_grouped_parameters.append(
    # {'params': [p for n, p in model.named_parameters() if ('bert.encoder.layer.10' in n or 'bert.encoder.layer.11' in n) and any(id(p) == id(dp) for dp in decay_params)],
    #  'weight_decay': 0.01, 'lr': 5e-5}
    # )
    # optimizer_grouped_parameters.append(
    # {'params': [p for n, p in model.named_parameters() if ('bert.encoder.layer.10' in n or 'bert.encoder.layer.11' in n) and any(id(p) == id(ndp) for ndp in no_decay_params)],
    #  'weight_decay': 0.0, 'lr': 5e-5}
    # )

    # optimizer_grouped_parameters.append(
    # {'params': [p for n, p in model.named_parameters() if ('bert.encoder.layer.6.' in n or 'bert.encoder.layer.7.' in n or 'bert.encoder.layer.8.' in n or 'bert.encoder.layer.9.' in n) and any(id(p) == id(dp) for dp in decay_params)],
    #  'weight_decay': 0.01, 'lr': 3e-5}
    # )
    # optimizer_grouped_parameters.append(
    # {'params': [p for n, p in model.named_parameters() if ('bert.encoder.layer.6.' in n or 'bert.encoder.layer.7.' in n or 'bert.encoder.layer.8.' in n or 'bert.encoder.layer.9.' in n) and any(id(p) == id(ndp) for ndp in no_decay_params)],
    #  'weight_decay': 0.0, 'lr': 3e-5}
    # )
 
    # optimizer_grouped_parameters.append(
    # {'params': [p for n, p in model.named_parameters() if ('bert.encoder.layer.0.' in n or 'bert.encoder.layer.1.' in n or 'bert.encoder.layer.2.' in n or 'bert.encoder.layer.3.' in n or 'bert.encoder.layer.4.' in n or 'bert.encoder.layer.5.' in n) and any(id(p) == id(dp) for dp in decay_params)],
    #  'weight_decay': 0.01, 'lr': 1e-5}
    # )
    # optimizer_grouped_parameters.append(
    # {'params': [p for n, p in model.named_parameters() if ('bert.encoder.layer.0.' in n or 'bert.encoder.layer.1.' in n or 'bert.encoder.layer.2.' in n or 'bert.encoder.layer.3.' in n or 'bert.encoder.layer.4.' in n or 'bert.encoder.layer.5.' in n) and any(id(p) == id(ndp) for ndp in no_decay_params)],
    #  'weight_decay': 0.0, 'lr': 1e-5}
    # )
 
    # optimizer_grouped_parameters.append(
    # {'params': [p for n, p in model.named_parameters() if ('bert.embeddings' in n or 'classifier' in n or 'bert.pooler' in n) and any(id(p) == id(dp) for dp in decay_params)],
    #  'weight_decay': 0.01, 'lr': 1e-6}
    # )
    # optimizer_grouped_parameters.append(
    # {'params': [p for n, p in model.named_parameters() if ('bert.embeddings' in n or 'classifier' in n or 'bert.pooler' in n) and any(id(p) == id(ndp) for ndp in no_decay_params)],
    #  'weight_decay': 0.0, 'lr': 1e-6}
    # )
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=cfg.lr)
    #optimizer = AdamW(params=optimizer_grouped_parameters)
    #optimizer = AdamW(params=optimizer_grouped_parameters, lr=cfg.lr,betas=(0.9, 0.98),eps=1e-8,amsgrad=False)
    #optimizer = SGD(params=optimizer_grouped_parameters,lr=cfg.lr,momentum=0.9,dampening=0.0,nesterov=True)
    #scheduler = get_custom_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    scheduler= get_custom_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,num_decay_steps=cfg.num_decay_steps,min_lr=1e-6)
    logger.info("starting training.")
    rank={}
    best_accuracy = 0.0  
    best_model_path = None  
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        total_accuracy=0
        total_batch=0
        optimizer.zero_grad()
        model.train()
        for batch_idx, sample in enumerate(train_loader):
            inputs=sample[0]
            labels=sample[1].squeeze(1)
            outputs= model.mlc_train_step(inputs,labels,device)
            loss=outputs.loss/cfg.gradient_accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if (batch_idx + 1) % cfg.gradient_accumulation == 0:
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
                # batch_accuracy,pred_probs=compute_accuracy_mlc2(vali_outputs.logits,labels)
                total_accuracy+=batch_accuracy
                total_batch+=1
                logger.info("batch {}/{}".format(batch_idx + 1, vali_loader.__len__()))
        overal_accuracy=total_accuracy/total_batch
        epoch_id='epoch{}'.format(epoch+1)
        rank[epoch_id]=overal_accuracy
        logger.info("epoch {} ,accuracy:{}.".format(epoch + 1,overal_accuracy))
        logger.info("epoch {} validated finished.".format(epoch + 1))
        if task_type==0:
           best_model_path = os.path.join(cfg.saved_model_path, 'best_ori_MLC_model')
        elif task_type==1:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prompt1_ori_MLC_model')
        elif task_type==2:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prompt2_ori_MLC_model')
        elif task_type==3:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prompt3_ori_MLC_model')
        elif task_type==4:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prompt4_ori_MLC_model')
        if overal_accuracy > best_accuracy:
           best_accuracy = overal_accuracy
           model_file_path = best_model_path + '/Prompt_MlcModel.pth'
           if os.path.exists(model_file_path):
                os.remove(model_file_path)
           if not os.path.exists(best_model_path):
               os.mkdir(best_model_path)
           torch.save(model.state_dict(), best_model_path + '/Prompt_MlcModel.pth')
           logger.info("New best model saved with accuracy: {}".format(best_accuracy))
        epoch_finish_time = datetime.now()
        logger.info("time for epoch{} : {}".format(epoch + 1,epoch_finish_time - epoch_start_time))
    logger.info("Train Finished ")
    logger.info("Best Accuracy: {}".format(best_accuracy))
    logger.info("Train Rank:{}".format(rank))
    print(rank)

def train_seqcls(model,train_dataset,train_loader,vali_loader,device,cfg,task_type):
    total_steps = int(train_dataset.__len__() * cfg.epochs / (cfg.batch_size*cfg.gradient_accumulation))
    logger.info("We will process {} steps.".format(total_steps))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    # optimizer_grouped_parameters = [
    # {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
    #  'weight_decay': 0.05},
    # {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    #  'weight_decay': 0.0}
    # ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=cfg.lr)
    #optimizer = AdamW(params=optimizer_grouped_parameters, lr=cfg.lr,betas=(0.9, 0.98),eps=1e-8,amsgrad=False)
    #optimizer = SGD(params=optimizer_grouped_parameters,lr=cfg.lr,momentum=0.9,dampening=0.0,nesterov=True)
    #scheduler = get_custom_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    scheduler= get_custom_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,num_decay_steps=cfg.num_decay_steps,min_lr=1e-6)
    logger.info("starting training.")
    rank={}
    best_accuracy = 0.0  
    best_model_path = None 
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        total_accuracy=0
        total_batch=0
        optimizer.zero_grad()
        model.train()
        for batch_idx, sample in enumerate(train_loader):
            inputs=sample[0]
            labels=sample[1]
            outputs= model.scl_train_step(inputs,labels,device)
            loss=outputs.loss/cfg.gradient_accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if (batch_idx + 1) % cfg.gradient_accumulation == 0:
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
        if task_type==0:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_ori_SQL_model')
        elif task_type==1:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prompt1_SQL_model')
        elif task_type==2:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prompt2_SQL_model')
        elif task_type==3:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prompt3_SQL_model')
        elif task_type==4:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prompt4_SQL_model')
        elif task_type==5:
            best_model_path = os.path.join(cfg.saved_model_path, 'best_prompt5_SQL_model')
        if overal_accuracy > best_accuracy:
           best_accuracy = overal_accuracy
           model_file_path = best_model_path + '/Prompt_SqlModel.pth'
           if os.path.exists(model_file_path):
               os.remove(model_file_path)
           if not os.path.exists(best_model_path):
               os.mkdir(best_model_path)
           torch.save(model.state_dict(), best_model_path + '/Prompt_SqlModel.pth')
           logger.info("New best model saved with accuracy: {}".format(best_accuracy))
        epoch_finish_time = datetime.now()
        logger.info("time for epoch{} : {}".format(epoch + 1,epoch_finish_time - epoch_start_time))
    logger.info("Train Finished ")
    logger.info("Best Accuracy: {}".format(best_accuracy))
    logger.info("Train Rank:{}".format(rank))
    print(rank)

def main():
    global logger
    cfg = CONFIG.CONFIG(3)
    device = ysy_util.device_info(0)
    logger = create_logger(cfg.log_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    model = Prompt_Model(cfg.pretrained_model_path,3)
    # model.load_state_dict(torch.load('/home/chenzhichen/projects/medical/save_model/best_prompt1_ori_MLC_model/Prompt_MlcModel.pth'))
    model.to(device)
    num_parameters = ysy_util.model_paramters_num(model)
    logger.info("number of model parameters:{}".format(num_parameters))
    train_set = read_json(cfg.train_data_path)
    train_dataset = Train_ClassDataset(train_set,tokenizer,'multichoice3')
    train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.num_workers)
    vali_set=read_json(cfg.dev_data_path)
    vali_dataset=Train_ClassDataset(vali_set,tokenizer,'multichoice3')
    vali_loader=DataLoader(vali_dataset,batch_size = cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
    # train_seqcls(model, train_dataset, train_loader, vali_loader,device, cfg,1)
    #train_QA(model, train_dataset, train_loader, vali_loader,device, cfg,5)
    train_MultiChoice(model, train_dataset, train_loader, vali_loader,device, cfg,1)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
if __name__ == "__main__":
    print("Hello,Train")
    main()
