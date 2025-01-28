import torch
import re
import pickle
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import logging
import numpy as np
import json

pad_id = None
text_len_flexible = True
text_len_stable = 50


def device_info(device):
    result = "cpu"
    if torch.cuda.is_available():
        counter = torch.cuda.device_count()
        print("There are {} GPU(s) is available.".format(counter))
        for i in range(counter):
            print("GPU {} Name:{}".format(i, torch.cuda.get_device_name(i)))
        if device == "gpu":
            result = "cuda:0"
            print("We will use {}".format(result))
    return result

def create_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    # 用于写入日志文件
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # 将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    #console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def read_txt(data_dir):
    dataset = []
    with open(data_dir, 'r', encoding="utf-8") as f:
        for data in f.readlines():
            info, start, end, ques = data.replace('\n','').split('\t')
            dataset.append([info, start, end, ques])
    return dataset

def read_json(data_dir):
    dataset = []
    with open(data_dir, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())  
            dataset.append(data)  
    return dataset

def read_jsonl(data_dir):
    dataset = []
    with open(data_dir, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())  # 解析每一行JSON数据
            dataset.append(data)  # 将数据添加到列表中
    return dataset

def model_paramters_num(model):
    return sum(param.numel() for param in model.parameters())

    with open(path, 'wb') as fil:
        pickle.dump(en, fil)

class ClassDataset(Dataset):
    #max_topic_len check?
    def __init__(self, data, tokenizer,max_seq_len = 420):
        super(ClassDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len=max_seq_len
    def __len__(self):  
        return len(self.data)
    
    def find_index(self, big, small):
        s_len = len(small)
        b_len = len(big)
        for i in range(b_len):
            if big[i] == small[0]:
                if big[i:i+s_len] == small:
                    break
        return i

    # def __getitem__(self, idx): #dataset=ClassDataset(data, tokenizer),load=dataset,dataloader(16)
    #     sample = self.data[idx]
    #     info = sample[0]
    #     start_position = sample[1].split(',').index('1')
    #     end_position = sample[2].split(',').index('1')
    #     # ori_answer = info[start_position:end_position+1]
    #     # ori_answer_ids = self.tokenizer.encode(ori_answer)[1:-1]#self.tokenizer
    #     info_text_ids = []
    #     info_text_ids.extend(self.tokenizer.encode(info)[1:-1])
    #     # start = torch.tensor([self.find_index(info_text_ids, ori_answer_ids) + self.max_ques_len + 2])
    #     # end = torch.tensor([self.find_index(info_text_ids, ori_answer_ids) + self.max_ques_len + 2 + len(ori_answer_ids) - 1])
    #     # start = start_position + self.max_ques_len + 2
    #     # end = end_position + self.max_ques_len + 2 
    #     start = torch.tensor([start_position + self.max_ques_len + 2])
    #     end = torch.tensor([end_position + self.max_ques_len + 2 ])
    #     ques = sample[3]
    #     topic=sample[3].replace('的概念是什么？', '')
    #     ques_text_ids = []
    #     ques_text_ids.extend(self.tokenizer.encode(ques)[1:-1])
    #     topic_text_ids = []
    #     topic_text_ids.extend(self.tokenizer.encode(topic)[1:-1])
    #     if len(info_text_ids) <= self.max_seq_len:
    #         info_text_ids.extend([self.tokenizer.pad_token_id for i in range(self.max_seq_len - len(info_text_ids))])
    #     if len(ques_text_ids)  <= self.max_ques_len:
    #         ques_text_ids.extend([self.tokenizer.pad_token_id for i in range(self.max_ques_len - len(ques_text_ids))])
    #     if len(topic_text_ids)  <= self.max_topic_len:
    #         topic_text_ids.extend([self.tokenizer.pad_token_id for i in range(self.max_topic_len - len(topic_text_ids))])
    #     ques_embed = torch.tensor(ques_text_ids[:self.max_ques_len])
    #     info_embed = torch.tensor(info_text_ids[:self.max_seq_len])
    #     topic_embed = torch.tensor(topic_text_ids[:self.max_topic_len])
    #     inputs = self.tokenizer(self.tokenizer.decode(ques_text_ids[:self.max_ques_len]),self.tokenizer.decode(info_text_ids[:self.max_seq_len]),return_tensors = 'pt')
    #     inputs['input_ids']=inputs['input_ids'].squeeze(0)
    #     inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
    #     inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
    #     return inputs, ques_embed,topic_embed, info_embed, start, end
    
    # def __getitem__(self, idx): #multichoice
    #     sample = self.data[idx]
    #     question=sample['question']
    #     questions=[question]*4
    #     options=list(sample['options'].values())
    #     inputs=self.tokenizer(questions,options,truncation=True,max_length=self.max_seq_len,padding='max_length',return_tensors='pt')
    #     label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    #     label=label_map[sample['answer_idx']]
    #     label=torch.tensor([label])
    #     return inputs,label
    
    def __getitem__(self, idx): #yes/no
        sample = self.data[idx]
        question=sample['question']
        context=sample['context']
        answer=sample['answer']
        inputs=self.tokenizer(question,context,truncation=True,max_length=self.max_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label_map = {'no': 0, 'yes': 1}
        label=label_map[answer]
        label=torch.tensor([label])
        return inputs,label
    

class Test_ClassDataset(Dataset):
    def __init__(self, data, tokenizer, max_topic_len=6, max_ques_len=36, max_seq_len = 256):
        super(Test_ClassDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_ques_len = max_ques_len
        self.max_topic_len = max_topic_len
        self.max_seq_len = max_seq_len

    def __len__(self):  
        return len(self.data)

    def __getitem__(self, idx):  
        sample = self.data[idx]
        info = sample[0]
        start = torch.tensor([ 0 + self.max_ques_len + 2])
        end = torch.tensor([0  + self.max_ques_len + 2])
        info_text_ids = []
        info_text_ids.extend(self.tokenizer.encode(info)[1:-1])
        if len(info_text_ids) <= self.max_seq_len:
            info_text_ids.extend([self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_seq_len - len(info_text_ids))])
        ques = sample[3]
        topic=sample[3].replace('的概念是什么？', '')
        ques_text_ids = []
        ques_text_ids.extend(self.tokenizer.encode(ques)[1:-1])
        topic_text_ids = []
        topic_text_ids.extend(self.tokenizer.encode(topic)[1:-1])
        if len(ques_text_ids)  <= self.max_ques_len:
            ques_text_ids.extend([self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_ques_len - len(ques_text_ids))])
        if len(topic_text_ids)  <= self.max_topic_len:
            topic_text_ids.extend([self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_topic_len - len(topic_text_ids))])
        ques_embed = torch.tensor(ques_text_ids[:self.max_ques_len])
        info_embed = torch.tensor(info_text_ids[:self.max_seq_len])
        topic_embed = torch.tensor(topic_text_ids[:self.max_topic_len])
        inputs = self.tokenizer(self.tokenizer.decode(ques_text_ids[:self.max_ques_len]),self.tokenizer.decode(info_text_ids[:self.max_seq_len]),return_tensors = 'pt')
        return inputs, ques_embed,topic_embed, info_embed, start, end
    