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
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from functools import partial
import math
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
        if device == 0:
            result = "cuda:0"
            print("We will use {}".format(result))
        elif device == 1:
            result = "cuda:1"
            print("We will use {}".format(result))
        elif device == 2:
            result = "cuda:2"
            print("We will use {}".format(result))
        elif device == 3:
            result = "cuda:3"
            print("We will use {}".format(result))
    return result

def create_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    #console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def get_custom_linear_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_schedule_with_warmup_cosine_lr_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int, num_decay_steps: int,min_lr: float):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step < num_warmup_steps + num_decay_steps:
        return 1.0 - float(current_step - num_warmup_steps) / float(max(1, num_decay_steps))
    else:
        # Cosine annealing
        progress = float(current_step - num_warmup_steps - num_decay_steps) / float(max(1, num_training_steps - num_warmup_steps - num_decay_steps))
        return max(min_lr,0.5 * (1.0 + math.cos(math.pi * progress)))

def get_custom_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_decay_steps: int, min_lr: float=1e-6,last_epoch: int = -1):
    lr_lambda = partial(
        _get_schedule_with_warmup_cosine_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_decay_steps=num_decay_steps,
        min_lr=min_lr,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

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
            data = json.loads(line.strip())  
            dataset.append(data)  
    return dataset

def model_paramters_num(model):
    return sum(param.numel() for param in model.parameters())

    with open(path, 'wb') as fil:
        pickle.dump(en, fil)

def compute_accuracy_qa(pred_start, pred_end, true_start, true_end):
    pred_start=pred_start.to('cpu')
    pred_end=pred_end.to('cpu')
    true_start=true_start.squeeze(1)
    true_end=true_end.squeeze(1)
    pred_start=F.softmax(pred_start,dim=1)
    pred_end=F.softmax(pred_end,dim=1)
    pred_start_idx = pred_start.argmax(dim=1)#tensor([3, 3, 0]) 
    pred_end_idx = pred_end.argmax(dim=1)
    correct_start = pred_start_idx == true_start
    correct_end = pred_end_idx == true_end
    correct = correct_start & correct_end
    accuracy = correct.float().mean().item()
    return accuracy

def compute_accuracy_mlc(pred_logits,true_logits):
    pred_logits = pred_logits.to('cpu')
    true_logits = true_logits.to('cpu')
    pred_logits=F.softmax(pred_logits,dim=1)
    pred_logits = pred_logits.argmax(dim=1) 
    correct = pred_logits == true_logits
    accuracy = correct.float().mean().item()
    return accuracy

def compute_accuracy_mlc2(pred_logits, true_logits):
    pred_logits = pred_logits.to('cpu')
    true_logits = true_logits.to('cpu')
    pred_probs = F.softmax(pred_logits, dim=1)
    print("Softmax probabilities:\n", pred_probs)
    pred_classes = pred_probs.argmax(dim=1)
    correct = pred_classes == true_logits
    accuracy = correct.float().mean().item()
    return accuracy, pred_probs

def compute_accuracy_yn(pred_logits,true_logits):
    pred_logits = pred_logits.to('cpu')
    true_logits = true_logits.to('cpu')
    true_logits=true_logits.squeeze(1)
    pred_logits=F.softmax(pred_logits,dim=1)
    pred_logits = pred_logits.argmax(dim=1) 
    correct = pred_logits == true_logits
    accuracy = correct.float().mean().item()
    return accuracy

def compute_micro_f1(pred_logits, true_logits):
    pred_logits = pred_logits.to('cpu')
    true_logits = true_logits.to('cpu')
    pred_probs = torch.sigmoid(pred_logits)
    pred_labels = (pred_probs >= 0.5).int()
    true_labels = true_logits.int()
    tp = torch.sum((pred_labels == 1) & (true_labels == 1))  
    fp = torch.sum((pred_labels == 1) & (true_labels == 0))  
    fn = torch.sum((pred_labels == 0) & (true_labels == 1))  
    precision = tp / (tp + fp + 1e-9)  # Add small epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-9)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return micro_f1.item()

def compute_micro_f12(pred_logits, true_logits):
    pred_logits = pred_logits.to('cpu')
    true_logits = true_logits.to('cpu')
    pred_probs = torch.sigmoid(pred_logits)
    print("Sigmoid probabilities:\n",pred_probs)
    pred_labels = (pred_probs >= 0.5).int()
    true_labels = true_logits.int()
    tp = torch.sum((pred_labels == 1) & (true_labels == 1))  
    fp = torch.sum((pred_labels == 1) & (true_labels == 0))  
    fn = torch.sum((pred_labels == 0) & (true_labels == 1))  
    precision = tp / (tp + fp + 1e-9)  # Add small epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-9)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return micro_f1.item()

def compute_sample_accuracy(pred_logits, true_logits):
    pred_logits = pred_logits.to('cpu')
    true_logits = true_logits.to('cpu')
    pred_probs = torch.sigmoid(pred_logits)
    pred_labels = (pred_probs >= 0.5).int()
    correct_per_sample = (pred_labels == true_logits.int()).all(dim=1)
    accuracy = correct_per_sample.float().mean().item()  
    return accuracy

def compute_pearson(pred_logits, true_logits):
    pred_logits = pred_logits.to('cpu').float()
    true_logits = true_logits.to('cpu').float()
    pred_logits = pred_logits.squeeze(1)
    true_logits = true_logits.squeeze(1)
    mean_pred = torch.mean(pred_logits)
    mean_true = torch.mean(true_logits)
    cov = torch.mean((pred_logits - mean_pred) * (true_logits - mean_true))
    std_pred = torch.std(pred_logits)
    std_true = torch.std(true_logits)
    pearson_corr = cov / (std_pred * std_true)
    return pearson_corr.item()

def model_load_dict(model):
    state_dict = torch.load('/home/chenzhichen/projects/medical/save_model/best_QA_model/Prompt_QAModel.pth', map_location='cuda:1')
    new_state_dict = {}
    for key, value in state_dict.items():
        if key not in ['qa_outputs.weight', 'qa_outputs.bias']:
           new_state_dict[key] = value
    if 'qa_outputs.original_module.weight' not in new_state_dict:
       original_module_weight_shape = model.pre_model.qa_outputs.original_module.weight.shape
       new_state_dict['qa_outputs.original_module.weight'] = torch.randn(original_module_weight_shape)
    if 'qa_outputs.original_module.bias' not in new_state_dict:
       original_module_bias_shape = model.pre_model.qa_outputs.original_module.bias.shape
       new_state_dict['qa_outputs.original_module.bias'] = torch.randn(original_module_bias_shape)
    if 'qa_outputs.modules_to_save.default.weight' not in new_state_dict:
       default_weight_shape = model.pre_model.qa_outputs.modules_to_save.default.weight.shape
       new_state_dict['qa_outputs.modules_to_save.default.weight'] = torch.randn(default_weight_shape)
    if 'qa_outputs.modules_to_save.default.bias' not in new_state_dict:
       default_bias_shape = model.pre_model.qa_outputs.modules_to_save.default.bias.shape
       new_state_dict['qa_outputs.modules_to_save.default.bias'] = torch.randn(default_bias_shape)
    model.pre_model.load_state_dict(new_state_dict, strict=False)
    return model

class Train_ClassDataset(Dataset):
    #max_topic_len check?
    def __init__(self, data, tokenizer,task_type,max_mlc_seq_len = 512,max_scl_seq_len=512,max_qa_seq_len=512):
        super(ClassDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_mlc_seq_len=max_mlc_seq_len
        self.max_scl_seq_len=max_scl_seq_len
        self.max_qa_seq_len=max_qa_seq_len
        self.task_type=task_type
    
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

    def __getitem__(self, idx):
        if self.task_type == "QA":
            return self._get_item_qa(idx)
        elif self.task_type == "multichoice1":
            return self._get_item_multichoice1(idx)
        elif self.task_type == "multichoice2":
            return self._get_item_multichoice2(idx)
        elif self.task_type == "multichoice3":
            return self._get_item_multichoice3(idx)
        elif self.task_type == "multichoice4":
            return self._get_item_multichoice4(idx)
        elif self.task_type == "seq_cls1":
            return self._get_item_seq_cls1(idx)
        elif self.task_type == "seq_cls2":
            return self._get_item_seq_cls2(idx)
        elif self.task_type == "seq_cls3":
            return self._get_item_seq_cls3(idx)
        elif self.task_type == "seq_cls4":
            return self._get_item_seq_cls4(idx)
        elif self.task_type == "seq_cls5":
            return self._get_item_seq_cls5(idx)
        elif self.task_type == "seq_cls6":
            return self._get_item_seq_cls6(idx)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def _get_item_qa(self, idx): #QA
        sample = self.data[idx]
        question = sample['question']
        context=sample['context']
        answer=sample['answer']
        answer_ids=self.tokenizer.encode(answer,add_special_tokens=False)
        inputs_ids=self.tokenizer.encode(question,context,add_special_tokens=True)
        inputs=self.tokenizer(question,context,truncation=True,max_length=self.max_qa_seq_len,padding='max_length',return_tensors='pt')
        start=self.find_index(inputs_ids,answer_ids)
        end=start+len(answer_ids)-1
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        start=torch.tensor([start])
        end=torch.tensor([end])
        return inputs,start,end
    
    def choose_qprompt(self,meta_info):
      step1_prompt = ("Focus on the core medical knowledge relevant to the condition described in the question. "
                      "Answer based on the most common causes, treatments, or physiological mechanisms involved, without introducing external factors or assumptions. ")
        
      step2_3_prompt = ("Consider the multiple clinical factors and their interconnections in the patient's condition. "
                        "Focus on the most effective treatment strategies, based on current evidence and clinical guidelines, while accounting for potential outcomes and risks. ")
      if meta_info == "step1":
        return step1_prompt
      elif meta_info == "step2&3":
        return step2_3_prompt
      else:
        raise ValueError("Invalid meta_info value. Expected 'step1' or 'step2&3'.") 

    def choose_option1(self,question):
      if ("old" in question.lower()) and ("diagnosis" in question.lower() or "treatment" in question.lower()):
        return f"This is the most accurate diagnosis or treatment for the patient's case: "
      elif ("old" in question.lower()) and re.search(r'\b(caused|cause|causes|prevent|prevention|prevented|prevents)\b', question.lower()):
        return f"This is the most accurate prevention or explanation for the patient's case: "
      elif "old" in question.lower():
        return f"This is the most accurate answer for the patient's case: "
      elif "likely" in question.lower() or "most probable" in question.lower():
        return f"This is the most likely answer: "
      else:
        return f"This is the most accurate answer for medical reasoning: "
      
    def _get_item_multichoice1(self, idx): #multichoice medqa
        sample = self.data[idx]
        question=sample['question']
        options=list(sample['options'].values())
        meta_info=sample['meta_info']
        questions=[question]*5
        # inputs=self.tokenizer(questions,options,truncation=True,max_length=self.max_mlc_seq_len,padding='max_length',return_tensors='pt')
        
        questions_with_prompts = [
        f"{self.choose_qprompt(meta_info)}{question}." 
        for question in questions
        ]
        options_with_prompts = [
        f"{self.choose_option1(question)}{option}."
        for option in options
        ]
        
        inputs = self.tokenizer(questions_with_prompts, options, truncation=True, max_length=self.max_mlc_seq_len, padding='max_length', return_tensors='pt')
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,'E': 4}
        label=label_map[sample['answer_idx']]
        label=torch.tensor([label])
        return inputs,label
    
    def _get_item_multichoice2(self, idx): #multichoice mmlu 
        sample = self.data[idx]
        question=sample['question']
        options = sample['options']
        label=sample['label']
        questions=[question]*4
        # inputs=self.tokenizer(questions,options,truncation=True,max_length=self.max_mlc_seq_len,padding='max_length',return_tensors='pt')
        prompt1="Analyze the patient's symptoms and medical history carefully to determine the most likely diagnosis or next step in management. Focus on direct evidence provided in the case and prioritize medically validated causes while avoiding assumptions or irrelevant associations. "
        prompt2="Carefully evaluate each option based on the medical evidence and the patient's clinical context. Prioritize the options that directly address the symptoms and history while avoiding choices that introduce assumptions or rely on indirect associations. "
        questions_with_prompts = [
        f"{prompt1}{question}." 
        for question in questions
        ]
        options_with_prompts=[
        f"{prompt2}{option}." 
        for option in options
        ]
        # inputs = self.tokenizer(questions_with_prompts, options, truncation=True, max_length=self.max_mlc_seq_len, padding='max_length', return_tensors='pt')
        inputs = self.tokenizer(questions, options_with_prompts, truncation=True, max_length=self.max_mlc_seq_len, padding='max_length', return_tensors='pt')
        # inputs = self.tokenizer(questions_with_prompts, options_with_prompts, truncation=True, max_length=self.max_mlc_seq_len, padding='max_length', return_tensors='pt')
        label=torch.tensor([label])
        return inputs,label

    def _get_item_multichoice3(self, idx): #multichoice medqa prompt
        sample = self.data[idx]
        question=sample['question']
        options=list(sample['options'].values())
        questions=[question]*5
        prompt=sample["prompt"]
        questions_with_prompts = [
        f"{prompt} {question}." 
        for question in questions
        ]
        inputs = self.tokenizer(questions_with_prompts, options, truncation=True, max_length=self.max_mlc_seq_len, padding='max_length', return_tensors='pt')
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,'E': 4}
        label=label_map[sample['answer_idx']]
        label=torch.tensor([label])
        return inputs,label

    def _get_item_multichoice4(self, idx): #multichoice mmlu prompt
        sample = self.data[idx]
        question=sample['question']
        options = sample['options']
        label=sample['label']
        prompt=sample['prompt']
        questions=[question]*4
        questions_with_prompts = [
        f"{prompt} {question}." 
        for question in questions
        ]
        inputs = self.tokenizer(questions_with_prompts, options, truncation=True, max_length=self.max_mlc_seq_len, padding='max_length', return_tensors='pt')
        label=torch.tensor([label])
        return inputs,label

    def _get_item_seq_cls1(self, idx): #seq_cls,single_label_classification
        sample = self.data[idx]
        question=sample['sentence1']
        context=sample['sentence2']
        answer=sample['label']
        inputs=self.tokenizer(question,context,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label_map = {'no': 0, 'yes': 1}
        # label_map = {'no': 0, 'yes': 1,'maybe':2}
        label=label_map[answer]
        label=torch.tensor([label])
        return inputs,label

    def _get_item_seq_cls2(self, idx): #seq_cls,single_label_classification,prompt
        sample = self.data[idx]
        question=sample['sentence1']
        context=sample['sentence2']
        answer=sample['label']
        prompt=sample['prompt']
        context_with_prompts=prompt+' '+context
        # context_with_prompts1="Identify the most relevant entities and relationships in the context. Carefully integrate multiple pieces of evidence to answer the question, ensuring that the conclusion follows logically from the data. "+context
        # context_with_prompts2='Step 1: Identify the most relevant entities and relationships in the context. Step 2: Integrate these relationships to reason through the evidence. Step 3: Answer the question based on the supported evidence. '+context
        # context_with_prompts3="Identify the key entities and relationships in the provided context. Integrate these relationships logically to answer the question, ensuring that all pieces of evidence support the conclusion without relying on prior knowledge. "+context
        # context_with_prompts4="Identify the key entities and relationships in the provided context. Integrate these relationships logically, ensuring that they form a valid causal chain rather than mere correlations. Use only the explicit evidence provided in the text, and ensure that the reasoning steps are logically connected and supported by the data, avoiding assumptions or prior knowledge about cause and effect. "+context
        # context_with_prompts5="Focus on the core evidence and avoid irrelevant associations. Determine the most accurate answer based only on the direct relationships and evidence in the context." "Examine the evidence for clear causal relationships and determine the answer based on the strongest causal chains."
        # context_with_prompts6="From the context in the second part, determine the most accurate answer to the question in the first part. Base your conclusion only on the evidence provided, without assuming additional knowledge."
        inputs=self.tokenizer(question, context_with_prompts,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label_map = {'no': 0, 'yes': 1}
        # label_map = {'no': 0, 'yes': 1,'maybe':2}
        label=label_map[answer]
        label=torch.tensor([label])
        return inputs,label

    def _get_item_seq_cls3(self, idx): #seq_cls,multi_label_classification
        sample = self.data[idx]
        sentence=sample['sentence']
        label=sample['label']
        inputs=self.tokenizer(sentence,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label=torch.tensor(label,dtype=torch.float)
        return inputs,label
    
    def _get_item_seq_cls4(self, idx): #seq_cls,multi_label_classification,prompt
        sample = self.data[idx]
        sentence=sample['sentence']
        label=sample['label']
        prompt=sample['prompt']
        sentence_with_prompts=prompt+' '+sentence
        # sentence_with_prompts2='Analyze the following sentence for biological mechanisms and determine how they align with specific hallmarks of cancer. Focus on evidence-based processes described in the text, avoiding assumptions based on unrelated terms or high-frequency words.'+' '+sentence
        # sentence_with_prompts3='Identify which hallmarks of cancer this text describes based on the biological mechanisms and processes mentioned.'+' '+sentence
        inputs=self.tokenizer(sentence_with_prompts,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label=torch.tensor(label,dtype=torch.float)
        return inputs,label
    
    def _get_item_seq_cls5(self, idx): #seq_cls,regression,prompt
        sample = self.data[idx]
        sentence1=sample['sentence1']
        sentence2=sample['sentence2']
        label=sample['label']
        sentence2_with_prompts1="Compare the original sentence with the simplified version. Focus on the key scientific facts and relationships, and assess how much information is preserved. Do not introduce assumptions beyond what is explicitly stated in the sentences. "+sentence2
        sentence2_with_prompts2='Compare the following two sentences and assess how similar they are in terms of the core biological concepts and mechanisms discussed. Focus on the key scientific details and relationships while ignoring superficial differences. Your task is to determine the degree of similarity based on the content provided, not based on external or prior knowledge. '+sentence2
        inputs=self.tokenizer(sentence1,sentence2_with_prompts2,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label=torch.tensor([label])
        return inputs,label

    def _get_item_seq_cls6(self, idx): #seq_cls,single_label_classification,prompt,ablation
        sample = self.data[idx]
        question=sample['sentence1']
        context=sample['sentence2']
        answer=sample['label']
        context_with_prompts1="Focus on the direct relationships within the context and avoid irrelevant associations. Identify the evidence supporting clear causal relationships and use it to determine the most accurate answer. "+context
        context_with_prompts2="Carefully examine the context for direct causal relationships. Focus only on the evidence that clearly establishes cause and effect, and disregard any irrelevant associations or misleading correlations. Your answer should be based strictly on the strongest causal evidence present in the context. "+context
        context_with_prompts3="In the context provided, identify the direct causal links that lead to the answer. Do not rely on statistical correlations or unrelated associations. Focus on the cause-effect relationships that are explicitly supported by the evidence, and base your conclusion solely on those. "+context
        context_with_prompts4='Based solely on the evidence in the second part, determine the most accurate answer to the question. The provided context contains all necessary information, so do not incorporate any external knowledge. '+context
        context_with_prompts5="Only use the information provided in the context to answer the question. Do not rely on any external knowledge or prior information beyond what is explicitly mentioned in the context."
        inputs=self.tokenizer(question, context_with_prompts5,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label_map = {'no': 0, 'yes': 1}
        # label_map = {'no': 0, 'yes': 1,'maybe':2}
        label=label_map[answer]
        label=torch.tensor([label])
        return inputs,label

class Test_ClassDataset(Dataset):
    def __init__(self, data, tokenizer,task_type,max_mlc_seq_len = 512,max_scl_seq_len=512,max_qa_seq_len=512):
        super(Test_ClassDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_ques_len = max_ques_len
        self.max_topic_len = max_topic_len
        self.max_seq_len = max_seq_len

    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.task_type == "QA":
            return self._get_item_qa(idx)
        elif self.task_type == "medqa":
            return self._get_item_multichoice1(idx)
        elif self.task_type == "mmlu":
            return self._get_item_multichoice2(idx)
        elif self.task_type == "bioasq":
            return self._get_item_seq_cls1(idx)
        elif self.task_type == "pubmedqa":
            return self._get_item_seq_cls2(idx)
        elif self.task_type == "hoc":
            return self._get_item_seq_cls3(idx)
        elif self.task_type == "biosess":
            return self._get_item_seq_cls4(idx)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
    
     def _get_item_multichoice1(self, idx): #multichoice medqa prompt
        sample = self.data[idx]
        question=sample['question']
        options=list(sample['options'].values())
        questions=[question]*5
        prompt=sample["prompt"]
        questions_with_prompts = [
        f"{prompt} {question}." 
        for question in questions
        ]
        inputs = self.tokenizer(questions_with_prompts, options, truncation=True, max_length=self.max_mlc_seq_len, padding='max_length', return_tensors='pt')
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,'E': 4}
        label=label_map[sample['answer_idx']]
        label=torch.tensor([label])
        return inputs,label

    def _get_item_multichoice2(self, idx): #multichoice mmlu prompt
        sample = self.data[idx]
        question=sample['question']
        options = sample['options']
        label=sample['label']
        prompt=sample['prompt']
        questions=[question]*4
        questions_with_prompts = [
        f"{prompt} {question}." 
        for question in questions
        ]
        inputs = self.tokenizer(questions_with_prompts, options, truncation=True, max_length=self.max_mlc_seq_len, padding='max_length', return_tensors='pt')
        label=torch.tensor([label])
        return inputs,label

     def _get_item_seq_cls1(self, idx): #bioasq,prompt
        sample = self.data[idx]
        question=sample['sentence1']
        context=sample['sentence2']
        answer=sample['label']
        prompt=sample['prompt']
        context_with_prompts=prompt+' '+context
        inputs=self.tokenizer(question, context_with_prompts,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label_map = {'no': 0, 'yes': 1}
        label=label_map[answer]
        label=torch.tensor([label])
        return inputs,label

     def _get_item_seq_cls2(self, idx): #pubmedqa,prompt
        sample = self.data[idx]
        question=sample['sentence1']
        context=sample['sentence2']
        answer=sample['label']
        prompt=sample['prompt']
        context_with_prompts=prompt+' '+context
        inputs=self.tokenizer(question, context_with_prompts,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label_map = {'no': 0, 'yes': 1,'maybe':2}
        label=label_map[answer]
        label=torch.tensor([label])
        return inputs,label
    
    def _get_item_seq_cls3(self, idx): #hoc,prompt
        sample = self.data[idx]
        sentence=sample['sentence']
        label=sample['label']
        prompt=sample['prompt']
        sentence_with_prompts=prompt+' '+sentence
        inputs=self.tokenizer(sentence_with_prompts,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label=torch.tensor(label,dtype=torch.float)
        return inputs,label
    
    def _get_item_seq_cls5(self, idx): #biosess,prompt
        sample = self.data[idx]
        sentence1=sample['sentence1']
        sentence2=sample['sentence2']
        label=sample['label']
        sentence2_with_prompts1="Compare the original sentence with the simplified version. Focus on the key scientific facts and relationships, and assess how much information is preserved. Do not introduce assumptions beyond what is explicitly stated in the sentences. "+sentence2
        sentence2_with_prompts2='Compare the following two sentences and assess how similar they are in terms of the core biological concepts and mechanisms discussed. Focus on the key scientific details and relationships while ignoring superficial differences. Your task is to determine the degree of similarity based on the content provided, not based on external or prior knowledge. '+sentence2
        inputs=self.tokenizer(sentence1,sentence2_with_prompts2,truncation=True,max_length=self.max_scl_seq_len,padding='max_length',return_tensors='pt')
        inputs['input_ids']=inputs['input_ids'].squeeze(0)
        inputs['attention_mask']=inputs['attention_mask'].squeeze(0)
        inputs['token_type_ids']=inputs['token_type_ids'].squeeze(0)
        label=torch.tensor([label])
        return inputs,label
    