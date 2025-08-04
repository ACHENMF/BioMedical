from transformers import AutoTokenizer,AutoConfig,AutoModelForMultipleChoice,AutoModelForSequenceClassification,AutoModelForQuestionAnswering
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PrefixTuningConfig,get_peft_model,TaskType
class Prompt_Model(nn.Module):
    #max_topic_len check?
    def __init__(self, model_name,type):
        super(Prompt_Model, self).__init__()
        if type==0:
            #seq_cls single_label_classification
            config=AutoConfig.from_pretrained(model_name,hidden_dropout_prob=0.3,attention_probs_dropout_prob=0.3,problem_type = "single_label_classification",num_labels=2)
            self.model=AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
            # for name, param in self.model.named_parameters():
            #     if 'bert.encoder.layer.10.' in name or 'bert.encoder.layer.11.' in name or 'classifier' in name:
            #         param.requires_grad = True  
            #     else:
            #         param.requires_grad = False  
        if type==1:
            #seq_cls regression
            config=AutoConfig.from_pretrained(model_name,hidden_dropout_prob=0.3,attention_probs_dropout_prob=0.3,problem_type = "regression",num_labels=1)
            self.model=AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
        if type==2:
            #seq_cls multi_label_classification
            config=AutoConfig.from_pretrained(model_name,hidden_dropout_prob=0.3,attention_probs_dropout_prob=0.3,problem_type = "multi_label_classification",num_labels=10)
            self.model=AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
        if type==3:
            #mlc
            config=AutoConfig.from_pretrained(model_name,hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1)
            self.model=AutoModelForMultipleChoice.from_pretrained(model_name,config=config)
            # for name, param in self.model.named_parameters():
            #     if 'bert.encoder.layer.10.' in name or 'bert.encoder.layer.11.' in name or 'classifier' in name:
            #         param.requires_grad = True  
            #     else:
            #         param.requires_grad = False  
        if type==4:
            #Y/N,prefix
            config=AutoConfig.from_pretrained(model_name,hidden_dropout_prob=0.3,attention_probs_dropout_prob=0.3,problem_type = "single_label_classification",num_labels=2)
            pre_model=AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
            prefix_config=PrefixTuningConfig(task_type=TaskType.SEQ_CLS,num_virtual_tokens=20,prefix_projection=True)
            self.model=get_peft_model(model=pre_model,peft_config=prefix_config)
            for param in self.model.parameters():
                param.requires_grad=True
        if type==5:
            #QA,pv2
            config=AutoConfig.from_pretrained(model_name,hidden_dropout_prob=0.3,attention_probs_dropout_prob=0.3)
            pre_model=AutoModelForQuestionAnswering.from_pretrained(model_name,config=config)
            prefix_config=PrefixTuningConfig(task_type=TaskType.QUESTION_ANS,num_virtual_tokens=10,prefix_projection=False)
            self.model=get_peft_model(model=pre_model,peft_config=prefix_config)
            for param in self.model.parameters():
                param.requires_grad=True 

    def generate_train_inputs(self, batch, topic_embed, start,end,device):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        # input_ids = input_ids.squeeze(1)
        # attention_mask = attention_mask.squeeze(1)
        # token_type_ids = token_type_ids.squeeze(1)
        topic_embed = topic_embed.to(device)
        batch_size, max_topic_length = topic_embed.shape
        extended_input_ids = torch.cat((topic_embed, input_ids), dim=1).to(device)
        one_like=torch.ones(batch_size, max_topic_length, device=device)
        zero_like=torch.zeros(batch_size, max_topic_length,device=device)
        extended_attention_mask = torch.cat((one_like, attention_mask), dim=1)
        extended_token_type_ids = torch.cat((zero_like, token_type_ids), dim=1)
        extended_attention_mask = extended_attention_mask[:, :extended_input_ids.shape[1]].to(device)
        extended_token_type_ids = extended_token_type_ids[:, :extended_input_ids.shape[1]].to(device)
        #check?        
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids,'start_positions':start.to(device),'end_positions':end.to(device)}
        return inputs

    def generate_test_inputs(self, batch, topic_embed, device):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        # input_ids = input_ids.squeeze(1)
        # attention_mask = attention_mask.squeeze(1)
        # token_type_ids = token_type_ids.squeeze(1)
        #check?        
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids}
        return inputs

    def qa_train_step(self,inputs,start, end, device):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        start=start.to(device)
        end=end.to(device)
        inputs={'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids,'start_positions':start,'end_positions':end}
        output = self.model(**inputs)
        return output
    
    def qa_test_step(self, inputs, device):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        inputs={'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids}
        output = self.model(**inputs)
        return output
    
    def mlc_train_step(self,inputs,labels,device):#multichoice
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        labels=labels.to(device)
        inputs={'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids,'labels':labels}
        output=self.model(**inputs)
        return output
    
    def mlc_test_step(self,inputs,device):#multichoice
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        inputs={'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids}
        output=self.model(**inputs)
        return output
    
    def scl_train_step(self,inputs,labels,device):#seqcls
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        labels=labels.to(device)
        inputs={'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids,'labels':labels}
        output=self.model(**inputs)
        return output
    
    def scl_test_step(self,inputs,device):#seqcls
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        inputs={'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids}
        output=self.model(**inputs)
        return output
    



    
