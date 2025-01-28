from transformers import AutoTokenizer,AutoConfig,AutoModelForMultipleChoice,AutoModelForSequenceClassification
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PrefixTuningConfig,get_peft_model,TaskType
class Prompt_Model(nn.Module):
    #max_topic_len check?
    def __init__(self, model_name,max_seq_len = 416):
        super(Prompt_Model, self).__init__()
        config=AutoConfig.from_pretrained(model_name,hidden_dropout_prob=0.3,attention_probs_dropout_prob=0.3,problem_type = "single_label_classification",num_labels=2)
        pre_model=AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
        prefix_config=PrefixTuningConfig(task_type=TaskType.SEQ_CLS,num_virtual_tokens=20,prefix_projection=False)
        self.model=get_peft_model(pre_model,prefix_config)
        for param in self.model.parameters():
            param.requires_grad=True
        # self.model=AutoModelForQuestionAnswering.from_pretrained(model_name)
        # self.model.qa_outputs.weight.data = torch.randn_like(self.model.qa_outputs.weight)
        # self.model.qa_outputs.bias.data = torch.zeros_like(self.model.qa_outputs.bias)
        # qa_outputs_params = torch.load('D:\\vscode-projects\\medical\\qa_outputs.pth', map_location='cuda:0')
        # self.model.qa_outputs.weight=nn.Parameter(qa_outputs_params['qa_outputs.weight'])
        # self.model.qa_outputs.bias=nn.Parameter(qa_outputs_params['qa_outputs.bias'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_len=max_seq_len
    
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

    def mlm_train_step(self, batch, topic_embed, start, end, device):
        #inputs_prompt check?
        inputs_prompt = self.generate_train_inputs(batch, topic_embed, start,end,device)
        output = self.model(**inputs_prompt)
        #bert_out = self.model(**inputs_prompt, start_positions=start, end_positions=end)
        return output
    
    def mlm_test_step(self, batch, topic_embed, device):
        #inputs_prompt check?
        inputs_prompt = self.generate_test_inputs(batch, topic_embed, device)
        output = self.model(**inputs_prompt)
        return output
    
    def mlc_train_step(self,inputs,device):#multichoice
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
    



    
