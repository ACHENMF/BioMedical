class CONFIG:
    def __init__(self,type):
        if type==0:# Train seq_cls single-label
            self.log_path = "log.txt"
            self.tokenizer_path = "/home/chenzhichen/premodel/biolinkbert"
            self.device = "gpu"
            self.saved_model_path = "/home/chenzhichen/projects/CF-CKP/save_model"
            self.pretrained_model_path = "/home/chenzhichen/premodel/biolinkbert"
            self.test = True
            self.num_workers = 0
            self.train_data_path = "/home/chenzhichen/projects/CF-CKP/dataset/data/data/seqcls/pubmedqa_pr/train.json"
            self.dev_data_path = "/home/chenzhichen/projects/CF-CKP/dataset/data/data/seqcls/pubmedqa_pr/dev.json"
            self.epochs = 200
            self.lr=2e-5
            self.warmup_steps=100
            self.max_grad_norm=1.0
            self.gradient_accumulation=8
            self.num_decay_steps=700
            self.save_mode=True
            self.together=False
        if type==1:#Train multichoice
            self.log_path = "log.txt"
            self.tokenizer_path = "/home/chenzhichen/premodel/biolinkbert"
            self.device = "gpu"
            self.saved_model_path = "/home/chenzhichen/projects/CF-CKP/save_model"
            self.pretrained_model_path = "/home/chenzhichen/premodel/biolinkbert"
            self.test = True
            self.num_workers = 0
            self.train_data_path = "/home/chenzhichen/projects/CF-CKP/dataset/data/data/mc/medqa_prompt6/train.jsonl"
            self.dev_data_path = "/home/chenzhichen/projects/CF-CKP/dataset/data/data/mc/medqa_prompt6/dev.jsonl"
            self.batch_size=8
            self.epochs = 100
            self.lr=3e-5
            self.warmup_steps=1600
            self.max_grad_norm=2.0
            self.gradient_accumulation=8
            self.num_decay_steps=8000
            self.save_mode=True
            self.together=False
        if type==2:#Train seq_cls multi-label
            self.log_path = "log.txt"
            self.tokenizer_path = "/home/chenzhichen/premodel/pubmedbert"
            self.device = "gpu"
            self.saved_model_path = "/home/chenzhichen/projects/CF-CKP/save_model"
            self.pretrained_model_path = "/home/chenzhichen/premodel/pubmedbert"
            self.test = True
            self.num_workers = 0
            self.train_data_path = "/home/chenzhichen/projects/CF-CKP/dataset/data/data/seqcls/hoc_pr/train.json"
            self.dev_data_path = "/home/chenzhichen/projects/CF-CKP/dataset/data/data/seqcls/hoc_pr/dev.json"
            self.batch_size=8
            self.epochs = 100
            self.lr=3e-5
            self.warmup_steps=100
            self.max_grad_norm=1.0
            self.gradient_accumulation=8
            self.num_decay_steps=1010
            self.save_mode=True
            self.together=False
        if type==3:#Test multichoice
            self.log_path = "log.txt"
            self.tokenizer_path = "/home/chenzhichen/premodel/biolinkbert"
            self.device = "gpu"
            self.saved_model_path = "/home/chenzhichen/projects/CF-CKP/save_model"
            self.pretrained_model_path = "/home/chenzhichen/premodel/biolinkbert"
            self.test = True
            self.num_workers = 0
            self.test_data_path = "/home/chenzhichen/projects/CF-CKP/dataset/data/data/mc/medqa_prompt6/test.jsonl"
            self.batch_size=16
            self.epochs = 10
            self.gradient_accumulation=1
        
        
