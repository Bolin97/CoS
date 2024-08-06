from utils import read_cos_datasets, CoSDataset, read_hierarchy_cos_datasets, normal_metrics
from model import DistilBertScorer, CustomTrainer, CustomTrainingArguments
from transformers import DistilBertTokenizerFast, EarlyStoppingCallback
import warnings


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

        
if __name__ == '__main__':
    args_dict = {
        "output_dir": './task-model',  
        "num_train_epochs": 20,   
        "per_device_train_batch_size": 64,  
        "per_device_eval_batch_size": 64, 
        "warmup_steps": 500,  
        "disable_tqdm": True,
        # "save_strategy": "no",
        "save_strategy": "epoch", # 按epoch保存
        "evaluation_strategy": "epoch",  
        "eval_steps": None,
        "learning_rate": 5e-05,
        "weight_decay": 0.001,
        "fp16": True, 
        "fp16_opt_level": 'O1',
        "ddp_find_unused_parameters": False,
        "load_best_model_at_end": True, # 早停机制必设参数
        # "loss_type": 'triplet|contrastive',
        "loss_type": "triplet",
        "dis_type": "pairwise",
        "T_margin": 3.0,
        "T_alpha": 2.0,
        "C_margin": 0.0,
        "C_alpha": 0.0,
        "binary_loss": False,
        "use_ce":True,
        "num_labels":6,
        "in_region": False,
        'tripletN': True
    }
    
    # Best Hyperparameters: {'margin': 3.0, 'loss_type': 'contrastive', 'dis_type': 'pairwise', 'alpha': 2.0, 'binary_loss': False}

    tokenizer = DistilBertTokenizerFast.from_pretrained('/data/CoS/ptm/distilbert-base-uncased')
    min_samples_per_class = 10
    train_texts, train_labels = read_cos_datasets('/data/CoS/task-train.json', args_dict["per_device_train_batch_size"], min_samples_per_class, balanced=False)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    train_dataset = CoSDataset(train_encodings, train_labels)

    test_texts, test_labels = read_cos_datasets('/data/CoS/task-test.json', args_dict["per_device_eval_batch_size"], min_samples_per_class, balanced=False)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    test_dataset = CoSDataset(test_encodings, test_labels)

    training_args = CustomTrainingArguments(**args_dict)
    print("train epoch samples", len(train_labels))
    print("min_samples_per_class", min_samples_per_class)
    print(args_dict)
    model = DistilBertScorer.from_pretrained("/data/CoS/ptm/distilbert-base-uncased", output_hidden_states=True, num_labels=6)
    trainer = CustomTrainer(
        model=model,  
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=test_dataset, 
        # compute_metrics=custom_metrics,
        compute_metrics=normal_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer.train()
        
    test_result = trainer.evaluate()
    print(f"Metrics on Test: {test_result}")

    
    
# nohup python phaseI_train.py > phaseI.log 2>&1 &

# 多卡并行训练    
# accelerate launch --num_processes 3 ce_bert.py 

