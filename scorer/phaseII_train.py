from utils import read_cos_datasets, CoSDataset, read_hierarchy_cos_datasets, normal_metrics, merged_class_metrics
from model import DistilBertScorer, CustomTrainer, CustomTrainingArguments, DistilBertScorerII
from transformers import DistilBertTokenizerFast, EarlyStoppingCallback
import warnings


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

        
if __name__ == '__main__':
    args_dict = {
        "output_dir": './scale-model',  
        "num_train_epochs": 20,   
        "per_device_train_batch_size": 70,  
        "per_device_eval_batch_size": 70, 
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
        "num_labels":10,
        "in_region": True,
        'tripletN': True
    }
    

    tokenizer = DistilBertTokenizerFast.from_pretrained('/data/CoS/ptm/distilbert-base-uncased')
    min_samples_per_class = 7
    train_texts, train_labels = read_cos_datasets('/data/CoS/task-10-train.json', args_dict["per_device_train_batch_size"], min_samples_per_class, balanced=True)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    train_dataset = CoSDataset(train_encodings, train_labels)

    test_texts, test_labels = read_cos_datasets('/data/CoS/task-10-test.json', args_dict["per_device_eval_batch_size"], min_samples_per_class, balanced=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    test_dataset = CoSDataset(test_encodings, test_labels)

    training_args = CustomTrainingArguments(**args_dict)
    print("train epoch samples", len(train_labels))
    print("min_samples_per_class", min_samples_per_class)
    print(args_dict)
    model_path = "/data/CoS/scorer/task-model/checkpoint-366"
    model = DistilBertScorerII.from_pretrained(model_path, output_hidden_states=True, num_labels=10)
    trainer = CustomTrainer(
        model=model,  
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=test_dataset, 
        compute_metrics=merged_class_metrics,
        # compute_metrics=normal_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train() 
        
    test_result = trainer.evaluate()
    print(f"Metrics on Test: {test_result}")

    


# nohup python phaseII_train.py > phaseII.log 2>&1 &