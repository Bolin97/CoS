from utils import read_cos_datasets, CoSDataset, normal_metrics, search_params
from model import DistilBertScorer, CustomTrainer, CustomTrainingArguments
from transformers import DistilBertTokenizerFast, EarlyStoppingCallback, DistilBertForSequenceClassification
import warnings
from accelerate import Accelerator

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

        
if __name__ == '__main__':
    args_dict = {
        "output_dir": './ce-notask',  
        "num_train_epochs": 15,   
        "per_device_train_batch_size": 64,  
        "per_device_eval_batch_size": 64, 
        "warmup_steps": 500,  
        "disable_tqdm": True,
        "save_strategy": "no",
        # "save_strategy": "epoch",
        "evaluation_strategy": "epoch",  
        "eval_steps": None,
        "learning_rate": 5e-05,
        "weight_decay": 0.001,
        "fp16": True, 
        "fp16_opt_level": 'O1',
        "ddp_find_unused_parameters": False,
        # "load_best_model_at_end": True,
        # "loss_type": 'triplet',
        # "dis_type": "pairwise",
        # "margin": 5.0,
        # "alpha": 2.0,
        # "binary_loss": False,
        "use_ce":True
    }

    tokenizer = DistilBertTokenizerFast.from_pretrained('/data/CoS/ptm/distilbert-base-uncased')
    min_samples_per_class = 10
    train_texts, train_labels = read_cos_datasets('/data/CoS/task-train.json', args_dict["per_device_train_batch_size"], min_samples_per_class)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    train_dataset = CoSDataset(train_encodings, train_labels)
    # train_loader = DataLoader(train_dataset, batch_size=args_dict["per_device_train_batch_size"], shuffle=False)
    print("train epoch samples", len(train_labels))
    test_texts, test_labels = read_cos_datasets('/data/CoS/task-test.json', args_dict["per_device_eval_batch_size"], min_samples_per_class)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    test_dataset = CoSDataset(test_encodings, test_labels)
    # 因为output_hidden_states=True使得模型变大! 考虑使用原生DistilBert
    # model = DistilBertForSequenceClassification.from_pretrained("/data/CoS/ptm/distilbert-base-uncased", output_hidden_states=True, num_labels=6)
    search_space = {
        "margin": [1.0, 2.0, 3.0, 4.0, 5.0],
        "loss_type": ["triplet", "contrastive"],
        "dis_type": ["pairwise", "cosine"],
        "alpha": [1.0, 1.5, 2.0], 
        "binary_loss": [True, False]
    }
    eval_accuracy = 0
    best_params = {}
    best_experiment = 0
    train_args, searched_args = search_params(search_space, args_dict)
    for i in range(len(train_args)):
        print("****Experiment{}".format(i))
        print("Hyperparameters:", searched_args[i])
        training_args = CustomTrainingArguments(**train_args[i])
        model = DistilBertScorer.from_pretrained("/data/CoS/ptm/distilbert-base-uncased", output_hidden_states=True, num_labels=6)
        trainer = CustomTrainer(
            model=model,  
            args=training_args,  
            train_dataset=train_dataset,  
            eval_dataset=test_dataset, 
            compute_metrics=normal_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()
        
        test_result = trainer.evaluate()
        print(f"Metrics on Test: {test_result}")
        if test_result["eval_accuracy"] > eval_accuracy:
            eval_accuracy = test_result["eval_accuracy"]
            best_params = searched_args[i]
            best_experiment = i
            
    print("Best eval_accuracy:", eval_accuracy)
    print("Best Hyperparameters:", best_params)
    print("Best Experiment:", best_experiment)
    
    
    # trainer = TripletTrainer(
    #     model=model,  
    #     args=training_args,  
    #     train_dataset=train_dataset,  
    #     eval_dataset=test_dataset, 
    #     compute_metrics=custom_metrics,
    # )
    # trainer.compute_loss = contrastive_loss
    # 初始化 Accelerator 对象
    # accelerator = Accelerator()
    # # 使用 Accelerator 分布式训练
    # trainer = accelerator.prepare(trainer)
    
    
# nohup python search_cl.py > search-ce.log 2>&1 &

# 多卡并行训练    
# accelerate launch --num_processes 3 ce_bert.py 

