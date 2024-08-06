from utils import simple_read_cos_datasets, CoSDataset, normal_metrics, read_cos_datasets
from model import DistilBertScorer, CustomTrainer, CustomTrainingArguments
from transformers import DistilBertTokenizerFast, EarlyStoppingCallback
import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

def train_on_model_dataset(llm_name, model, training_args, tokenizer):
    min_samples_per_class = 32
    train_texts, train_labels = read_cos_datasets('/data/CoS/bi-model-data/{}-train.json'.format(llm_name), batch_size=args_dict["per_device_train_batch_size"],
                                                  min_samples_per_class=min_samples_per_class, balanced=True)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    train_dataset = CoSDataset(train_encodings, train_labels)

    print("train epoch samples", len(train_labels))

    # test_texts, test_labels = simple_read_cos_datasets('/data/CoS/mistralai_mistral-7b-v0.1-test.json')
    test_texts, test_labels = read_cos_datasets('/data/CoS/bi-model-data/{}-test.json'.format(llm_name), batch_size=args_dict["per_device_eval_batch_size"],
                                                  min_samples_per_class=min_samples_per_class, balanced=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    test_dataset = CoSDataset(test_encodings, test_labels)

    trainer = CustomTrainer(
        model=model,  
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=test_dataset, 
        compute_metrics=normal_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer.train()

        
if __name__ == '__main__':
    args_dict = {
        "output_dir": './legalbench-binary',  
        "num_train_epochs": 30,   
        "per_device_train_batch_size": 64,  
        "per_device_eval_batch_size": 64, 
        "warmup_steps": 500,  
        "disable_tqdm": True,
        "save_strategy": "no",
        # "save_strategy": "epoch", # 按epoch保存
        "evaluation_strategy": "epoch",  
        "eval_steps": None,
        "learning_rate": 5e-05,
        "weight_decay": 0.001,
        "fp16": True, 
        "fp16_opt_level": 'O1',
        "ddp_find_unused_parameters": False,
        # "load_best_model_at_end": True, # 早停机制必设参数
        # "loss_type": 'triplet|contrastive',
        "loss_type": None,
        "dis_type": "pairwise",
        "T_margin": 3.0,
        "C_margin": 0.0,
        "T_alpha": 2.0,
        "C_alpha": 0.0,
        "binary_loss": False,
        "use_ce": True,
        "num_labels":2,
        'tripletN': False,
        'in_region': False
    }
    
    models = ['meta_llama-2-7b', 'openai_gpt-4-1106-preview', 'cohere_command', 'openai_gpt-3.5-turbo-0613', 'meta_llama-2-70b', 
          'cohere_command-light', 'tiiuae_falcon-40b', 'ai21_j2-jumbo', 'anthropic_claude-2.0', 'openai_gpt-4-0613', 
          'anthropic_claude-instant-1.2', 'ai21_j2-grande', 'mistralai_mistral-7b-v0.1', 'tiiuae_falcon-7b']
    tokenizer = DistilBertTokenizerFast.from_pretrained('/data/CoS/ptm/distilbert-base-uncased')
    
    training_args = CustomTrainingArguments(**args_dict)

    bert_model = DistilBertScorer.from_pretrained("/data/CoS/ptm/distilbert-base-uncased", output_hidden_states=True, num_labels=2)
        
    for llm in models:
        print("training the binary scorer of {}".format(llm))
        train_on_model_dataset(llm_name=llm, model=bert_model, training_args=training_args, tokenizer=tokenizer)
        print("***********************************")
    
    
# nohup python bi-scorer.py > llm-bi-None.log 2>&1 &

# 多卡并行训练    
# accelerate launch --num_processes 3 ce_bert.py 

