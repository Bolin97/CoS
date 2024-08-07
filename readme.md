## Data and Codes for CoS.

## Data Access

You can download our dataset through 

https://drive.google.com/drive/folders/1e94bfQ9D_7h4SvZ4RpuVN7Cg7i5tOHoY?usp=sharing

## Environment Setup

```bash
conda create --n cos python==3.7
```
```bash
pip install -r requirements.txt
```

```bash
conda activate cos
```

## Run the Program

Three steps for training the scorer:

1-step

```bash
cd ./scorer/
```

2-step

customized these params in train_cl.py

```python
args_dict = {
        "output_dir": './task-6-tripletN-use_ce-pairwise-ep30',  
        "num_train_epochs": 30,   
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
        # "load_best_model_at_end": True, # 早停机制
        "loss_type": "contrastive", # 对比学习策略, triplet:三元组损失, contrastive随机二元对比
        "dis_type": "pairwise", # 距离计算方法
        "T_margin": 3.0,
        "C_margin": 0.0,
        "T_alpha": 2.0,
        "C_alpha": 0.0,
        "binary_loss": False,
        "use_ce":True,
        "num_labels":6,
        'tripletN': False,
        'in_region': False
    }
```
3-step

```bash
nohup python train_cl.py > {name}.log 2>&1 &
```

