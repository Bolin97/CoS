import json
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
import warnings

# 有文本分类模型DistilBertForSequenceClassification训练过程如下，请使用TSNE将测试集中每个句子进行可视化，要求不同颜色代表不同的label

# 忽略UserWarning类型的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 有文本分类模型DistilBertForSequenceClassification训练过程如下，请增加一段代码使用对比学习计算损失函数，要求类别相同的样本距离，类别不同的样本距离越远越好

class CoSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_cos_datasets(file_name):
    texts = []
    labels = []
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
        texts.append("model:{} query:{}".format(d["model"], d["query"]))
        labels.append(d["label"])
    return texts, labels


def custom_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average=None)
    for i in range(6):
        print(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    
    precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')

    micro_precision = precision_score(labels, preds, average='micro')
    micro_recall = recall_score(y_true=labels, y_pred=preds, average='micro')
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    
    return {"accuracy": accuracy, "precision": precision, "micro_precision": micro_precision, "recall": recall, "micro_recall": micro_recall, "f1": f1, "micro_f1": micro_f1}


if __name__ == '__main__':
    
    train_texts, train_labels = read_cos_datasets('/data/CoS/task-train.json')
    test_texts, test_labels = read_cos_datasets('/data/CoS/task-test.json')
 
    tokenizer = DistilBertTokenizerFast.from_pretrained('/data/CoS/ptm/distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    train_dataset = CoSDataset(train_encodings, train_labels)

    test_dataset = CoSDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir='./multi-notask',  
        num_train_epochs=20,   
        per_device_train_batch_size=64,  
        per_device_eval_batch_size=32, 
        warmup_steps=500,  
        disable_tqdm=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",  
        eval_steps=None,
        learning_rate=5e-05,
        weight_decay=0.001
    )
    model = DistilBertForSequenceClassification.from_pretrained("/data/CoS/ptm/distilbert-base-uncased", num_labels=6)

    trainer = Trainer(
        model=model,  
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=test_dataset, 
        compute_metrics=custom_metrics
    )

    trainer.train()
    test_result = trainer.evaluate(test_dataset)
    print(f"Metrics on Test: {test_result}")
    
    # learning_rate=5e-05, num_train_epochs=20, weight_decay=0.001, warmup_steps=500 per_device_eval_batch_size=32, per_device_train_batch_size=16,

    

# nohup python multi-bert.py > notask-best.log 2>&1 &
