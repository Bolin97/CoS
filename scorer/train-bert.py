import json
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


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
        if d['subset']:
            task = d["task"]+"-"+d["subset"]
        else:
            task = d["task"]
        # texts.append("#query:{}#model:{}#task:{}".format(d["query"], d["model"], task))
        # use the following text as new input
        texts.append("model:{} task:{} query:{}".format(d["model"], task, d["query"]))
        labels.append(0 if d["is_right"] is False else 1)
    return texts, labels


def custom_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == '__main__':
    # load train-dev-test dataset
    train_texts, train_labels = read_cos_datasets('/data/CoS/datasets/train.json')
    test_texts, test_labels = read_cos_datasets('/data/CoS/datasets/test.json')
    # train_texts.extend(test_texts)
    # train_labels.extend(test_labels)
    # train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    # load tokenizer and convert text sequence to id sequence with Fixed length
    tokenizer = DistilBertTokenizerFast.from_pretrained('/data/CoS/ptm/distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    # val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    # convert to PyTorch dataset
    train_dataset = CoSDataset(train_encodings, train_labels)
    # val_dataset = CoSDataset(val_encodings, val_labels)
    test_dataset = CoSDataset(test_encodings, test_labels)

    # finetune bert

    training_args = TrainingArguments(
        output_dir='./general-results',  # output directory
        num_train_epochs=50,   # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        disable_tqdm=True,
        # logging_dir='./logs',  # directory for storing logs
        # logging_steps=100,
        # save_steps=1000,
        save_strategy="epoch",
        evaluation_strategy="steps",  # 每隔指定步数进行验证
        # eval_steps=100  # 设置验证频率
    )

    model = DistilBertForSequenceClassification.from_pretrained("/data/CoS/ptm/distilbert-base-uncased", num_labels=2)

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=custom_metrics
    )

    trainer.train()
    test_result = trainer.evaluate(test_dataset)
    print(f"Metrics on Test: {test_result}")

# nohup python train-bert.py > train-general-epoch-50.log 2>&1 &
