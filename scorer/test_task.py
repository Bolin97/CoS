from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from datasets import load_dataset
# model = "anthropic_claude-2.0" label:true  prediction:[[0.50550926 0.49449074]]


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
    labels = pred["label_ids"]
    preds = pred["predictions"]
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate(model, metric_func, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    dataset_like = {"label_ids": all_labels, "predictions": all_preds}
    results = metric_func(dataset_like)
    return results


if __name__ == '__main__':
    tasks = ["gsm", "legalbench", "math", "med_qa", "mmlu"]
    # 加载本地模型
    model = DistilBertForSequenceClassification.from_pretrained("/data/CoS/scorer/general-results/checkpoint-14750")
    model.to('cuda')
    tokenizer = DistilBertTokenizerFast.from_pretrained('/data/CoS/ptm/distilbert-base-uncased')

    for task in tasks:
        test_file = "/data/CoS/datasets/{}-test.json".format(task)
        val_texts, val_labels = read_cos_datasets(test_file)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
        val_dataset = CoSDataset(val_encodings, val_labels)

        # 对数据集进行验证
        results = evaluate(model, custom_metrics, val_dataset)
        print("matrix on {}".format(task))
        print(results)
        print("********************")

