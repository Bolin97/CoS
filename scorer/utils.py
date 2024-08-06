import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
import itertools


def get_label(sample):
    if "label" in sample.keys():
        # 多分类能力区间划分数据集
        label = sample['label']
    else:
        # 二分类任务数据集
        label = 0 if sample["is_right"] is False else 1
    return label


def trans_binary_labels_torch(labels):
    # 将标签转换为二分类标签
    binary_labels = torch.zeros(len(labels), dtype=labels.dtype)  # 初始化二分类标签为0
    binary_labels[labels != 0] = 1  # 将原始标签不为0的样本标记为1
    binary_labels = binary_labels.to(labels.device)
    return binary_labels

def trans_binary_labels(labels):
    # 将标签转换为二分类标签
    binary_labels = np.zeros_like(labels)  # Initialize binary labels as 0
    binary_labels[labels != 0] = 1  # Set samples with non-zero original labels to 1
    return binary_labels


def normal_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    num_labels = len(set(labels))
    
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average=None)
    for i in range(num_labels):
        print(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

    
    precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')

    micro_precision = precision_score(labels, preds, average='micro')
    micro_recall = recall_score(y_true=labels, y_pred=preds, average='micro')
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    
    # merged_labels = labels
    # merged_preds = preds

    # right_labels = np.array([i for i in range(num_labels//2)])
    # error_labels = np.array([i+num_labels//2 for i in range(num_labels//2)])
    
    # merged_labels[np.isin(labels, right_labels)] = 1
    # merged_labels[np.isin(labels, error_labels)] = 0
    # merged_preds[np.isin(preds, right_labels)] = 1
    # merged_preds[np.isin(preds, error_labels)] = 0
    
    return {
        "epoch samples": len(labels),
        "accuracy": accuracy, #"binary_accuracy": accuracy_score(y_true=merged_labels, y_pred=merged_preds),
        "precision": precision, "micro_precision": micro_precision, 
        "recall": recall, "micro_recall": micro_recall, 
        "f1": f1, "micro_f1": micro_f1
        }


def merged_class_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    # 将每个能力区域内的负样本合并为相同label: num_labels//2 + 1
    merged_labels = labels
    merged_preds = preds
    num_labels = len(set(labels))

    right_labels = np.array([i for i in range(num_labels//2)])
    error_labels = np.array([i+num_labels//2 for i in range(num_labels//2)])
    merged_labels[np.isin(merged_labels, error_labels)] = num_labels//2 + 1
    merged_preds[np.isin(merged_preds, error_labels)] = num_labels//2 + 1
    
    accuracy = accuracy_score(y_true=merged_labels, y_pred=merged_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=merged_labels, y_pred=merged_preds, average=None)
    for i in range(num_labels//2+1):
        print(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

    precision = precision_score(y_true=merged_labels, y_pred=merged_preds, average='macro')
    recall = recall_score(y_true=merged_labels, y_pred=merged_preds, average='macro')
    f1 = f1_score(y_true=merged_labels, y_pred=merged_preds, average='macro')

    micro_precision = precision_score(merged_labels, y_pred=merged_preds, average='micro')
    micro_recall = recall_score(y_true=merged_labels, y_pred=merged_preds, average='micro')
    micro_f1 = f1_score(y_true=merged_labels, y_pred=merged_preds, average='micro')
    

    labels[np.isin(labels, right_labels)] = 1
    labels[np.isin(labels, error_labels)] = 0
    preds[np.isin(preds, right_labels)] = 1
    preds[np.isin(preds, error_labels)] = 0
    return {
        "epoch samples": len(labels),
        "{}-label-accuracy".format(num_labels//2 + 1): accuracy, "binary_accuracy": accuracy_score(y_true=labels, y_pred=preds),
        "precision": precision, "micro_precision": micro_precision, 
        "recall": recall, "micro_recall": micro_recall, 
        "f1": f1, "micro_f1": micro_f1
        }
    
    
def unbalanced_batch_construct(num_samples, batch_size, class_indices, min_samples_per_class):
    # 从每个类别中随机选择样本组成批次
    data_indices = []
    final_label = len(class_indices.keys())-1
    for _ in range(num_samples // batch_size):
        batch_indices = []
        remain_size = 0
        for label, label_indices in class_indices.items():
            if label != final_label:
                # 采样正例
                sample_size = max(min_samples_per_class, int(batch_size*(len(label_indices) / num_samples)))
                remain_size += sample_size
                if len(label_indices) >= sample_size:
                    selected_indices = np.random.choice(label_indices, size=sample_size, replace=False)
                    batch_indices.extend(selected_indices)
                    # 从类别样本索引中移除已经选择的样本，避免重复采样
                    for idx in selected_indices:
                        label_indices.remove(idx)
            else:
                # 采样负例
                if len(label_indices) >= (batch_size-remain_size):
                    selected_indices = np.random.choice(label_indices, size=batch_size-remain_size, replace=False)
                    batch_indices.extend(selected_indices)
                    for idx in selected_indices:
                        label_indices.remove(idx)
        if len(batch_indices) == batch_size:
            data_indices.extend(batch_indices)
    return data_indices


def balanced_batch_construct(batch_size, class_indices, min_samples_per_class):
    # 从每个类别中随机选择样本组成批次,每个类别的样本数量相等
    min_samples_num = min([len(samples) for samples in class_indices.values()])
    data_indices = []
    for _ in range(min_samples_num // min_samples_per_class):
        batch_indices = []
        for label, label_indices in class_indices.items():
            if len(label_indices) >= min_samples_per_class:
                selected_indices = np.random.choice(label_indices, size=min_samples_per_class, replace=False)
                batch_indices.extend(selected_indices)
                # 从类别样本索引中移除已经选择的样本，避免重复采样
                for idx in selected_indices:
                    label_indices.remove(idx)
            if len(batch_indices) == batch_size:
                data_indices.extend(batch_indices)
    return data_indices

    
def read_cos_datasets(file_name, batch_size, min_samples_per_class, balanced=False):
    """读取两类数据集(类别平衡和类别不平衡), 类别平衡:bs=70, samples_per_class=10, class_num=7
    Args:
        file_name (str): json文件目录
        batch_size (int): 批次大小
        min_samples_per_class (int): 每个类别至少包含的样本数
        balanced (bool): 是否类别平衡
    Returns:
        list: texts, labels 
    """
    texts = []
    labels = []
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    num_samples = len(data)
    # 获取每个类别的样本索引
    class_indices = {}
    for i, sample in enumerate(data):
        label = sample['label']
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    if balanced:
        data_indices = balanced_batch_construct(batch_size, class_indices, min_samples_per_class)
    else:
        # 按label升序排列,最大的label是false样本
        class_indices = dict(sorted(class_indices.items(), key=lambda item: item[0], reverse=False)) 
        data_indices = unbalanced_batch_construct(num_samples, batch_size, class_indices, min_samples_per_class)
    
    for idx in data_indices:
        texts.append("model:{} query:{}".format(data[idx]["model"], data[idx]["query"]))
        labels.append(data[idx]["label"])
    return texts, labels 
        


def read_hierarchy_cos_datasets(file_name, batch_size, min_samples_per_class):
    """读取层次分类的数据集0-5, 1-6, 2-7, 3-8, 4-9
    Args:
        file_name (_type_): json文件目录
        batch_size (_type_): 批次大小
        min_samples_per_class (_type_): 每个类别至少包含的样本数

    Returns:
        list: texts, labels 
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 获取每个类别的样本索引
    class_indices = {}
    for i, sample in enumerate(data):
        label = get_label(sample)
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    class_indices = dict(sorted(class_indices.items(), key=lambda item: item[0], reverse=True))
    # 从每个类别中随机选择样本组成批次
    data_indices = []
    num_samples = len(data)
    for _ in range(num_samples // batch_size):
        batch_indices = []
        for label, label_indices in class_indices.items():
            if len(label_indices) >= min_samples_per_class:
                selected_indices = np.random.choice(label_indices, size=min_samples_per_class, replace=False)
                batch_indices.extend(selected_indices)
                # 从类别样本索引中移除已经选择的样本，避免重复采样
                for idx in selected_indices:
                    label_indices.remove(idx)
        if len(batch_indices) == batch_size:
            data_indices.extend(batch_indices)
    texts, labels = [], []
    for idx in data_indices:
        texts.append("model:{} query:{}".format(data[idx]["model"], data[idx]["query"]))
        labels.append(get_label(data[idx]))
    return texts, labels 


def simple_read_cos_datasets(file_name):
    texts = []
    labels = []
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
        texts.append("model:{} query:{}".format(d["model"], d["query"]))
        labels.append(d["label"])
    return texts, labels



class CoSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 等价于下面两行
        # item['input_ids'] = torch.tensor(self.encodings['input_ids'][idx])
        # item['attention_mask'] = torch.tensor(self.encodings['attention_mask'][idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def search_params(search_space, default_args):
    train_args = []
    searched_args = []
    # 获取所有待搜索参数的列表
    params = list(search_space.keys())
    # 获取所有待搜索参数的选项值列表
    param_values = list(search_space.values())

    # 使用itertools.product生成所有可能的组合
    # combinations is a <class 'itertools.product'> object and is not subscriptable
    combinations = itertools.product(*param_values)

    for combination in combinations:
        # combination is a <class 'tuple'> object
        search_args = {}
        for i in range(len(combination)):
            search_args[params[i]] = combination[i]
        searched_args.append(search_args)
        train_args.append({**default_args, **search_args})

    return train_args, searched_args
           

if __name__ == '__main__':
    # train_encodings = {"input_ids":[nums, max_length], "attention_mask": [nums, max_length]}
    # tokenizer = DistilBertTokenizerFast.from_pretrained('/data/CoS/ptm/distilbert-base-uncased')
    # bs = 16
    # train_texts, train_labels = read_cos_datasets('/data/CoS/task-train.json', batch_size=bs, min_samples_per_class=2)
    # train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    # train_dataset = CoSDataset(train_encodings, train_labels)    

    # train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
    
    # # 打印训练数据集中每个批次的数据
    # print("Training Data:")
    # for batch in train_loader:
    #     print(batch)
    #     print(batch['labels'].shape)
    #     break  # 仅打印第一个批次的数据，方便查看结构
        
    # 定义超参数搜索空间
    search_space = {
        "margin": [1.0, 2.0, 3.0, 4.0, 5.0],
        "loss_type": ["triplet", "contrastive"],
        "dis_type": ["pairwise", "cosine"],
        "alpha": [1.0, 1.5, 2.0]
    }
    args_dict = {
        "learning_rate": 5e-05,
        "weight_decay": 0.001,
        "fp16": True, 
        "fp16_opt_level": 'O1',
        "use_ce":True
    }
    train_args, searched_args = search_params(search_space, args_dict)
    for i in range(len(train_args)):
        print(train_args[i])
        print(searched_args[i])
        break
