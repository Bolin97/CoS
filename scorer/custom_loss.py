import torch
import torch.nn.functional as F
from utils import trans_binary_labels_torch
import pdb


def calc_contrastive_loss(embeddings, labels, dis_type, margin):
    # 实现对比损失函数,要求同类样本距离越近越好, 不同类样本距离越远越好
    if dis_type == "pairwise":
        euclidean_distance = F.pairwise_distance(embeddings.unsqueeze(1), embeddings.unsqueeze(0))
    if dis_type == "cosine":
        euclidean_distance = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        
    # 根据标签是否相同创建目标值
    target = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()

    # 计算loss
    c_loss = torch.mean((target * torch.pow(euclidean_distance, 2)) +
                                ((1 - target) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)))
    return c_loss


def calc_triplet_loss(anchors, positives, negatives, margin, dis_type):
    if dis_type == 'pairwise':
        distance_positive = F.pairwise_distance(anchors, positives, 2)
        distance_negative = F.pairwise_distance(anchors, negatives, 2)
    if dis_type == 'cosine':
        distance_positive = 1 - F.cosine_similarity(anchors, positives, dim=0)
        distance_negative = 1 - F.cosine_similarity(anchors, negatives, dim=0)
    return F.relu(distance_positive - distance_negative + margin).mean()


def calc_triplet_loss_N(anchors, label_groups, dis_type, margin):
    """计算1:1:N三元组损失, len(anchors) = len(label_groups.keys())
    Args:
        anchors (_type_): list, 每个类别的锚点(几何中心点)
        label_groups (_type_): dict, key是类别, value是list,包含属于该类的所有embedding
        dis_type (_type_): 距离类型
        margin (_type_): 间隔
    """
    pos_distances = []
    neg_distances = []
    for label, anchor in anchors.items():
        # 计算类内距离: 当前类的中心点到类内任一点的距离
        if dis_type == 'pairwise':
            pos_distances.append(F.pairwise_distance(anchor, label_groups[label][0])) 
        if dis_type == 'cosine':
            pos_distances.append(F.cosine_similarity(anchor, label_groups[label][0], dim=0))
        
        # 计算类间距离
        negative_labels = [key for key in label_groups.keys() if key != label]
        negatives = []
        for label in negative_labels:
            negatives.extend(label_groups[label]) 
        if dis_type == 'pairwise':
            stacked_distance_neg = torch.stack([F.pairwise_distance(anchor, negative) for negative in negatives])
        if dis_type == 'cosine':
            stacked_distance_neg = torch.stack([F.cosine_similarity(anchor, negative, dim=0) for negative in negatives])
        neg_distances.append(torch.mean(stacked_distance_neg, dim=0)) 
        del negative_labels
        del negatives
        del stacked_distance_neg
    triplet_loss = F.relu( torch.tensor(pos_distances) - torch.tensor(neg_distances) + torch.tensor(margin)).mean() 
    return triplet_loss



def get_triplet(embeddings, labels, drop_state=None):
    """根据标签将embeddings划分为anchors、positives和negatives
    Args:
        embeddings (_type_): cls向量
        drop_state (tensor) use the output of  dropout(rate=0.2) layer as anchor
        labels (_type_): 类别
    """
    batch_size = len(labels)
    anchors = []
    positives = []
    negatives = []
    for i in range(batch_size):
        if drop_state:
            anchor = drop_state[i]
        else: 
            anchor = embeddings[i]
        label = labels[i]
        same_label_idx = (labels == label).nonzero().squeeze()
        diff_label_idx = (labels != label).nonzero().squeeze()
        # 从同类样本中随机采样一个作为positive
        if len(same_label_idx) > 1:
            pos_idx = same_label_idx[torch.randperm(len(same_label_idx))[1]]
            positive = embeddings[pos_idx]
        else:
            positive = anchor
        
        # 从异类样本中随机采样一个作为negative
        if len(diff_label_idx) > 0:
            neg_idx = diff_label_idx[torch.randperm(len(diff_label_idx))[1]]
            negative = embeddings[neg_idx]
        else:
            negative = anchor
        
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    return anchors, positives, negatives


def group_batch(labels, embeddings):
    # 按照标签{01234}对样本进行分组
    label_groups = {}
    for i, label in enumerate(labels):
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(embeddings[i])
    # 类内中心点作为锚点
    anchors = {}  
    for label in label_groups.keys():
        stacked_tensors = torch.stack(label_groups[label])
        anchors[label] = torch.mean(stacked_tensors, dim=0)
    return label_groups, anchors


def triplet_contrastive_loss(model, inputs, return_outputs, T_margin, C_margin, T_alpha, C_alpha, dis_type, use_ce, num_labels):
    # 不同的能力空间之间用三元组损失, 同一个能力空间之内的正负例用01对比损失
    outputs = model(**inputs)
    labels = inputs['labels']
    right_labels = [i for i in range(num_labels//2)]
    embeddings = outputs["hidden_states"]["last_hidden_state"]
    ce_loss = torch.mean(outputs.loss)
    
    # 同一能力空间下的正负例样本合并到一起
    region_labels = []
    for label in labels.tolist():
        if label in right_labels:
            region_labels.append(label)
        else:
            region_labels.append(label-num_labels//2)
    
    region_groups, anchors = group_batch(region_labels, embeddings)
    
    # 计算类别{0-5, 1-6, 2-7, 3-8, 4-9}五大类之间的三元组对比损失
    triplet_loss = calc_triplet_loss_N(anchors, region_groups, dis_type, T_margin)
    del region_groups
    del anchors
    del region_labels
    
    c_losses = []
    # 同一能力空间下正负样本之间的对比损失
    # labels = torch.tensor(inputs['labels'])
    for label in right_labels:
        mask = (labels == torch.tensor(label)) | (labels == torch.tensor(label+int(num_labels//2)))
        # 使用布尔索引从原始张量中取出满足条件的元素
        current_label = labels[mask]
        current_label_embeddings = embeddings[mask]
        c_losses.append(calc_contrastive_loss(current_label_embeddings, current_label, dis_type, C_margin))
    c_loss = torch.tensor(c_losses).mean()
    del right_labels
    del c_losses
    
    if use_ce:
        total_loss = ce_loss + T_alpha * triplet_loss + C_alpha * c_loss
    else:
        total_loss = T_alpha * triplet_loss.to(ce_loss.device) + C_alpha * c_loss.to(ce_loss.device)
        total_loss.requires_grad = True
    
    return (total_loss, outputs) if return_outputs else total_loss


def triplet_loss(model, inputs, return_outputs, margin, alpha, dis_type, use_ce, num_labels, tripletN, drop_as_anchor, binary_contrastive, in_region):
    outputs = model(**inputs)
    labels = inputs['labels']
    embeddings = outputs["hidden_states"]["last_hidden_state"]
    if binary_contrastive:
        # 将标签转换为二分类标签, 仅在正负例之间使用对比学习
        labels = trans_binary_labels_torch(labels)
    # error_labels = torch.tensor([i+num_labels//2 for i in range(num_labels//2)], dtype=labels.dtype)
    # error_labels = error_labels.to(labels.device)
    # label_final = torch.tensor(num_labels//2, dtype=labels.dtype)
    # label_final = label_final.to(labels.device)

    # # 使用clone()方法创建一个新的标签张量
    # new_labels = labels.clone()

    # # 在新的标签张量上执行操作
    # new_labels[torch.isin(new_labels, error_labels)] = label_final

    # # 将新的标签张量赋值回原始标签张量
    # labels = new_labels
    # del new_labels
    ce_loss = torch.mean(outputs.loss)
    if in_region:
        # 同一能力空间下的正负例样本合并到一起
        region_labels = []
        right_labels = [i for i in range(num_labels//2)]
        for label in labels.tolist():
            if label in right_labels:
                region_labels.append(label)
            else:
                region_labels.append(label-num_labels//2)   
        if tripletN:
            # 1:1:N (锚点, 正例, 负例)三元组损失
            region_groups, anchors = group_batch(region_labels, embeddings)
            triplet_loss = calc_triplet_loss_N(anchors, region_groups, dis_type, margin)
            del region_groups, anchors
        else:
            # 1:1:1 三元组损失
            drop_state = outputs["hidden_states"]["drop_state"] if drop_as_anchor else None
            anchors, positives, negatives = get_triplet(embeddings, region_labels, drop_state)
            triplet_loss = calc_triplet_loss(anchors, positives, negatives, margin, dis_type)
            del anchors, positives, negatives
        del region_labels
    if tripletN:
        # 1:1:N (锚点, 正例, 负例)三元组损失
        label_groups, anchors = group_batch(labels, embeddings)
        triplet_loss = calc_triplet_loss_N(anchors, label_groups, dis_type, margin)
        del label_groups, anchors
    else:
        # 1:1:1 三元组损失
        drop_state = outputs["hidden_states"]["drop_state"] if drop_as_anchor else None
        anchors, positives, negatives = get_triplet(embeddings, labels, drop_state)
        triplet_loss = calc_triplet_loss(anchors, positives, negatives, margin, dis_type)
        del anchors, positives, negatives
    if use_ce:
        total_loss = ce_loss + alpha * triplet_loss
        # print("triplet_loss:", triplet_loss)
        # print("ce_loss:", ce_loss)
    else:
        total_loss = triplet_loss.to(ce_loss.device)
        total_loss.requires_grad = True

    return (total_loss, outputs) if return_outputs else total_loss



def contrastive_loss(model, inputs, return_outputs, margin, alpha, dis_type, binary_loss, use_ce, num_labels, in_region):
    outputs = model(**inputs)
    labels = inputs['labels']
    if binary_loss:
        # 将标签转换为二分类标签
        labels = trans_binary_labels_torch(labels)
    ce_loss = torch.mean(outputs.loss)
    embeddings = outputs["hidden_states"]["last_hidden_state"]
    if in_region:
        right_labels = [i for i in range(num_labels//2)]
        c_losses = []
        # 同一能力空间下正负样本之间的对比损失
        # labels = torch.tensor(inputs['labels'])
        for label in right_labels:
            mask = (labels == torch.tensor(label)) | (labels == torch.tensor(label+int(num_labels//2)))
            # 使用布尔索引从原始张量中取出满足条件的元素
            current_label = labels[mask]
            current_label_embeddings = embeddings[mask]
            c_losses.append(calc_contrastive_loss(current_label_embeddings, current_label, dis_type, margin))
        c_loss = torch.tensor(c_losses).mean()
        del right_labels, c_losses
    else:
        c_loss = calc_contrastive_loss(embeddings, labels, dis_type, margin)
    if use_ce:
        total_loss = ce_loss + alpha * c_loss
    else:
        total_loss = c_loss.to(ce_loss.device)
        total_loss.requires_grad = True
    total_loss = ce_loss + alpha * c_loss
    return (total_loss, outputs) if return_outputs else total_loss
    
    
if __name__ == '__main__':
    pass