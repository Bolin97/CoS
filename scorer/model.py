from torch.nn import CrossEntropyLoss
from transformers import DistilBertPreTrainedModel, DistilBertModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from custom_loss import triplet_loss, contrastive_loss, triplet_contrastive_loss
from transformers import TrainingArguments, Trainer


class DistilBertScorer(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        drop4pos = nn.Dropout(0.1)

        pos_state = drop4pos(distilbert_output.last_hidden_state)[:, 0, :]
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states={"last_hidden_state":distilbert_output.last_hidden_state[:, 0, :], "drop_state": pos_state} ,
            attentions=distilbert_output.attentions,
        )


class DistilBertScorerII(DistilBertPreTrainedModel):
    # 阶段二训练所用模型:在能力区间内进行正负例的划分,数据集中的label变多，但loss计算方式与阶段一保持一致
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels//2+1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # labels=[0,1,2,3],将label=0视为一类，将label=1,2,3视为另外一类，计算二分类任务的交叉熵损失函数
        # if self.config.num_labels == 2:
        #     # 将标签转换为二分类标签
        #     binary_labels = torch.zeros(len(labels), dtype=labels.dtype)  # 初始化二分类标签为0
        #     binary_labels[labels != 0] = 1  # 将原始标签为0的样本标记为1
        #     binary_labels = binary_labels.to(labels.device)
        # else:
        
        # 将标签转换为n//2+1分类标签
        merged_labels = labels
        error_labels = torch.tensor([i+self.config.num_labels//2 for i in range(self.config.num_labels//2)], dtype=labels.dtype)
        error_labels = error_labels.to(labels.device)
        label_final = torch.tensor(self.config.num_labels//2, dtype=labels.dtype)
        label_final = label_final.to(labels.device)
        merged_labels[torch.isin(labels, error_labels)] = label_final

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        drop4pos = nn.Dropout(0.1)

        pos_state = drop4pos(distilbert_output.last_hidden_state)[:, 0, :]
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss_fct = CrossEntropyLoss()
        
        # if self.config.num_labels == 2:
        #     loss = loss_fct(logits.view(-1, self.num_labels), binary_labels.view(-1))
        # elif self.config.num_labels == 10:
        #     loss = loss_fct(logits.view(-1, self.config.num_labels//2+1), six_labels.view(-1))
        # else:
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        loss = loss_fct(logits.view(-1, self.config.num_labels//2+1), merged_labels.view(-1))

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output
        # torch.equal(distilbert_output.hidden_states[-1][:, 0, :], distilbert_output.last_hidden_state[:, 0, :]) = True
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states={"last_hidden_state":distilbert_output.last_hidden_state[:, 0, :], "drop_state": pos_state} ,
            attentions=distilbert_output.attentions,
        )


class CustomTrainingArguments(TrainingArguments):
    def __init__(self, loss_type, dis_type, num_labels, T_margin, C_margin, T_alpha, C_alpha, binary_loss, use_ce, tripletN, in_region, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dis_type = dis_type
        self.T_margin = T_margin
        self.C_margin = C_margin
        self.T_alpha = T_alpha
        self.C_alpha = C_alpha
        # loss 计算方案 triplet or contrastive
        self.loss_type = loss_type
        # 是否使用cross_entropy
        self.use_ce = use_ce
        self.binary_loss = binary_loss
        self.num_labels = num_labels
        # in_region=Ture, 同一能力空间的正负例进行对比学习
        self.in_region = in_region
        # 使用自实现的三元组损失1:1:N
        self.tripletN = tripletN
        


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.loss_type == "triplet":
            return triplet_loss(
                model, inputs, return_outputs, margin=self.args.T_margin, alpha=self.args.T_alpha,
                dis_type=self.args.dis_type, use_ce=self.args.use_ce, num_labels=self.args.num_labels, 
                tripletN=self.args.tripletN, drop_as_anchor=False, binary_contrastive=False, in_region=self.args.in_region
                )
        elif self.args.loss_type == "contrastive":
            return contrastive_loss(
            model, inputs, return_outputs=return_outputs, 
            margin=self.args.C_margin, alpha=self.args.C_alpha, 
            dis_type=self.args.dis_type, binary_loss=self.args.binary_loss, 
            use_ce=self.args.use_ce, num_labels=self.args.num_labels,
            in_region=self.args.in_region
        ) 
            
        elif self.args.loss_type == "triplet|contrastive":
            return triplet_contrastive_loss(
            model, inputs, return_outputs=return_outputs,
            T_margin=self.args.T_margin,  C_margin=self.args.C_margin, 
            T_alpha=self.args.T_alpha, C_alpha=self.args.C_alpha,
            dis_type=self.args.dis_type, use_ce=self.args.use_ce,
            num_labels=self.args.num_labels
        ) 
        else:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        
        
