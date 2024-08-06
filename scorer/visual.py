import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import read_cos_datasets, read_hierarchy_cos_datasets
import numpy as np


# 推断测试集中的每个句子并获取嵌入向量
def infer_and_get_embeddings(texts, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs.to(device)
        outputs = model(**inputs)
        # print(outputs['hidden_states'][-1].shape) #torch.Size([100, 512, 768])
        embeddings = outputs["hidden_states"][-1][:, 0, :].squeeze().cpu().numpy()
    return embeddings



# 加载模型和标记器
# model_path = "/data/CoS/ptm/distilbert-base-uncased"
model_path = "/data/CoS/task-10-class/checkpoint-880"
# model_path = "/data/CoS/scorer/multi-notask/checkpoint-12969"
# model_path = "/data/CoS/scorer/task-model/checkpoint-366"

model = DistilBertForSequenceClassification.from_pretrained(model_path, output_hidden_states=True)
tokenizer = DistilBertTokenizerFast.from_pretrained("/data/CoS/ptm/distilbert-base-uncased")
test_texts, test_labels = read_cos_datasets('/data/CoS/task-10-test.json', batch_size=120, min_samples_per_class=12, balanced=True)
# test_texts, test_labels = test_texts[:240], test_labels[:240]

embeddings = []
for i in range(0, len(test_labels), 120):
    batch_text_embeddings = infer_and_get_embeddings(test_texts[i:i+120], model, tokenizer)
    embeddings.extend(batch_text_embeddings)
print(len(embeddings))
print(set(test_labels))
# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
embeddings_array = np.array(embeddings)
embedded_test_sentences = tsne.fit_transform(embeddings_array)
# 绘制可视化图表
plt.figure(figsize=(10, 6))
for i, label in enumerate(set(test_labels)):
    indices = [idx for idx, l in enumerate(test_labels) if l == label]
    plt.scatter(embedded_test_sentences[indices, 0], embedded_test_sentences[indices, 1], label=label)
plt.legend()
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Capacity Space Partitioning Visualization for LLM on Test Set')
plt.savefig('Cl-Bert_visualization.pdf')  # 保存为 PDF 文件
plt.show()
