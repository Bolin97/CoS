import numpy as np
from scipy.stats import spearmanr

# 假设你有5个模型在任务1，任务2和任务3上的性能排名数据
# 每个任务的性能排名（假设为一个列表，每个模型一个排名）
task1_rankings = [2, 4, 1, 5, 3]  # 任务1
task2_rankings = [3, 2, 5, 1, 4]  # 任务2
task3_rankings = [1, 3, 2, 4, 5]  # 任务3

# 计算Spearman等级相关系数
correlation_coefficient, _ = spearmanr([task1_rankings, task2_rankings, task3_rankings])

print("Spearman等级相关系数:", correlation_coefficient)
