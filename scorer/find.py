file_path = '/data/CoS/scorer/search.log'  # 替换为你的文件路径
import json

# Metrics on Test: {'eval_loss': 0.5254903435707092, 'eval_accuracy': 0.7735275401258573, 'eval_precision': 0.776712563659272, 'eval_micro_precision': 0.7735275401258573, 'eval_recall': 0.7403297047028906, 'eval_micro_recall': 0.7735275401258573, 'eval_f1': 0.7570098675863468, 'eval_micro_f1': 0.7735275401258573, 'eval_runtime': 15.9576, 'eval_samples_per_second': 886.286, 'eval_steps_per_second': 9.275, 'epoch': 20.0}
# 打开文件
with open(file_path, 'r') as file:
    # 按行读取文件内容
    lines = file.readlines()

eval_loss = []
eval_accuracy = []
count = 0
for line in lines:
    line = line.strip()
    if "Metrics on Test: {'eval_loss':" in line.strip():
        data = line.split('Metrics on Test: ')[-1]
        data = eval(data)
        eval_loss.append(data['eval_loss'])
        eval_accuracy.append(data['eval_accuracy'])
        if count == 19 or count == 27:
            print(line.split('Metrics on Test: ')[-1])
        count += 1
# 找到最小值
min_loss = min(eval_loss)

# 找到最小值的索引（下标）
min_loss_index = eval_loss.index(min_loss)

print("最小值：", min_loss)
print("最小值的索引（下标）：", min_loss_index)


# 找到最大值
max_acc = max(eval_accuracy)

# 找到最小值的索引（下标）
max_acc_index = eval_accuracy.index(max_acc)

print("最大值：", max_acc)
print("最大值的索引（下标）：", max_acc_index)

            
        
        
        
