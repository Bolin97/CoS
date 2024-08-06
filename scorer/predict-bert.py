from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
import torch
import json

# model = "anthropic_claude-2.0" label:true  prediction:[[0.50550926 0.49449074]]


def prediction(model, input):
    tokenizer = DistilBertTokenizerFast.from_pretrained('/data/CoS/ptm/distilbert-base-uncased')
    inputs = tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()
    results = predictions.flatten().tolist()
    predict = True
    if results[0] > results[1]:
        predict = False
    return predict, results


if __name__ == '__main__':
    model_2 = DistilBertForSequenceClassification.from_pretrained("/data/CoS/scorer/overlap-results/checkpoint-1770")
    model_2.to('cuda')

    with open('/data/CoS/datasets/test.json', 'r', encoding="utf8") as f:
        tests = json.load(f)
    dic = {}
    for d in tests:
        if d['subset']:
            task = d["task"] + "-" + d["subset"]
        else:
            task = d["task"]
        text = "model:{} task:{} query:{}".format(d["model"], task, d["query"])
        pre, prob = prediction(model_2, text)
        if pre and pre == d["is_right"]:
            sample = d["id"]
            record = dict(
                model=d['model'],
                prob=prob
            )
            if sample in dic:
                dic[sample].append(record)
            else:
                dic[sample] = list()
    for key, value in dic.items():
        if len(value) > 1:
            print(key, value)
            print('**************')
    with open("/data/CoS/scorer/right_prob.json", 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)


