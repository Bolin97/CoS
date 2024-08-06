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


def get_text(data):
    if data['subset']:
        task = data["task"] + "-" + data["subset"]
    else:
        task = data["task"]
    text = "model:{} task:{} query:{}".format(data["model"], task, data["query"])
    return text


if __name__ == '__main__':
    model_2 = DistilBertForSequenceClassification.from_pretrained("/data/CoS/scorer/overlap-results/checkpoint-1770")
    model_2.to('cuda')

    data = {}
    with open('/data/CoS/datasets/test.json', 'r', encoding="utf8") as f:
        tests = json.load(f)
    for d in tests:
        id = d['id']
        if id in data:
            record = dict(
                model=d['model'],
                text=get_text(d),
                label=d['is_right']
            )
            data[id].append(record)
        else:
            data[id] = []
    with open("/data/CoS/datasets/test_text.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    results = []
    for id, texts in data.items():
        goods, bads = [], []
        for text in texts:
            pre, prob = prediction(model_2, text['text'])
            if pre:
                if text['label']:
                    # 正确预测
                    goods.append(dict(model=text['model'], prob=prob[1]))
                else:
                    # 错误预测
                    bads.append(dict(model=text['model'], prob=prob[1]))
        results.append(dict(id=id, good_models=goods, bad_models=bads))
    with open("/data/CoS/scorer/test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)





