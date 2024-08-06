from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
import torch
import json
# model = "anthropic_claude-2.0" label:true  prediction:[[0.50550926 0.49449074]]


def cost_delay_by_model(data, id, model):
    for d in data:
        if d['id'] == id and d['model'] == model:
            return d['cost'], d['delay']


if __name__ == '__main__':
    with open('/data/CoS/datasets/test.json', 'r', encoding="utf8") as f:
        data = json.load(f)
    with open('/data/CoS/scorer/test_results.json', 'r', encoding="utf8") as f:
        tests = json.load(f)
    cost_delay = []
    results = [r for r in tests if len(r['good_models'])>0 or len(r['bad_models'])>0]

    for result in results:
        id = result['id']
        if len(result["good_models"]) == 0:
            # 预测错误
            models = [r["model"] for r in result["bad_models"]]
            probs = [r["prob"] for r in result["bad_models"]]
            max_prob = max(probs)
            model = models[probs.index(max_prob)]
            cost, delay = cost_delay_by_model(data, id, model)
            cost_delay.append(
                dict(id=id, flag=False, min_cost=cost, cost_model=model, cost_model_delay=delay,
                     min_delay=delay, delay_model=model, delay_model_cost=cost)
            )
            # cost_delay.append(
            #     dict(id=id, flag=False, min_cost=cost, min_delay=delay, model=model)
            # )
        else:
            # 预测正确
            models2 = [r["model"] for r in result["good_models"]]
            group_id = [d for d in data if d['id'] == id]
            group_model = [d for d in group_id if d['model'] in models2]
            costs = [d['cost'] for d in group_model]
            delays = [d['delay'] for d in group_model]
            models = [d['model'] for d in group_model]
            # 花费最低优先
            min_cost = min(costs)
            cost_model = models[costs.index(min_cost)]
            cost_model_delay = delays[costs.index(min_cost)]
            # 时延最低优先
            min_delay = min(delays)
            delay_model = models[delays.index(min_delay)]
            delay_model_cost = costs[delays.index(min_delay)]

            cost_delay.append(
                dict(id=id, flag=True, min_cost=min_cost, cost_model=cost_model, cost_model_delay=cost_model_delay,
                     min_delay=min_delay, delay_model=delay_model, delay_model_cost=delay_model_cost)
            )

    with open("/data/CoS/scorer/cost_delay.json", 'w', encoding='utf-8') as f:
        json.dump(cost_delay, f, ensure_ascii=False, indent=4)







