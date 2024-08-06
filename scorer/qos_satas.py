import json

with open('/data/CoS/scorer/cost_delay.json', 'r', encoding="utf8") as f:
    results = json.load(f)
cost = [r['min_cost'] for r in results]
delay = [r['cost_model_delay'] for r in results]

cost2 = [r['min_delay'] for r in results]
delay2 = [r['delay_model_cost'] for r in results]

rights = [r for r in results if r['flag']]

# print(sum(cost)*100/len(cost), sum(delay)/len(delay))
# print(sum(cost2)*100/len(cost2), sum(delay2)/len(delay2))
print(len(rights)/len(results))