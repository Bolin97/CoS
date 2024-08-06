import requests
import json

def download_json(url, local_path):
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        # 下载成功，保存到本地文件
        with open(local_path, 'w', encoding='utf-8') as file:
            json.dump(response.json(), file, ensure_ascii=False, indent=4)
        print(f"JSON file downloaded and saved at {local_path}")
    else:
        # 下载失败，打印错误信息
        print(f"Failed to download JSON file. Status code: {response.status_code}")
        
if __name__ == '__main__':
    # download to extract model_name and task_name
    # schema_url = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.2.0/schema.json"
    # download_json(schema_url, "/data/CoS/crawler/schema.json")

    # load schema to fill-in the scenario slot of rank_urls 
    # with open('/data/CoS/crawler/schema.json', encoding='utf8') as f:
    #     data = json.load(f)
    # scenarios = data["run_groups"][0]["subgroups"]
    # for scenario in scenarios:
    #     rank_url = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.2.0/groups/{}.json".format(scenario)
    #     print(rank_url)
    with open('/data/CoS/crawler/{}.json'.format('narrative_qa'), encoding='utf8') as f:
        data = json.load(f)