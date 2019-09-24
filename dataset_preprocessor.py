import json
import csv
import numpy as np
from collections import defaultdict

class dataset_preprocessor:
    def __init__(self, name):
        self.name = name.replace(' ', '-')
        self.apiKey = "RGAPI-593e0a66-6d8d-440b-9fa6-c09d1091d2b7"
        self.headers = {
            "Origin": "https://developer.riotgames.com",
            "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Riot-Token": self.apiKey,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,zh;q=0.5",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"
        }
        m = open('./documents/dummy_variable_mapping.json', 'r')
        self.map = json.load(m)

        raw_dataset_csv = open('./result/raw_dataset_{0}.csv'.format(self.name))        
        self.raw_dataset = csv.reader(raw_dataset_csv)
    
    def mapping(self, pre_data, dataset_type): 
        dataset = list()
        champion_picks = [0 for i in range(145)]
        
        for data in pre_data[1:]:
            champion_picks[self.map[str(data)]] = 1

        dataset.append(pre_data[0])
        dataset.extend(champion_picks)
        return dataset

    def worker(self, raw_data, dataset_type, container, index):       
        pre_data = list()
        pre_data.append(raw_data[0])

        if dataset_type == "myPick":
            pre_data.append(raw_data[int(raw_data[1]) + 1])
            container.append(self.mapping(pre_data, dataset_type))
        else:
            pre_data.extend(raw_data[index["begin"]:index["end"]])
            container.append(self.mapping(pre_data, dataset_type))   

    def processor(self):
        dataset = defaultdict(list)

        for raw_data in self.raw_dataset:
            self.worker(raw_data, "myPick", dataset["myPick"], {"begin": int(raw_data[1])+1, "end": int(raw_data[1])+2})
            
            if int(raw_data[1]) <= 5:
                myTeam_index = {"begin": 2, "end": 7}
                enemy_index = {"begin": 7, "end": 13}
            else:
                myTeam_index = {"begin": 7, "end": 13}
                enemy_index = {"begin": 2, "end": 7}
            self.worker(raw_data, "myTeam", dataset["myTeam"], myTeam_index)
            self.worker(raw_data, "enemy", dataset["enemy"], enemy_index)
            self.worker(raw_data, "totalGame", dataset["totalGame"], {"begin": 2, "end": 13})

        for d_set in dataset.items():
            self.save(d_set[0], d_set[1])

    def save(self, dataset_type, dataset):
        f = open('./dataset/dataset_{0}_{1}.csv'.format(self.name, dataset_type), 'w', newline='')
        writer = csv.writer(f)
        for data in dataset: writer.writerow(data)

if __name__ == "__main__":
    preprocessor = dataset_preprocessor("hide on bush")
    preprocessor.processor()
