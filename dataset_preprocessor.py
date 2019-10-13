import json
import csv
import numpy as np
from collections import defaultdict

class dataset_preprocessor:
    def __init__(self, name):
        self.name = name.replace(' ', '-')
        m = open('./documents/dummy_variable_mapping.json', 'r')
        self.map = json.load(m)

        history = open('./history/history_{0}.csv'.format(self.name))
        self.history = csv.reader(history)
    
    def mapping(self, pre_data): 
        dataset = list()
        champion_picks = [0 for i in range(145)]
        
        for data in pre_data[1:]:
            champion_picks[self.map[str(data)]] = 1

        dataset.append(int(pre_data[0]))
        dataset.append(1)   # Bias Column
        dataset.extend(champion_picks)
        return dataset

    def worker(self, raw_data, dataset_type, container, index):       
        pre_data = list()
        pre_data.append(raw_data[0])

        pre_data.extend(raw_data[index["begin"]:index["end"]])
        container.append(self.mapping(pre_data))

    def processor(self):
        dataset = defaultdict(list)
        dataset_my_average = [0 for i in range(145)]

        for raw_data in self.history:              
            if int(raw_data[1]) <= 5:
                myTeam_index = {"begin": 2, "end": 7}
                enemy_index = {"begin": 7, "end": 13}
            else:
                myTeam_index = {"begin": 7, "end": 13}
                enemy_index = {"begin": 2, "end": 7}
            
            #self.worker(raw_data, "myPick", dataset["myPick"], {"begin": int(raw_data[1])+1, "end": int(raw_data[1])+2})
            self.worker(raw_data, "myTeam", dataset["myTeam"], myTeam_index)
            self.worker(raw_data, "enemy", dataset["enemy"], enemy_index)

        for d_set in dataset.items():
            self.save(d_set[0], d_set[1])
        
    def save(self, dataset_type, dataset):
        f = open('./dataset/dataset_{0}_{1}.csv'.format(self.name, dataset_type), 'w', newline='')
        writer = csv.writer(f)
        columns = ['x' + str(i) for i in range(146)]
        columns.insert(0, 'y')
        writer.writerow(columns)
        for data in dataset: writer.writerow(data)

    def my_average(self):
        dataset_my_average = [0 for i in range(145)]
        temp_total = [0 for i in range(145)]
        temp_win = [0 for i in range(145)]
        for data in self.history:
            raw_myPick = data[int(data[1])]
            mapped_myPick = self.map[raw_myPick]

            temp_total[int(mapped_myPick)] += 1
            
            if int(data[0]) == 1:
                temp_win[int(mapped_myPick)] += 1
        
        for i in range(145):
            if temp_total[i]==0 or temp_win==0:
                continue
            dataset_my_average[i] = temp_win[i]/temp_total[i]
        
        with open('./dataset/dataset_{0}_average.csv'.format(self.name), 'w', newline='') as f:
            writer = csv.writer(f)
            columns = ['c' + str(i) for i in range(145)]
            writer.writerow(columns)
            writer.writerow(dataset_my_average)



if __name__ == "__main__":
    preprocessor = dataset_preprocessor("hide on bush")
    preprocessor.processor()
    preprocessor.my_average()