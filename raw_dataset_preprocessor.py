import json
import csv
import numpy as np
import pandas as pd

### dataset structure ###
# dataset[0:144] : Champion picks 
# dataset[145] : result {win = 1, lost = 0}
# dataset[146] : myPick



# dummy variable mapping
'''
f = open('./documents/game_document.csv', 'r')
reader = csv.reader(f)

dummy_variable_mapping = dict()

for i, rd in enumerate(reader):
    dummy_variable_mapping[int(rd[0])] = i

g = open('./documents/dummy_variable_mapping.json', 'w', newline='')
json.dump(dummy_variable_mapping, g, indent=4)
'''

m = open('./documents/dummy_variable_mapping.json', 'r')
mapping = json.load(m)

myName = "hide-on-bush"


#raw_dataset = [1, 6, 103, 112, 164, 126, 412, 142, 266, 555, 79, 145]
raw_dataset = np.load('./result/raw_dataset_{}.npy'.format(myName))

dataset = list()
d = open('./dataset/dataset_{}.csv'.format(myName), 'w', newline='')
writer = csv.writer(d)

for raw_data in raw_dataset:
    dataset = list()
         
    champion_picks = [0 for i in range(145)]

    for data in raw_data[2:]:
        #print(mapping[str(data)])
        champion_picks[mapping[str(data)]] = 1

    dataset.extend(champion_picks)
    dataset.append(raw_data[0])     # result

    myPickIndex = raw_data[1]
    dataset.append(mapping[str(raw_data[myPickIndex+1])])       # myPick

    writer.writerow(dataset)
    