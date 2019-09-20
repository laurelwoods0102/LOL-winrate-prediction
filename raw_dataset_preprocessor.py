import json
import csv
import numpy as np
import pandas as pd

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
mapping = json.dump(m)

#raw_dataset = np.load('./result/raw_dataset_hide-on-bush.npy')
raw_dataset = [1, 6, 51, 112, 164, 126, 412, 142, 266, 555, 79, 145]

champion_picks = np.zeros((145,), dtype=int)

for data in raw_dataset[2:]:
    champion_picks[data] = 1


