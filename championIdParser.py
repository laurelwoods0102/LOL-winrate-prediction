import json
import csv

f = open('./json/ddragon_champions.json', encoding='utf-8')
ddragon = json.loads(f.read())

data = ddragon['data']

championIdList = list()

with open('./documents/game_document.csv', 'w', newline='') as d:
    wr = csv.writer(d)
    for dk in data.keys():
        temp = list()
        temp.append(int(data[dk]["key"]))
        temp.append(dk)
        wr.writerow(temp)