import json
from collections import defaultdict

f = open('./ddragon_champions.json', encoding='utf-8')
ddragon = json.loads(f.read())

data = ddragon['data']

championIdList = list()

temp = dict()
for dk in data.keys():
    temp[int(data[dk]["key"])] = dk

print(temp)