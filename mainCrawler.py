import requests
import time
import numpy as np
import json
import csv

class GameResultCrawler():
    def __init__(self):
        self.dataset = list()
        self.apiKey = "RGAPI-c9181828-a8dd-4261-9af3-d185320704fd"
        self.myName = 'hide on bush'
        self.headers = {
            "Origin": "https://developer.riotgames.com",
            "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Riot-Token": self.apiKey,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,zh;q=0.5",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"
        }
        

        URL_id = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}".format(self.myName)
        res_id = requests.get(URL_id, headers=self.headers)
        self.accountId = res_id.json()['accountId']


    def crawlMatchlists(self, counter, season=13, queue=420):
        URL_matchlists = "https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/{0}?season={1}&endIndex={2}&beginIndex={3}".format(self.accountId, season, (counter+1)*100+1, counter*100+1)
        res_matchlists = requests.get(URL_matchlists, headers=self.headers)
        
        matchlist = [mtc["gameId"] for mtc in res_matchlists.json()["matches"]]

        return matchlist


    def crawlMatch(self, match):
        URL_match = "https://kr.api.riotgames.com/lol/match/v4/matches/{}".format(match)
        res_match = requests.get(URL_match, headers=self.headers)
        data = res_match.json()

        if res_match.status_code == 429:
            print("pending")
            time.sleep(int(res_match.headers['Retry-After']))

            res_match = requests.get(URL_match, headers=self.headers)
            data = res_match.json()


        myId = None     # id to identify myPick in match
        queueId = data['queueId']       # 420 for Rank game

        participantIdentities = data['participantIdentities']
        result = None       # win = 1
        for pi in participantIdentities:
            if pi['player']['summonerName'].lower() == self.myName.lower():
                myId = pi['participantId']
                if pi['participantId'] <= 5:
                    if data['teams'][0]['win'] == 'Win': result = 1
                    else:  result = 0
                else:
                    if data['teams'][1]['win'] == 'Win': result = 1
                    else:  result = 0


        participants = data['participants']
        champions = list()
        for ptc in participants:
            champions.append(ptc['championId'])


        # win, myId, [picks1~10]
        dataset = list()
        dataset.append(result)
        dataset.append(myId)
        dataset.extend(champions)

        #with open('./{}.json'.format(match), 'w') as j:
         #   json.dump(data, j, indent=4)
        
        return dataset



if __name__ == "__main__":
    crawler = GameResultCrawler()
    list_counter = 0        # matchlist counter
    matchlist = list()
    dataset = list()
    '''
    try:
        print("Collecting Matchlist")
        while True:        
            matchlist.extend(crawler.crawlMatchlists(list_counter))
            list_counter += 1
    except Exception as e:
        print(e)
    finally:
        print("Complete collecting Matchlist")
        with open('./result/matchlist_{}.json'.format(crawler.myName), 'w') as mt:
            json.dump(matchlist, mt, indent=4)

    '''
    j = open('./result/matchlist_hide on bush.json', 'r')
    matchlist = json.load(j)

    try:
        print("Collecting Match results")
        for match in matchlist:
            try:
                data = crawler.crawlMatch(match)
                print(data)
                dataset.append(data)
            except Exception as e:
                print("{0} : {1}".format(match, e))
    except Exception as e:
        print(e)
    finally:
        #np.save('./raw_dataset_{}.npy'.format(crawler.myName.replace(' ', '-')), dataset)
        f = open('./result/raw_dataset_{}.csv'.format(crawler.myName.replace(' ', '-')), 'w', newline='')
        writer = csv.writer(f)
        for data in dataset: writer.writerow(data)