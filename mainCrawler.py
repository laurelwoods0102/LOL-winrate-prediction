import requests
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import json
from pprint import pprint

class GameResultCrawler():
    def __init__(self):
        self.dataset = list()
        self.apiKey = "RGAPI-177af57d-a184-43fc-bfae-969769c15ebb"
        self.myName = 'Laurelwoods'
        self.headers = {
            "Origin": "https://developer.riotgames.com",
            "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Riot-Token": self.apiKey,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,zh;q=0.5",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"
        }
        self.time_limit = timedelta(seconds=4)
        self.totalGames = list()
        

        URL_id = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/laurelwoods"
        res_id = requests.get(URL_id, headers=self.headers)
        self.accountId = res_id.json()['accountId']


    def crawlMatchlists(self, counter):
        URL_matchlists = "https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/{0}?endIndex={1}&beginIndex={2}".format(self.accountId, (counter+1)*100+1, counter*100+1)
        res_matchlists = requests.get(URL_matchlists, headers=self.headers)
        
        matchlist = [mtc["gameId"] for mtc in res_matchlists.json()["matches"]]

        self.totalGames.append(res_matchlists.json()["totalGames"])

        return matchlist


    def crawlMatch(self, match):
        URL_match = "https://kr.api.riotgames.com/lol/match/v4/matches/{}".format(match)
        res_match = requests.get(URL_match, headers=self.headers)
        data = res_match.json()

        if res_match.status_code == 429:
            print("pending")
            print(res_match.headers['Retry-After'])
            time.sleep(int(res_match.headers['Retry-After']))

            res_match = requests.get(URL_match, headers=self.headers)
            data = res_match.json()


        myId = None
        queueId = data['queueId']       # 420 for Rank game

        participantIdentities = data['participantIdentities']
        result = None
        for pi in participantIdentities:
            if pi['player']['summonerName'] == self.myName:
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


        # gametype, win, myId, picks1~10
        dataset = list()
        dataset.append(queueId)
        dataset.append(result)
        dataset.append(myId)
        dataset.extend(champions)

        
        return dataset

if __name__ == "__main__":
    crawler = GameResultCrawler()
    print(crawler.accountId)
    list_counter = 0
    matchlist = list()
    dataset = list()
    match_omission = list()
    
    try:
        while True:        
            matchlist.extend(crawler.crawlMatchlists(list_counter))
            list_counter += 1
    except Exception as e:
        print(e)
    finally:
        print("Complete Matchlist")

    try:
        for i, match in enumerate(matchlist):
            try:
                dataset.append(crawler.crawlMatch(match))
                print(i)
            except Exception as e:
                print("{0} : {1}".format(match, e))
    except Exception as e:
        print(e)
    finally:
        np.save('./dataset.npy', dataset)
        print(crawler.totalGames)