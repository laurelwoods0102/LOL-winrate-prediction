import requests
import time
import numpy as np
import json
import csv

class GameResultCrawler():
    def __init__(self, name):
        self.dataset = list()
        self.apiKey = "RGAPI-44189125-9503-474a-93c0-61f22a3ddbdf"
        self.myName = name
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

    def crawlMatchlists(self, counter, season, queue=420): # queue=420 for Rank game
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

        participantIdentities = data['participantIdentities']
        result = None       # win = 1
        for pi in participantIdentities:
            if pi['player']['accountId'] == self.accountId:
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
        
        return dataset

    def main_crawler(self, exist_matchlist=False):
        season = 13
        matchlist = list()
        dataset = list()

        if exist_matchlist:
            m = open('./result/matchlist_{}.json'.format(self.myName.replace(' ', '-')), 'r')
            matchlist = json.load(m)
        else:
            list_counter = 0        # matchlist counter
            try:
                print("Collecting Matchlist")
                while True:        
                    matchlist.extend(crawler.crawlMatchlists(list_counter, season))
                    list_counter += 1
            except Exception as e:
                print(e)
            finally:
                print("Complete collecting Matchlist")
                with open('./result/matchlist_{}.json'.format(self.myName.replace(' ', '-')), 'w') as mt:
                    json.dump(matchlist, mt, indent=4)

        try:
            print("Collecting Match results")
            for match in matchlist:
                try:
                    data = crawler.crawlMatch(match)
                    #print(data)
                    dataset.append(data)
                except Exception as e:
                    print("{0} : {1}".format(match, e))
        except Exception as e:
            print(e)
        finally:
            with open('./history/history_{}.csv'.format(self.myName.replace(' ', '-')), 'w', newline='') as f:
                writer = csv.writer(f)
                for data in dataset: writer.writerow(data)




if __name__ == "__main__":
    crawler = GameResultCrawler('hide on bush')
    crawler.main_crawler(exist_matchlist=True)