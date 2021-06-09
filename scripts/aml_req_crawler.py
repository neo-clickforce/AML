import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import requests
import random
import time
import re

def crawler(x):
    try:
        headers = {'User-Agent': ua["google chrome"]}
        ruq = requests.get(x,headers=headers,allow_redirects=False)
        time.sleep(random.uniform(1, 5))
        ruq.encoding = "unicode"
        innertext = BeautifulSoup(ruq.text,"lxml")
        if "http://domestic.judicial.gov.tw" in x:
            ruq = requests.get(x)
            ruq.encoding = "Big5"
            innertext = BeautifulSoup(ruq.text,"lxml")
            getText = innertext.select("pre")
        elif 'https://mops.twse.com.tw' in x:
            getText = innertext.select("pre")
        elif "https://hk.on.cc" in x:
            getText = innertext.select("div.paragraph")
        elif "https://www.nownews.com" in x:
            getText = innertext.select("div.newsMsgText")
        else:
            getText = innertext.select("p")
        response = [cop.sub('',key.text) for key in getText]
        #response = [r for r in response if len(r) > 12 and len(r)<500]
    except:
        response = []
    return response

if __name__ =="__main__":
    df=pd.read_csv("tbrain_train_final_0610.csv")
    df['domain']=df.hyperlink.apply(lambda x: '/'.join(x.split('/')[:3]))
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^1-9^！^。^，^？^、^]")
    ua = UserAgent()
    result=df.hyperlink.apply(crawler)
    df['crawler'] = result
    df['crawler']=df.crawler.apply(lambda x: ''.join(x))
    #data=df.loc[~(df.crawler==''),:]
    label = [1 if len(i)>2 else 0 for i in df.name]
    df['label'] = label
    df.to_csv('../lib/crawler_data.csv',index=False, encoding='utf-8')
