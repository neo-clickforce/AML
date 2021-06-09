from selenium import webdriver
import pandas as pd
import time
import re

def get_text(driver, selector, count=0):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^1-9^！^。^，^？^]")
    try:
        texts = driver.find_elements_by_tag_name(selector)
        text_list=[cop.sub("",t.text) for t in texts if len(t.text)>12 and len(t.text)<600]
        return ''.join(text_list)
    except:
        time.sleep(2)
        count+=1
        if(count <2):
            get_text(driver, selector,count)
        else:
            print("cannot locate element " + selector)

if __name__ =="__main__":
    df = pd.read_csv("crawler_data.csv")
    df.fillna("empty",inplace=True)
    df["len"] = df.crawler.apply(len)
    df_textlost=df.loc[df.len<200,:]
    df = df.loc[~(df.len<200),:]
    df_textlost.reset_index(drop=True,inplace=True)
    
    texts = []
    for idx in range(df_textlost.shape[0]):
    try:
        link = df_textlost.hyperlink[idx]
        driver = webdriver.Chrome("/Users/Moose/Desktop/work/chromedriver")
        driver.get(link)
        time.sleep(5)
        if "http://domestic.judicial.gov.tw" in link:
            texts.append(df_textlost.crawler[idx])
        elif 'https://mops.twse.com.tw' in link:
            texts.append(get_text(driver, "pre"))
        elif "https://hk.on.cc" in link:
            texts.append(get_text(driver, "div.paragraph"))
        elif "https://www.nownews.com" in link:
            texts.append(get_text(driver, "div.newsMsgText"))
        else:
            texts.append(get_text(driver, "p"))
        driver.close()
    except:
        texts.append("empty")
        
    df_textlost["crawler"]=texts
    df_textlost.fillna("empty",inplace=True)
    df_textlost["len"]=df_textlost.crawler.apply(len)
    df=df.append(df_textlost).sort_values(by="news_ID")
    df=df.loc[~(df.len<30),:]
    df["crawler"]=df.crawler.apply(lambda x: x.replace("為達最佳瀏覽效果，建議使用ChromeFirefox或MicrosoftEdge的瀏覽器。",""))
    df.to_csv("data.csv", index=False, encoding="utf-8")

