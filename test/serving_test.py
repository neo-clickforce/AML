#from tensorflow import keras
import json
import pickle
import numpy as np
import requests
from sklearn import preprocessing
#from ckiptagger import WS
import re
#ws = WS("../lib/ckip_data",disable_cuda=False)

#def stopwords(PATH):
#    stopword_set = set()
#    with open(PATH,'r', encoding='utf-8') as stopwords:
#        for stopword in stopwords:
#            stopword_set.add(stopword.strip('\n'))
#    return stopword_set

#stopword_set=stopwords("../lib/corpus-stopwords.txt")
#cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^1-9^！^。^，^？^]")

#def word_preprocess(text):
#    text = cop.sub("",text)
#    seg=[]
#    words = ws([text],
#        sentence_segmentation=True, # To consider delimiters
#        segment_delimiter_set = {",", "。", ":", "?", "!", ";","，"})
#    for w in words[0]:
#        if w not in stopword_set:
#            seg.append(w)
#    return ' '.join(seg)

#with open('../scripts/tokenizer.pickle', 'rb') as handle:
#    tokenizer = pickle.load(handle)

response = {
    "esun_uuid":"399d69f7-8199-454e-bd79-c0c9571bc98b",
    "server_uuid":"9c324806-fc89-465f-9748-0c194f1873c0",
    "esun_timestamp":1590493849,
    "news":"""自稱房產幽默大師的王派宏，涉吸金捲款25億落跑！他自稱炒房專家，在全台授課分享理財，卻遭指控，4月28號突然聯繫不上，也沒上課，顧問公司以為他發生危險立刻報案，但人卻已經離開台灣，被害的投資人懷疑，王派宏已經捲款，急忙報警追人。自稱房產幽默大師的王派宏，涉吸金捲款25億落跑！圖東森新聞被害人小吳他有提到從台灣，帶黃金到印度，個月會有3分紅，除了他介紹以外，透過很多學員都已經開始投資，而且過去23年都ok，都有如期拿到利息，的確有固定利息，從小金額1多萬，到後面越投越多，老師也一直鼓吹，所以投到後來變1千多萬。4多歲的受害人，因為遭裁員決定把畢生積蓄，拿去投資黃金變粉末賣到印度，尤其王派宏還鼓吹學員，甚至還簽了本票讓所有人深信不疑。自稱炒房專家，在全台授課分享理財，卻遭指控4月28號突然聯繫不上。圖東森新聞被害人小吳有跟我們看工廠的狀況，有看到實際的，每個月有拿到利息，外面很多投資案，第一個他是老師，很多學員畢竟時間很久了也有出過書，被人家採訪過，他又開本票，其他投資案沒有的，給了我們很大的放心。知名度加上人人說好，以為是一線希望反而血本無歸。圖東森新聞顧問公司也透漏，不少學員和員工都是受害者，如果學員上課受影響需要退費會幫忙處理，這回誆稱投資卻遭指控吸金大會，被害人只希望能盡快他們口中的王老師，能夠盡早面對""",
    "retry":2
}

#seg = word_preprocess(response["news"])
#sequen=tokenizer.texts_to_sequences([seg])
#data = keras.preprocessing.sequence.pad_sequences(sequen,
#                   maxlen=1000)

#payload = {
#    "instances": data.tolist()
#}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:9001/inference', json=response)
pred = json.loads(r.content.decode('utf-8'))

print(pred)
