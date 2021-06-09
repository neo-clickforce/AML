from tensorflow import keras
import json
import pickle
import numpy as np
import requests
from sklearn import preprocessing
from ckiptagger import WS
import re
ws = WS("../lib/ckip_data",disable_cuda=False)

def stopwords(PATH):
    stopword_set = set()
    with open(PATH,'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    return stopword_set

stopword_set=stopwords("../lib/corpus-stopwords.txt")
cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^1-9^！^。^，^？^、^]")

with open('../scripts/suspect_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

response = {
    "esun_uuid":"399d69f7-8199-454e-bd79-c0c9571bc98b",
    "server_uuid":"9c324806-fc89-465f-9748-0c194f1873c0",
    "esun_timestamp":1590493849,
    "news":["房產 幽默 大師 的 王派宏 ， 涉吸金 捲款 25億 的 投資人 懷疑 ， 王派宏 已經 捲款 ， 急忙 房產 幽默 大師 的 王派宏 ， 涉吸金 捲款 25億 賣到 印度 ， 尤其 王派宏 還 鼓吹 學員 ，",
            "都市 計畫 委員會 委員 李威儀 與 其 學生 藍秀琪 花蓮 高分院 更三 審判 李威儀 12 年 徒刑 藍秀琪 召集人 台科大 建築系 副教授 李威儀 搭上線 ， 相約 在 西華 飯店 碰面 。 李威儀 在 西華 當面 向 。 幾經 運作 ， 李威儀 趕在 他 的 委員 偵辦 ， 27年 起訴 李威儀 等 人 。 案 審理 12 年 ， 李威儀 等 人 一度 獲判"],
    "retry":2
}

sequen=tokenizer.texts_to_sequences(response["news"])
data = keras.preprocessing.sequence.pad_sequences(sequen,
                   maxlen=100)

payload = {
    "instances": data.tolist()
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:9527/v1/models/suspectClassifier:predict', json=payload)
pred = json.loads(r.content.decode('utf-8'))

print(pred['predictions'])
