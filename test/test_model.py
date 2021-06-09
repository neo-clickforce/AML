from tensorflow import keras
import json
import pickle
import numpy as np
import requests
from sklearn import preprocessing
from ckiptagger import WS
import re
import sys
ws = WS("../lib/ckip_data",disable_cuda=False)

def stopwords(PATH):
    stopword_set = set()
    with open(PATH,'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    return stopword_set

stopword_set=stopwords("../lib/corpus-stopwords.txt")
cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^1-9^！^。^，^？^]")

def word_preprocess(text):
    text = cop.sub("",text)
    seg=[]
    words = ws([text],
        sentence_segmentation=True, # To consider delimiters
        segment_delimiter_set = {",", "。", ":", "?", "!", ";","，"})
    for w in words[0]:
        if w not in stopword_set:
            seg.append(w)
    return ' '.join(seg)

with open('../scripts/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

response = {
    "esun_uuid":"399d69f7-8199-454e-bd79-c0c9571bc98b",
    "server_uuid":"9c324806-fc89-465f-9748-0c194f1873c0",
    "esun_timestamp":1590493849,
    "news":sys.argv[1],
    "retry":2
}

seg = word_preprocess(response["news"])
#seg = response["news"]
sequen=tokenizer.texts_to_sequences([seg])
data = keras.preprocessing.sequence.pad_sequences(sequen,
                   maxlen=1000)

payload = {
    "instances": data.tolist()
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:4040/v1/models/amlClassifier:predict', json=payload)
pred = json.loads(r.content.decode('utf-8'))

print(pred)
