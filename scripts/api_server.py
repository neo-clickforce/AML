from ckiptagger import WS, POS, NER
from flask import Flask
from flask import request
from flask import jsonify
import tensorflow as tf
from tensorflow import keras
from collections import Counter
import json
import time
import requests
import pickle
import hashlib
import numpy as np
import pandas as pd
import os
import re
import sys

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('suspect_tokenizer.pickle', 'rb') as handle:
    suspect_tokenizer = pickle.load(handle)

app = Flask(__name__)

ws = WS("../lib/ckip_data",disable_cuda=True)
pos = POS("../lib/ckip_data",disable_cuda=True)
ner = NER("../lib/ckip_data",disable_cuda=True)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'theone1346@gmail.com'  #
SALT = 'SaveMe9527'                            #
#########################################


def stopwords(PATH):
    stopword_set = set()
    with open(PATH,'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    return stopword_set

stopword_set=stopwords("../lib/corpus-stopwords.txt")
cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^1-9^！^。^，^？^、^]")

def journalist(seg):
    if "記者" in seg or "翻攝" in seg or "撰文" in seg:
        for i,s in enumerate(seg):
            if s == "記者":
                return seg[i+1]
            elif s in ["撰文","翻攝"]:
                return seg[i-1]

def word_preprocess(text):
    text = cop.sub("",text)
    seg=[]
    words = ws([text],
        sentence_segmentation=True, # To consider delimiters
        segment_delimiter_set = {",", "。", ":", "?", "!", ";","，","、","？"})
    for w in words[0]:
        if w not in stopword_set:
            seg.append(w)
    return ' '.join(seg), words

def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)
    t = int(time.time())
    ts = str(t)
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    server_timestamp = t
    return jsonify({'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, \
                'captain_email': CAPTAIN_EMAIL, 'server_timestamp': server_timestamp})

@app.route('/inference', methods=['POST'])
def inference():
    data = request.get_json(force=True)
    esun_timestamp = data['esun_timestamp'] #自行取用
    with open("/home/worker/aml/lib/test_data/day6/%s.log"%esun_timestamp,"w",encoding='utf-8') as output:
        output.write(data["news"])
    timestamp = int(time.time())
    ts = str(int(timestamp))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)

    try:
        seg, word_sentence=word_preprocess(data['news'][:2000])
        sequen=tokenizer.texts_to_sequences([seg])
        input = keras.preprocessing.sequence.pad_sequences(sequen,maxlen=1000)
        payload = {
               "instances": input.tolist()
        }
        r = requests.post('http://localhost:4040/v1/models/amlClassifier:predict', json=payload)
        pred = json.loads(r.content.decode('utf-8'))
        if pred['predictions'][0][0]<0.5:
            answer = []
        else:
            pos_sentence = pos(word_sentence)
            entity_sentence = ner(word_sentence, pos_sentence)
            all_name=[object[3] for object in entity_sentence[0] if object[2] == 'PERSON']
            target = list({i for i in all_name if len(i)>=3})
            if len(target)==1:
                answer = target
            else:
                name_data=[]
                for t in target:
                    ind=[idx for idx, word in enumerate(word_sentence[0]) if word ==t]
                    name_data.append(' '.join([' '.join(word_sentence[0][i-6:i+7]) for i in ind]))
                seq = suspect_tokenizer.texts_to_sequences(name_data)
                input = keras.preprocessing.sequence.pad_sequences(seq,maxlen=100)
                payload = {
                   "instances": input.tolist()
                    }
                r = requests.post('http://localhost:9527/v1/models/suspectClassifier:predict', json=payload)
                pred = json.loads(r.content.decode('utf-8'))
                answer = [target[num] for num, pre in enumerate(pred['predictions']) if pre[0]>0.5]
    except:
        raise ValueError('Model error.')
    server_timestamp = timestamp

    return jsonify({'esun_timestamp': data['esun_timestamp'], 'server_uuid': server_uuid, \
                    'answer': answer, 'server_timestamp': server_timestamp, 'esun_uuid': data['esun_uuid']})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9001, debug=True)
