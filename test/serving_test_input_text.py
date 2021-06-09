import json
import pickle
import numpy as np
import requests
from sklearn import preprocessing
import time
import re
import sys

response = {
    "esun_uuid":"399d69f7-8199-454e-bd79-c0c9571bc98b",
    "server_uuid":"9c324806-fc89-465f-9748-0c194f1873c0",
    "esun_timestamp":time.time(),
    "news":sys.argv[1],
    "retry":2
}

st=time.time()
r = requests.post('http://localhost:9001/inference', json=response)
pred = json.loads(r.content.decode('utf-8'))
en = time.time()

print("prediction: ", pred, "time: ",en-st)
