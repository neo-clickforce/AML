#!/bin/bash
cd /home/worker/aml/scripts

/home/worker/aml/venv/bin/python3 /home/worker/aml/scripts/api_server_new.py \
>> /home/worker/aml/log/api.log 2>&1 &

tensorflow_model_server --model_base_path=/home/worker/aml/model/ --rest_api_port=4040 --model_name=amlClassifier

tensorflow_model_server --model_base_path=/home/worker/aml/suspect_model/ --rest_api_port=9527 --model_name=suspectClassifier
