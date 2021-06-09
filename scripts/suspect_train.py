from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Concatenate, Dense, Conv1D, Attention, \
                                    GlobalAveragePooling1D, Dropout, MaxPool1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ =="__main__":
    df =pd.read_csv("/home/worker/aml/lib/suspect2new.csv")
    x_train, x_test, y_train, y_test = train_test_split(df.sentence,df.label,test_size=0.1,stratify = df.label,
                                                        random_state=9001,shuffle = True)
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=6000)
    tokenizer.fit_on_texts(x_train)
    with open("./suspect_tokenizer.pickle", 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    x = tokenizer.texts_to_sequences(x_train)
    x_train = keras.preprocessing.sequence.pad_sequences(x,
                   maxlen=200,)
    xt= tokenizer.texts_to_sequences(x_test)
    x_test = keras.preprocessing.sequence.pad_sequences(xt,
                   maxlen=200,)
    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

    NUM_CLASSES = 1

# 在語料庫裡有多少詞彙
    MAX_NUM_WORDS = 6000

# 一個標題最長有幾個詞彙
    MAX_SEQUENCE_LENGTH = 200

# 一個詞向量的維度
    NUM_EMBEDDING_DIM = 256

    model = Sequential()
    model.add(Embedding(MAX_NUM_WORDS,NUM_EMBEDDING_DIM))
    model.add(Conv1D(48, kernel_size=5,padding='same',activation="relu"))
    model.add(MaxPool1D(3,strides=1))
    model.add(Conv1D(48, kernel_size=5,padding='same',activation="relu"))
    model.add(MaxPool1D(3,strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

    BATCH_SIZE = 512
    NUM_EPOCHS =100
    checkpoint = ModelCheckpoint("./suspect_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True,
                mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(
        # 輸入是兩個長度為 20 的數字序列
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=callbacks_list,
        # 每個 epoch 完後計算驗證資料集
        # 上的 Loss 以及準確度
        validation_data=(
            x_test,
            y_test
        ),
        # 每個 epoch 隨機調整訓練資料集
        # 裡頭的數據以讓訓練過程更穩定
        shuffle=True
    )

