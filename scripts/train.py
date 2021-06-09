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
from ckiptagger import WS
import pickle
import os

def stopwords(PATH):
    stopword_set = set()
    with open(PATH,'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    return stopword_set

def word_preprocess(text):
    seg=[]
    words = ws([text],
        sentence_segmentation=True, # To consider delimiters
        segment_delimiter_set = {",", "。", ":", "?", "!", ";","，","！","？","、"})
    for w in words[0]:
        if w not in stopword_set:
            seg.append(w)
    return ' '.join(seg)

class Preprocess:
    def load_data(self):
        df = pd.read_csv("../lib/data.csv")
        self.raw = df
        self.df = df.loc[:,["seg","label"]]

    def split(self):
        VALIDATION_RATIO = 0.1
        RANDOM_STATE = 9527

        self.x_train, self.x_test, self.y_train, self.y_test = \
        train_test_split(self.df.seg, self.df.label, 
                        test_size=VALIDATION_RATIO,
                        stratify = self.df.label,
                        random_state=RANDOM_STATE, 
                        shuffle=True)
        self.x_train.to_csv("preprocessed_data.csv", index=False, encoding='utf-8')
        
    def data_preprocess(self, max_num_words, max_sequence_length):
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_num_words)
        tokenizer.fit_on_texts(self.x_train)
        with open('../scripts/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        #with open("./tokenizer.pickle", 'wb') as handle:
        #    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.tokenizer = tokenizer
        x = tokenizer.texts_to_sequences(self.x_train)
        self.x_train = keras.preprocessing.sequence.pad_sequences(x,
                   maxlen=max_sequence_length,)
        
        xt=tokenizer.texts_to_sequences(self.x_test)
        self.x_test = keras.preprocessing.sequence.pad_sequences(xt,
                   maxlen=max_sequence_length,)
        
    def main(self, max_num_words, max_sequence_length):
        self.load_data()
        self.split()
        self.data_preprocess(max_num_words,max_sequence_length)
        return self.x_train, np.asarray(self.y_train).astype('float32').reshape((-1,1)), \
    self.x_test, np.asarray(self.y_test).astype('float32').reshape((-1,1))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ws = WS("../lib/ckip_data",disable_cuda=False)
stopword_set=stopwords("../lib/corpus-stopwords.txt")

if __name__ =="__main__":
    preprocess = Preprocess()
    x_train, y_train, x_test, y_test = preprocess.main(100000,1200)
    NUM_CLASSES = 1

    MAX_NUM_WORDS = 100000

    MAX_SEQUENCE_LENGTH = 1200

    NUM_EMBEDDING_DIM = 300

    model = Sequential()
    model.add(Embedding(MAX_NUM_WORDS,NUM_EMBEDDING_DIM))
    model.add(Conv1D(256, kernel_size=13,padding='same'))
    model.add(MaxPool1D(7))
    model.add(Dropout(0.4))
    model.add(Conv1D(512, kernel_size=13,padding='same'))
    model.add(MaxPool1D(7))
    model.add(Dropout(0.4))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1,activation='sigmoid'))

    model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

    BATCH_SIZE = 512
    NUM_EPOCHS = 120
    checkpoint = ModelCheckpoint("./model_new.hdf5", monitor='val_acc', verbose=1, save_best_only=True,
                mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=callbacks_list,
        validation_data=(
            x_test,
            y_test
        ),
        shuffle=True
    )
