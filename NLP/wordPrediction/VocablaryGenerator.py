import numpy as nm
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences




class WordPredictionDataSet:
    def __init__(self):
        self.fectDataSet("F:\Hindi_English_Truncated_Corpus.csv")

    def fectDataSet(self,path):
        data = pd.read_csv(path,encoding='utf-8')
        data=data[data['source']=='ted']
        englishVocabData=data['english_sentence'].sample(frac=.1)
       #for handling memory issue
        self.createEnglishVocab(englishVocabData)

    def createEnglishVocab(self,englishVocabData):
        complete_sentence=''
        for sentence in englishVocabData:
                complete_sentence=complete_sentence+" "+sentence
        print('Sentence creation is Success processing towards cleaning and vocablary creartion')
        self.token=[]
        cleaned_text=word_tokenize(re.sub(r'/W+','',complete_sentence).lower())
        train_length=4
        for i in range(train_length,len(cleaned_text)):
            self.token.append(cleaned_text[i-train_length:i])
        self.sequence={}
        count=1

        for i in range(len(cleaned_text)):
            if cleaned_text[i] not in self.sequence:
                self.sequence[cleaned_text[i]]=count
                count+=1


        self.keras_tokenizer=Tokenizer()
        self.keras_tokenizer.fit_on_texts(self.token)
        self.sequence=self.keras_tokenizer.texts_to_sequences(self.token)

        self.vocablary_size=len(self.keras_tokenizer.word_counts)+1
        self.n_sequence=nm.empty([len(self.sequence),train_length],dtype='uint8')

        for i in range (len(self.sequence)):
            self.n_sequence[i]=self.sequence[i]

        #print(self.n_sequence)
        #print('sucess')
        ##Check what it does
        self.train_input=self.n_sequence[:,:-1]
        #print(self.train_input)
        self.train_target=self.n_sequence[:,-1]
        self.train_target=to_categorical(self.train_target,num_classes=self.vocablary_strain_targetize)
        self.seq_len = self.train_input.shape[1]
        print('Vocablary creation SucssFull Processing Towoards Model creation')
        self.processTraing()

    def processTraing(self):
        self.model=Sequential()
        self.model.add(Embedding(self.vocablary_size, self.seq_len, input_length=self.seq_len))
        self.model.add(LSTM(250, return_sequences=True))
        self.model.add(LSTM(250))
        self.model.add(Dense(250, activation='relu'))
        self.model.add(Dense(self.vocablary_size, activation='softmax'))
        # compiling the network
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.train_input, self.train_target, epochs=50, verbose=1)
        print('Training of Model is Complete')

    def testData(self,text):
        str=text
        str.strip().lower();
        encoded_text= self.keras_tokenizer.texts_to_sequences([str])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=self.seq_len, truncating='pre')
        print('endoed Text= ',encoded_text,'pad_encoded= ', pad_encoded)
        for i in (self.model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
            pred_word = self.keras_tokenizer.index_word[i]
            print("Next word suggestion:", pred_word)





if __name__ == '__main__':
    wr= WordPredictionDataSet()
    str="I am going to check"
    wr.testData(str)









