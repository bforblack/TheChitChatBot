from pathlib import  Path
#from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import numpy as nm
import os
import string
from string import digits
import matplotlib.pyplot as plt
#%matplotlib inline
import re

import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model

def loadData(dataPath):
    data=pd.read_csv(dataPath,encoding='utf-8')
    #pd=pd[pd["ted"]]
    data = data[data['source'] == 'ted']
    pd.isnull(data).sum()
    data[~pd.isnull(data['english_sentence'])]
    data.drop_duplicates(inplace=True)
    cleanData(data)




def cleanData(data):
   #selecting data for traing
    data=data.sample(frac=0.7,random_state=42)
    #data=data.sample(random_state=42)
   #changing to lowercase
    data['english_sentence']=data['english_sentence'].apply(lambda x:x.lower())
    data['hindi_sentence']=data['hindi_sentence'].apply(lambda  x:x.lower())
     #remove special charaters

    data['english_sentence']=data['english_sentence'].apply(lambda x: re.sub("'",'',x))
    data['hindi_sentence']=data['hindi_sentence'].apply(lambda  x: re.sub("'",'',x))

    exclude=set(string.punctuation)

    data['english_sentence']=data['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))


    remove_digits = str.maketrans('', '', digits)
    data['english_sentence'] = data['english_sentence'].apply(lambda x: x.translate(remove_digits))
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: x.translate(remove_digits))

    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# Remove extra spaces
    data['english_sentence'] = data['english_sentence'].apply(lambda x: x.strip())
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: x.strip())
    data['english_sentence'] = data['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))


    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: 'START_ ' + x + ' _END')
    createVocablary(data)


def createVocablary(data):
    english_vocab=set()
    hindi_vocab=set()
    for eng in data['english_sentence']:
        for engWords in eng.split():
            if engWords not in english_vocab:
                english_vocab.add(engWords)

    for hindi in data['hindi_sentence']:
        for hind_word in hindi.split():
            if hind_word not in hindi_vocab:
                hindi_vocab.add(hind_word)

    data['length_english_sentance']=data['english_sentence'].apply(lambda x:len(x.split(' ')))
    data['length_hindi_sentance']=data['hindi_sentence'].apply(lambda x:len(x.split(' ')))

    print(data.head)


if __name__ == '__main__':
    loadData("F:\Hindi_English_Truncated_Corpus.csv")

