import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class sentiment:
    def __init__(self):
        print("Initilizing Sentiment Model")
       # self.prepareData()
        self.createSentimentModel()

    def prepareData(self):
        sentimentPath = "F:\DataSet\IMDBDataset.csv"
        data = pd.read_csv(sentimentPath)
        data['review'] = data['review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x).lower())
        data['setiment_modified']=self.changeData(data)
        return data

    def changeData(self,data):
        dataList=[]
        for i in data['sentiment'].values:
            if i =='positive':
                dataList.append(1)
            elif i=='negative':
                dataList.append(-1)
            else:
                dataList.append(0)
        return dataList


    def createSentimentModel(self):
       self.model=Sequential()
       tokenizer= Tokenizer()
       data=self.prepareData()
       print("data Fetched and Prepared----->Creating Model")
       tokenizer.fit_on_texts(data['review'].values)
       X=tokenizer.texts_to_sequences(data['review'].values)
       X = pad_sequences(X)
       Y=pd.get_dummies(data['review'])
       self.max_rows=len(data['review'].values.max())
       self.max_columns = X.shape[1]
       self.model.add(Embedding(self.max_rows,self.max_columns,input_length=self.max_columns))
       self.model.add(LSTM(150,return_sequences=True))
       self.model.add(LSTM(150))
       self.model.add(Dense(150,activation='relu'))
       self.model.add(Dense(150,activation='softmax'))
       self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
       x_train,y_train,x_test,y_test=train_test_split(X,Y, test_size = 0.33, random_state = 42)
       self.model.fit(x_train,y_train,epochs=50, verbose=1)

    print('model Traing is Completed')


    def predict(self,text):
        text=re.re.sub('[^a-zA-z0-9\s]', '',text).lower()
        encoded_text = self.keras_tokenizer.texts_to_sequences([str])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=self.max_columns, truncating='pre')
        print('outPut is==',self.model.predict(pad_encoded)[0])





if __name__ == '__main__':
    sentimentObject=sentiment()
    sentimentObject.predict("I am feeling bad for this movie")

