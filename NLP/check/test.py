import pandas as pd


def show():
    data=pd.read_csv("F:\DataSet\IMDBDataset.csv")
    X=data['sentiment']
    Y=pd.get_dummies(data['sentiment']).values
    print(Y)




if __name__ == '__main__':
  show()