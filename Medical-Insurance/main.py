import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/HP/Desktop/insurance.csv')

from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def run():
    ax = input("Choose one:\nAge\nBMI\nChildren\n")
    x = df[f'{ax}'].values.reshape(-1,1)
    y = df['charges']
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.7, random_state=0)
    x_train.shape, x_test.shape, y_train.shape, y_test.shape
    Lin_reg = LinearRegression()
    Lin_reg.fit(x_train, y_train)

    train = plt
    train.scatter(x_train, y_train, color='red')
    train.plot(x_train, Lin_reg.predict(x_train), color='green')
    train.title("Medical Insurance")
    train.xlabel(f'{ax}')
    train.ylabel("Charges")
    plt.show()

if __name__=="__main__":
    run()
