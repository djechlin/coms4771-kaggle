import csv
import pandas as pd
from sklearn.linear_model import Perceptron
df = pd.read_csv('original-data.csv')
df_feature = df.ix[:, df.columns != 'label']
df_label = df['label']
quiz = pd.read_csv('quiz.csv')
len = 126387
mega = pd.concat([df, quiz])
full = pd.get_dummies(mega)
multi_category_columns = ['0', '5', '7',
                          '8', '9', '14',
                          '16', '17', '18',
                          '20', '23', '25',
                          '26', '56', '57',
                          '58']