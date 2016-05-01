import csv
import pandas as pd
from sklearn.linear_model import Perceptron

one_hot_cols =  ['0', '5', '7',
        '8', '9', '14',
        '16', '17', '18',
        '20', '23', '25',
        '26', '56', '57',
        '58']

def perceptron_from_nothing():
    df = pd.read_csv('original-data.csv')
    quiz = pd.read_csv('quiz.csv')
    jammed = [df, quiz]
    mega = pd.concat(jammed)
    print('Getting dummies')
    full = pd.get_dummies(mega, columns=one_hot_cols)
    len = 126837
    print('Sectioning array')
    top = full[0:len]
    bottom = full[len:]
    bottom_features = bottom.ix[:, bottom.columns != 'label']
    print('Starting perceptron')
    pout = perceptron(top)
    return pout.predict(bottom_features)


def perceptron(df):
    p = Perceptron(n_iter=15, verbose=True)
    return p.fit(X=df.ix[:, df.columns != 'label'], y=df['label'])


def write_prediction(array, filename):
    with open(filename, 'w') as out_file:
        w = csv.writer(out_file)
        w.writerow(['Id', 'Prediction'])
        w.writerows([i+1, int(val)] for i, val in enumerate(array))