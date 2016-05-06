import csv
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import time

multi_category_columns = ['0', '5', '7',
        '8', '9', '14',
        '16', '17', '18',
        '20', '23', '25',
        '26', '56', '57',
        '58']


def adaboost_forest(train, test):
    print("Start")
    train_length = len(train)
    train_feature = train.ix[:, train.columns != 'label']
    label = train['label']
    mega = pd.concat([train_feature, test]).apply(LabelEncoder().fit_transform)
    labelled_train_feature = mega[0:train_length]
    labelled_test = mega[train_length:]

    forest_clf = RandomForestClassifier(n_jobs=4, n_estimators=70)
    adaboost_clf = AdaBoostClassifier(base_estimator=forest_clf, n_estimators=3)
    print("Fitting")
    adaboost_clf.fit(X=labelled_train_feature, y=label)
    print("Predicting")
    return adaboost_clf.predict(labelled_test)


def cross_validate_adaboost_on_forest(train):

    predictions = []

    train_feature = train.ix[:, train.columns != 'label']
    label = train['label']
    labelled_train = train_feature.apply(LabelEncoder().fit_transform)
    labelled_train_feature = labelled_train.ix[:, labelled_train.columns != 'label']
    adaboost_parameters = [3,4]
    forest_parameters = [65,70,75,80]
    criteria = ['gini']
    max_features = ['auto']

    start_time = time.time()
    print('MaxFeature,Criterion,AdaboostParam,ForestParam,Mean,Std,ElapsedSeconds')
    for max_feature in max_features:
        for criterion in criteria:
            for adaboost_parameter in adaboost_parameters:
                for forest_parameter in forest_parameters:
                    forest_clf = RandomForestClassifier(n_estimators=forest_parameter, n_jobs=4, max_features=max_feature, criterion=criterion)
                    adaboost_clf = AdaBoostClassifier(base_estimator=forest_clf, n_estimators=adaboost_parameter)
                    prediction = cross_val_score(adaboost_clf, X=labelled_train_feature, y=label, cv=10)
                    predictions.append(prediction)
                    print("%s,%s,%d,%d,%.6f,%.6f,%.1f" %
                          (max_feature, criterion, adaboost_parameter, forest_parameter, prediction.mean(), prediction.std(), time.time() - start_time))
    return predictions


def random_forest(train, test, n_estimators=160, n_jobs=3):
    train_length = len(train)
    train_feature = train.ix[:, train.columns != 'label']
    label = train['label']
    mega = pd.concat([train_feature, test]).apply(LabelEncoder().fit_transform)
    labelled_train_feature = mega[0:train_length]
    labelled_test = mega[train_length:]
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, verbose=True)
    clf.fit(X=labelled_train_feature, y=label)
    return clf.predict(labelled_test)


def perceptron_from_nothing():
    df = pd.read_csv('original-data.csv')
    quiz = pd.read_csv('quiz.csv')
    jammed = [df, quiz]
    mega = pd.concat(jammed)
    print('Getting dummies')
    full = pd.get_dummies(mega, columns=multi_category_columns)
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