import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Get data from the database file.
def Get_Data():
    cnx = sqlite3.connect('database.sqlite')
    country_df = pd.read_sql_query("select * from Country", cnx)
    league_df = pd.read_sql_query("select * from League", cnx)
    match_df = pd.read_sql_query("select * from Match", cnx)
    player_df = pd.read_sql_query("select * from Player", cnx)
    player_att_df = pd.read_sql_query("select * from Player_Attributes", cnx)
    team_df = pd.read_sql_query("select * from Team", cnx)
    team_att_df = pd.read_sql_query("select * from Team_Attributes", cnx)
    return country_df, league_df, match_df, player_df, player_att_df, team_df, team_att_df

# Clean data.
def Clean_Data(match_df, team_att_df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    match_fil_df = match_df[['home_team_api_id', 'stage', 'home_team_goal', 'away_team_goal', 'B365H', 'B365A', 'BWH', 'BWA', 'IWH', 'IWA', 'LBH', 'LBA', 'PSH', 'PSA']]

    X = team_att_df[['team_api_id', 'date', 'buildUpPlaySpeed', 'buildUpPlayDribbling', 'buildUpPlayPassing',
                     'chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting', 'defencePressure',
                     'defenceAggression', 'defenceTeamWidth']]
    # X = X.groupby(['team_api_id'])['date'].transform(max)
    X = X.sort_values('date', ascending=True).drop_duplicates(['team_api_id'])
    # X = X.sort_values('team_api_id')
    # TimeWeight(X)
    X = X.drop('date', axis=1)
    X = pd.merge(match_fil_df, X, left_on='home_team_api_id', right_on='team_api_id')
    X = X.drop(['home_team_api_id', 'team_api_id'], axis=1)
    X['mean_rows'] = X.mean(axis=1)

    for column in X:
        X[column].fillna(X.mean_rows, inplace=True)

    return X

# Get bookkeepers' accuracy results to compare with.
def Bookkeeper_Result(X, y):
    # get bookkeepers' labels
    y365 = pd.DataFrame()
    y365['365_result'] = np.where(X['B365H'] < X['B365A'], 1, 0)
    yBW = pd.DataFrame()
    yBW['BW_result'] = np.where(X['BWH'] < X['BWA'], 1, 0)
    yIW = pd.DataFrame()
    yIW['IW_result'] = np.where(X['IWH'] < X['IWA'], 1, 0)
    yLB = pd.DataFrame()
    yLB['LB_result'] = np.where(X['LBH'] < X['LBA'], 1, 0)
    yPS = pd.DataFrame()
    yPS['PS_result'] = np.where(X['PSH'] < X['PSA'], 1, 0)

    # print bookkeeper's accuracy
    bet_score = accuracy_score(y, y365)
    print("Bet365's accuracy", bet_score)
    bet_score = accuracy_score(y, yBW)
    print("BW's accuracy", bet_score)
    bet_score = accuracy_score(y, yIW)
    print("IW's accuracy", bet_score)
    bet_score = accuracy_score(y, yLB)
    print("LB's accuracy", bet_score)
    bet_score = accuracy_score(y, yPS)
    print("PS's accuracy", bet_score)

# The data is sorted by the date, the more recent the data, the higher the weight of the data.
# Could not get the weight of the date correctly
def TimeWeight(X):
    w = pd.DataFrame(X['date'])
    # w['weight'] = (w['date'] != w['date'].shift()).any(axis=0).cumsum()
    w['weight'] = (w['date'] != w['date'].shift())
    w['weight'] = w['weight'].any(axis=0)
    w['weight'] = w['weight'].cumsum()

    print(w.head(20))

    

# Labelling result of matches.
def Result_Label(X):
    y = pd.DataFrame()
    y['match_result'] = np.where(X['home_team_goal'] > X['away_team_goal'], 1, 0)
    return y

def main():
    # Get data from the database file.
    country_df, league_df, match_df, player_df, player_att_df, team_df, team_att_df = Get_Data()

    # Clean data.
    X = Clean_Data(match_df, team_att_df)

    # get labels
    y = Result_Label(X)

    # Get bookkeepers' accuracy results to compare with.
    Bookkeeper_Result(X, y)

    # Drop unused feature columns.
    X = X.drop(['home_team_goal', 'away_team_goal', 'mean_rows', 'B365H', 'B365A', 'BWH', 'BWA', 'IWH', 'IWA', 'LBH', 'LBA', 'PSH', 'PSA'], axis=1)

    # Spliting data
    X_train = X[:int(X.shape[0] * 0.9)]
    X_test = X[int((X.shape[0] * 0.9)):]

    y_train = y[:int(X.shape[0] * 0.9)]
    y_test = y[int(X.shape[0] * 0.9):]

    clf = Perceptron(validation_fraction=0.2, random_state=0)
    dtc = DecisionTreeClassifier(random_state=0)
    gnbc = GaussianNB()
    # TRAINING
    clf.fit(X_train, y_train.values.ravel())
    dtc.fit(X_train, y_train.values.ravel())
    gnbc.fit(X_train, y_train.values.ravel())
    # TESTING
    y_pred_clf = clf.predict(X_test)
    y_pred_dtc = dtc.predict(X_test)
    y_pred_gnbc = gnbc.predict(X_test)
    clf_score = accuracy_score(y_test, y_pred_clf)
    dtc_score = accuracy_score(y_test, y_pred_dtc)
    gnbc_score = accuracy_score(y_test, y_pred_gnbc)

    # Accuracy was somewhat close across 3 different classifiers, with the worst performing one being the
    # decision tree, and the best one being Gaussian Naives Bayes, barely beating out the Perceptron classifier.
    # However, when the training data size is changed to 80% from 90% that is when the difference
    # is more notable and Gaussian Naives Bayes becomes the clear winner.
    print("\nPerceptron testing accuracy", clf_score)
    print("Decision Tree testing accuracy", dtc_score)
    print("Gaussian Naives Bayes  testing accuracy", gnbc_score)
    print('\nPerceptron was', clf_score - dtc_score, '% more accurate than decision tree')
    print('Perceptron was', clf_score - gnbc_score, '% more accurate than gaussian naives bayes')


    # print(X.head(10))
    # print(y.head(10))


if __name__ == "__main__":
    main()