import sqlite3
import pandas as pd
import numpy as np


def main():
    cnx = sqlite3.connect('database.sqlite')
    country_df = pd.read_sql_query("select * from Country", cnx)
    league_df = pd.read_sql_query("select * from League", cnx)
    match_df = pd.read_sql_query("select * from Match", cnx)
    player_df = pd.read_sql_query("select * from Player", cnx)
    player_att_df = pd.read_sql_query("select * from Player_Attributes", cnx)
    team_df = pd.read_sql_query("select * from Team", cnx)
    team_att_df = pd.read_sql_query("select * from Team_Attributes", cnx)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    match_fil_df = match_df[['home_team_api_id', 'stage', 'home_team_goal', 'away_team_goal']]

    X = team_att_df[['team_api_id', 'date', 'buildUpPlaySpeed', 'buildUpPlayDribbling', 'buildUpPlayPassing',
                     'chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting', 'defencePressure',
                     'defenceAggression', 'defenceTeamWidth']]
    # X = X.groupby(['team_api_id'])['date'].transform(max)
    X = X.sort_values('date', ascending=False).drop_duplicates(['team_api_id'])
    X = X.sort_values('team_api_id')
    X = X.drop('date', axis=1)
    X = pd.merge(match_fil_df, X, left_on='home_team_api_id', right_on='team_api_id')
    X = X.drop(['home_team_api_id', 'team_api_id'], axis=1)

    X_train = X[:int(len(X) * 0.8)]
    X_test = X[int((len(X) * 0.8)):]

    y = pd.DataFrame()
    y['match_result'] = np.where(X['home_team_goal'] > X['away_team_goal'], 1, 0)

    y_train = y[:int(len(y) * 0.8)]
    y_test = y[int(len(y) * 0.8):]

    print(X.head(10))
    print(y.head(10))


if __name__ == "__main__":
    main()