import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron


def matches_label(matches):
    labels = pd.DataFrame()
    labels.insert(0, 'match_api_id', matches['match_api_id'])
    labels['result'] = np.where(matches['home_team_goal'] > matches['away_team_goal'], 'Win', 'Lose')
    return labels

def main():
    # Parsing from sql file to panda's df's
    cnx = sqlite3.connect('database.sqlite')
    country_df = pd.read_sql_query("select * from Country", cnx)
    league_df = pd.read_sql_query("select * from League", cnx)
    match_df = pd.read_sql_query("select * from Match", cnx)
    player_df = pd.read_sql_query("select * from Player", cnx)
    player_att_df = pd.read_sql_query("select * from Player_Attributes", cnx)
    team_df = pd.read_sql_query("select * from Team", cnx)
    team_att_df = pd.read_sql_query("select * from Team_Attributes", cnx)

    # Create labels for matches
    #print(match_df['id'])
    labels = matches_label(match_df)
    print(labels)



if __name__ == "__main__":
    main()