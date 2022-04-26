import sqlite3
import pandas as pd


def main():
    cnx = sqlite3.connect('database.sqlite')
    country_df = pd.read_sql_query("select * from Country", cnx)
    league_df = pd.read_sql_query("select * from League", cnx)
    match_df = pd.read_sql_query("select * from Match", cnx)
    player_df = pd.read_sql_query("select * from Player", cnx)
    player_att_df = pd.read_sql_query("select * from Player_Attributes", cnx)
    team_df = pd.read_sql_query("select * from Team", cnx)
    team_att_df = pd.read_sql_query("select * from Team_Attributes", cnx)



    print(country_df)
    print(league_df)
    print(match_df)
    print(player_df)
    print(player_att_df)
    print(team_df)
    print(team_att_df)


if __name__ == "__main__":
    main()