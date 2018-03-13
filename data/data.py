import pandas as pd
import random
import numpy as np

team_stats = {}
stat_fields = ['score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or', 'dr',
    'ast', 'to', 'stl', 'blk', 'pf', 'o_score', 'o_fgm', 'o_fga','o_fgm3', 'o_fga3',
    'o_ftm', 'o_fta', 'o_or', 'o_dr', 'o_ast', 'o_to', 'o_stl','o_blk','o_pf']

# return csv as a pandas dataframe
def format_as_df(csv_file):
    df = pd.read_csv(csv_file)
    return df

# get temporary team stats for a current point in the season while generating test cases
def get_stat_temp(season, team, field):
    try:
        stat = team_stats[season][team][field]
        return sum(stat) / float(len(stat))
    except:
        return 0

# get final team stats for a season, passing in an array of team stats, to use for prediction
def get_stat_final(season, team, field, team_stats):
    try:
        stat = team_stats[season][team][field]
        return sum(stat) / float(len(stat))
    except:
        return 0

# update team stats based on most recent game
def update_stats(season, team, fields):
    if team not in team_stats[season]:
        team_stats[season][team] = {}

    for key, value in fields.items():
        # Make sure we have the field
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []

        # we only want to keep track of n most recent games, so get rid of the oldest if necessary
        if len(team_stats[season][team][key]) >= 15:
            team_stats[season][team][key].pop()

        team_stats[season][team][key].append(value)

# use outside of this file when we want to get game features so that we can predict using any model
def get_game_features(team_1, team_2, loc, season, all_stats):
    # both teams are "away" since it is a tourney game
    if (loc == 1):
        # first team is home, second is away
        features = [0, 1]
    elif (loc == -1):
        # first team is away, second is home
        features = [1, 0]
    else:
        # both teams are "away" (neutral)
        features = [1, 1]

    # Team 1
    for stat in stat_fields:
        features.append(get_stat_final(season, team_1, stat, all_stats))

    # Team 2
    for stat in stat_fields:
        features.append(get_stat_final(season, team_2, stat, all_stats))

    return features

# get the ids of all teams in the tourney and a map of team ids to their actual name
def get_tourney_teams(year):
    seeds = pd.read_csv('../data' + tourneyYear + '/TourneySeeds.csv')
    tourney_teams = []
    for index, row in seeds.iterrows():
        if row['season'] == year:
            tourney_teams.append(row['Team'])

    team_id_map = get_team_dict()

    return tourney_teams, team_id_map

# get a map of team ids to team names
def get_team_dict():
    teams = pd.read_csv('../data' + tourneyYear + '/Teams.csv')
    team_map = {}
    for index, row in teams.iterrows():
        team_map[row['Team_Id']] = row['Team_Name']
    return team_map

# build test cases for all seasons
def build_season_data(data):
    X = []
    y = []

    for index, row in data.iterrows():

        skip = 0
        firstLoc = 0
        secondLoc = 0
        randNum = random.random()
        row.index = row.index.str.lower()

        # home = 0, away and neutral = 1
        if (randNum > 0.5):
            if (row['wloc'] == 'H'):
                # first team is home, second is away
                firstLoc = 0
                secondLoc = 1
            elif (row['wloc'] == 'A'):
                # first team is away, second is home
                firstLoc = 1
                secondLoc = 0
            else:
                # both teams are "away" (neutral)
                firstLoc = 1
                secondLoc = 1
        else:
            if (row['wloc'] == 'H'):
                # first team is away, second is home
                firstLoc = 1
                secondLoc = 0
            elif (row['wloc'] == 'A'):
                # first team is home, second is away
                firstLoc = 0
                secondLoc = 1
            else:
                # both teams are "away" (neutral)
                firstLoc = 1
                secondLoc = 1

        loc = [firstLoc, secondLoc]

        team_1_features = []
        team_2_features = []

        # get all other team statistics
        for field in stat_fields:
            team_1_stat = get_stat_temp(row['season'], row[row.index.str.contains('wteam')][0], field)
            team_2_stat = get_stat_temp(row['season'], row[row.index.str.contains('lteam')][0], field)

            if team_1_stat is not 0 and team_2_stat is not 0:
                team_1_features.append(team_1_stat)
                team_2_features.append(team_2_stat)
            else:
                skip = 1

        # if skip = 0, it is the first game of the season so we have no prior statistics (everything is 0)
        # we label as '0' if team1 won, label as '1' if team2 won
        if skip == 0:
            if randNum > 0.5:
                X.append(loc + team_1_features + team_2_features)
                y.append(0)
            else:
                X.append(loc + team_2_features + team_1_features)
                y.append(1)

        # Update teams' overall stats so that they can later be averaged and used to make predictions
        stat_1_fields = {
            'score': row['wscore'],
            'fgm': row['wfgm'],
            'fga': row['wfga'],
            'fgm3': row['wfgm3'],
            'fga3': row['wfga3'],
            'ftm': row['wftm'],
            'fta': row['wfta'],
            'or': row['wor'],
            'dr': row['wdr'],
            'ast': row['wast'],
            'to': row['wto'],
            'stl': row['wstl'],
            'blk': row['wblk'],
            'pf': row['wpf'],
            'o_score': row['lscore'],
            'o_fgm': row['lfgm'],
            'o_fga': row['lfga'],
            'o_fgm3': row['lfgm3'],
            'o_fga3': row['lfga3'],
            'o_ftm': row['lftm'],
            'o_fta': row['lfta'],
            'o_or': row['lor'],
            'o_dr': row['ldr'],
            'o_ast': row['last'],
            'o_to': row['lto'],
            'o_stl': row['lstl'],
            'o_blk': row['lblk'],
            'o_pf': row['lpf']
        }
        stat_2_fields = {
            'score': row['lscore'],
            'fgm': row['lfgm'],
            'fga': row['lfga'],
            'fgm3': row['lfgm3'],
            'fga3': row['lfga3'],
            'ftm': row['lftm'],
            'fta': row['lfta'],
            'or': row['lor'],
            'dr': row['ldr'],
            'ast': row['last'],
            'to': row['lto'],
            'stl': row['lstl'],
            'blk': row['lblk'],
            'pf': row['lpf'],
            'o_score': row['wscore'],
            'o_fgm': row['wfgm'],
            'o_fga': row['wfga'],
            'o_fgm3': row['wfgm3'],
            'o_fga3': row['wfga3'],
            'o_ftm': row['wftm'],
            'o_fta': row['wfta'],
            'o_or': row['wor'],
            'o_dr': row['wdr'],
            'o_ast': row['wast'],
            'o_to': row['wto'],
            'o_stl': row['wstl'],
            'o_blk': row['wblk'],
            'o_pf': row['wpf']
        }
        update_stats(row['season'], row[row.index.str.contains('wteam')][0], stat_1_fields)
        update_stats(row['season'], row[row.index.str.contains('lteam')][0], stat_2_fields)

    trainX = np.array(X)
    trainY = np.array(y)

    return trainX, trainY

def get_data(tourney_year):
    year = tourney_year

    for i in range(1985, year+1):
        team_stats[i] = {}

    season_detailed_results = format_as_df('./data/raw/data' + str(year) + '/RegularSeasonDetailedResults.csv')
    tourney_detailed_results = format_as_df('./data/raw/data' + str(year) + '/TourneyDetailedResults.csv')
    frames = [season_detailed_results, tourney_detailed_results]
    data = pd.concat(frames)

    X, Y = build_season_data(data)

    return X, Y, team_stats

if __name__ == "__main__":
    trainingX, trainingY, team_stats = get_data(2018)
