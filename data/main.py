import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import shotchartdetail, leaguedashplayerstats
import time
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Constants
API_DELAY = 1
RAW_OUTPUT_FILE = 'warriors_player_shots_with_defense.csv'
PROCESSED_OUTPUT_FILE = 'processed_data.csv'

def get_warriors_player_ids(season):
    warriors = [team for team in teams.get_teams() if team['full_name'] == "Golden State Warriors"][0]
    warriors_id = warriors['id']
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    warriors_players = player_stats[player_stats['TEAM_ID'] == warriors_id][['PLAYER_ID', 'PLAYER_NAME']]
    return warriors_players.rename(columns={'PLAYER_ID': 'id', 'PLAYER_NAME': 'full_name'})

def get_most_recent_season():
    current_year = pd.Timestamp.today().year
    return f"{current_year - 1}-{str(current_year)[-2:]}"

def fetch_player_shots(player_id, player_name, season):
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            player_id=player_id,
            season_nullable=season,
            team_id=0,
            context_measure_simple='FGA'
        ).get_data_frames()[0]
        if shot_data.empty:
            return pd.DataFrame()
        shot_data['PLAYER_NAME'] = player_name
        shot_data = shot_data[['GAME_ID', 'PLAYER_ID', 'PLAYER_NAME', 'LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 
                               'SHOT_MADE_FLAG']]
        time.sleep(API_DELAY)
        return shot_data
    except Exception as e:
        print(f"Error fetching shot data for {player_name} (ID: {player_id}): {e}")
        return pd.DataFrame()

def fetch_defensive_data(player_id, player_name, season):
    try:
        defense_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced"
        ).get_data_frames()[0]
        player_defense = defense_stats[defense_stats['PLAYER_ID'] == player_id]
        if player_defense.empty:
            return pd.DataFrame()
        player_defense = player_defense[['PLAYER_ID', 'PLAYER_NAME', 'DEF_RATING']]
        time.sleep(API_DELAY)
        return player_defense
    except Exception as e:
        print(f"Error fetching defensive data for {player_name} (ID: {player_id}): {e}")
        return pd.DataFrame()

def fetch_all_data(player_df, season):
    shot_results, defense_results = [], []
    for _, row in tqdm(player_df.iterrows(), total=len(player_df), desc="Processing Players"):
        shot_data = fetch_player_shots(row['id'], row['full_name'], season)
        if not shot_data.empty:
            shot_results.append(shot_data)
        defense_data = fetch_defensive_data(row['id'], row['full_name'], season)
        if not defense_data.empty:
            defense_results.append(defense_data)
    all_shots_df = pd.concat(shot_results, ignore_index=True) if shot_results else pd.DataFrame()
    all_defense_df = pd.concat(defense_results, ignore_index=True) if defense_results else pd.DataFrame()
    return all_shots_df, all_defense_df

def merge_data(shot_df, defense_df):
    if defense_df.empty:
        return shot_df
    return pd.merge(shot_df, defense_df, how='left', on=['PLAYER_NAME'])

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # --- Normalize Numerical Features ---
    numerical_features = ["LOC_X", "LOC_Y"]
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # --- Interaction Term: Shot Attempted with Location ---
    df["LOC_XY_ATTEMPTED"] = (df["LOC_X"] ** 2 + df["LOC_Y"] ** 2) ** 0.5 * df["SHOT_ATTEMPTED_FLAG"]

    # --- Save Processed Data ---
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    season = get_most_recent_season()
    player_df = get_warriors_player_ids(season)
    all_shots_df, all_defense_df = fetch_all_data(player_df, season)
    final_df = merge_data(all_shots_df, all_defense_df)
    final_df.to_csv(RAW_OUTPUT_FILE, index=False)
    preprocess_data(RAW_OUTPUT_FILE, PROCESSED_OUTPUT_FILE)
