import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import shotchartdetail, playercareerstats, leaguedashplayerstats
import time
from tqdm import tqdm

# Constants
API_DELAY = 1  # Delay in seconds to respect API rate limits
OUTPUT_FILE = 'warriors_player_shots_with_defense.csv'  # File to save the dataset

# Step 1: Retrieve Player IDs for Warriors Only
def get_warriors_player_ids(season):
    # Get Warriors Team ID
    warriors = [team for team in teams.get_teams() if team['full_name'] == "Golden State Warriors"][0]
    warriors_id = warriors['id']

    # Fetch all players on the Warriors roster for the given season
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    warriors_players = player_stats[player_stats['TEAM_ID'] == warriors_id][['PLAYER_ID', 'PLAYER_NAME']]
    
    return warriors_players.rename(columns={'PLAYER_ID': 'id', 'PLAYER_NAME': 'full_name'})

# Step 2: Fetch Most Recent Season
def get_most_recent_season():
    current_year = pd.Timestamp.today().year
    return f"{current_year - 1}-{str(current_year)[-2:]}"  # Format: '2022-23'

# Step 3: Fetch Shot Data for a Single Player (Both Made and Missed Shots)
def fetch_player_shots(player_id, player_name, season):
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            player_id=player_id,
            season_nullable=season,
            team_id=0,  # Ensure all teams are included
            context_measure_simple='FGA'  # Fetch all shot attempts (made and missed)
        ).get_data_frames()[0]
        
        if shot_data.empty:
            return pd.DataFrame()  # Return empty DataFrame if no shot data
        
        shot_data['PLAYER_NAME'] = player_name
        shot_data = shot_data[['GAME_ID', 'PLAYER_ID', 'PLAYER_NAME', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA',
                               'SHOT_ZONE_RANGE', 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 
                               'SHOT_MADE_FLAG', 'GAME_DATE']]  # Ensure all relevant columns are included
        time.sleep(API_DELAY)  # Respect API rate limits
        return shot_data
    except Exception as e:
        print(f"Error fetching shot data for {player_name} (ID: {player_id}): {e}")
        return pd.DataFrame()  


# Step 4: Fetch Defensive Data for a Single Player
def fetch_defensive_data(player_id, player_name, season):
    try:
        defense_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced"
        ).get_data_frames()[0]
        
        # Filter for the specific player
        player_defense = defense_stats[defense_stats['PLAYER_ID'] == player_id]

        if player_defense.empty:
            print(f"No defensive data for {player_name}")  # Debugging message
            return pd.DataFrame()

        # Select only relevant columns
        player_defense = player_defense[['PLAYER_ID', 'PLAYER_NAME', 'DEF_RATING']]
        time.sleep(API_DELAY)  # Respect API rate limits
        return player_defense
    except Exception as e:
        print(f"Error fetching defensive data for {player_name} (ID: {player_id}): {e}")
        return pd.DataFrame()


# Step 5: Fetch All Data for Warriors Players
def fetch_all_data(player_df, season):
    shot_results = []
    defense_results = []

    print("Fetching Shot Data...")
    for _, row in tqdm(player_df.iterrows(), total=len(player_df), desc="Shots Processed"):
        shot_data = fetch_player_shots(row['id'], row['full_name'], season)
        if not shot_data.empty:  # Only add if there's data
            shot_results.append(shot_data)

    print("Fetching Defensive Data...")
    for _, row in tqdm(player_df.iterrows(), total=len(player_df), desc="Defense Processed"):
        defense_data = fetch_defensive_data(row['id'], row['full_name'], season)
        if not defense_data.empty:  # Only add if there's data
            defense_results.append(defense_data)

    # Combine results
    all_shots_df = pd.concat(shot_results, ignore_index=True) if shot_results else pd.DataFrame()
    all_defense_df = pd.concat(defense_results, ignore_index=True) if defense_results else pd.DataFrame()

    return all_shots_df, all_defense_df

# Step 6: Merge Shot Data with Defensive Data
def merge_data(shot_df, defense_df):
    if defense_df.empty:
        print("Warning: Defensive data is empty. Merging without defensive stats.")
        return shot_df  # Return shots data only if no defensive data is available

    merged_df = pd.merge(
        shot_df,
        defense_df,
        how='left',
        on=['PLAYER_NAME']
    )
    return merged_df


# Main Workflow
if __name__ == "__main__":
    # Get Most Recent NBA Season
    season = get_most_recent_season()
    print(f"Using season: {season}")

    # Step 1: Get Warriors Player IDs
    player_df = get_warriors_player_ids(season)
    print(f"Retrieved {len(player_df)} Warriors players.")

    # Step 2: Fetch Shot and Defensive Data for Warriors Players
    print("Fetching shot and defensive data...")
    all_shots_df, all_defense_df = fetch_all_data(player_df, season)
    print(f"Retrieved {len(all_shots_df)} shots and {len(all_defense_df)} defensive records.")

    # Step 3: Merge Data
    print("Merging data...")
    final_df = merge_data(all_shots_df, all_defense_df)
    print(f"Merged dataset contains {len(final_df)} records.")

    # Step 4: Save Data to CSV
    print(f"Saving data to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Process completed. Data saved to CSV.")
