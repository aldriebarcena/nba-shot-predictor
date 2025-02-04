import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail, playercareerstats, leaguedashplayerstats
from multiprocessing import Pool
import time
from tqdm import tqdm

# Constants
API_DELAY = 1  # Delay in seconds to respect API rate limits
OUTPUT_FILE = 'nba_player_shots_with_defense.csv'  # File to save the dataset

# Step 1: Retrieve Player IDs
def get_player_ids():
    nba_players = players.get_players()
    return pd.DataFrame(nba_players)

# Step 2: Fetch Most Recent Season
def get_most_recent_season():
    current_year = pd.Timestamp.today().year
    return f"{current_year - 1}-{str(current_year)[-2:]}"  # Format: '2022-23'

# Step 3: Fetch Shot Data for a Single Player
def fetch_player_shots(player_id, player_name, season):
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            player_id=player_id,
            season_nullable=season,
            team_id=0  # REQUIRED FIX: Missing `team_id` argument (0 = all teams)
        ).get_data_frames()[0]
        
        if shot_data.empty:
            return pd.DataFrame()  # Return empty DataFrame if no shot data
        
        shot_data['PLAYER_NAME'] = player_name
        time.sleep(API_DELAY)  # Respect API rate limits
        return shot_data
    except Exception as e:
        print(f"Error fetching shot data for {player_name} (ID: {player_id}): {e}")
        return pd.DataFrame()  # **Return empty DataFrame instead of None**

# Step 4: Fetch Defensive Data for a Single Player
def fetch_defensive_data(player_id, player_name):
    try:
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
        
        if career_stats.empty or 'BLK_PCT' not in career_stats.columns or 'DEF_RATING' not in career_stats.columns:
            return pd.DataFrame()  # Return empty DataFrame if stats are missing

        defensive_stats = career_stats[['PLAYER_ID', 'BLK_PCT', 'DEF_RATING']].groupby('PLAYER_ID').mean().reset_index()
        defensive_stats['PLAYER_NAME'] = player_name
        time.sleep(API_DELAY)  # Respect API rate limits
        return defensive_stats
    except Exception as e:
        print(f"Error fetching defensive data for {player_name} (ID: {player_id}): {e}")
        return pd.DataFrame()  # **Return empty DataFrame instead of None**

# Step 5: Parallel Processing for Data Collection
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

    # Step 1: Get Player IDs
    player_df = get_player_ids()
    print(f"Retrieved {len(player_df)} players.")

    # Step 2: Fetch Shot and Defensive Data for All Players (Parallel Processing)
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
