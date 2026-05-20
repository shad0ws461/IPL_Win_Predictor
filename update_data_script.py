import os
import pandas as pd
import numpy as np
from model_engine import retrain_model_pipeline

def generate_mock_seasons_data(data_dir=None):
    """
    Generates and appends structured mock match and ball-by-ball delivery datasets 
    for IPL seasons 2020 to 2026 to update the model footprint.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
    matches_path = os.path.join(data_dir, "matches.csv")
    deliveries_path = os.path.join(data_dir, "deliveries.csv")
    
    if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
        raise FileNotFoundError("Missing matches.csv or deliveries.csv inside target data directory.")
        
    existing_matches = pd.read_csv(matches_path)
    existing_deliveries = pd.read_csv(deliveries_path)
    
    max_match_id = existing_matches['id'].max()
    print(f"Current database has {len(existing_matches)} matches. Max ID: {max_match_id}")
    
    # Check if we already have 2020-2026 matches to prevent duplicate appends
    if existing_matches['season'].max() >= 2026:
        print("Database already contains data up to season 2026. Bypassing append.")
        return
        
    new_matches = []
    new_deliveries = []
    
    # 2020 - 2026 seasons setup
    new_seasons = list(range(2020, 2027))
    new_franchises = ['Gujarat Titans', 'Lucknow Super Giants', 'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore']
    cities = {
        'Ahmedabad': 'Narendra Modi Stadium',
        'Lucknow': 'Ekana Cricket Stadium',
        'Mumbai': 'Wankhede Stadium',
        'Chennai': 'MA Chidambaram Stadium',
        'Bangalore': 'M Chinnaswamy Stadium'
    }
    
    current_new_id = max_match_id + 1
    
    for season in new_seasons:
        for idx in range(3): # Generate 3 matches per season
            mid = current_new_id
            current_new_id += 1
            
            # Select teams and city
            if season >= 2022:
                # Introduce GT and LSG after 2022
                t1 = 'Gujarat Titans' if idx == 0 else ('Lucknow Super Giants' if idx == 1 else 'Chennai Super Kings')
                t2 = 'Mumbai Indians' if idx == 0 else ('Royal Challengers Bangalore' if idx == 1 else 'Lucknow Super Giants')
            else:
                t1 = 'Chennai Super Kings' if idx == 0 else 'Mumbai Indians'
                t2 = 'Royal Challengers Bangalore' if idx == 0 or idx == 1 else 'Kolkata Knight Riders'
                
            city = 'Ahmedabad' if t1 == 'Gujarat Titans' or t2 == 'Gujarat Titans' else (
                    'Lucknow' if t1 == 'Lucknow Super Giants' or t2 == 'Lucknow Super Giants' else 'Mumbai')
            venue = cities[city]
            
            winner = t1 if idx % 2 == 0 else t2
            
            # Append match row
            match_row = {
                'id': mid,
                'season': season,
                'city': city,
                'date': f"{season}-05-15",
                'team1': t1,
                'team2': t2,
                'toss_winner': t2,
                'toss_decision': 'field',
                'result': 'normal',
                'dl_applied': 0,
                'winner': winner,
                'win_by_runs': 0,
                'win_by_wickets': 6,
                'player_of_match': 'Player XYZ',
                'venue': venue,
                'umpire1': 'Umpire A',
                'umpire2': 'Umpire B',
                'umpire3': 'Umpire C'
            }
            new_matches.append(match_row)
            
            # Generate ball-by-ball deliveries
            # First Innings (Batting: team1, Bowling: team2)
            total_runs_inn1 = 160 + (mid % 30) # 160-190 runs
            # Distribute runs across 20 overs
            for over in range(1, 21):
                for ball in range(1, 7):
                    is_last_ball = (over == 20 and ball == 6)
                    # Approx 1.3 runs per ball to reach total
                    runs = 1 if not is_last_ball else 4
                    if (over + ball) % 9 == 0:
                        runs = 4
                    elif (over + ball) % 15 == 0:
                        runs = 6
                    
                    new_deliveries.append({
                        'match_id': mid,
                        'inning': 1,
                        'batting_team': t1,
                        'bowling_team': t2,
                        'over': over,
                        'ball': ball,
                        'batsman': 'Batsman A',
                        'non_striker': 'Batsman B',
                        'bowler': 'Bowler X',
                        'is_super_over': 0,
                        'wide_runs': 0,
                        'bye_runs': 0,
                        'legbye_runs': 0,
                        'noball_runs': 0,
                        'penalty_runs': 0,
                        'batsman_runs': runs,
                        'extra_runs': 0,
                        'total_runs': runs,
                        'player_dismissed': np.nan,
                        'dismissal_kind': np.nan,
                        'fielder': np.nan
                    })
            
            # Second Innings (Chasing: team2, Bowling: team1)
            target = total_runs_inn1 + 1
            current_score = 0
            wickets = 0
            
            # Generate chasing delivery records
            for over in range(1, 21):
                if current_score >= target or wickets >= 10:
                    break
                for ball in range(1, 7):
                    if current_score >= target or wickets >= 10:
                        break
                        
                    runs = 1
                    player_dismissed = np.nan
                    
                    # Add boundaries and wickets to make it realistic
                    if (over * 6 + ball) % 11 == 0:
                        runs = 4
                    elif (over * 6 + ball) % 17 == 0:
                        runs = 6
                    elif (over * 6 + ball) % 25 == 0:
                        wickets += 1
                        player_dismissed = 'Batsman Out'
                        runs = 0
                        
                    current_score += runs
                    
                    new_deliveries.append({
                        'match_id': mid,
                        'inning': 2,
                        'batting_team': t2,
                        'bowling_team': t1,
                        'over': over,
                        'ball': ball,
                        'batsman': 'Chaser A',
                        'non_striker': 'Chaser B',
                        'bowler': 'Bowler Y',
                        'is_super_over': 0,
                        'wide_runs': 0,
                        'bye_runs': 0,
                        'legbye_runs': 0,
                        'noball_runs': 0,
                        'penalty_runs': 0,
                        'batsman_runs': runs,
                        'extra_runs': 0,
                        'total_runs': runs,
                        'player_dismissed': player_dismissed,
                        'dismissal_kind': 'caught' if not pd.isna(player_dismissed) else np.nan,
                        'fielder': 'Fielder Z' if not pd.isna(player_dismissed) else np.nan
                    })
                    
    # Convert and append to CSV files
    new_matches_df = pd.DataFrame(new_matches)
    new_deliveries_df = pd.DataFrame(new_deliveries)
    
    new_matches_df.to_csv(matches_path, mode='a', header=False, index=False)
    new_deliveries_df.to_csv(deliveries_path, mode='a', header=False, index=False)
    
    print(f"Appended {len(new_matches_df)} matches and {len(new_deliveries_df)} balls to CSV dataset templates (2020-2026).")

if __name__ == "__main__":
    print("Initializing Data Preparation Pipeline...")
    generate_mock_seasons_data()
    print("Triggering retraining pipeline with updated CSV database...")
    retrain_model_pipeline()
    print("Finished successfully.")
