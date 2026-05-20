import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Starting Model Training Pipeline...")
    
    # 1. Load data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    matches_path = os.path.join(data_dir, "matches.csv")
    deliveries_path = os.path.join(data_dir, "deliveries.csv")
    
    if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
        raise FileNotFoundError("Dataset files matches.csv or deliveries.csv not found in data/ folder!")
        
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    
    print(f"Matches shape: {matches.shape}")
    print(f"Deliveries shape: {deliveries.shape}")
    
    # 2. Compute first innings scores
    # Group deliveries by match_id and inning to sum the runs
    # inning == 1 represents the batting team in the first innings
    total_score_df = deliveries[deliveries['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    
    # The chasing target is first innings total runs + 1
    total_score_df['target_runs'] = total_score_df['total_runs'] + 1
    print(f"First innings scores computed. Target runs sample:\n{total_score_df.head()}")
    
    # Merge matches with the target score
    matches_df = matches.merge(total_score_df[['match_id', 'target_runs']], left_on='id', right_on='match_id')
    print(f"Merged shape: {matches_df.shape}")
    
    # 3. Clean and standardize team names
    # We focus on the 8 core current franchises
    teams = [
        'Sunrisers Hyderabad',
        'Mumbai Indians',
        'Royal Challengers Bangalore',
        'Kolkata Knight Riders',
        'Kings XI Punjab',
        'Chennai Super Kings',
        'Rajasthan Royals',
        'Delhi Capitals'
    ]
    
    # Map old names to new ones
    matches_df['team1'] = matches_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    matches_df['team2'] = matches_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    
    matches_df['team1'] = matches_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    matches_df['team2'] = matches_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    
    # Filter to only keep matches where both teams are in our list of 8 teams
    matches_df = matches_df[matches_df['team1'].isin(teams)]
    matches_df = matches_df[matches_df['team2'].isin(teams)]
    
    # Remove rain-affected matches (Duckworth-Lewis applied)
    matches_df = matches_df[matches_df['dl_applied'] == 0]
    print(f"Matches shape after filtering rain-affected matches and standardizing teams: {matches_df.shape}")
    
    # Keep only matches of interest
    matches_df = matches_df[['id', 'city', 'winner', 'target_runs', 'venue']]
    
    # 4. Merge deliveries with cleaned match metadata
    delivery_df = matches_df.merge(deliveries, left_on='id', right_on='match_id')
    
    # Filter for second innings (the chasing phase)
    delivery_df = delivery_df[delivery_df['inning'] == 2]
    print(f"Chasing deliveries (inning 2) shape before team cleaning: {delivery_df.shape}")
    
    delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    delivery_df['bowling_team'] = delivery_df['bowling_team'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    
    delivery_df['batting_team'] = delivery_df['batting_team'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    delivery_df['bowling_team'] = delivery_df['bowling_team'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    
    delivery_df = delivery_df[delivery_df['batting_team'].isin(teams)]
    delivery_df = delivery_df[delivery_df['bowling_team'].isin(teams)]
    
    print(f"Chasing deliveries (inning 2) shape after team cleaning: {delivery_df.shape}")
    
    # 5. Clean city names and handle missing values
    # In standard IPL data, matches played in Dubai or Sharjah might have NaN city
    # Let's fill NaNs in city based on the venue name
    print(f"Missing cities before filling: {delivery_df['city'].isna().sum()}")
    is_null_city = delivery_df['city'].isna()
    venue_cities = np.where(delivery_df['venue'].str.contains('Dubai', case=False, na=False), 'Dubai',
                            np.where(delivery_df['venue'].str.contains('Sharjah', case=False, na=False), 'Sharjah', 'Unknown'))
    delivery_df['city'] = np.where(is_null_city, venue_cities, delivery_df['city'])
    # Drop remaining rows where city is still 'Unknown' or null (if any)
    delivery_df = delivery_df[delivery_df['city'] != 'Unknown']
    
    # 6. Feature Engineering
    print("Engineering features...")
    
    # Cumulative score of batting team at each delivery
    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs'].cumsum()
    
    # Runs left to chase
    delivery_df['runs_left'] = delivery_df['target_runs'] - delivery_df['current_score']
    
    # Balls left in the innings
    # standard deliveries has over (1-20) and ball (1-6)
    delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])
    
    # Cumulative wickets fallen
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
    delivery_df['is_wicket'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == "0" else 1)
    wickets = delivery_df.groupby('match_id')['is_wicket'].cumsum()
    delivery_df['wickets_left'] = 10 - wickets
    
    # Current Run Rate (CRR)
    # balls_bowled = 120 - balls_left
    # CRR = (current_score * 6) / balls_bowled
    balls_bowled = 120 - delivery_df['balls_left']
    delivery_df['crr'] = np.where(balls_bowled > 0, (delivery_df['current_score'] * 6) / balls_bowled, 0.0)
    
    # Required Run Rate (RRR)
    # RRR = (runs_left * 6) / balls_left
    delivery_df['rrr'] = np.where(delivery_df['balls_left'] > 0, (delivery_df['runs_left'] * 6) / delivery_df['balls_left'], 0.0)
    
    # Target label: 1 if batting team (chasing team) wins, 0 if they lose
    def check_winner(row):
        return 1 if row['winner'] == row['batting_team'] else 0
        
    delivery_df['result'] = delivery_df.apply(check_winner, axis=1)
    
    # Filter and rename columns for final dataset
    final_df = delivery_df[[
        'batting_team', 'bowling_team', 'city', 
        'runs_left', 'balls_left', 'wickets_left', 
        'target_runs', 'crr', 'rrr', 'result'
    ]]
    
    # Drop rows where balls_left == 0 (as match is over)
    final_df = final_df[final_df['balls_left'] > 0]
    
    # Drop missing values
    final_df = final_df.dropna()
    
    # Shuffle dataset
    final_df = final_df.sample(final_df.shape[0])
    
    print(f"Final training dataset shape: {final_df.shape}")
    print("Final dataset features sample:")
    print(final_df.head())
    
    # 7. Model Training
    X = final_df.drop(columns=['result'])
    y = final_df['result']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Build scikit-learn Pipeline
    # Nominal Categorical: batting_team, bowling_team, city
    cf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')
    
    pipe = Pipeline(steps=[
        ('preprocessor', cf),
        ('classifier', LogisticRegression(solver='liblinear'))
    ])
    
    print("Fitting Logistic Regression model...")
    pipe.fit(X_train, y_train)
    
    # 8. Evaluation
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training completed.")
    print(f"Accuracy Score: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the pipeline to disk
    model_dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipe.pkl")
    with open(model_dest, 'wb') as f:
        pickle.dump(pipe, f)
    print(f"Model pipeline successfully serialized and saved to: {model_dest}")

if __name__ == "__main__":
    main()
