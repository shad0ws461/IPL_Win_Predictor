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

# Unified standards for active IPL franchises (updated to 2026)
TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad',
    'Gujarat Titans',
    'Lucknow Super Giants'
]

# Team name mapping to handle historical variations, rebranding, and new franchises
TEAM_NAME_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Punjab Kings': 'Kings XI Punjab',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'
}

def clean_and_standardize_teams(df, team_cols):
    """Replaces obsolete team names with active counterparts and standardizes casing/strings"""
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Map re-branded teams
            for old_name, new_name in TEAM_NAME_MAPPING.items():
                df[col] = df[col].str.replace(old_name, new_name, regex=False)
    return df

def retrain_model_pipeline(data_dir=None, model_path=None):
    """
    Loads matches.csv and deliveries.csv, applies feature engineering, 
    and trains a Logistic Regression pipeline with OOV categorical guards.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipe.pkl")
        
    matches_path = os.path.join(data_dir, "matches.csv")
    deliveries_path = os.path.join(data_dir, "deliveries.csv")
    
    if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
        raise FileNotFoundError(f"Source datasets not found under: {data_dir}")
        
    print(f"Loading datasets from {data_dir}...")
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    
    # 1. Clean and merge matches metadata
    matches = clean_and_standardize_teams(matches, ['team1', 'team2', 'winner'])
    
    # Filter for target franchises only
    matches = matches[matches['team1'].isin(TEAMS) & matches['team2'].isin(TEAMS)]
    matches = matches[matches['dl_applied'] == 0] # Avoid rain-shortened matches
    
    # Calculate first innings totals to determine chasing targets (Score + 1)
    first_inn_runs = deliveries[deliveries['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    first_inn_runs['target_runs'] = first_inn_runs['total_runs'] + 1
    
    matches_df = matches.merge(first_inn_runs[['match_id', 'target_runs']], left_on='id', right_on='match_id')
    matches_df = matches_df[['id', 'city', 'winner', 'target_runs', 'venue']]
    
    # 2. Merge with ball-by-ball deliveries
    delivery_df = matches_df.merge(deliveries, left_on='id', right_on='match_id')
    delivery_df = delivery_df[delivery_df['inning'] == 2] # Chasing innings only
    
    delivery_df = clean_and_standardize_teams(delivery_df, ['batting_team', 'bowling_team'])
    delivery_df = delivery_df[delivery_df['batting_team'].isin(TEAMS) & delivery_df['bowling_team'].isin(TEAMS)]
    
    # 3. Handle city missing values dynamically
    is_null_city = delivery_df['city'].isna()
    venue_cities = np.where(delivery_df['venue'].astype(str).str.contains('Dubai', case=False, na=False), 'Dubai',
                            np.where(delivery_df['venue'].astype(str).str.contains('Sharjah', case=False, na=False), 'Sharjah', 
                            np.where(delivery_df['venue'].astype(str).str.contains('Navi Mumbai', case=False, na=False), 'Mumbai', 'Unknown')))
    delivery_df['city'] = np.where(is_null_city, venue_cities, delivery_df['city'])
    delivery_df = delivery_df[delivery_df['city'] != 'Unknown']
    
    # 4. Feature Engineering
    print("Performing feature engineering...")
    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs'].cumsum()
    delivery_df['runs_left'] = np.maximum(0, delivery_df['target_runs'] - delivery_df['current_score'])
    delivery_df['balls_left'] = np.maximum(0, 126 - (delivery_df['over'] * 6 + delivery_df['ball']))
    
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
    delivery_df['is_wicket'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == "0" else 1)
    delivery_df['wickets_left'] = np.maximum(0, 10 - delivery_df.groupby('match_id')['is_wicket'].cumsum())
    
    balls_bowled = 120 - delivery_df['balls_left']
    delivery_df['crr'] = np.where(balls_bowled > 0, (delivery_df['current_score'] * 6) / balls_bowled, 0.0)
    delivery_df['rrr'] = np.where(delivery_df['balls_left'] > 0, (delivery_df['runs_left'] * 6) / delivery_df['balls_left'], 0.0)
    
    delivery_df['result'] = np.where(delivery_df['winner'] == delivery_df['batting_team'], 1, 0)
    
    final_df = delivery_df[[
        'batting_team', 'bowling_team', 'city',
        'runs_left', 'balls_left', 'wickets_left',
        'target_runs', 'crr', 'rrr', 'result'
    ]].dropna()
    
    final_df = final_df[final_df['balls_left'] > 0] # Filter finished balls
    
    # 5. Pipeline construction & training
    X = final_df.drop(columns=['result'])
    y = final_df['result']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use handle_unknown='ignore' to avoid out-of-vocabulary categorical errors
    preprocessor = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear'))
    ])
    
    print("Fitting model...")
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Retraining successful! Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Serialize and save pipeline weights
    with open(model_path, 'wb') as f:
        pickle.dump(pipe, f)
    print(f"Model saved to: {model_path}")
    return accuracy

def append_new_match_data(new_matches_df, new_deliveries_df, data_dir=None):
    """Appends incoming match & ball records to main project database CSV files"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
    matches_path = os.path.join(data_dir, "matches.csv")
    deliveries_path = os.path.join(data_dir, "deliveries.csv")
    
    # Standardize headers and team names before appending
    new_matches_df = clean_and_standardize_teams(new_matches_df, ['team1', 'team2', 'winner'])
    new_deliveries_df = clean_and_standardize_teams(new_deliveries_df, ['batting_team', 'bowling_team'])
    
    new_matches_df.to_csv(matches_path, mode='a', header=not os.path.exists(matches_path), index=False)
    new_deliveries_df.to_csv(deliveries_path, mode='a', header=not os.path.exists(deliveries_path), index=False)
    print("New data successfully integrated into matches.csv and deliveries.csv.")

def check_and_update_model(new_data_dir, data_dir=None, model_path=None):
    """
    Checks if new match datasets are dropped in new_data_dir, appends them, 
    triggers automated model retraining, and archives/deletes processed CSV dumps.
    """
    if not os.path.exists(new_data_dir):
        print(f"Directory {new_data_dir} does not exist. No new data checked.")
        return False
        
    new_matches_file = os.path.join(new_data_dir, "new_matches.csv")
    new_deliveries_file = os.path.join(new_data_dir, "new_deliveries.csv")
    
    if os.path.exists(new_matches_file) and os.path.exists(new_deliveries_file):
        print("New match data dump detected! Processing update...")
        new_m = pd.read_csv(new_matches_file)
        new_d = pd.read_csv(new_deliveries_file)
        
        # Append data to template
        append_new_match_data(new_m, new_d, data_dir)
        
        # Retrain model pipeline to update serialized weights
        retrain_model_pipeline(data_dir, model_path)
        
        # Clean up processed dump
        os.remove(new_matches_file)
        os.remove(new_deliveries_file)
        print("Data updates processed. New data file dumps deleted.")
        return True
    
    print("No complete match CSV data templates (new_matches.csv, new_deliveries.csv) found.")
    return False
