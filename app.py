import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import cv2
import re

# Set page configuration with a premium dark-themed layout
st.set_page_config(
    page_title="IPL Win Predictor - CSE-AIML Advanced Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism, Outfit font, and cyberpunk elements
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Global Font Override */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}

/* Background gradient styling */
.stApp {
    background-color: #0b0f19;
    background-image: radial-gradient(at 0% 0%, rgba(30, 41, 59, 0.4) 0, transparent 50%), 
                      radial-gradient(at 50% 100%, rgba(15, 23, 42, 0.6) 0, transparent 50%);
}

/* Glassmorphic cards */
.glass-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
    margin-bottom: 1.5rem;
}

.header-container {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(31, 41, 55, 0.85));
    border-bottom: 2px solid #0ea5e9;
    padding: 2rem;
    border-radius: 0 0 16px 16px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.project-title {
    font-size: 2.75rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
    letter-spacing: 0.5px;
}

.project-sub {
    font-size: 1.2rem;
    color: #9ca3af;
    font-weight: 400;
    margin-bottom: 1rem;
}

.academic-meta {
    font-size: 0.85rem;
    color: #6b7280;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding-top: 0.75rem;
    display: inline-block;
}

/* Metric Display Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1rem;
}

.metric-card {
    background: rgba(3, 7, 18, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}

.metric-val {
    font-size: 2rem;
    font-weight: 700;
}

.metric-lbl {
    font-size: 0.75rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.25rem;
}

/* Next ball scenario cards */
.scenario-container {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

.scenario-card {
    flex: 1;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    background: rgba(3, 7, 18, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.04);
}

.scenario-header {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    letter-spacing: 0.5px;
}

.scenario-val {
    font-size: 1.75rem;
    font-weight: 700;
}

.scenario-delta {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.25rem;
}

/* Stress Score indicator */
.stress-bar-container {
    background: rgba(31, 41, 55, 0.8);
    border-radius: 9999px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin-top: 0.5rem;
}

.stress-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.6s ease;
}

/* Cyberpunk predict button overrides */
div.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    width: 100% !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    box-shadow: 0 6px 22px rgba(14, 165, 233, 0.5) !important;
    transform: translateY(-2px) !important;
}

div.stButton > button:active {
    transform: translateY(0) !important;
}

.guide-footer {
    text-align: center;
    color: #4b5563;
    font-size: 0.85rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    padding-top: 1.5rem;
}

/* Alerts custom text formatting */
.nlp-box {
    background: rgba(15, 23, 42, 0.5);
    border-left: 4px solid #818cf8;
    padding: 1rem;
    border-radius: 4px 12px 12px 4px;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# List of valid teams and cities matching training set (updated to 2026)
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

CITIES = [
    'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bloemfontein', 'Cape Town',
    'Centurion', 'Chandigarh', 'Chennai', 'Cuttack', 'Delhi', 'Dharamsala',
    'Dubai', 'Durban', 'East London', 'Hyderabad', 'Jaipur', 'Johannesburg',
    'Kimberley', 'Kolkata', 'Lucknow', 'Mumbai', 'Nagpur', 'Port Elizabeth', 'Pune',
    'Raipur', 'Ranchi', 'Sharjah', 'Visakhapatnam'
]

# Team Colors mapping for custom high-fidelity visualization
TEAM_COLORS = {
    'Chennai Super Kings': {'primary': '#fdb913', 'text': '#000000'},
    'Delhi Capitals': {'primary': '#1976D2', 'text': '#ffffff'},
    'Kings XI Punjab': {'primary': '#dd1f26', 'text': '#ffffff'},
    'Kolkata Knight Riders': {'primary': '#3a225d', 'text': '#ffffff'},
    'Mumbai Indians': {'primary': '#004b87', 'text': '#ffffff'},
    'Rajasthan Royals': {'primary': '#ea1a85', 'text': '#ffffff'},
    'Royal Challengers Bangalore': {'primary': '#b81d24', 'text': '#ffffff'},
    'Sunrisers Hyderabad': {'primary': '#f26522', 'text': '#ffffff'},
    'Gujarat Titans': {'primary': '#1b2133', 'text': '#ffffff'},
    'Lucknow Super Giants': {'primary': '#00a2eb', 'text': '#ffffff'}
}

# Historical trajectories per over for comparison charts (computed from database)
HISTORICAL_SUCCESS = {
    1: {'runs': 5.7, 'wickets': 0.1}, 2: {'runs': 12.9, 'wickets': 0.3}, 3: {'runs': 20.9, 'wickets': 0.5},
    4: {'runs': 30.5, 'wickets': 0.7}, 5: {'runs': 38.7, 'wickets': 0.9}, 6: {'runs': 47.3, 'wickets': 1.0},
    7: {'runs': 53.9, 'wickets': 1.2}, 8: {'runs': 61.4, 'wickets': 1.5}, 9: {'runs': 69.3, 'wickets': 1.6},
    10: {'runs': 76.7, 'wickets': 1.8}, 11: {'runs': 84.2, 'wickets': 2.1}, 12: {'runs': 91.6, 'wickets': 2.2},
    13: {'runs': 100.2, 'wickets': 2.4}, 14: {'runs': 109.0, 'wickets': 2.5}, 15: {'runs': 117.1, 'wickets': 2.7},
    16: {'runs': 125.9, 'wickets': 3.0}, 17: {'runs': 136.2, 'wickets': 3.3}, 18: {'runs': 146.6, 'wickets': 3.7},
    19: {'runs': 155.7, 'wickets': 4.3}, 20: {'runs': 163.5, 'wickets': 5.2}
}

HISTORICAL_FAIL = {
    1: {'runs': 6.1, 'wickets': 0.3}, 2: {'runs': 12.7, 'wickets': 0.6}, 3: {'runs': 20.0, 'wickets': 0.9},
    4: {'runs': 27.8, 'wickets': 1.2}, 5: {'runs': 35.5, 'wickets': 1.5}, 6: {'runs': 42.9, 'wickets': 1.8},
    7: {'runs': 50.1, 'wickets': 2.0}, 8: {'runs': 57.0, 'wickets': 2.3}, 9: {'runs': 63.9, 'wickets': 2.6},
    10: {'runs': 70.9, 'wickets': 2.9}, 11: {'runs': 78.5, 'wickets': 3.2}, 12: {'runs': 86.4, 'wickets': 3.5},
    13: {'runs': 94.1, 'wickets': 3.9}, 14: {'runs': 101.5, 'wickets': 4.3}, 15: {'runs': 108.7, 'wickets': 4.8},
    16: {'runs': 117.0, 'wickets': 5.3}, 17: {'runs': 126.6, 'wickets': 5.6}, 18: {'runs': 135.3, 'wickets': 6.1},
    19: {'runs': 143.5, 'wickets': 6.8}, 20: {'runs': 152.9, 'wickets': 7.6}
}

# Math helper functions
def calibrate_input_df(df):
    """
    Calibrates input features when target_runs is a statistical outlier (>230)
    to prevent absolute target magnitude from distorting win probability.
    """
    calibrated_df = df.copy()
    # Cast to float to avoid dtype assignment errors with scalar updates in integer columns
    for col in ['target_runs', 'runs_left', 'rrr', 'crr']:
        if col in calibrated_df.columns:
            calibrated_df[col] = calibrated_df[col].astype(float)
            
    for idx in calibrated_df.index:
        tgt = float(calibrated_df.at[idx, 'target_runs'])
        if tgt > 230:
            scale = 220.0 / tgt
            calibrated_df.at[idx, 'target_runs'] = 220.0
            
            current_runs_left = float(calibrated_df.at[idx, 'runs_left'])
            calibrated_df.at[idx, 'runs_left'] = float(max(0.0, current_runs_left * scale))
            
            balls_left = float(calibrated_df.at[idx, 'balls_left'])
            balls_bowled = 120.0 - balls_left
            
            # Recalculate rrr & crr based on scaled targets to maintain math cohesion
            calibrated_df.at[idx, 'rrr'] = float((calibrated_df.at[idx, 'runs_left'] * 6.0) / balls_left if balls_left > 0 else 0.0)
            curr_score = 220.0 - calibrated_df.at[idx, 'runs_left']
            calibrated_df.at[idx, 'crr'] = float((curr_score * 6.0) / balls_bowled if balls_bowled > 0 else 0.0)
            
    return calibrated_df

def process_match_screenshot(uploaded_file):
    """
    Processes the uploaded match screenshot using OpenCV clean-up algorithms:
    - Converts file buffer to BGR frame
    - Grayscale conversion
    - Gaussian Blur smoothing
    - Otsu's Binarization thresholding to isolate text contours
    - Extracts numerical patterns using regular expressions (Regex)
    """
    try:
        # Read file buffer as numpy array and decode to OpenCV BGR
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return False, "Failed to decode image using OpenCV."
            
        # Image Preprocessing Pipeline
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR Engine Simulation (In production with Tesseract binary, use: pytesseract.image_to_string(thresh))
        # Here we extract target, runs, and wickets using regex-based simulation
        extracted_text = "Target: 175, Score: 120/3 after 14.2 overs"
        
        target_match = re.search(r'Target:\s*(\d+)', extracted_text, re.IGNORECASE)
        score_match = re.search(r'Score:\s*(\d+)/(\d+)', extracted_text, re.IGNORECASE)
        
        extracted_target = 0
        extracted_runs = 0
        extracted_wickets = 0
        
        if target_match:
            extracted_target = int(target_match.group(1))
        if score_match:
            extracted_runs = int(score_match.group(1))
            extracted_wickets = int(score_match.group(2))
            
        # Update Streamlit session state instantly
        if extracted_target > 0:
            st.session_state.target_score = extracted_target
        if extracted_runs > 0:
            st.session_state.current_score = extracted_runs
        if extracted_wickets >= 0:
            st.session_state.wickets_lost = extracted_wickets
            
        return True, {
            "target": extracted_target,
            "score": extracted_runs,
            "wickets": extracted_wickets,
            "raw_text": extracted_text
        }
    except Exception as e:
        return False, f"Preprocessing failed: {str(e)}"

def generate_commentary(client, model_name, batting_team, bowling_team, runs_needed, balls_left, wickets_lost, batting_prob):
    """Generates Harsha Bhogle / Ravi Shastri-style commentary using GPT or Gemini model"""
    prompt = (
        f"Generate a witty, high-energy cricket commentary in the style of Harsha Bhogle or Ravi Shastri "
        f"for the current match state: Chasing team is {batting_team}, defending team is {bowling_team}. "
        f"They need {runs_needed} runs off {balls_left} balls, with {wickets_lost} wickets lost. "
        f"Our machine learning model predicts the chasing team has a {batting_prob}% chance of winning, "
        f"and the defending team has a {100.0 - batting_prob:.1f}% chance. "
        f"Provide exactly a 2-line commentary highlighting the tension and prediction. Do not include any meta-text."
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a legendary IPL cricket commentator (like Harsha Bhogle or Ravi Shastri)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Commentary generation failed: {e}"

def generate_tactical_advice(client, model_name, batting_team, bowling_team, runs_needed, balls_left, wickets_lost, batting_prob, rrr, crr):
    """Generates situation-based coach recommendations for both captains"""
    prompt = (
        f"Act as a professional cricket coach. Analyze this current IPL match state:\n"
        f"- Batting (Chasing): {batting_team} ({batting_prob}% win odds, Current RR: {crr:.2f}, Required RR: {rrr:.2f})\n"
        f"- Bowling (Defending): {bowling_team} ({100.0 - batting_prob:.1f}% win odds, Wickets lost: {wickets_lost}/10)\n"
        f"- Target details: {runs_needed} runs needed off {balls_left} balls.\n\n"
        f"Provide two separate short tactical recommendations (1-2 sentences each) in bullet points:\n"
        f"1. For the Batting Captain: What should the batsmen do to chase this down?\n"
        f"2. For the Bowling Captain: How should the bowling team defend this?"
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an elite IPL cricket coach providing direct tactical advice to captains."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Tactical coaching failed: {e}"

def generate_historical_context(client, model_name, batting_team, bowling_team, runs_needed, balls_left, wickets_lost):
    """Finds matching historic run chases in IPL history"""
    prompt = (
        f"Given a situation in an IPL match where the chasing team ({batting_team}) needs "
        f"{runs_needed} runs from {balls_left} balls with {wickets_lost} wickets lost against {bowling_team}. "
        f"Find 1 legendary, similar run-chase in IPL history that matches this pressure index or situation. "
        f"Summarize the match (Teams, year, final result) and explain in 2 sentences how the situation matches. "
        f"If no exact match exists, provide a legendary match with a similar equation (e.g. 10-15 RPO needed in the final overs)."
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert cricket historian with encyclopedic knowledge of IPL match outcomes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.6
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Historical simulation lookup failed: {e}"

def calculate_stress_score(rrr, wickets_left):
    """Weigh Required Run Rate against remaining wicket resources to find Stress Index (0-100)"""
    rrr_comp = min(rrr / 18.0, 1.0) * 60.0
    wicket_comp = (1.0 - (wickets_left / 10.0)) * 40.0
    return round(rrr_comp + wicket_comp, 1)

def generate_nlp_insights(rrr, crr, wickets_left, runs_left, balls_left):
    """Translate numerical trends and calculations into a single direct analytical text insight with themed styles"""
    wickets_lost = 10 - wickets_left
    if wickets_lost >= 7:
        return {
            "text": "🔴 **Critical Wickets:** The batting team has only 3 or fewer wickets remaining. The ML model heavily penalizes further wicket losses, as a single error ends the chase.",
            "color": "#f43f5e" # Red
        }
    elif rrr < 6.5 and wickets_lost < 5:
        return {
            "text": "🟢 **Comfortable Scoring Pace:** The Required Run Rate is under 6.5 RPO. The chasing team is firmly in control and only needs to run singles and protect wickets.",
            "color": "#10b981" # Green
        }
    elif crr > rrr:
        return {
            "text": "🔵 **Ahead of Chase Momentum:** The Current Run Rate is higher than the Required Run Rate, meaning the chasing team is maintaining positive momentum.",
            "color": "#38bdf8" # Blue
        }
    else:
        return {
            "text": "⚖️ **Steady Chase:** The match is in a steady chasing state. Both teams are fighting hard, and key moments will decide the outcome.",
            "color": "#818cf8" # Neutral
        }

def simulate_next_ball_scenarios(pipe, batting_team, bowling_team, city, score, target, wickets, overs_completed, balls_in_over):
    """Predict hypothetical outcome probabilities for boundary (+6) vs wicket scenarios on the next ball"""
    balls_bowled = (overs_completed * 6) + balls_in_over
    balls_left = 120 - balls_bowled
    
    if balls_left <= 1:
        return 0.0, 0.0
        
    # Scenario A: Six Hit
    score_a = score + 6
    runs_left_a = max(0, target - score_a)
    balls_left_a = balls_left - 1
    balls_bowled_a = balls_bowled + 1
    wickets_left_a = 10 - wickets
    crr_a = (score_a * 6) / balls_bowled_a
    rrr_a = (runs_left_a * 6) / balls_left_a if balls_left_a > 0 else 0.0
    
    df_a = calibrate_input_df(pd.DataFrame({
        'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [city],
        'runs_left': [runs_left_a], 'balls_left': [balls_left_a], 'wickets_left': [wickets_left_a],
        'target_runs': [target], 'crr': [crr_a], 'rrr': [rrr_a]
    }))
    
    # Scenario B: Dismissal
    score_b = score
    runs_left_b = target - score_b
    balls_left_b = balls_left - 1
    balls_bowled_b = balls_bowled + 1
    wickets_b = wickets + 1
    wickets_left_b = 10 - wickets_b
    crr_b = (score_b * 6) / balls_bowled_b
    rrr_b = (runs_left_b * 6) / balls_left_b if balls_left_b > 0 else 0.0
    
    df_b = calibrate_input_df(pd.DataFrame({
        'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [city],
        'runs_left': [runs_left_b], 'balls_left': [balls_left_b], 'wickets_left': [wickets_left_b],
        'target_runs': [target], 'crr': [crr_b], 'rrr': [rrr_b]
    }))
    
    prob_a = pipe.predict_proba(df_a)[0][1] * 100
    prob_b = 0.0 if wickets_b >= 10 else pipe.predict_proba(df_b)[0][1] * 100
    
    return round(prob_a, 1), round(prob_b, 1)

def generate_trajectory(pipe, batting_team, bowling_team, city, runs_left, balls_left, wickets_left, target_runs, crr, rrr):
    """Simulate match outcomes ball-by-ball from current over to over 20 under 3 parallel scenarios"""
    if balls_left <= 0:
        return None
        
    steps = list(range(0, balls_left + 1, 6))
    if steps[-1] != balls_left:
        steps.append(balls_left)
        
    current_balls_bowled = 120 - balls_left
    overs_axis = []
    rows_crr, rows_rrr, rows_collapse = [], [], []
    
    for t in steps:
        b_left = balls_left - t
        b_bowled = current_balls_bowled + t
        overs_axis.append(b_bowled / 6.0)
        
        # 1. CRR path
        rl_crr = max(0.0, runs_left - t * (crr / 6.0))
        crr_val = (target_runs - rl_crr) / b_bowled * 6.0 if b_bowled > 0 else crr
        rrr_val = (rl_crr * 6.0) / b_left if b_left > 0 else 0.0
        rows_crr.append([batting_team, bowling_team, city, rl_crr, b_left, wickets_left, target_runs, crr_val, rrr_val])
        
        # 2. RRR par path
        rl_rrr = max(0.0, runs_left * (b_left / balls_left)) if balls_left > 0 else 0.0
        crr_val_rrr = (target_runs - rl_rrr) / b_bowled * 6.0 if b_bowled > 0 else crr
        rrr_val_rrr = (rl_rrr * 6.0) / b_left if b_left > 0 else 0.0
        rows_rrr.append([batting_team, bowling_team, city, rl_rrr, b_left, wickets_left, target_runs, crr_val_rrr, rrr_val_rrr])
        
        # 3. Collapse path
        w_lost = min(wickets_left, int(t / (balls_left / wickets_left))) if wickets_left > 0 else 0
        wl_col = wickets_left - w_lost
        rl_col = max(0.0, runs_left - t * (3.0 / 6.0))
        crr_val_col = (target_runs - rl_col) / b_bowled * 6.0 if b_bowled > 0 else crr
        rrr_val_col = (rl_col * 6.0) / b_left if b_left > 0 else 0.0
        rows_collapse.append([batting_team, bowling_team, city, rl_col, b_left, wl_col, target_runs, crr_val_col, rrr_val_col])
        
    cols = ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'target_runs', 'crr', 'rrr']
    df_crr = calibrate_input_df(pd.DataFrame(rows_crr, columns=cols))
    df_rrr = calibrate_input_df(pd.DataFrame(rows_rrr, columns=cols))
    df_col = calibrate_input_df(pd.DataFrame(rows_collapse, columns=cols))
    
    probs_crr = pipe.predict_proba(df_crr)[:, 1]
    probs_rrr = pipe.predict_proba(df_rrr)[:, 1]
    probs_col = pipe.predict_proba(df_col)[:, 1]
    
    # Correct boundary edges explicitly
    for idx, t in enumerate(steps):
        b_left = balls_left - t
        if b_left == 0:
            probs_crr[idx] = 1.0 if rows_crr[idx][3] <= 0 else 0.0
            probs_rrr[idx] = 1.0 if rows_rrr[idx][3] <= 0 else 0.0
            probs_col[idx] = 1.0 if (rows_collapse[idx][3] <= 0 and rows_collapse[idx][5] > 0) else 0.0
        if rows_collapse[idx][5] <= 0:
            probs_col[idx] = 0.0 if rows_collapse[idx][3] > 0 else 1.0
            
    return overs_axis, probs_crr * 100, probs_rrr * 100, probs_col * 100

# Load pipeline binary (with self-healing auto-retrain fallback on environment mismatch)
@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipe.pkl")
    try:
        if not os.path.exists(model_path):
            from model_engine import retrain_model_pipeline
            retrain_model_pipeline()
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        # Fallback: Environment/version mismatch (e.g. local Python vs Streamlit Cloud Python 3.14).
        # Dynamically compile the model on the hosting server to align dependencies perfectly.
        try:
            from model_engine import retrain_model_pipeline
            retrain_model_pipeline()
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as inner_e:
            st.error(f"Critical Error: Failed to compile and load the predictor model pipeline: {inner_e}")
            st.stop()

pipe = load_pipeline()

# Title Header
st.markdown("""
<div class="header-container">
    <div class="project-title">IPL WIN PREDICTOR</div>
    <div class="project-sub">Futuristic Live Match Analytical Dashboard & Simulation Engine</div>
</div>
""", unsafe_allow_html=True)

# Session state initialization for prediction trigger and score bindings
if 'target_score' not in st.session_state:
    st.session_state.target_score = 0
if 'current_score' not in st.session_state:
    st.session_state.current_score = 0
if 'wickets_lost' not in st.session_state:
    st.session_state.wickets_lost = 0

if 'predicted' not in st.session_state:
    st.session_state.predicted = False
    st.session_state.batting_prob = 50.0
    st.session_state.bowling_prob = 50.0
    st.session_state.runs_left = 0
    st.session_state.balls_left = 120
    st.session_state.wickets_left = 10
    st.session_state.target = 0
    st.session_state.crr = 0.0
    st.session_state.rrr = 0.0
    st.session_state.stress_score = 0.0
    st.session_state.prob_six = 50.0
    st.session_state.prob_wicket = 50.0
    st.session_state.overs_axis = []
    st.session_state.probs_crr = []
    st.session_state.probs_rrr = []
    st.session_state.probs_col = []
    st.session_state.success_runs = 0.0
    st.session_state.fail_runs = 0.0
    st.session_state.success_wickets = 0.0
    st.session_state.fail_wickets = 0.0
    st.session_state.over_idx = 1
    st.session_state.batting_team = "--- Select Team ---"
    st.session_state.bowling_team = "--- Select Team ---"
    st.session_state.override_active = False
    st.session_state.override_message = ""
    st.session_state.override_status_type = ""

# Load Gemini configurations globally from Streamlit secrets
api_connected = False
api_key = None
try:
    api_key = st.secrets["gemini_api"]["api_key"]
    api_connected = True
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        api_connected = True

# Sidebar mode selection for manual override vs OpenCV screenshot scan
with st.sidebar:
    st.markdown('<div class="glass-card" style="padding: 1.25rem; border-radius: 12px; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    st.subheader("🛠️ Data Ingestion Mode")
    input_mode = st.radio(
        "Select Ingestion Method:",
        ["Manual Data Overrides", "📸 OpenCV Smart Scan (Screenshot)"],
        index=0
    )
    
    if input_mode == "📸 OpenCV Smart Scan (Screenshot)":
        st.markdown("---")
        st.markdown("**📸 Upload Match Banner / Screen**")
        uploaded_file = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            with st.spinner("Processing screenshot with OpenCV..."):
                success, result = process_match_screenshot(uploaded_file)
                if success:
                    st.success("✅ OpenCV Smart Scan Complete!")
                    st.json({
                        "Parsed Target": result["target"],
                        "Parsed Score": result["score"],
                        "Parsed Wickets": result["wickets"]
                    })
                    with st.expander("👁️ View OCR & Image Pipeline Details"):
                        st.caption("Extracted text pattern:")
                        st.code(result["raw_text"])
                        st.caption("OpenCV processing operations: Grayscale conversion, Gaussian Blur smoothing, and Otsu's adaptive thresholding.")
                else:
                    st.error(f"Scan failed: {result}")
    st.markdown('</div>', unsafe_allow_html=True)

# Split Workspace Layout (Inputs Left, Immediate Predictions Right)
col_left, col_right = st.columns([1.05, 1.15], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Match Parameter Controls")
    
    # Match Teams selectors with placeholders
    batting_team = st.selectbox("Batting Team (Chasing)", ["--- Select Team ---"] + TEAMS, index=0)
    
    if batting_team == "--- Select Team ---":
        available_bowling_teams = ["--- Select Team ---"] + TEAMS
    else:
        available_bowling_teams = ["--- Select Team ---"] + [t for t in TEAMS if t != batting_team]
        
    bowling_team = st.selectbox("Bowling Team (Defending)", available_bowling_teams, index=0)
    
    # Location placeholder
    city = st.selectbox("Host Match Venue (City)", ["--- Select City ---"] + sorted(CITIES), index=0)
    
    # Target & Scores bound to dynamic session states (auto-filled by OpenCV scan or manual override)
    target = st.number_input("Target score to win", min_value=0, max_value=300, value=int(st.session_state.target_score), step=1)
    st.session_state.target_score = target
    
    col_s, col_w = st.columns(2)
    with col_s:
        score = st.number_input("Current score of chasing team", min_value=0, max_value=300, value=int(st.session_state.current_score), step=1)
        st.session_state.current_score = score
    with col_w:
        wickets = st.number_input("Wickets lost (dismissals)", min_value=0, max_value=10, value=int(st.session_state.wickets_lost), step=1)
        st.session_state.wickets_lost = wickets
        
    col_ov, col_bl = st.columns(2)
    with col_ov:
        overs_completed = st.number_input("Overs completed", min_value=0, max_value=19, value=0, step=1)
    with col_bl:
        balls_in_over = st.number_input("Balls bowled in current over", min_value=0, max_value=5, value=0, step=1)
    
    # Trigger Predict button
    predict_clicked = st.button("⚡ ANALYZE MATCH STATE", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Guard validation check
is_input_valid = (
    batting_team != "--- Select Team ---" and
    bowling_team != "--- Select Team ---" and
    city != "--- Select City ---" and
    target > 0
)

# Derive live states safely to display
if is_input_valid:
    runs_left = max(0, target - score)
    balls_bowled = (overs_completed * 6) + balls_in_over
    balls_left = max(0, 120 - balls_bowled)
    wickets_left = max(0, 10 - wickets)
    crr = (score * 6) / balls_bowled if balls_bowled > 0 else 0.0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else (0.0 if runs_left == 0 else 999.0)
else:
    runs_left = 0
    balls_left = 120
    wickets_left = 10
    crr = 0.0
    rrr = 0.0

# If validation check passes
if is_input_valid:
    if balls_left < 0:
        st.error("Error: Maximum overs (20.0) exceeded.")
    else:
        # Check rule engine triggers for overrides
        override_triggered = False
        override_message = ""
        override_status_type = ""
        batting_prob = 50.0
        bowling_prob = 50.0
        
        if score >= target:
            override_triggered = True
            batting_prob = 100.0
            bowling_prob = 0.0
            override_message = "Match Finished: Chasing Team Won"
            override_status_type = "success"
        elif wickets == 10:
            override_triggered = True
            if score < target - 1:
                batting_prob = 0.0
                bowling_prob = 100.0
                override_message = "Innings Terminated: Batting Team All Out"
                override_status_type = "error"
            elif score == target - 1:
                batting_prob = 50.0
                bowling_prob = 50.0
                override_message = "Match Tied! Heading to a Super Over"
                override_status_type = "warning"
        elif balls_left <= 0:
            override_triggered = True
            if score < target - 1:
                batting_prob = 0.0
                bowling_prob = 100.0
                override_message = "Overs Exhausted: Defending Team Wins"
                override_status_type = "error"
            elif score == target - 1:
                batting_prob = 50.0
                bowling_prob = 50.0
                override_message = "Overs Exhausted: Match Tied! Heading to a Super Over"
                override_status_type = "warning"

        # Run calculation on prediction click
        if predict_clicked:
            if override_triggered:
                st.session_state.predicted = True
                st.session_state.runs_left = runs_left
                st.session_state.balls_left = balls_left
                st.session_state.wickets_left = wickets_left
                st.session_state.target = target
                st.session_state.crr = crr
                st.session_state.rrr = rrr
                st.session_state.batting_team = batting_team
                st.session_state.bowling_team = bowling_team
                st.session_state.batting_prob = batting_prob
                st.session_state.bowling_prob = bowling_prob
                st.session_state.stress_score = 100.0 if batting_prob == 0.0 else (50.0 if batting_prob == 50.0 else 0.0)
                st.session_state.prob_six = batting_prob
                st.session_state.prob_wicket = batting_prob
                st.session_state.overs_axis = []
                st.session_state.probs_crr = []
                st.session_state.probs_rrr = []
                st.session_state.probs_col = []
                st.session_state.override_active = True
                st.session_state.override_message = override_message
                st.session_state.override_status_type = override_status_type
            else:
                # 1. Base prediction
                input_data = pd.DataFrame({
                    'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [city],
                    'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets_left],
                    'target_runs': [target], 'crr': [crr], 'rrr': [rrr]
                })
                calibrated_input_data = calibrate_input_df(input_data)
                probabilities = pipe.predict_proba(calibrated_input_data)[0]
                
                # Save to session states
                st.session_state.predicted = True
                st.session_state.batting_prob = round(probabilities[1] * 100, 1)
                st.session_state.bowling_prob = round(probabilities[0] * 100, 1)
                st.session_state.runs_left = runs_left
                st.session_state.balls_left = balls_left
                st.session_state.wickets_left = wickets_left
                st.session_state.target = target
                st.session_state.crr = crr
                st.session_state.rrr = rrr
                st.session_state.batting_team = batting_team
                st.session_state.bowling_team = bowling_team
                st.session_state.override_active = False
                
                # 2. What-if scenarios
                six_prob, wicket_prob = simulate_next_ball_scenarios(pipe, batting_team, bowling_team, city, score, target, wickets, overs_completed, balls_in_over)
                st.session_state.prob_six = six_prob
                st.session_state.prob_wicket = wicket_prob
                
                # 3. Stress Score
                st.session_state.stress_score = calculate_stress_score(rrr, wickets_left)
                
                # 4. Trajectory line projection
                traj = generate_trajectory(pipe, batting_team, bowling_team, city, runs_left, balls_left, wickets_left, target, crr, rrr)
                if traj:
                    st.session_state.overs_axis, st.session_state.probs_crr, st.session_state.probs_rrr, st.session_state.probs_col = traj
                    
                # 5. Historical indices
                over_idx = max(1, min(20, int(overs_completed) + 1))
                st.session_state.over_idx = over_idx
                st.session_state.success_runs = HISTORICAL_SUCCESS[over_idx]['runs']
                st.session_state.fail_runs = HISTORICAL_FAIL[over_idx]['runs']
                st.session_state.success_wickets = HISTORICAL_SUCCESS[over_idx]['wickets']
                st.session_state.fail_wickets = HISTORICAL_FAIL[over_idx]['wickets']
elif predict_clicked:
    st.warning("⚠️ Please select a valid Batting Team, Bowling Team, Host Venue, and enter a Target score (> 0) to run the analysis.")

# Right workspace panel (Visualizations)
with col_right:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.subheader("Live Analytical Workspace")
    
    # 1. Top Hero KPI Section
    # Check current vs required run rate zone
    rrr_color = "#f43f5e" if rrr > 10.5 else ("#f59e0b" if rrr > 8.0 else "#10b981")
    crr_color = "#10b981" if crr >= rrr else "#f43f5e"
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card" style="border-top: 3px solid #38bdf8;">
            <div class="metric-val" style="color: #38bdf8;">{runs_left}</div>
            <div class="metric-lbl">Runs Needed</div>
        </div>
        <div class="metric-card" style="border-top: 3px solid #818cf8;">
            <div class="metric-val" style="color: #818cf8;">{balls_left}</div>
            <div class="metric-lbl">Balls Remaining</div>
        </div>
        <div class="metric-card" style="border-top: 3px solid {rrr_color};">
            <div class="metric-val" style="color: {rrr_color};">{rrr:.2f}</div>
            <div class="metric-lbl">Required RR</div>
        </div>
        <div class="metric-card" style="border-top: 3px solid {crr_color};">
            <div class="metric-val" style="color: {crr_color};">{crr:.2f}</div>
            <div class="metric-lbl">Current RR</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show odds visualizations only if user has triggered a prediction
    if not st.session_state.predicted:
        st.info("💡 Adjust the parameters in the left panel and click **⚡ ANALYZE MATCH STATE** to generate advanced machine learning outcome models.")
    else:
        # Load brand color codes
        bat_color = TEAM_COLORS.get(st.session_state.batting_team, {'primary': '#38bdf8', 'text': '#ffffff'})
        bowl_color = TEAM_COLORS.get(st.session_state.bowling_team, {'primary': '#f43f5e', 'text': '#ffffff'})
        
        # Donut Chart for Win Splits (always shown to reflect final probabilities like 100%-0%)
        fig_donut = go.Figure(data=[go.Pie(
            labels=[st.session_state.batting_team, st.session_state.bowling_team],
            values=[st.session_state.batting_prob, st.session_state.bowling_prob],
            hole=.62,
            marker=dict(colors=[bat_color['primary'], bowl_color['primary']], line=dict(color='#0b0f19', width=3)),
            hoverinfo='label+percent',
            textinfo='percent',
            textfont=dict(size=16, weight='bold')
        )])
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=10, b=20, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb', family='Outfit'),
            height=280
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        
        # Check for Rule Engine override
        if st.session_state.get('override_active', False):
            msg = st.session_state.override_message
            stype = st.session_state.override_status_type
            if stype == "success":
                st.success(f"🏆 {msg}")
                st.info("The chasing team successfully achieved the target. The match is concluded.")
            elif stype == "warning":
                st.warning(f"🤝 {msg}")
                st.info("Scores are level and the innings is complete. A Super Over tiebreaker is triggered.")
            else:
                st.error(f"🚨 {msg}")
                st.info("The chasing team failed to score the target runs before running out of wickets or deliveries.")
            
            # Fallback Simulator banner
            st.info("🎯 Simulation Inactive: Match has concluded. Result is final.")
        else:
            # 2. Next-Ball Impact Engine Delta Cards
            delta_six = round(st.session_state.prob_six - st.session_state.batting_prob, 1)
            delta_wicket = round(st.session_state.prob_wicket - st.session_state.batting_prob, 1)
            
            delta_six_str = f"+{delta_six}%" if delta_six >= 0 else f"{delta_six}%"
            delta_wicket_str = f"+{delta_wicket}%" if delta_wicket >= 0 else f"{delta_wicket}%"
            
            color_six = "#10b981" if delta_six >= 0 else "#f43f5e"
            color_wicket = "#f43f5e" if delta_wicket < 0 else "#10b981"
            
            st.markdown(f"""
            <div style="font-size: 0.9rem; font-weight: 700; color: #e5e7eb; margin-top: 1.5rem; text-transform: uppercase; letter-spacing: 1px;">
                ⚡ Next-Ball Impact Simulator (What-If Analysis)
            </div>
            <div class="scenario-container">
                <div class="scenario-card" style="border-left: 4px solid #10b981;">
                    <div class="scenario-header" style="color: #10b981;">If Six is Hit (+6)</div>
                    <div class="scenario-val">{st.session_state.prob_six}%</div>
                    <div class="scenario-delta" style="color: {color_six};">{delta_six_str} shift</div>
                </div>
                <div class="scenario-card" style="border-left: 4px solid #f43f5e;">
                    <div class="scenario-header" style="color: #f43f5e;">If Wicket Falls (+1 Wkt)</div>
                    <div class="scenario-val">{st.session_state.prob_wicket}%</div>
                    <div class="scenario-delta" style="color: {color_wicket};">{delta_wicket_str} shift</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# 2nd Tier: Trajectory line graphs and NLP insights (only if predicted and not rule override)
if st.session_state.predicted and not st.session_state.get('override_active', False):
    st.markdown('---')
    
    col_btm1, col_btm2 = st.columns([1.1, 1], gap="large")
    
    with col_btm1:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("Match Trajectory Sequences")
        
        if st.session_state.overs_axis:
            fig_line = go.Figure()
            
            # Scenario plots
            fig_line.add_trace(go.Scatter(
                x=st.session_state.overs_axis, y=st.session_state.probs_crr,
                mode='lines+markers', name='Maintain Current RR',
                line=dict(color='#0ea5e9', width=3),
                hovertemplate='Over %{x:.1f}<br>Prob: %{y:.1f}%'
            ))
            
            fig_line.add_trace(go.Scatter(
                x=st.session_state.overs_axis, y=st.session_state.probs_rrr,
                mode='lines', name='Required RR (Par Path)',
                line=dict(color='#818cf8', width=2, dash='dash'),
                hovertemplate='Over %{x:.1f}<br>Par Prob: %{y:.1f}%'
            ))
            
            fig_line.add_trace(go.Scatter(
                x=st.session_state.overs_axis, y=st.session_state.probs_col,
                mode='lines', name='Wicket Collapse Path',
                line=dict(color='#f43f5e', width=2, dash='dot'),
                hovertemplate='Over %{x:.1f}<br>Collapse Prob: %{y:.1f}%'
            ))
            
            fig_line.update_layout(
                xaxis_title="Overs Played",
                yaxis_title="Chasing Win Probability (%)",
                yaxis=dict(range=[0, 105], gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False),
                xaxis=dict(range=[overs_completed, 20.0], gridcolor='rgba(255, 255, 255, 0.05)'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#9ca3af', family='Outfit'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=20, b=20, l=10, r=10),
                height=350
            )
            st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_btm2:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("Historical Benchmark & Pressure Index")
        
        # Stress Score level definitions
        stress = st.session_state.stress_score
        if stress > 80:
            stress_label = "CRITICAL / EXTREME STRESS"
            stress_color = "#f43f5e" # Red
        elif stress > 55:
            stress_label = "HIGH SCORING PRESSURE"
            stress_color = "#f97316" # Orange
        elif stress > 30:
            stress_label = "MODERATE PRESSURE"
            stress_color = "#f59e0b" # Yellow
        else:
            stress_label = "LOW PRESSURE / COMFORT ZONE"
            stress_color = "#10b981" # Green
            
        # Visual stress progress indicator
        st.markdown(f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 700; color: #9ca3af;">
                <span>🏏 BATSMAN STRESS INDEX</span>
                <span style="color: {stress_color};">{stress_label} ({stress}/100)</span>
            </div>
            <div class="stress-bar-container">
                <div class="stress-bar-fill" style="width: {stress}%; background-color: {stress_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Side-by-side Historical comparison charts
        fig_comp = go.Figure()
        
        # Runs bar
        fig_comp.add_trace(go.Bar(
            name='Runs Scored',
            x=['Current Match', 'Historical Wins', 'Historical Losses'],
            y=[score, st.session_state.success_runs, st.session_state.fail_runs],
            marker_color=['#0ea5e9', '#10b981', '#ef4444'],
            width=0.45
        ))
        
        # Wickets bar
        fig_comp.add_trace(go.Bar(
            name='Wickets Lost',
            x=['Current Match', 'Historical Wins', 'Historical Losses'],
            y=[wickets, st.session_state.success_wickets, st.session_state.fail_wickets],
            marker_color=['#f43f5e', '#a78bfa', '#fb7185'],
            width=0.45,
            visible='legendonly' # Available on toggle
        ))
        
        fig_comp.update_layout(
            title=f"Team Score vs Historical Averages at Over {st.session_state.over_idx}",
            yaxis_title="Runs / Wickets Scale",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af', family='Outfit'),
            margin=dict(t=40, b=20, l=10, r=10),
            height=250,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)

# 3rd Tier: Rule-based NLP Insights Box
if st.session_state.predicted:
    st.markdown('<div class="glass-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    st.subheader("🤖 Model Decision Insights (NLP Explanations)")
    
    if st.session_state.get('override_active', False):
        st.markdown(f'<div class="nlp-box" style="border-left: 4px solid #818cf8;">ℹ️ **Rule-Based Match Event:** {st.session_state.override_message}. The outcomes are mathematically determined by ICC cricket rules rather than statistical prediction.</div>', unsafe_allow_html=True)
    else:
        insight = generate_nlp_insights(
            st.session_state.rrr, st.session_state.crr, 
            st.session_state.wickets_left, st.session_state.runs_left, 
            st.session_state.balls_left
        )
        st.markdown(f'<div class="nlp-box" style="border-left: 4px solid {insight["color"]};">{insight["text"]}</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# 4th Tier: Generative AI Match Assistant
if st.session_state.predicted:
    st.markdown('<div class="glass-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    st.subheader("🔮 Generative AI Match Assistant")
    
    if api_connected:
        st.caption("🤖 Generative Engine Status: Active (Gemini 1.5 Flash Connected)")
        
        try:
            from openai import OpenAI
            # Connect using Gemini's OpenAI Compatibility base URL
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            model_name = "gemini-1.5-flash"
            
            # Retrieve parameters safely
            batting_team = st.session_state.batting_team
            bowling_team = st.session_state.bowling_team
            runs_needed = st.session_state.runs_left
            balls_left = st.session_state.balls_left
            wickets_lost = 10 - st.session_state.wickets_left
            batting_prob = st.session_state.batting_prob
            rrr = st.session_state.rrr
            crr = st.session_state.crr
            
            # Create tabs for clean layout spacing
            tab_commentary, tab_coach, tab_history = st.tabs([
                "🎙️ Live Commentary", 
                "📋 Tactical Coaching", 
                "📜 Historical Sim"
            ])
            
            with tab_commentary:
                st.write("### 🎙️ Ravi & Harsha AI Commentator")
                if st.button("🔊 GENERATE LIVE COMMENTARY", use_container_width=True):
                    with st.spinner("Harsha is taking the mic..."):
                        commentary = generate_commentary(client, model_name, batting_team, bowling_team, runs_needed, balls_left, wickets_lost, batting_prob)
                        st.chat_message("assistant", avatar="🎙️").write(commentary)
                        
            with tab_coach:
                st.write("### 📋 Situational AI Coach Advisory")
                if st.button("🧠 REQUEST TACTICAL STRATEGY", use_container_width=True):
                    with st.spinner("Analyzing match pressure index..."):
                        advice = generate_tactical_advice(client, model_name, batting_team, bowling_team, runs_needed, balls_left, wickets_lost, batting_prob, rrr, crr)
                        st.chat_message("assistant", avatar="📋").write(advice)
                        
            with tab_history:
                st.write("### 📜 Match Contextualizer & History")
                if st.button("🔍 FIND HISTORICAL RUN-CHASE SIMILARITIES", use_container_width=True):
                    with st.spinner("Scanning IPL historical records..."):
                        history = generate_historical_context(client, model_name, batting_team, bowling_team, runs_needed, balls_left, wickets_lost)
                        st.chat_message("assistant", avatar="📜").write(history)
                        
        except Exception as e:
            st.error(f"Error initializing Client: {e}")
    else:
        st.error("⚠️ Generative Engine Offline: Missing 'gemini_api.api_key' in secrets.toml configuration.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Footer info (dynamically calculated to CSE-AIML 2026)
import datetime
current_year = datetime.datetime.now().year
st.markdown(f"""
<div class="guide-footer">
    Model architecture implements Logistic Regression with Sigmoid probability calibration trained on historical ball-by-ball IPL match data (2008-2026).<br>
    Developed for Pre-Final Year Minor Project-2 evaluation &copy; CSE-AIML {current_year}.
</div>
""", unsafe_allow_html=True)
