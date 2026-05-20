# IPL Win Predictor: Real-Time Match Outcome Prediction Using Machine Learning

Minor Project-2 presentation and source code repository. Developed for the Pre-Final Year evaluation under the Department of Computer Science & Engineering (Artificial Intelligence and Machine Learning).

## Academic Metadata
*   **Project Title:** IPL Win Predictor: Real-Time Match Outcome Prediction Using Machine Learning
*   **Presented By:** Saurabh Kumar, Niraj Kumar, Kishan Mahato
*   **Branch:** Computer Science & Engineering (Artificial Intelligence and Machine Learning)
*   **Guided By:** Prof. Uma Vishwakarma
*   **HOD:** Prof. Ayonija Pathre

---

## Technical Details

### 1. Core Objective
Predict the winning probability of the batting (chasing) team and the bowling (defending) team ball-by-ball during the 2nd innings of an IPL match.

### 2. Dataset
Sourced historical data containing:
*   `matches.csv` (756 match-level insights)
*   `deliveries.csv` (179,078 ball-by-ball details)

**Preprocessing:**
*   Removed Duckworth-Lewis (rain-affected) matches to avoid run rate skewness.
*   Standardized historic franchises (e.g. `Deccan Chargers` $\rightarrow$ `Sunrisers Hyderabad`, `Delhi Daredevils` $\rightarrow$ `Delhi Capitals`).
*   Isolated 2nd innings data.

### 3. Feature Engineering (AIML Core)
Raw ball-by-ball sequences are converted into 5 core features:
1.  `runs_left` = Target Score - Current Cumulative Score
2.  `balls_left` = 120 - Balls Delivered
3.  `wickets_left` = 10 - Wickets Out
4.  `crr` (Current Run Rate) = $\frac{\text{Current Score} \times 6}{\text{Balls Bowled}}$
5.  `rrr` (Required Run Rate) = $\frac{\text{Runs Left} \times 6}{\text{Balls Left}}$

Target label `result` represents win (1) or loss (0) of the chasing team.

### 4. Model Choice
We implement a **Logistic Regression** pipeline (with `liblinear` solver). 
**Why?** Rather than tree-based classifiers (e.g. Random Forest) which predict discrete probabilities and produce step-like jumps, Logistic Regression uses the Sigmoid function to output smooth, continuous, and well-calibrated probabilities between 0 and 1. This accurately reflects the momentum of a cricket match.

---

## Project Structure
```
IPL_Win_Predictor/
│
├── data/
│   ├── matches.csv            # Sourced match-level details
│   └── deliveries.csv         # Sourced ball-by-ball details
│
├── download_data.py           # Programmatic dataset downloader
├── train.py                   # Data cleaning, feature engineering & model training
├── app.py                     # High-fidelity Streamlit user interface
├── requirements.txt           # Python dependency requirements
├── pipe.pkl                   # Serialized ML pipeline (created after running train.py)
└── README.md                  # Project documentation (this file)
```

---

## Setup & Execution Instructions

### 1. Install Dependencies
Make sure you have Python 3.10+ installed. Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Download Datasets
Download the matching CSV datasets from the verified repository:
```bash
python download_data.py
```
This downloads `matches.csv` and `deliveries.csv` directly into the `data/` directory.

### 3. Model Training & Pickling
Execute the training script to run cleaning, feature engineering, and model training:
```bash
python train.py
```
This prints the classification report and saves the model pipeline as `pipe.pkl` in the workspace root.

### 4. Run the Streamlit Application
Launch the web interface locally:
```bash
streamlit run app.py
```
Open the provided URL (usually `http://localhost:8501`) in your browser to interact with the predictor dashboard.
