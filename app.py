import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Veri yükleme: geçmiş Süper Lig sezonlarının verilerini indiriyoruz
@st.cache_data
def load_data():
    urls = {
        "2021-22": "https://www.football-data.co.uk/mmz4281/2122/TUR.csv",
        "2022-23": "https://www.football-data.co.uk/mmz4281/2223/TUR.csv",
        "2023-24": "https://www.football-data.co.uk/mmz4281/2324/TUR.csv"
    }
    columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    dfs = []
    for season, url in urls.items():
        df = pd.read_csv(url)
        df = df[columns]
        df['Season'] = season
        dfs.append(df)
    df_all = pd.concat(dfs)
    df_all.dropna(inplace=True)
    df_all['Date'] = pd.to_datetime(df_all['Date'], dayfirst=True)
    df_all.sort_values('Date', inplace=True)
    df_all['Result'] = df_all['FTR'].map({'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'})
    return df_all

# Özellik mühendisliği: takımların son formu ve ortalama golleri hesaplanıyor
def feature_engineering(df):
    team_form = {}
    features = []

    for i, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']

        def get_last_results(team):
            return team_form.get(team, [])[-5:]

        def points_from_results(results):
            return sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in results])

        home_form = get_last_results(home)
        away_form = get_last_results(away)

        features.append({
            'HomePointsLast5': points_from_results(home_form),
            'AwayPointsLast5': points_from_results(away_form),
            'HomeWinsLast5': home_form.count('W'),
            'AwayWinsLast5': away_form.count('W')
        })

        if row['FTR'] == 'H':
            team_form.setdefault(home, []).append('W')
            team_form.setdefault(away, []).append('L')
        elif row['FTR'] == 'A':
            team_form.setdefault(home, []).append('L')
            team_form.setdefault(away, []).append('W')
        else:
            team_form.setdefault(home, []).append('D')
            team_form.setdefault(away, []).append('D')

    features_df = pd.DataFrame(features)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, features_df], axis=1)

    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    avg_goals = {}
    for team in teams:
        home_goals = df[df['HomeTeam'] == team]['FTHG'].sum()
        away_goals = df[df['AwayTeam'] == team]['FTAG'].sum()
        matches = len(df[df['HomeTeam'] == team]) + len(df[df['AwayTeam'] == team])
        avg_goals[team] = (home_goals + away_goals) / matches if matches > 0 else 0

    df['HomeAvgGoals'] = df['HomeTeam'].map(avg_goals)
    df['AwayAvgGoals'] = df['AwayTeam'].map(avg_goals)
    return df

# Model eğitimi: RandomForest ile sonuç tahmini
@st.cache_resource
def train_model(df):
    features = ['HomePointsLast5', 'AwayPointsLast5', 'HomeWinsLast5', 'AwayWinsLast5', 'HomeAvgGoals', 'AwayAvgGoals']
    X = df[features]
    y = df['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=250, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def main():
    st.title("Turkish Süper Lig Match Outcome Predictor")
    st.write("Geçmiş verilere dayanarak maç sonucu tahmin edebilirsiniz.")

    df = load_data()
    df = feature_engineering(df)
    model, X_test, y_test = train_model(df)

    st.subheader("Model Accuracy on Test Data")
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    st.write(f"Accuracy: {acc:.2%}")

    # Kullanıcıdan maç bilgisi al
    st.subheader("Predict a New Match")
    teams = sorted(pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel()))
    home_team = st.selectbox("Select Home Team", teams)
    away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team])

    def get_team_features(team):
        recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)
        points = 0
        wins = 0
        for idx, r in recent.iterrows():
            if r['HomeTeam'] == team:
                if r['FTR'] == 'H':
                    wins += 1
                    points += 3
                elif r['FTR'] == 'D':
                    points += 1
            else:
                if r['FTR'] == 'A':
                    wins += 1
                    points += 3
                elif r['FTR'] == 'D':
                    points += 1
        return points, wins

    h_points, h_wins = get_team_features(home_team)
    a_points, a_wins = get_team_features(away_team)

    h_avg_goals = df[df['HomeTeam'] == home_team]['FTHG'].sum() + df[df['AwayTeam'] == home_team]['FTAG'].sum()
    h_matches = len(df[df['HomeTeam'] == home_team]) + len(df[df['AwayTeam'] == home_team])
    h_avg_goals = h_avg_goals / h_matches if h_matches > 0 else 0

    a_avg_goals = df[df['HomeTeam'] == away_team]['FTHG'].sum() + df[df['AwayTeam'] == away_team]['FTAG'].sum()
    a_matches = len(df[df['HomeTeam'] == away_team]) + len(df[df['AwayTeam'] == away_team])
    a_avg_goals = a_avg_goals / a_matches if a_matches > 0 else 0

    input_data = pd.DataFrame([{
        'HomePointsLast5': h_points,
        'AwayPointsLast5': a_points,
        'HomeWinsLast5': h_wins,
        'AwayWinsLast5': a_wins,
        'HomeAvgGoals': h_avg_goals,
        'AwayAvgGoals': a_avg_goals
    }])

    if st.button("Predict Outcome"):
        prediction = model.predict(input_data)[0]
        st.write(f"### Predicted Match Result: {prediction}")

if __name__ == "__main__":
    main()
