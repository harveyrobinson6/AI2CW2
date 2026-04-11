import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_models():
    print("Training models...")

    os.makedirs("models", exist_ok=True)

    # Load segment-level data
    segment_df = pd.read_csv("train_segments.csv")

    route = ['WEY','DCH','WRM','HAM','POO','PKS','BSM','BMH','BCU','SOU','SOA','WIN','BSK','WOK','GLD','CLJ','WAT']

    # Only valid adjacent segments
    valid_segments = set()
    for i in range(len(route)-1):
        valid_segments.add((route[i], route[i+1]))
        valid_segments.add((route[i+1], route[i]))

    segment_df = segment_df[
        segment_df.apply(lambda r: (r['from_station'], r['to_station']) in valid_segments, axis=1)
    ]

    # Feature engineering
    segment_df['planned_dep'] = pd.to_datetime(segment_df['planned_dep'])
    segment_df['planned_dep_hour'] = segment_df['planned_dep'].dt.hour

    models_trained = 0

    groups = segment_df.groupby(['from_station','to_station'])
    for (from_station, to_station), group in groups:

        if len(group) < 100:
            continue  # skip too small data

        group = group.copy()

        # Learn incremental delay
        group['delay_change'] = group['delay'] - group['prev_delay']

        # Clip extreme changes
        group['delay_change'] = group['delay_change'].clip(-5,15)

        X = group[['planned_dep_hour','day_of_week','month','prev_delay','num_stops_remaining']]
        y = group['delay_change']

        split = int(len(group) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Train model
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Save model
        filename = f"models/model_{from_station}_{to_station}.pkl"
        joblib.dump(rf, filename)
        models_trained += 1

    print(f"Training complete. Models trained: {models_trained}")