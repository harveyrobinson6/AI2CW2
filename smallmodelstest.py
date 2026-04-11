import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
os.makedirs("models", exist_ok=True)

# ---------- LOAD DATA ----------
segment_df = pd.read_csv("train_segments.csv")

# ---------- DEFINE VALID ROUTE ----------
route = ['WEY','DCH','WRM','HAM','POO','PKS','BSM','BMH','BCU','SOU','SOA','WIN','BSK','WOK','GLD','CLJ','WAT']

# Create valid adjacent segments
valid_segments = set()
for i in range(len(route)-1):
    valid_segments.add((route[i], route[i+1]))
    valid_segments.add((route[i+1], route[i]))  # include reverse direction

# ---------- FILTER DATA ----------
segment_df = segment_df[
    segment_df.apply(
        lambda row: (row['from_station'], row['to_station']) in valid_segments,
        axis=1
    )
]

print("Remaining valid segments:", len(segment_df))

# ---------- FEATURE ENGINEERING ----------
segment_df['planned_dep'] = pd.to_datetime(segment_df['planned_dep'])
segment_df['planned_dep_hour'] = segment_df['planned_dep'].dt.hour

# ---------- GROUP BY SEGMENT ----------
groups = segment_df.groupby(['from_station', 'to_station'])

models_trained = 0
mae_list = []

# ---------- TRAIN ONE MODEL PER SEGMENT ----------
for (from_station, to_station), group in groups:

    # Skip small datasets (important!)
    if len(group) < 100:
        continue

    # Features (NO station columns anymore)
    X = group[['planned_dep_hour', 'day_of_week', 'month', 'prev_delay', 'num_stops_remaining']]
    y = group['delay']

    # Train/test split
    split_index = int(len(group) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Train model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Evaluate
    if len(X_test) > 0:
        y_pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)
        print(f"{from_station}->{to_station} MAE: {mae:.2f} mins")

    # Save model
    filename = f"models/model_{from_station}_{to_station}.pkl"
    joblib.dump(rf, filename)

    models_trained += 1

# ---------- SUMMARY ----------
print("\nTotal models trained:", models_trained)

if mae_list:
    print("Average MAE across segments:", round(sum(mae_list)/len(mae_list), 2))