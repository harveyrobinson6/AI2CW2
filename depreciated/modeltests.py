import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

#load data
segment_df = pd.read_csv("train_segments.csv")

#feature enginerring
#convert planned departure to datetime and extract hour
segment_df['planned_dep'] = pd.to_datetime(segment_df['planned_dep'])
segment_df['planned_dep_hour'] = segment_df['planned_dep'].dt.hour

#select features and target
feature_cols = [
    'from_station',
    'to_station',
    'planned_dep_hour',
    'day_of_week',
    'month',
    'prev_delay',
    'num_stops_remaining'
]
target_col = 'delay'

X = segment_df[feature_cols]
y = segment_df[target_col]

#encode categorical features
categorical_cols = ['from_station', 'to_station']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X_encoded = pd.DataFrame(
    encoder.fit_transform(X[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols)
)

#combine encoded and numerical features
X_final = pd.concat(
    [X_encoded, X.drop(columns=categorical_cols).reset_index(drop=True)],
    axis=1
)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

#train model
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

#evaluate
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Random Forest MAE: {mae:.2f} minutes")

#save model and encoder
joblib.dump(rf, "segment_delay_rf_model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model saved as segment_delay_rf_model.pkl")
print("Encoder saved as encoder.pkl")