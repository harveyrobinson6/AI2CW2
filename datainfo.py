import pandas as pd

#LOAD & COMBINE DATA

files = ["2022.csv", "2023.csv", "2024.csv", "2025.csv"]
df_list = []

for file in files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

df['journey_id'] = df['rid'].astype(str) + "_" + df['date_of_service'].astype(str)

#print(df.columns)

#CLEAN TIME COLUMNS

time_cols = [
    'planned_departure_time',
    'actual_departure_time',
    'planned_arrival_time',
    'actual_arrival_time'
]

for col in time_cols:
    df[col] = pd.to_datetime(df[col], format='%H:%M', errors='coerce')


print("Total rows:", len(df))
print("Total journeys (rid):", df['rid'].nunique())

#BUILD JOURNEY DATASET

journeys = []
valid_journeys = 0

for jid, group in df.groupby('journey_id'):

    #sort stops in correct order
    group = group.sort_values(
    ['planned_departure_time', 'planned_arrival_time'])

    #drop rows with no useful timing
    group = group.dropna(subset=['planned_departure_time', 'planned_arrival_time'], how='all')

    if len(group) < 2:
        continue

    #START OF JOURNEY
    start_row = group.iloc[0]
    end_row = group.iloc[-1]

    start_station = start_row['location']
    end_station = end_row['location']

    planned_dep = start_row['planned_departure_time']
    actual_dep = start_row['actual_departure_time']

    planned_arr = end_row['planned_arrival_time']
    actual_arr = end_row['actual_arrival_time']

    #skip if critical data missing
    if pd.isna(planned_arr):
        continue

    if pd.isna(actual_arr):
        total_delay = 0  #assume on time or skip later
    else:
        if actual_arr < planned_arr:
            actual_arr += pd.Timedelta(days=1)
        total_delay = (actual_arr - planned_arr).total_seconds() / 60

    #MIDNIGHT ROLLOVER
    if actual_arr < planned_arr:
        actual_arr += pd.Timedelta(days=1)

    if actual_dep < planned_dep:
        actual_dep += pd.Timedelta(days=1)

    #CALC DELAYS
    total_delay = (actual_arr - planned_arr).total_seconds() / 60

    departure_delay = None
    if pd.notna(actual_dep):
        departure_delay = (actual_dep - planned_dep).total_seconds() / 60

    #BUILD RECORD
    journeys.append({
        'jid': jid,
        'start': start_station,
        'end': end_station,
        'route': f"{start_station}-{end_station}",
        'departure_hour': planned_dep.hour,
        'day_of_week': planned_dep.dayofweek,
        'month': planned_dep.month,
        'num_stops': len(group),
        'departure_delay': departure_delay,
        'delay': total_delay   #TARGET
    })

    valid_journeys += 1

print("Valid journeys:", valid_journeys)

#convert to DataFrame
journey_df = pd.DataFrame(journeys)

#CLEAN FINAL DATASET

#drop missing values
journey_df = journey_df.dropna(subset=['delay'])

#remove unrealistic delays
journey_df = journey_df[journey_df['delay'].between(-5, 120)]

#INFO

print("\nRoute counts:")
print(journey_df['route'].value_counts())

print("\nDelay stats:")
print(journey_df['delay'].describe())

#SAVE CLEAN DATA

journey_df.to_csv("processed_journeys.csv", index=False)

print("\nProcessing complete. Saved as processed_journeys.csv")