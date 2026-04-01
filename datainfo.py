import pandas as pd

#load and combine data
files = ["2022.csv", "2023.csv", "2024.csv", "2025.csv"]
df_list = []

for file in files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

#clean and parse
#strip whitespace from location
df['location'] = df['location'].str.strip()

#convert date_of_service to datetime
df['date_of_service'] = pd.to_datetime(df['date_of_service'], errors='coerce', dayfirst=True)

#combine date and time columns into full datetime
time_cols = ['planned_departure_time', 'actual_departure_time', 'planned_arrival_time', 'actual_arrival_time']
for col in time_cols:
    df[col] = pd.to_datetime(
        df['date_of_service'].astype(str) + ' ' + df[col].astype(str),
        errors='coerce',
        dayfirst=True
    )

#sort by date and planned departure
df = df.sort_values(['date_of_service', 'planned_departure_time'])

#extract journeys as dataframes
all_journeys = []
current_journey = []

for _, row in df.iterrows():
    loc = row['location']

    #Start a new journey at WEY
    if loc == 'WEY' and not current_journey:
        current_journey = [row]
    elif current_journey:
        current_journey.append(row)

        #End journey if WAT reached
        if loc == 'WAT':
            group = pd.DataFrame(current_journey)

            #skip journeys where the train didnt run
            if group['actual_departure_time'].isna().all() and group['actual_arrival_time'].isna().all():
                current_journey = []
                continue

            all_journeys.append(group)  #store journey as DataFrame
            current_journey = []

#create journey level dataset
journey_records = []
for journey_df in all_journeys:
    start_row = journey_df.iloc[0]
    end_row = journey_df.iloc[-1]

    planned_dep = start_row['planned_departure_time']
    actual_dep = start_row['actual_departure_time']
    planned_arr = end_row['planned_arrival_time']
    actual_arr = end_row['actual_arrival_time']

    #handle arrival delay
    if pd.notna(actual_arr):
        if actual_arr < planned_arr:
            actual_arr += pd.Timedelta(days=1)
        total_delay = (actual_arr - planned_arr).total_seconds() / 60
    else:
        total_delay = None

    #handle departure delay
    departure_delay = None
    if pd.notna(actual_dep) and pd.notna(planned_dep):
        if actual_dep < planned_dep:
            actual_dep += pd.Timedelta(days=1)
        departure_delay = (actual_dep - planned_dep).total_seconds() / 60

    journey_records.append({
        'start': 'WEY',
        'end': 'WAT',
        'route': 'WEY-WAT',
        'departure_hour': planned_dep.hour if pd.notna(planned_dep) else None,
        'day_of_week': start_row['date_of_service'].dayofweek,
        'month': start_row['date_of_service'].month,
        'num_stops': len(journey_df),
        'departure_delay': departure_delay,
        'delay': total_delay
    })

journey_df = pd.DataFrame(journey_records)
journey_df = journey_df.dropna(subset=['delay'])
journey_df = journey_df[journey_df['delay'].between(-5, 120)]
journey_df.to_csv("processed_journeys.csv", index=False)

print("Journey-level dataset saved as processed_journeys.csv")
print("Valid journeys:", len(journey_df))

#create segment level dataset, replace with lots of smaller models soon
segments = []

for journey_df in all_journeys:
    if journey_df['actual_arrival_time'].isna().all() and journey_df['actual_departure_time'].isna().all():
        continue

    for i in range(len(journey_df) - 1):
        start_row = journey_df.iloc[i]
        end_row = journey_df.iloc[i+1]

        if pd.isna(end_row['actual_arrival_time']):
            continue

        prev_delay = 0
        if pd.notna(start_row['actual_departure_time']) and pd.notna(start_row['planned_departure_time']):
            prev_delay = (start_row['actual_departure_time'] - start_row['planned_departure_time']).total_seconds() / 60

        delay = None
        if pd.notna(end_row['planned_arrival_time']) and pd.notna(end_row['actual_arrival_time']):
            delay = (end_row['actual_arrival_time'] - end_row['planned_arrival_time']).total_seconds() / 60

        segments.append({
            'from_station': start_row['location'],
            'to_station': end_row['location'],
            'planned_dep': start_row['planned_departure_time'],
            'planned_arr': end_row['planned_arrival_time'],
            'actual_dep': start_row['actual_departure_time'],
            'actual_arr': end_row['actual_arrival_time'],
            'day_of_week': start_row['date_of_service'].dayofweek,
            'month': start_row['date_of_service'].month,
            'prev_delay': prev_delay,
            'delay': delay,
            'num_stops_remaining': len(journey_df) - i - 1
        })

segment_df = pd.DataFrame(segments)
segment_df = segment_df.dropna(subset=['delay'])
segment_df = segment_df[segment_df['delay'].between(-5, 120)]
segment_df.to_csv("train_segments.csv", index=False)

print("Segment-level dataset saved as train_segments.csv")
print("Total segments:", len(segment_df))
print("\nExample rows:")
print(segment_df.head())