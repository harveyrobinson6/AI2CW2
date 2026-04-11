import pandas as pd

def run_segmentation():

    print("Running segmentation...")

    files = ["2022.csv", "2023.csv", "2024.csv", "2025.csv"]
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    df['location'] = df['location'].str.strip()

    df['date_of_service'] = pd.to_datetime(df['date_of_service'], errors='coerce', dayfirst=True)

    time_cols = [
        'planned_departure_time',
        'actual_departure_time',
        'planned_arrival_time',
        'actual_arrival_time'
    ]

    for col in time_cols:
        df[col] = pd.to_datetime(
            df['date_of_service'].astype(str) + ' ' + df[col].astype(str),
            errors='coerce',
            dayfirst=True
        )

    df = df.sort_values(['date_of_service', 'planned_departure_time'])

    # ---------- EXTRACT JOURNEYS ----------
    all_journeys = []
    current_journey = []

    for _, row in df.iterrows():
        loc = row['location']

        if loc == 'WEY' and not current_journey:
            current_journey = [row]

        elif current_journey:
            current_journey.append(row)

            if loc == 'WAT':
                group = pd.DataFrame(current_journey)

                if group['actual_departure_time'].isna().all() and group['actual_arrival_time'].isna().all():
                    current_journey = []
                    continue

                all_journeys.append(group)
                current_journey = []

    # ---------- SEGMENTS ----------
    segments = []

    for journey_df in all_journeys:
        for i in range(len(journey_df) - 1):

            start_row = journey_df.iloc[i]
            end_row = journey_df.iloc[i + 1]

            if pd.isna(end_row['actual_arrival_time']):
                continue

            prev_delay = 0
            if pd.notna(start_row['actual_departure_time']) and pd.notna(start_row['planned_departure_time']):
                prev_delay = (
                    start_row['actual_departure_time']
                    - start_row['planned_departure_time']
                ).total_seconds() / 60

            delay = (
                end_row['actual_arrival_time']
                - end_row['planned_arrival_time']
            ).total_seconds() / 60

            segments.append({
                'from_station': start_row['location'],
                'to_station': end_row['location'],
                'planned_dep': start_row['planned_departure_time'],
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

    print("Segmentation complete.")