import joblib
import pandas as pd
import sys

#load model and encoder
model = joblib.load("segment_delay_rf_model.pkl")
encoder = joblib.load("encoder.pkl")

print("Model and encoder loaded successfully!")

#route def
route = ['WEY','DCH','WRM','HAM','POO','PKS','BSM','BMH','BCU','SOU','SOA','WIN','BSK','WOK','GLD','CLJ','WAT']

#predict single segment
def predict_segment(from_station, to_station, hour, day, month, prev_delay, stops_remaining):
    
    df = pd.DataFrame([{
        'from_station': from_station,
        'to_station': to_station,
        'planned_dep_hour': hour,
        'day_of_week': day,
        'month': month,
        'prev_delay': prev_delay,
        'num_stops_remaining': stops_remaining
    }])
    
    #encode stations
    encoded = pd.DataFrame(
        encoder.transform(df[['from_station', 'to_station']]),
        columns=encoder.get_feature_names_out(['from_station', 'to_station'])
    )
    
    #combine features
    X = pd.concat(
        [encoded, df.drop(columns=['from_station','to_station']).reset_index(drop=True)],
        axis=1
    )
    
    prediction = model.predict(X)[0]
    
    #clamp unrealistic values
    prediction = max(-5, min(prediction, 120))
    
    return prediction

#predict full journeey
def predict_journey(from_station, to_station, hour, day, month):
    
    if from_station not in route or to_station not in route:
        print("Invalid station name.")
        return None
    
    start = route.index(from_station)
    end = route.index(to_station)
    
    if start >= end:
        print("Invalid journey direction.")
        return None
    
    journey_route = route[start:end+1]
    
    total_delay = 0
    prev_delay = 0
    
    for i in range(len(journey_route)-1):
        delay = predict_segment(
            journey_route[i],
            journey_route[i+1],
            hour,
            day,
            month,
            prev_delay,
            len(journey_route) - i - 1
        )
        
        increment = delay - prev_delay
        total_delay += increment
        prev_delay = delay
    
    return total_delay

#command line
def main():
    print("\nTrain Delay Predictor\n")
    
    while True:
        from_station = input("From station: ").strip().upper()
        to_station = input("To station: ").strip().upper()
        hour = int(input("Departure hour (0-23): "))
        day = int(input("Day of week (0=Mon, 6=Sun): "))
        month = int(input("Month (1-12): "))
        
        result = predict_journey(from_station, to_station, hour, day, month)
        
        if result is not None:
            print(f"\nPredicted total delay: {result:.2f} minutes\n")

        while True:
            again = input("Continue?")
            if again.lower() == "y":
                break
            else:
                sys.exit()

main()