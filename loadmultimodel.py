import joblib
import pandas as pd
import os
import sys

# ---------- LOAD ALL MODELS ----------
models = {}

for file in os.listdir("models"):
    if file.startswith("model_") and file.endswith(".pkl"):
        parts = file.replace("model_", "").replace(".pkl", "").split("_")
        models[(parts[0], parts[1])] = joblib.load(f"models/{file}")

print("Models loaded:", len(models))

# ---------- ROUTE DEFINITION ----------
route = ['WEY','DCH','WRM','HAM','POO','PKS','BSM','BMH','BCU','SOU','SOA','WIN','BSK','WOK','GLD','CLJ','WAT']

# ---------- PREDICT SINGLE SEGMENT ----------
def predict_segment(from_station, to_station, hour, day, month, prev_delay, stops_remaining):
    
    seg = (from_station, to_station)
    
    # If model doesn't exist
    if seg not in models:
        print(f"No model for segment {seg}, skipping...")
        return prev_delay  # assume delay stays the same
    
    model = models[seg]
    
    X = pd.DataFrame([{
        'planned_dep_hour': hour,
        'day_of_week': day,
        'month': month,
        'prev_delay': prev_delay,
        'num_stops_remaining': stops_remaining
    }])
    
    prediction = model.predict(X)[0]
    
    # Clamp unrealistic values
    prediction = max(-5, min(prediction, 60))
    
    return prediction

# ---------- PREDICT FULL JOURNEY ----------
def predict_journey_live(from_station, to_station, current_delay, hour, day, month):

    if from_station not in route or to_station not in route:
        print("Invalid station")
        return None

    start = route.index(from_station)
    end = route.index(to_station)

    if start >= end:
        print("Invalid direction")
        return None

    journey_route = route[start:end+1]

    total_delay = current_delay
    prev_delay = current_delay

    for i in range(len(journey_route)-1):
        seg = (journey_route[i], journey_route[i+1])

        if seg not in models:
            continue

        model = models[seg]

        X = pd.DataFrame([{
            'planned_dep_hour': hour,
            'day_of_week': day,
            'month': month,
            'prev_delay': prev_delay,
            'num_stops_remaining': len(journey_route)-i-1
        }])

        delay = model.predict(X)[0]

        increment = delay - prev_delay
        total_delay += increment
        prev_delay = delay

    return total_delay

# ---------- COMMAND LINE ----------
def main():
    print("\nTrain Delay Chatbot\n")

    while True:
        current_station = input("Where are you now? ").strip().upper()
        destination = input("Where are you going? ").strip().upper()
        
        try:
            current_delay = float(input("Current delay (minutes): "))
            hour = int(input("Current hour (0-23): "))
            day = int(input("Day of week (0=Mon): "))
            month = int(input("Month (1-12): "))
        except:
            print("Invalid input")
            continue

        predicted = predict_journey_live(
            current_station,
            destination,
            current_delay,
            hour,
            day,
            month
        )

        if predicted is not None:
            print(f"\nEstimated delay at destination: {predicted:.2f} minutes\n")

        again = input("Continue? (y/n): ").lower()
        if again != "y":
            sys.exit()

# ---------- RUN ----------
main()

'''
From: WEY
To: WRM
Hour: 8
Day: 0
Month: 1

From: HAM
To: WOK
Hour: 17
Day: 4
Month: 11

From: WEY
To: WAT
Hour: 9
Day: 2
Month: 6

From: BMH
To: SOU
Hour: 8
Day: 1
Month: 3

From: SOU
To: WAT
Hour: 22
Day: 5
Month: 12
'''