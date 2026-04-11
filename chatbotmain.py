from segmentation import run_segmentation
from train_models import train_models
import os
import sys
import joblib
import pandas as pd
import json
import random
import spacy
from datetime import datetime
import warnings   
import re

warnings.filterwarnings("ignore")

conversation_state = {
    "from_station": None,
    "to_station": None,
    "delay": None,
    "hour": None,
    "day": None,
    "month": None
}

# ---------- ROUTE DEFINITION ----------
route = ['WEY','DCH','WRM','HAM','POO','PKS','BSM','BMH','BCU','SOU','SOA','WIN','BSK','WOK','GLD','CLJ','WAT']

station_map = {
    "WEYMOUTH": "WEY",
    "DORCHESTER": "DCH",
    "WOOL": "WRM",
    "HAMWORTHY": "HAM",
    "POOLE": "POO",
    "PARKSTONE": "PKS",
    "BOURNEMOUTH": "BMH",
    "SOUTHAMPTON": "SOU",
    "WINCHESTER": "WIN",
    "BASINGSTOKE": "BSK",
    "WOKING": "WOK",
    "CLAPHAM": "CLJ",
    "WATERLOO": "WAT"
}

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

# ---------- PREDICT SINGLE JOURNEY ----------
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

    print("\n--- Segment predictions ---")
    for i in range(len(journey_route)-1):
        seg = (journey_route[i], journey_route[i+1])
        if seg not in models:
            print(f"{seg} : no model, assuming delay unchanged")
            continue

        model = models[seg]
        X = pd.DataFrame([{
            'planned_dep_hour': hour,
            'day_of_week': day,
            'month': month,
            'prev_delay': prev_delay,
            'num_stops_remaining': len(journey_route)-i-1
        }])

        change = model.predict(X)[0]
        change = max(-5, min(change,15))  # clamp same as training

        total_delay += change
        prev_delay += change

        total_delay = max(0, min(total_delay, 30))
        #total_delay = max(0, min(total_delay, 60))

        print(f"{seg} predicted change: {change:.2f}, total delay: {total_delay:.2f}")

    return total_delay

def models_exist():
    if not os.path.exists("models"):
        return False
    return any(f.endswith(".pkl") for f in os.listdir("models"))

# ---------- AUTO SETUP ----------
if not models_exist():
    print("No models found. Running full pipeline...\n")
    run_segmentation()
    train_models()
else:
    print("Models already exist. Skipping training.\n")



intentions_path = "intentions.json"

# text file that contains sentences for time/date detection
sentences_path = "sentences.txt"

# variables to store all time-related sentences
time_sentences = ''

# variables to store all date-related sentences
date_sentences = ''

# open the file that contains example sentences
with open(sentences_path) as file:

    # read the file line by line
    for line in file:

        # split each line using the separator " | "
        # parts[0] = label (time or date)
        # parts[1] = the sentence
        parts = line.split(' | ')

        # if the sentence is labelled as time
        if parts[0] == 'time':

            # add the sentence to the time_sentences string
            time_sentences = time_sentences + ' ' + parts[1].strip()

        # if the sentence is labelled as date
        elif parts[0] == 'date':

            # add the sentence to the date_sentences string
            date_sentences = date_sentences + ' ' + parts[1].strip()


# print all collected time sentences
print(time_sentences)

# print separator line
print('*' * 50)

# print all collected date sentences
print(date_sentences)


# -----------------------------------------------------------
# Step 2: Prepare sentences for NLP processing
# -----------------------------------------------------------

# load the spaCy English model
# this model allows us to perform sentence segmentation and similarity
nlp = spacy.load('en_core_web_sm')

# list that will store labels (time or date)
labels = []

# list that will store individual sentences
sentences = []


# -----------------------------------------------------------
# Step 3: Process time sentences
# -----------------------------------------------------------

# convert the text into a spaCy document
doc = nlp(time_sentences)

# spaCy automatically detects sentence boundaries
for sentence in doc.sents:

    # store the label "time"
    labels.append("time")

    # store the sentence in lowercase format
    sentences.append(sentence.text.lower().strip())


# -----------------------------------------------------------
# Step 4: Process date sentences
# -----------------------------------------------------------

# convert date sentences into spaCy document
doc = nlp(date_sentences)

# extract each sentence
for sentence in doc.sents:

    # store the label "date"
    labels.append("date")

    # store the sentence text
    sentences.append(sentence.text.lower().strip())


# -----------------------------------------------------------
# Step 5: Display the labelled dataset
# -----------------------------------------------------------

# this loop shows which sentence belongs to which category
for label, sentence in zip(labels, sentences):

    print(label + " : " + sentence)

# open the JSON file
with open(intentions_path) as f:

    # load the JSON data into a Python dictionary
    intentions = json.load(f)

# print the loaded intentions so we can see the structure
print(json.dumps(intentions, indent=4))

def check_intention_by_keyword(sentence):
    for word in sentence.split():
        for type_of_intention in intentions:
            if word.lower() in intentions[type_of_intention]["patterns"]:
                print("BOT: " + random.choice(intentions[type_of_intention]["responses"]))      
                if type_of_intention == 'greeting':
                    print("BOT: We can talk about the time, date, and train tickets.\n(Hint: What time is it?)")
                return type_of_intention
    return None

def lemmatize_and_clean(text):
    doc = nlp(text.lower())
    out = ""

    for token in doc:
        if not token.is_stop and not token.is_punct:
            out = out + token.lemma_ + " "

    return out.strip()

def date_time_response(user_input):

    cleaned_user_input = lemmatize_and_clean(user_input)
    doc_1 = nlp(cleaned_user_input)

    similarities = {}

    for idx, sentence in enumerate(sentences):

        cleaned_sentence = lemmatize_and_clean(sentence)
        doc_2 = nlp(cleaned_sentence)

        similarity = doc_1.similarity(doc_2)

        similarities[idx] = similarity

    max_similarity_idx = max(similarities, key=similarities.get)

    min_similarity = 0.75

    if similarities[max_similarity_idx] > min_similarity:

        if labels[max_similarity_idx] == 'time':
            print("BOT: " + "It’s " + str(datetime.now().strftime('%H:%M:%S')))

            #if final_chatbot:
            print("BOT: You can also ask me what the date is today. (Hint: What is the date today?)")

        elif labels[max_similarity_idx] == 'date':
            print("BOT: " + "It’s " + str(datetime.now().strftime('%Y-%m-%d')))

            #if final_chatbot:
            print("BOT: Now can you tell me where you want to go? "
                  "(Hints: you can type in a city's name, or an organisation. "
                  "I am going to London or I want to visit the University of East Anglia.)")

        return True

    return False

def extract_travel_info(user_input):
    text = user_input.lower()

    # ---------- STATIONS ----------
    for name, code in station_map.items():
        if name.lower() in text or code.lower() in text:
            if conversation_state["from_station"] is None:
                conversation_state["from_station"] = code
                print(f"BOT: Got it, you're at {code}")
            elif conversation_state["to_station"] is None:
                conversation_state["to_station"] = code
                print(f"BOT: Travelling to {code}")

    # ---------- DELAY ----------
    delay_match = re.search(r'(\d+)\s*(min|mins|minutes)?', text)
    if delay_match and ("delay" in text or "late" in text):
        delay = float(delay_match.group(1))
        conversation_state["delay"] = delay
        print(f"BOT: Delay noted: {delay} minutes")

    # ---------- TIME (HOUR) ----------
    # matches "15", "15:30", "3pm", etc.
    time_match = re.search(r'(\d{1,2})(:\d{2})?\s*(am|pm)?', text)
    if time_match and ("time" in text or "hour" in text or "pm" in text or "am" in text):
        hour = int(time_match.group(1))

        if time_match.group(3):  # am/pm handling
            if time_match.group(3) == "pm" and hour != 12:
                hour += 12
            elif time_match.group(3) == "am" and hour == 12:
                hour = 0

        conversation_state["hour"] = hour
        print(f"BOT: Time noted: {hour}:00")

    # ---------- DAY OF WEEK ----------
    days = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }

    for day_name, day_num in days.items():
        if day_name in text:
            conversation_state["day"] = day_num
            print(f"BOT: Day noted: {day_name.capitalize()}")

    # ---------- MONTH ----------
    months = {
        "january": 1, "february": 2, "march": 3,
        "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12
    }

    for month_name, month_num in months.items():
        if month_name in text:
            conversation_state["month"] = month_num
            print(f"BOT: Month noted: {month_name.capitalize()}")

def extract_all_info(user_input):
    text = user_input.lower()

    # ---------- STATIONS ----------
    for name, code in station_map.items():
        if name.lower() in text or code.lower() in text:
            if conversation_state["from_station"] is None:
                conversation_state["from_station"] = code
            elif conversation_state["to_station"] is None:
                conversation_state["to_station"] = code

    # ---------- DELAY ----------
    delay_match = re.search(r'(\d+)\s*(min|mins|minutes)?\s*(late|delay)?', text)
    if delay_match:
        conversation_state["delay"] = float(delay_match.group(1))

    # ---------- TIME ----------
    time_match = re.search(r'(\d{1,2})(:\d{2})?\s*(am|pm)?', text)
    if time_match:
        hour = int(time_match.group(1))

        if time_match.group(3):
            if time_match.group(3) == "pm" and hour != 12:
                hour += 12
            elif time_match.group(3) == "am" and hour == 12:
                hour = 0

        conversation_state["hour"] = hour

    # ---------- DAY ----------
    days = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }

    for d, val in days.items():
        if d in text:
            conversation_state["day"] = val

    # ---------- MONTH ----------
    months = {
        "january": 1, "february": 2, "march": 3,
        "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12
    }

    for m, val in months.items():
        if m in text:
            conversation_state["month"] = val

def fill_missing_with_current_time():
    now = datetime.now()

    if conversation_state["hour"] is None:
        conversation_state["hour"] = now.hour

    if conversation_state["day"] is None:
        conversation_state["day"] = now.weekday()

    if conversation_state["month"] is None:
        conversation_state["month"] = now.month

def ask_for_missing():
    if conversation_state["from_station"] is None:
        print("BOT: Where are you now?")
        return True
    if conversation_state["to_station"] is None:
        print("BOT: Where are you going?")
        return True
    if conversation_state["delay"] is None:
        print("BOT: What is your current delay?")
        return True
    if conversation_state["hour"] is None:
        print("BOT: What hour is it? (0–23)")
        return True
    if conversation_state["day"] is None:
        print("BOT: What day of week? (0=Mon)")
        return True
    if conversation_state["month"] is None:
        print("BOT: What month?")
        return True

    return False

# ---------- START CHATBOT ----------
print("Chatbot ready")

# ---------- LOAD ALL MODELS ----------
models = {}

for file in os.listdir("models"):
    if file.startswith("model_") and file.endswith(".pkl"):
        parts = file.replace("model_", "").replace(".pkl", "").split("_")
        models[(parts[0], parts[1])] = joblib.load(f"models/{file}")

print("Models loaded:", len(models))

print("\nTrain Delay Chatbot\n")

'''
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
'''
flag = True
'''

print("BOT: Hi there! How can I help you?.\n(If you want to exit, just type bye!)")

# keep chatting while flag is True
while flag == True:

    # read input from the user
    user_input = input()

    # first try keyword matching
    intention = check_intention_by_keyword(user_input)

    # if the user says goodbye, stop the chatbot
    if intention == 'goodbye':
        flag = False

    # if no keyword intention is found, try the other modules
    elif intention == None:

        # first try NER
        #if not ner_response(user_input):

            # then try time/date similarity
            if not date_time_response(user_input):

                # then try ticket expert system
                #if not expert_response(user_input):

                    # if still no answer is found
                    print("BOT: Sorry I don't understand that. Please rephrase your statement.")
'''
while flag:

    user_input = input()

    # store extracted info
    extract_all_info(user_input)
    # keyword / NLP stuff
    intention = check_intention_by_keyword(user_input)

    if intention == 'goodbye':
        break

    if intention is None:
        if not date_time_response(user_input):

            #  CHECK IF ENOUGH DATA
            if not ask_for_missing():

                #  RUN MODEL
                predicted = predict_journey_live(
                    conversation_state["from_station"],
                    conversation_state["to_station"],
                    conversation_state["delay"],
                    int(conversation_state["hour"]),
                    int(conversation_state["day"]),
                    int(conversation_state["month"])
                )

                print(f"\nEstimated delay: {predicted:.2f} minutes\n")

                # reset for next query
                for key in conversation_state:
                    conversation_state[key] = None