import transformers
import torch
import sys 
import pickle
import numpy as np
from groq import Groq
import os
import time

requests_per_minute = 30
def llm_assign_preferences(feature_pairs, goal, feature_names, model_name="llama3-70b-8192"):
    """ 
    Assigns preferences between segments based on groq api LLM query 
       0 = first segment
       1 = second segment
       -1 = equally preferable
    """ 
    
    client = Groq(api_key='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')
        
    preferences = []
    requests_made = 0
    WAIT_TIME = 60

    # individual queries
    for pair in feature_pairs:
        car_1_distance = round(pair[0][0], 2)
        car_2_distance = round(pair[1][0], 2)
        car_1_crash = "didn’t crash" if pair[0][1] == 0.0 else "crashed"
        car_2_crash = "didn’t crash" if pair[1][1] == 0.0 else "crashed"

        query = f"""Goal: We're trying to drive safely and efficiently on a highway. We have two cars with information about how far each traveled (in meters) and whether they crashed.
    Car 1: Moved {car_1_distance} meters toward its goal and {car_1_crash}.
    Car 2: Moved {car_2_distance} meters toward its goal and {car_2_crash}.
    Question: Which car performed better in terms of safe and efficient driving?
    If Car 1 did better, return 0.
    If Car 2 did better, return 1.
    If there's no preference, return -1.
    Important: Return only the number (0, 1, or -1) as your answer - do not include any other words."""

        print(query)  

        # make sure query goes through
        while True:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    temperature=0,
                    max_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                )

                # parse  preference response as int
                pref = int(completion.choices[0].message.content.strip())
                print("Preference is", pref)
                preferences.append(pref)
                requests_made += 1

                # wait for rate-limiting 
                if requests_made >= requests_per_minute:
                    print(f"Reached {requests_per_minute} requests. Waiting for {WAIT_TIME} seconds...")
                    time.sleep(WAIT_TIME)
                    requests_made = 0 

                break  # exit the while loop when successful

            except Exception as e:
                print(f"Error occurred: {e}. Waiting for {WAIT_TIME} seconds before retrying...")
                time.sleep(WAIT_TIME)
                requests_made = 0

        # # OLD INDIV QUERY
        # base_string = "We have two segments with the following total sums: "
        # base_string += ", ".join(feature_names) + ". "
        # segment_0 = "Segment 0 has " + ", ".join([f"{feature_names[i]} of {pair[0][i]}" for i in range(len(feature_names))]) + ". "
        # segment_1 = "Segment 1 has " + ", ".join([f"{feature_names[i]} of {pair[1][i]}" for i in range(len(feature_names))]) + ". "
        # comparison_string = f"Which is better? Take note of the differences in the two trajectories and our key goal of {goal}. If segment 0 is better for the goal, output 0, or if segment 1 is better, output 1, else if both trajectories are equally preferable, output -1. IMPORTANT: Please return your output as ONLY an integer - 0 for segment 0 preferred, 1 for segment 1 preferred, and -1 for equally preferable. DO NOT INCLUDE ANY OTHER WORDS"
        # final_string = base_string + segment_0 + segment_1 + comparison_string
        

    # # GROUP QUERY 
    # query = "Goal: We're trying to drive safely and efficiently on a highway. Below are 45 pairs of cars, with information about how far each traveled (in meters) and whether they crashed.\n\n"

    # query += "For each pair: Please tell us which car performed better in terms of safe and efficient driving.\n"
    # query += "If Car 1 did better, return 0.\n"
    # query += "If Car 2 did better, return 1.\n"
    # query += "If there's no preference, return -1.\n"
    # query += f"Important: Please return your output as a sequence of integers (0, 1, or -1) (ONE NUMBER FOR EACH AND EVERY PAIR - should have exactly {len(feature_pairs)} in order). For example: 0 1 -1 0 1 ...\n*DO NOT RETURN ANY OTHER TEXT*\n"

    # for idx, pair in enumerate(feature_pairs):
    #     car_1 = f"Car 1: Moved {round(pair[0][0],2)} meters toward its goal and {'didn’t crash' if pair[0][1] == 0.0 else 'crashed'}."
    #     car_2 = f"Car 2: Moved {round(pair[1][0],2)} meters toward its goal and {'didn’t crash' if pair[1][1] == 0.0 else 'crashed'}."
    #     query += f"Pair {idx + 1}:\n{car_1}\n{car_2}\n\n"
    
    print(f"num feature pairs is {len(feature_pairs)} and num preferences is {len(preferences)}.")
    if len(feature_pairs) != len(preferences):
        raise ValueError("Wrong number of preferences")
    return preferences



def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("command should be formatted as: python llm_assign_preferences_llama.py <input_file_name> <output_file_name> <optional: modelname>, you only provided", len(sys.argv))
        sys.exit(1)

    # read inputs
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) == 4 else "llama3-70b-8192"

    with open(input_file_name, 'rb') as f:
        pairs, feature_sums = pickle.load(f)
    print(f"done loading: dataset has {len(feature_sums)} pairs")

    # this is just bc our first two features in generated dataset are redundant
    feature_sums = [(
        pair1[1:],  
        pair2[1:]  
    ) for pair1, pair2 in feature_sums]
    feature_sums = feature_sums[:100]
    
    prefs = []
    for index, pair in enumerate(feature_sums):
        if index >= 45:
            car_1_distance = round(pair[0][0], 2)
            car_2_distance = round(pair[1][0], 2)
            car_1_crash = "didn’t crash" if pair[0][1] == 0.0 else "crashed"
            car_2_crash = "didn’t crash" if pair[1][1] == 0.0 else "crashed"

            print(f"INDEX {index} \nCar 1: Moved {car_1_distance} meters toward its goal and {car_1_crash}. \nCar 2: Moved {car_2_distance} meters toward its goal and {car_2_crash}.")
            try:
                preference = int(input("Enter 0, 1, or -1 preference:"))
            except Exception as e:
                print(f"Error occurred: {e}. here are curr prefs")
                print(prefs)
            prefs.append(preference)
    if len(prefs) == 55:
        print("preferences are \n", prefs)
    else:
        print("WRONG NUM PREFS...")
    # # can switch this out as we vary inputs
    # feature_names = ["car displacement (distance)","crashed"]
    # preferences = llm_assign_preferences(feature_sums, "to drive safely and efficiently across a highway", feature_names, model_name)

    # print("preferences found")
    # print(preferences)
    # with open(output_file_name, 'wb') as f:
    #     pickle.dump((preferences), f)
    
if __name__ ==  "__main__":
    main()



# gemini
import google.generativeai as genai
import google.api_core.exceptions

def llm_assign_preferences_gemini(feature_pairs, goal, feature_names):
    """ 
    Original llm pref generation code - new one is in llm_assign_preferences_llama.py
    -- Using gemini llm to assign preferences based on llm query on which segment is better 
       0 = first segment
       1 = second segment
       -1 = equally preferable
    """ 
    genai.configure(api_key="AIzaSyD_KERnQ2CGAxL1zXb-2d0HizkUrniCEao")
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    preferences = []
    requests_made = 0
    WAIT_TIME = 60

    for pair in feature_pairs:
        # make query string
        base_string = "We have two segments with the following total sums: "
        base_string += ", ".join(feature_names) + ". "
        segment_0 = "Segment 0 has " + ", ".join([f"{feature_names[i]} of {pair[0][i]}" for i in range(len(feature_names))]) + ". "
        segment_1 = "Segment 1 has " + ", ".join([f"{feature_names[i]} of {pair[1][i]}" for i in range(len(feature_names))]) + ". "
        comparison_string = f"Which is better? Take note of the differences in the two trajectories and our key goal of {goal}. If segment 0 is better for the goal, output 0, or if segment 1 is better, output 1, else if both trajectories are equally preferable, output -1. IMPORTANT: Please return your output as just the integer - 0 for segment 0 preferred, 1 for segment 1 preferred, and -1 for equally preferable."
        final_string = base_string + segment_0 + segment_1 + comparison_string
        print(final_string)
        

        while True:
            try:
                response = model.generate_content(final_string)
                pref = int(response.text.strip())
                print("preference is", pref)
                preferences.append(pref)
                requests_made += 1

                if requests_made >= 15:
                    print(f"Reached 15 requests. Waiting for {WAIT_TIME} seconds...")
                    time.sleep(WAIT_TIME)
                    requests_made = 0 

                break 

            except google.api_core.exceptions.ResourceExhausted as e:
                print(f"Quota exceeded. Waiting for {WAIT_TIME} seconds before retrying...")
                time.sleep(WAIT_TIME)
                requests_made = 0 

    print(f"num feature pairs is {len(feature_pairs)} and num preferences is {len(preferences)}.")
    if len(feature_pairs) != len(preferences):
        raise ValueError("Wrong number of preferences")
    return preferences
