import sys 
import pickle
import numpy as np
import google.generativeai as genai
import os
import time
import google.api_core.exceptions

def llm_assign_preferences(feature_pairs, goal, feature_names):
    """ 
    assigns preferences based on  assigns preferences based on llm query on which segment is better
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



def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("command should be formatted as: python assign_preferences.py <input_file_name> <output_file_name> <optional: weights>, you only provided", len(sys.argv))
        sys.exit(1)

    # read in file
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    with open(input_file_name, 'rb') as f:
        pairs, feature_sums = pickle.load(f)
    print(f"done loading: dataset has {len(feature_sums)} pairs")

    # this is just bc our first two features in generated dataset are redundant
    feature_sums = [(
        pair1[1:],  
        pair2[1:]  
    ) for pair1, pair2 in feature_sums]
    feature_sums = feature_sums[:45]

    # can switch this out as we vary inputs
    feature_names = ["car displacement (distance)","crashed"]
    preferences = assign_preferences(feature_sums, "to drive safely and efficiently across a highway", feature_names)
    print("preferences found")
    print(preferences)
    with open(output_file_name, 'wb') as f:
        pickle.dump((preferences), f)
    
if __name__ ==  "__main__":
    main()