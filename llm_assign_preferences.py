import cdd
import numpy as np
from groq import Groq
import pickle
import time
import sys
from possible_subspace import check_and_remove_conflicts, resolve_conflicts, find_full_weight_space
import volume
# from visualize import visualize_polyhedron
import random
from scipy.optimize import linprog

REQUESTS_PER_MINUTE = 30
WAIT_TIME = 60


def llm_assign_single_pref(pair):
    client = Groq(api_key='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')  # initialize client

    prompt_prefix = "Goal: We're trying to drive safely and efficiently on a highway. We have two cars with information about how far each traveled (in kilometers) and whether they crashed. You are acting as a human, so please make decisions according to what would be expected in society."
    prompt_suffix = ("Question: Which car performed better in terms of safe and efficient driving? "
                     "If Car 1 did better, return 0. If Car 2 did better, return 1. "
                     "If there's no preference, return -1. "
                     "Important: Return only a number (from the options 0, 1, or -1) as your answer - do not include any words. ")
    model_name = "llama3-70b-8192"
    car_1_distance = round(pair[0][0], 2)
    car_2_distance = round(pair[1][0], 2)
    car_1_crash = "didn’t crash" if pair[0][1] == 0.0 else "crashed"
    car_2_crash = "didn’t crash" if pair[1][1] == 0.0 else "crashed"

    query = f"""{prompt_prefix} {prompt_suffix}\nHere is the information for the two cars:\n
    Car 1: Moved {car_1_distance} kilometers toward its goal and {car_1_crash}.
    Car 2: Moved {car_2_distance} kilometers toward its goal and {car_2_crash}.
    """

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

            # parse preference response as float
            pref_str = completion.choices[0].message.content.strip()
            pref = int(pref_str)
            if pref not in [0, 1, -1]:
                print(f"Invalid preference value received: {pref}")
            return pref
        except Exception as e:
                print(f"Error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(60)  # Wait before retrying


def user_assign_single_pref(pair):
    car_1_distance = round(pair[0][0], 2)
    car_2_distance = round(pair[1][0], 2)
    car_1_crash = "didn’t crash" if pair[0][1] == 0.0 else "crashed"
    car_2_crash = "didn’t crash" if pair[1][1] == 0.0 else "crashed"

    print("\n--- Compare the following two cars ---")
    print(f"Car 1: Moved {car_1_distance} km and {car_1_crash}")
    print(f"Car 2: Moved {car_2_distance} km and {car_2_crash}")
    print("Which car performed better in terms of safe and efficient driving?")
    print("Enter 0 if Car 1 is better, 1 if Car 2 is better, or -1 if there is no preference.")

    while True:
        user_input = input("Your choice (0, 1, or -1): ").strip()
        if user_input in {"0", "1", "-1"}:
            return int(user_input)
        else:
            print("Invalid input. Please enter 0, 1, or -1.")

def llm_assign_preferences_group(feature_pairs, prompt_prefix, prompt_suffix, model_name, chunk_size=10):
    """
    assign preferences for feature pairs in smaller groups (batch the queries)
    """
    preferences = []
    requests_made = 0
    first = True
    client = Groq(api_key='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')

    # chunk the feature pairs
    for i in range(0, len(feature_pairs), chunk_size):
        current_chunk = feature_pairs[i:i+chunk_size]
        
        # construct prompt
        query = (
            f"Goal: We're trying to drive safely and efficiently on a highway. We have {len(current_chunk)} pairs of cars with information about how far each traveled (in kilometers) and whether they crashed.\n\n"
            f"Output: Please identify which cars performed better in terms of safe and efficient driving for each pair - if Car 1 did better, represent your preference as 0. "
            f"If Car 2 did better, use 1. If there's no preference, use -1. "
            f"Return a sequence of {len(current_chunk)} comma-separated integers 0, 1, -1 representing the better car in every pair. Do not include any words or explanations (even at the start or end).\n\n"
            f"Example Input: \nPair 1: \nCar 1: 0.75m, didn’t crash.\nCar 2: 4.89m, didn’t crash.\nPair 2:\nCar 1: 4.0m, didn’t crash.\nCar 2: 5.0m, crashed.\nPair 3:\nCar 1: 2.38m, didn’t crash.\nCar 2: 2.38m, didn’t crash.\nExample Output: 1, 0, -1\n\n"
            f"Here are the actual input pairs:\n"
        )
    
        # add each pair to the query
        for idx, pair in enumerate(current_chunk):
            car_1 = f"Car 1: {round(pair[0][0], 2)}m, {'didn’t crash' if pair[0][1] == 0.0 else 'crashed'}."
            car_2 = f"Car 2: {round(pair[1][0], 2)}m, {'didn’t crash' if pair[1][1] == 0.0 else 'crashed'}."
            query += f"Pair {idx + 1}:\n{car_1}\n{car_2}\n"
        query += "\nYour Output Here:"

        if first:
            first = False
            print("Group Query:", query)
        
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
                response_text = completion.choices[0].message.content.strip()
                new_pref = [int(pref.strip()) for pref in response_text.replace('\n', ',').split(',') if pref.strip()]
                
                if len(new_pref) != len(current_chunk):
                    raise ValueError(f"Expected {len(current_chunk)} preferences, but got {len(new_pref)}.")
                
                preferences.extend(new_pref)
                requests_made += 1
                
                if requests_made >= REQUESTS_PER_MINUTE:
                    print(f"Reached {REQUESTS_PER_MINUTE} requests. Waiting for {WAIT_TIME} seconds...")
                    time.sleep(WAIT_TIME)
                    requests_made = 0 

                break  

            except Exception as e:
                print(f"Error occurred: {e}. Retrying in 60 seconds...")
                time.sleep(60) # wait in case of timeout
    print("all preferences:", preferences)
    return preferences

def assign_preferences_consistently(feature_pairs, og_preferences=None):
    """assign preferences one by one and resolve conflicts after each assignment"""
    preferences = []
    requests_made = 0
    
    for idx, pair in enumerate(feature_pairs):
        if requests_made >= 30:
            print(f"Reached 30 requests. Waiting for 60 seconds...")
            time.sleep(60)
            requests_made = 0 

        if og_preferences:
            pref = og_preferences[idx] 
        else: 
            pref = llm_assign_single_pref(pair)
            requests_made += 1 
        preferences.append(pref)
         
        # check for conflicts
        conflicts = check_and_remove_conflicts(feature_pairs, preferences)
        if conflicts:
            if requests_made >= 30:
                print(f"Reached 30 requests. Waiting for 60 seconds...")
                time.sleep(60)
                requests_made = 0 
            preferences = resolve_conflicts(conflicts, feature_pairs, preferences)
            requests_made += 1

    return preferences

def final_assign_preferences(feature_pairs):
    # just a variation of individual where I was playing with the prompt
    client = Groq(api_key='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')
    preferences = []
    requests_made = -1
    prompt_prefix = "Goal: We're trying to drive safely and efficiently on a highway. We have two cars with information about how far each traveled (in kilometers) and whether they crashed."
    prompt_suffix = ("Question: Which car performed better in terms of safe and efficient driving? "
                     "If Car 1 did better, return 0. If Car 2 did better, return 1. "
                     "If there's no preference, return -1. "
                     "Important: Return only a number (from the options 0, 1, or -1) as your answer - do not include any words. ")
    model_name = "llama3-70b-8192"
 
    for idx, pair in enumerate(feature_pairs):
        car_1_distance = round(pair[0][0], 2)
        car_2_distance = round(pair[1][0], 2)
        car_1_crash = "didn’t crash" if pair[0][1] == 0.0 else "crashed"
        car_2_crash = "didn’t crash" if pair[1][1] == 0.0 else "crashed"

        query = f"""{prompt_prefix} {prompt_suffix}\nHere is the information for the two cars:\n
        Car 1: Moved {car_1_distance} kilometers toward its goal and {car_1_crash}.
        Car 2: Moved {car_2_distance} kilometers toward its goal and {car_2_crash}.
        """

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
                    max_tokens=10,  # Adjusted to receive only the number
                    top_p=1,
                    stream=False,
                    stop=None,
                )

                # parse preference response as float
                pref_str = completion.choices[0].message.content.strip()
                pref = int(pref_str)
                if pref not in [0, 1, -1]:
                    print(f"Invalid preference value received: {pref}")
                
                preferences.append(pref)
                requests_made += 1
                print(f"Assigned preference for pair {idx + 1}: {pref}")

                # Handle rate limiting
                if requests_made >= REQUESTS_PER_MINUTE:
                    print(f"Reached {REQUESTS_PER_MINUTE} requests. Waiting for {WAIT_TIME} seconds...")
                    time.sleep(WAIT_TIME)
                    requests_made = 0 

                break  # exit the while loop when successful

            except Exception as e:
                print(f"Error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(5)  # Wait before retrying
    return preferences

def run_initial_tests():
    # modify here for testing various different prefixes / suffixes
    ind_prompt_prefix = [
        "Goal: We're trying to drive safely and efficiently on a highway. We have two cars with information about how far each traveled (in kilometers) and whether they crashed.",
    ]
    ind_prompt_suffix = [
        "Question: Which car performed better in terms of safe and efficient driving? If Car 1 did better, return 0. If Car 2 did better, return 1. If there's no preference, return -1. Important: Return only a number (from the options 0, 1, or -1) as your answer - do not include any words. "
    ]

    grp_prompt_prefix = [
        "Goal: We're trying to drive safely and efficiently on a highway. We have 45 pairs of two cars with information about how far each traveled (in kilometers) and whether they crashed.",
    ]
    grp_prompt_suffix = [
        "Question: Which cars performed better in terms of safe and efficient driving? Return the result as a sequence of integers 0, 1, -1 for each of the pairs: do not include any other words"
    ]
    model_options = [
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "llama-3.1-70b-versatile",
        "llama-3.2-90b-vision-preview",
        "gemma2-9b-it",
    ]

    best_accuracy = 0
    best_group_prompt = ""
    best_individual_prompt = ""
    best_model = ""

    # Placeholder for true preferences
    true_preferences = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 
                        0, 1, 0, 1, 1, 1, 1, -1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
                        -1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 
                        -1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 
                        0, 1, 1, 1, 0, 1, 1, 0, -1, 1, 0, 1, 1, 0, 0, 0, 0, 
                        1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]

    # test group
    for prefix in grp_prompt_prefix:
        for suffix in grp_prompt_suffix:
            for model_name in model_options:
                print(f"*** GROUP PROMPT for model {model_name} with prefix '{prefix[:30]}...' and suffix '{suffix[:30]}...'")
                preferences = llm_assign_preferences_group(feature_pairs, prompt_prefix, prompt_suffix, model_name)
                group_accuracy = calculate_accuracy(preferences, true_preferences)
                print(f"Model {model_name} achieved accuracy {group_accuracy}")

                if group_accuracy > best_accuracy:
                    best_accuracy = group_accuracy
                    best_group_prompt = (prefix, suffix)
                    best_model = model_name

    # Uncomment and implement individual prompt testing if needed
    for prefix in ind_prompt_prefix:
        for suffix in ind_prompt_suffix:
            for model_name in model_options:
                print(f"*** INDIVIDUAL prompt for model {model_name} with prefix '{prefix[:30]}...' and suffix '{suffix[:30]}...'")
                preferences = llm_assign_preferences_individual(feature_pairs, prompt_prefix, prompt_suffix, model_name)
                individual_accuracy = calculate_accuracy(preferences, true_preferences)
                print(f"Model {model_name} achieved accuracy {individual_accuracy}")
                if individual_accuracy > best_accuracy:
                    best_accuracy = individual_accuracy
                    best_individual_prompt = (prefix, suffix)
                    best_model = model_name

    print("Best group prompt:", best_group_prompt)
    print("Best individual prompt:", best_individual_prompt)
    print("Best model:", best_model)
    print("Best accuracy:", best_accuracy)
    return best_group_prompt, best_individual_prompt, best_model

def calculate_accuracy(predicted_preferences, true_preferences, use_0_5_for_indifference=False):
    if len(predicted_preferences) == 0:
        return 0
    if use_0_5_for_indifference:
        predicted_preferences = [0 if p == 0.0 else (1 if p == 1.0 else -1) for p in predicted_preferences]
    
    correct_predictions = 0
    incorrect_indices = []  
    for idx, (p, t) in enumerate(zip(predicted_preferences, true_preferences)):
        if p == t:
            correct_predictions += 1
        else:
            incorrect_indices.append(idx)
    
    print(f"incorrect prediction indices: {incorrect_indices}")
    accuracy = correct_predictions / len(true_preferences)
    return accuracy

def main():
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    with open(input_file_name, 'rb') as f:
        pairs, feature_pairs = pickle.load(f)
    print(f"Done loading: dataset has {len(feature_pairs)} pairs")
    feature_pairs = [(
        pair1[1:],  
        pair2[1:]  
    ) for pair1, pair2 in feature_pairs]
    feature_pairs = feature_pairs[:100]
    true_preferences = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 
                        0, 1, 0, 1, 1, 1, 1, -1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
                        -1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 
                        -1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 
                        0, 1, 1, 1, 0, 1, 1, 0, -1, 1, 0, 1, 1, 0, 0, 0, 0, 
                        1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]

    # Run tests in order
    # print("True preference length:", len(true_preferences))
    # best_group_prompt, best_individual_prompt, best_model = run_initial_tests()
    preferences = final_assign_preferences(feature_pairs)
    # preferences = assign_preferences_consistently(feature_pairs)
    accuracy = calculate_accuracy(preferences, true_preferences)
    print(f"Final accuracy: {accuracy}")
    # with open(output_file_name, 'wb') as f:
    #     pickle.dump((preferences), f)
if __name__ == '__main__':
    main()
