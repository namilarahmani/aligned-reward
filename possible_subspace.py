from scipy.optimize import linprog
import numpy as np
import sys
import pickle
# from time import time
import time
from itertools import combinations
import cdd
import matplotlib.pyplot as plt
from groq import Groq

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from volume import calculate_polyhedron_volume

def find_feasible_weights(pairs, preferences):
    '''
    Representing each preference as an inequality constraint and using scipy linprog implementation to find a set of 
    feasible weights (or identify if there are no feasible weights)
    '''
    A_neq = []
    b_neq = []
    A_eq = []
    b_eq = []

    epsilon = 1e-5  # small margin to have < instead of <= ; this is kinda iffy though so will look for better options

    for (features_0, features_1), pref in zip(pairs, preferences):
        delta_f = features_1 - features_0
        
        if pref == 0:
            A_neq.append(delta_f)
            b_neq.append(-epsilon)
        elif pref == 1:
            A_neq.append(-delta_f)
            b_neq.append(-epsilon)
        elif pref == -1:
            A_eq.append(delta_f)
            b_eq.append(0)
    
    # convert lists to arrays for linprog
    A_neq = np.array(A_neq) if A_neq else None
    b_neq = np.array(b_neq) if b_neq else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    
    # solve using linprog
    n_features = len(pairs[0][0])  # number of features
    result = linprog(c=np.zeros(n_features), A_ub=A_neq, b_ub=b_neq, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))

    return result

def find_feasible_weight_space(pairs, preferences, test_weight=None):
    '''
    * Returns boolean for if feasible weight space exists *

    Using pycddlib 2.1.8 double description implementation to represent feasible weight space as Polyhedron
    docs here: https://pycddlib.readthedocs.io/en/2.1.8.post1/index.html 
    '''

    num_features = len(pairs[0][0])
    pref_matrix = []
    epsilon = 1e-5

    # constructing inequality matrix
    for (features_0, features_1), pref in zip(pairs, preferences):
        delta_f = features_1 - features_0
        
        if pref == 0: 
            # w · (features_1 - features_0) >= epsilon 
            row = [-epsilon] + list(delta_f)
            pref_matrix.append(row)
        elif pref == 1:  
            # w · (features_1 - features_0) <= epsilon
            row = [-epsilon] + list(-delta_f)
            pref_matrix.append(row)
        elif pref == -1:  # strict equality
            row1 = [0] + list(delta_f)
            row2 = [0] + list(-delta_f)
            pref_matrix.extend([row1, row2])

    mat = cdd.Matrix(pref_matrix, number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY

    # convert to polyhedron representation
    poly = cdd.Polyhedron(mat)
    if poly.get_generators().row_size == 0:
        # print("no feasible space!")
        return False  
    
    # check if test weights r in polyhedron - this is just for testing
    if test_weight is not None:
        inequalities = poly.get_inequalities()
        
        # each row of inequalities is in the form [b, -a1, -a2, ..., -an]
        # Representing the inequality: a1*x1 + a2*x2 + ... + an*xn <= b
        for inequality in inequalities:
            b = inequality[0]
            a = np.array(inequality[1:])
            
            # check if the inequality holds for test_weight
            if np.dot(a, test_weight) > b:
                raise ValueError("the feasible weight is NOT within the polyhedron defined by the preference constraints")
                # return False

        # print("the test weight lies within the feasible space.")
        return True
    
    # print("polyhedron of feasible weightspace is", poly.get_generators())
    return True

def find_full_weight_space(pairs, preferences, basic_bounds=None, test_weight=None, num_features = 2):
    '''
    * Returns the whole polyhedron of feasible weight space *
    '''
    pref_matrix = []
    epsilon = 1e-5
    
    if(len(pairs) != 0):
        num_features = len(pairs[0][0])
        for (f0, f1), pref in zip(pairs, preferences):
            delta_f = f1 - f0
            if pref == 0:
                pref_matrix.append([-epsilon] + list(-delta_f))
            elif pref == 1:
                pref_matrix.append([-epsilon] + list(delta_f))
            elif pref == -1:
                pref_matrix.extend([[0] + list(delta_f), [0] + list(-delta_f)])

    if basic_bounds is not None:
        if len(basic_bounds) != num_features:
            raise ValueError("basic_bounds length must match number of features")
        for i, (L, U) in enumerate(basic_bounds):
            row_lb = [-L] + [1 if j == i else 0 for j in range(num_features)]
            row_ub = [U] + [-1 if j == i else 0 for j in range(num_features)]
            pref_matrix.extend([row_lb, row_ub])

    mat = cdd.Matrix(pref_matrix, number_type='fraction')
    # volume = calculate_polyhedron_volume(mat)
    # print(volume)
    mat.rep_type = cdd.RepType.INEQUALITY
    # print("matrix issss" , mat)
    return cdd.Polyhedron(mat)


def find_conflicts(pairs, preferences):
    ''' just returns all subsets of 1,2,3 pairs that are creating conflicts '''
    weights = find_feasible_weights(pairs, preferences)
    weights = weights.x
    result = find_feasible_weight_space(pairs, preferences)
    conflicts = []

    # if result.success:
    if result:
        print("no conflicts yay")
        return None
        
    # if conflict exists, we'll try removing subsets of 1, 2, or 3 preferences
    for r in range(1, min(len(preferences), 4)):
        indices = range(len(preferences))
        for to_remove in combinations(indices, r):

            # dont reuse conflicts we alr found
            if any(set(conflict).issubset(to_remove) for conflict in conflicts):
                continue
            
            reduced_pairs = [pair for i, pair in enumerate(pairs) if i not in to_remove]
            reduced_preferences = [pref for i, pref in enumerate(preferences) if i not in to_remove]

            result = find_feasible_weight_space(reduced_pairs, reduced_preferences)
            
            # if result.success:
            if result:
                print(f"No feasible solution for the given preferences. Feasible solution can be found after removing preferences at indices {to_remove}.")
                conflicts.append(to_remove)

        if len(conflicts) == 0:
            print(f"No feasible solution exists even after removing all possible subsets of {r} preference(s).")

    print(f"Conflict can be resolved by removing any of the following subsets: {conflicts}")
    return conflicts

       
def check_and_remove_conflicts(pairs, preferences):
    weights = find_feasible_weights(pairs, preferences)
    weights = weights.x
    result = find_feasible_weight_space(pairs, preferences)
    
    # if result.success:
    if result:
        print("no conflicts yay")
        return pairs, preferences
        
    # if conflict exists, we'll try removing subsets of 1, 2, or 3 preferences
    conflicts = []
    for r in range(1, min(len(preferences), 4)):
        indices = range(len(preferences))
        for to_remove in combinations(indices, r):
            # print(to_remove)

            # dont reuse conflicts we alr found
            if any(set(conflict).issubset(to_remove) for conflict in conflicts):
                continue
            
            reduced_pairs = [pair for i, pair in enumerate(pairs) if i not in to_remove]
            reduced_preferences = [pref for i, pref in enumerate(preferences) if i not in to_remove]

            result = find_feasible_weight_space(reduced_pairs, reduced_preferences)
            
            # if result.success:
            if result:
                print(f"No feasible solution for the given preferences. Feasible solution can be found after removing preferences at indices {to_remove}.")
                # print(f"Weights for this solution are {result.x}")
                conflicts.append(to_remove)

        if len(conflicts) == 0:
            print(f"No feasible solution exists even after removing all possible subsets of {r} preference(s).")
            return [], []

    print(f"Conflict can be resolved by removing any of the following subsets: {conflicts}")
    # return reduced_pairs, reduced_preferences
    return resolve_conflicts(conflicts, pairs, preferences)

def resolve_conflicts(conflicts,feature_sums,preferences):
    """
    resolves conflicts by querying the LLM to decide which preferences to change

    Args:
        conflicts (List[Tuple[int, ...]]): List of conflict subsets.
        feature_sums (List[Tuple[Tuple[float, float], Tuple[float, float]]]): List of feature pairs.
        preferences (List[float]): Current list of preferences.

    Returns:
        List[float]: Updated preferences after conflict resolution.
    """
    client = Groq(api_key='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')
    requests_made = 0
    WAIT_TIME = 60  
    REQUESTS_PER_MINUTE = 60  

    max_iterations = 10 
    iteration = 0

    while conflicts and iteration < max_iterations:
        print(f"\nConflict Resolution Iteration: {iteration + 1}")
        option_number = 0
        for conflict_set in conflicts:
            options_text = ""
            for idx in conflict_set:
                car1_distance = round(feature_sums[idx][0][0], 2)
                car2_distance = round(feature_sums[idx][1][0], 2)
                car1_crash = "didn’t crash" if feature_sums[idx][0][1] == 0.0 else "crashed"
                car2_crash = "didn’t crash" if feature_sums[idx][1][1] == 0.0 else "crashed"

                option_number += 1 
                options_text += f"""
                Option {option_number}:
                    Preference {idx}:
                        Car 1: Moved {car1_distance} kilometers toward its goal and {car1_crash}.
                        Car 2: Moved {car2_distance} kilometers toward its goal and {car2_crash}.
                """

            prompt = (f"Goal: We're trying to drive safely and efficiently on a highway. We have two cars with information about how far each traveled (in kilometers) and whether they crashed.\n\n"
                      f"Preferences have been selected for various car trajectories. You are assisting in conflict resolution for these preferences. Below are some options of preferences that you can change to resolve the conflict:\n\n"
                      f"{options_text}\n"
                      f"Please select the option number that you would like to change to resolve the conflict while adhering best to the goal of safe highway driving. Return only a number (from the options) as your answer - do not include any words.")
            query = prompt.strip()
            while True:
                try:
                    completion = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {
                                "role": "user",
                                "content": query
                            }
                        ],
                        temperature=0,
                        max_tokens=10,
                        top_p=1,
                        stream=False,
                        stop=None,
                    )

                    # parse response
                    response = completion.choices[0].message.content.strip()
                    selected_option = int(response)
                    print(f"LLM selected to change Option {selected_option}.")

                    if selected_option < 1 or selected_option > len(feature_sums):
                        raise ValueError(f"Selected option {selected_option} is out of range.")

                    feature_sums = [pair for i, pair in enumerate(feature_sums) if i not in conflicts[selected_option]]
                    preferences = [pref for i, pref in enumerate(preferences) if i not in conflicts[selected_option]]
                    requests_made += 1
                    if requests_made >= REQUESTS_PER_MINUTE:
                        print(f"Reached {REQUESTS_PER_MINUTE} requests. Waiting for {WAIT_TIME} seconds...")
                        time.sleep(WAIT_TIME)
                        requests_made = 0

                    break  

                except ValueError as ve:
                    print(f"Invalid response from LLM: '{response}'. Error: {ve}. Skipping this conflict.")
                    break  
                except Exception as e:
                    print(f"Error occurred while querying LLM: {e}. Retrying in 5 seconds...")
                    time.sleep(5)  

        # re-check for conflicts after
        conflicts = find_conflicts(pairs=feature_sums, preferences=preferences)
        iteration += 1

    if conflicts:
        print("\nMaximum iterations reached. Some conflicts could not be resolved.")
    else:
        print("\nAll conflicts resolved successfully.")

    return feature_sums,preferences


def main():
    # read in data
    data_file = sys.argv[1]
    pref_file = sys.argv[2]

    with open(data_file, 'rb') as f:
        segments, feature_vectors = pickle.load(f)
    with open(pref_file, 'rb') as f:
        preferences = pickle.load(f)

    pairs = [(
        np.array(pair1[1:]),  # drop the first element each step bc we're only including displacement not velocity
        np.array(pair2[1:])   
    ) for pair1, pair2 in feature_vectors]

    # use subset of 50
    pairs = pairs[:50]
    preferences = preferences[:50]

    # ADD CONFLICT -----------------
    # pairs.append(pairs[-1])
    # opposite_pref = 1 if preferences[-1] == 0 else 0
    # preferences.append(opposite_pref)

    # pairs.append(pairs[5])
    # opposite_pref = 1 if preferences[5] == 0 else 0
    # preferences.append(opposite_pref)

    # pairs.append(pairs[3])
    # opposite_pref = 1 if preferences[3] == 0 else 0
    # preferences.append(opposite_pref)

    # print("pairs", pairs, "preferences", preferences)

    start_time = time()
    check_and_remove_conflicts(pairs, preferences)
    end_time = time()
    print(f"total time elapsed: {end_time - start_time}")


    # # check points
    poly = find_full_weight_space(pairs, preferences)

if __name__ ==  "__main__":
    main()