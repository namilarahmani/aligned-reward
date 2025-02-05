from scipy.optimize import linprog
import numpy as np
import sys
import pickle
from time import time
from itertools import combinations
import cdd

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
        print("no feasible space!")
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

        print("the test weight lies within the feasible space.")
        return True
    
    print("polyhedron of feasible weightspace is", poly.get_generators())
    return True

    
def check_and_remove_conflicts(pairs, preferences):
    weights = find_feasible_weights(pairs, preferences)
    weights = weights.x
    result = find_feasible_weight_space(pairs, preferences)
    
    # if result.success:
    if result:
        print("no conflicts yay")
        # check if weights are here
        if weights is not None:
            is_in_polyhedron = find_feasible_weight_space(pairs, preferences, test_weight=weights)
            if is_in_polyhedron:
                print("The feasible weight is within the polyhedron.")
            else:
                print("The feasible weight is NOT within the polyhedron.")
        return None
        
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

    print(f"Conflict can be resolved by removing any of the following subsets: {conflicts}")
    return conflicts

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

    # add a conflict
    pairs.append(pairs[-1])
    opposite_pref = 1 if preferences[-1] == 0 else 0
    preferences.append(opposite_pref)

    pairs.append(pairs[5])
    opposite_pref = 1 if preferences[5] == 0 else 0
    preferences.append(opposite_pref)

    pairs.append(pairs[3])
    opposite_pref = 1 if preferences[3] == 0 else 0
    preferences.append(opposite_pref)

    # print("pairs", pairs, "preferences", preferences)

    start_time = time()
    check_and_remove_conflicts(pairs, preferences)
    end_time = time()
    print(f"total time elapsed: {end_time - start_time}")

if __name__ ==  "__main__":
    main()