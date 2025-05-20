import sys 
import pickle
import numpy as np

def assign_preferences(feature_pairs, weight=None):
    """ 
    assigns preferences based on partial returns with some weight vector:
       0 = first segment
       1 = second segment
       -1 = equally preferable
    """ 
    
    if weight is None:
        number_of_features = len(feature_pairs[0][0])
        weight = np.ones(number_of_features) / number_of_features
        print("weight is", weight)
    
    preferences = []
    for pair in feature_pairs:
        partial_ret_0 = np.dot(pair[0], weight)
        partial_ret_1 = np.dot(pair[1], weight)
        pref =  0 if partial_ret_0 > partial_ret_1 else 1 if partial_ret_1 > partial_ret_0 else -1
        preferences.append(pref)
    return preferences

def confirm_preferences(correct_prefs, trained_prefs, feature_sums):
    mismatched_indices = []
    if len(correct_prefs) != len(trained_prefs):
        raise ValueError("Arrays must be of the same length.")
    for i in range(len(correct_prefs)):
        if correct_prefs[i] != trained_prefs[i]:
            mismatched_indices.append(i)
    print("number of incorrect prefs is ", len(mismatched_indices))
    print("correct values are", np.array(correct_prefs)[mismatched_indices], "and trained prefs are", np.array(trained_prefs)[mismatched_indices])
    print("feature vals of incorrect predictions", np.array(feature_sums)[mismatched_indices])
    return mismatched_indices

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

    # assign preferences (using predef weights to ensure consistent set)
    weights = [0, 0.5, -0.8]
    preferences = assign_preferences(feature_sums, weights)
    print("preferences found")
    print(preferences)
    with open(output_file_name, 'wb') as f:
        pickle.dump((preferences), f)
    
    if len(sys.argv) == 4:
        weights = [0] + sys.argv[3].flatten().tolist()
        print("weights are", weights)
        predicted_prefs = assign_preferences(feature_sums, weights)
        confirm_preferences(preferences, predicted_prefs, feature_sums)
    
if __name__ ==  "__main__":
    main()