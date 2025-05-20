# outline of pipeline
import pickle 
# import generate_sequences
# from assign_preferences import assign_preferences
# from llm_assign_preferences import llm_assign_preferences
from possible_subspace import check_and_remove_conflicts, resolve_conflicts
from llm_assign_preferences import llm_assign_preferences_individual, final_assign_preferences, llm_assign_single_pref, assign_preferences_consistently


def main():
    # STEP 1. define reward features (by hand for now)
    # STEP 2. find trajectories
    # number_of_pairs = 50
    # segment_length = 5
    # pair_file_name = 'test_dataset_pairs'
    # generate_sequences(number_of_pairs, segment_length, pair_file_name)

    # STEP 3. assign preferences to pairs
    print("assigning prefs")
    with open('test_segment_dataset', 'rb') as f:
        _, feature_pairs = pickle.load(f)

    feature_pairs = [(pair1[1:], pair2[1:]) for pair1, pair2 in feature_pairs]
    feature_pairs = feature_pairs[:100]


    # ----- manual option ----
    # preferences = assign_preferences(feature_pairs) 

    # ----- original llm option ----
    # goal = "to drive safely across the highway"
    # feature_names = ["displacement", "velocity", "is crashed (1 = yes)"]
    # preferences = llm_assign_preferences(feature_pairs, goal, feature_names)

    # ----- llama individual query option ----
    preferences = assign_preferences_consistently(feature_pairs)
    # with open('llm_segment_preferences', 'rb') as f:
    #     preferences = pickle.load(f)

    # detect inconsistencies
    # print("detecting inconsistencies")
    # conflicts = check_and_remove_conflicts(feature_pairs, preferences)

    # # resolve conflicts if any
    # print("resolving conflicts")
    # if conflicts:
    #     preferences = resolve_conflicts(conflicts, feature_pairs, preferences)

    # else:
    #     print("No conflicts detected. No resolution needed.")

    # ------- active learning preference loop -----
    feature_pairs, preferences = prefs_with_active_query_selection(feature_pairs)
    # Final conflict check
    final_conflicts = check_and_remove_conflicts(pairs=feature_pairs, preferences=preferences)
    if final_conflicts:
        print("Conflicts remain after resolution attempts.")
    else:
        print("All conflicts have been resolved.")

    # Save the resolved preferences
    # output_file_name = "new_preferences"
    # try:
    #     with open(output_file_name, 'wb') as f:
    #         pickle.dump((feature_pairs, preferences), f)
    #     print(f"Resolved preferences saved to '{output_file_name}'.")
    # except Exception as e:
    #     print(f"Failed to save output file '{output_file_name}': {e}")
    # return final weights

if __name__ == "__main__":
    main()