#  LLM-based design of aligned reward functions
## Project Goal
Implement feedback loop for autonomous reward design with LLMs, tuning reward function based on preferences. Overall goal is being able to design an aligned reward function for reinforcement learning from a flawed/incomplete natural language description. This aligned reward function will more accurately achieve the human’s interests and match their preferences over outcome distributions, interpreting the general incomplete goal with consideration for common sense concepts / societal standards that may not be explicitly stated. This project will aim to use preferences to formulate a linear reward function that aligns completely, identifying any contradicting preferences and presenting them back to the stakeholder for input on final resolutions.

## Overview of Key Steps & Associated Files
- **1) Identify Key Outcome Variables**
- **2) Find Trajectories**: currently generated randomly in `generate_sequences.py` -> in progress: will replace with active learning
- **3) Assign Preferences to Pairs**: Can assign through set weight vector or LLM
  - Weight Vector: `python assign_preferences.py <input_file_name> <output_file_name> <optional: weights>`
  - LLM: `python llm_assign_preferences.py <input_file_name> <output_file_name> <optional: modelname>`
- **4) Find feasible weights / detect inconsistencies**: currently weightspace is represented as convex polyhedron identified through pycddlib double description implementation and conflicts are identied through brute force combinations of up to 3 preferences in `possible_subspace.py`. Key functions are as follows:
  - `find_feasible_weights(pairs, preferences)`: represents each preference as an inequality constraint and uses scipy linprog implementation to find a feasible set of weights (or identify if there is none)
  - `find_feasible_weight_space(pairs, preferences, test_weight=None)`: represents each preference as an inequality constraint and uses pycddlib double description implementation to convert the inequality matrix to a convex polyhedron representing the feasible space of weights (or identify if there is none). If there is a feasible weight space, checks if test_weight is contained in that space.
  - `check_and_remove_conflicts(pairs, preferences)`: Checks if feasible space exists then brute force identifies conflicts of up to 3 preferences.
- **5) Remove conflicts**: present conflicts to LLM stakeholder for resolution (or possibly consider addition of features to explain the conflicts)
  - to-do
