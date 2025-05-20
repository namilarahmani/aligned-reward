import cdd
from groq import Groq
import random
import pickle
import time
from llm_assign_preferences import llm_assign_single_pref, user_assign_single_pref
import numpy as np
import sys
from scipy.optimize import linprog
from possible_subspace import check_and_remove_conflicts, resolve_conflicts, find_full_weight_space
import volume
# from visualize import visualize_polyhedron

''' hard coded for highway driving environment - only using crashed and distance traveled '''
REQUESTS_PER_MINUTE = 30
WAIT_TIME = 60

def get_feasible_poly_with_expansion(all_pairs, preferences, dim=2, signs=None):
    """
    creates feasible polyhedron for the weight space by:
      1) Starting with a small bounding box [-1, 1]^dim
      2) Building a polyhedron with those bounds + existing pairwise constraints
      3) Checking if it's unbounded. If unbounded, multiply bounds by 10 and repeat.
    returns a feasible bounded cdd polyhedron or empty.
    """
    scale = 1.0
    while True:
        weight_bounds = []
        if signs is None:
            # bounding box in each dimension as [-scale, scale]
            weight_bounds = [(-scale, scale) for _ in range(dim)]
        else:
            for j, s in enumerate(signs):
                if j == dim - 1:                
                    weight_bounds.append((s, s))
                else:                           
                    if s > 0:
                        weight_bounds.append((0, scale))
                    else:
                        weight_bounds.append((-scale, 0))

        # print(weight_bounds)
        # print(dim)
        poly = find_full_weight_space(all_pairs, preferences, basic_bounds=weight_bounds, num_features=dim)

        # check if poly is unbounded 
        generators = poly.get_generators()
        n_rays = sum(1 for row in generators if int(row[0]) == 0)
        if n_rays > 0:
            # unbounded, so expand further
            scale *= 10.0
        else:
            # bounded (or possibly empty) -> return
            return poly

def compute_min_and_max_dot(poly, direction):
    """
    returns (min_val, max_val) of w^T direction subject to w in 'poly' by
    solving two linear programs using the polyhedron's H-representation (inequalities)
    """
    H = poly.get_inequalities()
    A_ub = []
    b_ub = []
    for row in H:
        b_i = float(row[0])
        A_i = -np.array([float(x) for x in row[1:]], dtype=float)
        A_ub.append(A_i)
        b_ub.append(b_i)

    A_ub = np.array(A_ub, dtype=float)
    b_ub = np.array(b_ub, dtype=float)
    c = np.array(direction, dtype=float)

    # solve for max (c^T w) => min (-(c^T) w).
    res_max = linprog(c=-c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
    max_val = float(c.dot(res_max.x)) if res_max.success else float('inf')     

    # solve for min (c^T w)
    res_min = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
    min_val = float(c.dot(res_min.x)) if res_min.success else float('-inf')

    return min_val, max_val

def strict_assign__pref(pair):
    # llm assign preference for a pair without an option of indifference
    client = Groq(api_key='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')  
    prompt_prefix = "Goal: We're trying to drive safely and efficiently on a highway. We have two cars with information about how far each traveled (in kilometers) and whether they crashed. You are acting as a human, so please make decisions according to what would be expected in society."
    prompt_suffix = ("Question: Which car performed better in terms of safe and efficient driving? "
                     "If Car 1 did better, return 0. If Car 2 did better, return 1. "
                     "Important: Return only a number (from the options 0 or 1) as your answer - do not include any words. ")
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
                messages=[{
                    "role": "user",
                    "content": query
                }],
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
                print(f"Error occurred: {e}. retrying in 1 min...")
                time.sleep(60)  # wait for query limit

def find_signs(base_point):
    """find sign of all features"""   
    signs = []
    for i in range(len(base_point)):
        f_same  = base_point.copy()
        f_flip  = base_point.copy()

        # change the feature of interest
        f_same[i] = 0
        f_flip[i] = 1 

        pref = strict_assign__pref((f_flip, f_same))
        sign = +1 if pref == 0 else -1 if pref == 1 else 0
        signs.append(sign)
    print("signs", signs)
    return signs

def prefs_with_active_query_selection(
        stopping_num=25, 
        basic_bounds=None, # not used here
        n_samples=5000,
        volume_approx_func=None
    ):
    """
    iteratively select queries that reduce the feasible weight space and collect LLM preference
    
    params
    ----------
    stopping_num : int
        max number of queries to collect
    basic_bounds : list of (float, float)
        default bounding box for the feasible region
    n_samples : int
        number of samples to use for sampling-based volume approximation
    volume_approx_func : callable function or None
    """

    if volume_approx_func is None:
        volume_approx_func = lambda poly: volume.calculate_polyhedron_volume(poly.get_inequalities())

    preferences = []
    all_pairs = []

    distance_scale = 10.0
    found_reward = False
    reward_func = []
    first = True
    signs = []
    for iteration in range(stopping_num):
        # construct the current feasible weight-space polyhedron
        poly = get_feasible_poly_with_expansion(all_pairs, preferences, dim=2)
        # volume_before = volume_approx_func(poly)

        # check if polyhedron is effectively a single point (or empty)
        generators = poly.get_generators()
        tvals = [int(row[0]) for row in generators]  # t=1 => vertex, t=0 => ray
        n_vertices = sum(1 for t in tvals if t == 1)
        n_rays = sum(1 for t in tvals if t == 0)

        if (n_vertices == 1 and n_rays == 0):
            print("feasible region is now a single point; no further query can reduce it")
            break

        # find a pair that reduces the feasible region
        found_splitting_pair = False
        expansions = 0
        max_expansions = 20

        while (not found_splitting_pair) and (expansions < max_expansions):
            attempts = 100
            for _ in range(attempts):
                # randomly sample f1, f2 from [0, distance_scale], crash in {0,1}
                dist1 = random.uniform(0, distance_scale)
                crash1 = 1.0 if random.random() < 0.2 else 0.0

                dist2 = random.uniform(0, distance_scale)
                crash2 = 1.0 if random.random() < 0.2 else 0.0

                f1 = np.array([dist1, crash1], dtype=float)
                f2 = np.array([dist2, crash2], dtype=float)

                direction = f1 - f2
                min_val, max_val = compute_min_and_max_dot(poly, direction)

                # if direction crosses 0 => there's a feasible 'w' that prefers f1 over f2 
                # and another that prefers f2 over f1 => so this query "cleaves" the region
                if min_val < 0.0 and max_val > 0.0:
                    found_splitting_pair = True
                    pair_for_llm = (f1, f2)

                    # pref = user_assign_single_pref(pair_for_llm) 
                    # pref = llm_assign_single_pref(pair_for_llm)
                    pref = strict_assign__pref(pair_for_llm)

                    all_pairs.append(pair_for_llm)
                    preferences.append(pref)

                    print(f"\n[Iteration {iteration}] New Query Found:")
                    print(f"   f1 = {f1}, f2 = {f2}, user/LLM preference = {pref}")

                    if first:
                        first = False
                        signs = find_signs(f1)
    
                    final_poly = get_feasible_poly_with_expansion(all_pairs, preferences, dim=2, signs=signs)

                    # see if there's a feasible solution and print it out
                    H = final_poly.get_inequalities()
                    A_ub, b_ub = [], []
                    for row in H:
                        b_i = float(row[0])
                        A_i = -np.array([float(x) for x in row[1:]], dtype=float)
                        A_ub.append(A_i)
                        b_ub.append(b_i)
                    A_ub = np.array(A_ub, dtype=float) if A_ub else None
                    b_ub = np.array(b_ub, dtype=float) if b_ub else None
                    dim_w = A_ub.shape[1] if (A_ub is not None and len(A_ub) > 0) else 2
                    all_pairs, preferences = check_and_remove_conflicts(all_pairs, preferences)
                    result = linprog(c=[0]*dim_w, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
                    if result.success:
                        feasible_w = result.x
                        print("  A feasible weight vector for the updated region is:", feasible_w)
                        found_reward = True
                        reward_func = result.x

                    else:
                        print("  uh oh! No feasible solution found..")

                    # volume_after = volume_approx_func(final_poly)
                    # # print fraction removed
                    # if volume_before > 1e-12:
                    #     frac_removed_volume = 1.0 - (volume_after / volume_before)
                    # else:
                    #     frac_removed_volume = 1.0  # if region was ~0, we consider it all removed

                    # print(f"  volume-based check: {frac_removed_volume*100:.2f}% of feasible region removed.")
                    # print("-"*70)

                    break

            if not found_splitting_pair:
                # if we still haven't found a splitting pair, expand the distance range
                expansions += 1
                distance_scale *= 10
                print(f"No splitting pair found. Increasing distance_scale to {distance_scale} ")

        # if we haven't found a splitting pair after expansions, stop
        if not found_splitting_pair:
            print("No additional query can further cleave the current region. Stopping.")
            break

        # TODO: should check for a stopping criteria (volume reduction, certain volume, etc.)

    return all_pairs, preferences, reward_func

def main():
    pairs_asked, user_prefs, reward = prefs_with_active_query_selection(25)  
    
    print("All asked preference pairs:")
    for i, (pair, pref) in enumerate(zip(pairs_asked, user_prefs)):
        print(f" #{i+1}: {pair[0]} vs {pair[1]} => pref = {pref}")
    print("reward func is", reward)

if __name__ == '__main__':
    main()