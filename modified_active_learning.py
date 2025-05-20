import random
import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog
import pickle
import time
import numpy as np
from groq import Groq
import sys
from possible_subspace import check_and_remove_conflicts, resolve_conflicts, find_full_weight_space, mod_check_and_remove_conflicts
from active_learning import get_feasible_poly_with_expansion, compute_min_and_max_dot
import volume
# from visualize import visualize_polyhedron

REQUESTS_PER_MINUTE = 30
WAIT_TIME = 60

# functions to assign preferences ---------------------------------
def assign_single_pref(pair: Tuple[np.ndarray, np.ndarray],
    task_description: str,
    feature_names: List[str],
    binary_features: List[bool],
    model_name: str = "llama3-70b-8192",
    use_llm: bool = True) -> int:

    client = Groq(api_key='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')
    prompt_prefix = (
        f"Goal: {task_description}. You are a human evaluator, please act accordingly with societal values. "
        "You will be shown two options described by their feature values."
    )
    prompt_suffix = (
        "Question: Which option better satisfies the goal? "
        "Return 0 if Option 1, 1 if Option 2, or −1 if no preference. "
        "Return only a single number (0, 1, or -1)."
    )

    # build natural‑language feature listings
    def describe(option_vals):
        parts = []
        for name, val, is_bin in zip(feature_names, option_vals, binary_features):
            if is_bin:
                parts.append(f"{name}: {'yes' if val > 0.5 else 'no'}")
            else:
                parts.append(f"{name}: {round(val, 3)}")
        return "; ".join(parts)

    option_1_desc = describe(pair[0])
    option_2_desc = describe(pair[1])

    query = (
        f"{prompt_prefix} {prompt_suffix}\nHere are the two options:\n"
        f"Option 1 → {option_1_desc}\n"
        f"Option 2 → {option_2_desc}"
    )
    # print(query)
        
    while True:
        # user input 
        if not use_llm:
            print(query)
            user_input = input("Your choice (0, 1, or -1): ").strip()
            if user_input in {"0", "1", "-1"}:
                return int(user_input)
            else:
                print("invalid input. please enter 0, 1, or -1.")
        
        else:  # LLM input
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": query}],
                    temperature=0,
                    max_tokens=16,
                    top_p=1,
                )
                pref_str = completion.choices[0].message.content.strip()
                pref = int(pref_str)
                if pref not in {0, 1, -1}:
                    raise ValueError(f"invalid preference value: {pref_str}")
                return pref
            except Exception as e:
                print(f"LLM error: {e}. Retrying in 30 s…")
                time.sleep(30)


def strict_assign__pref(
    pair: Tuple[np.ndarray, np.ndarray],
    task_description: str,
    feature_names: List[str],
    binary_features: List[bool],
    model_name: str = "llama3-70b-8192") -> int:
    # same idea but just doesn't allow for -1

    client = Groq(api_key='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')
    prompt_prefix = (
        f"Goal: {task_description}. You are a human evaluator, please act accordingly with societal values. "
        "You will be shown two options described by their feature values."
    )
    prompt_suffix = (
        "Question: Which option better satisfies the goal? "
        "Return 0 if Option 1 or 1 if Option 2. "
        "Return only a single number (0 or 1)."
    )

    def describe(option_vals):
        parts = []
        for name, val, is_bin in zip(feature_names, option_vals, binary_features):
            if is_bin:
                parts.append(f"{name}: {'yes' if val > 0.5 else 'no'}")
            else:
                parts.append(f"{name}: {round(val, 3)}")
        return "; ".join(parts)

    option_1_desc = describe(pair[0])
    option_2_desc = describe(pair[1])

    query = (
        f"{prompt_prefix} {prompt_suffix}\nHere are the two options:\n"
        f"Option 1 → {option_1_desc}\n"
        f"Option 2 → {option_2_desc}"
    )

    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query}],
                temperature=0,
                max_tokens=16,
                top_p=1,
            )
            pref_str = completion.choices[0].message.content.strip()
            pref = int(pref_str)
            if pref not in {0, 1, -1}:
                raise ValueError(f"Invalid preference value: {pref_str}")
            return pref
        except Exception as e:
            print(f"LLM error: {e}. Retrying in 30 s…")
            time.sleep(30)

def find_signs(base_point, task_description, feature_names, binary_features):
    """find sign of all features"""   
    signs = []
    for i in range(len(base_point)):
        f_same  = base_point.copy()
        f_flip  = base_point.copy()
        f_same[i] = 0
        f_flip[i] = 1          # only last dimension differs (you can pick any Δ>0)

        pref = strict_assign__pref((f_flip, f_same), task_description, feature_names, binary_features)
        sign = +1 if pref == 0 else -1 if pref == 1 else 0
        signs.append(sign)
    print("signs", signs)
    return signs


def prefs_with_active_query_selection(
    task_description: str,
    feature_names: List[str],
    binary_features: List[bool],
    feature_ranges: List[Tuple[float, float]],
    stopping_num: int = 25,
    volume_approx_func=None,
):
    """ active‑learning loop that works for varying task & feature set """

    assert len(feature_names) == len(binary_features) == len(feature_ranges), ("feature_names, binary_features, and feature_ranges must have equal length")
    dim = len(feature_names)

    if volume_approx_func is None:
        volume_approx_func = lambda poly: volume.calculate_polyhedron_volume(poly.get_inequalities())

    preferences = []
    all_pairs = []

    default_scale = 10.0
    found_reward = False 
    reward_func = []
    first = True
    signs = []
    for iteration in range(stopping_num):
        # stop if current feasible region is a single point
        poly = get_feasible_poly_with_expansion(all_pairs, preferences, dim)
        generators = poly.get_generators()
        n_vertices = sum(1 for row in generators if int(row[0]) == 1)
        n_rays = sum(1 for row in generators if int(row[0]) == 0)
        if n_vertices == 1 and n_rays == 0:
            print("feasible region is now a single point; no further query can reduce it")
            break

        found_splitting_pair = False
        range_expansions = 0
        max_expansions = 20
        current_ranges = list(feature_ranges)

        while not found_splitting_pair and range_expansions < max_expansions:
            attempts = 100
            for _ in range(attempts):
                f1_vals, f2_vals = [], []
                for idx, (low, high) in enumerate(current_ranges):
                    if binary_features[idx]:
                        f1_vals.append(float(random.choice([0, 1])))
                        f2_vals.append(float(random.choice([0, 1])))
                    else:
                        f1_vals.append(random.uniform(low, high))
                        f2_vals.append(random.uniform(low, high))

                f1, f2 = np.array(f1_vals, dtype=float), np.array(f2_vals, dtype=float)
                direction = f1 - f2
                min_val, max_val = compute_min_and_max_dot(poly, direction)

                if min_val < 0.0 < max_val:
                    found_splitting_pair = True
                    # pref = assign_single_pref((f1, f2), task_description, feature_names, binary_features, use_llm=True)
                    pref = strict_assign__pref((f1, f2), task_description, feature_names, binary_features)
                    
                    all_pairs.append((f1, f2))
                    preferences.append(pref)

                    print(f"\n[Iteration {iteration}] New Query Found:")
                    print(f"   f1 = {f1}, f2 = {f2}, user/LLM preference = {pref}")

                    if first:
                        first = False
                        signs = find_signs(f1, task_description, feature_names, binary_features)
                    
                    final_poly = get_feasible_poly_with_expansion(all_pairs, preferences, dim, signs)
                    
                    # see if there's a feasible solution and print it out
                    H = final_poly.get_inequalities()
                    A_ub, b_ub = [], []
                    for row in H:
                        b_i = float(row[0])
                        A_i = -np.array([float(x) for x in row[1:]], dtype=float)
                        A_ub.append(A_i)
                        b_ub.append(b_i)
                    # if A_ub:
                    #     A_ub = np.array(A_ub, dtype=float)
                    #     b_ub = np.array(b_ub, dtype=float)
                    #     lp_res = linprog(c=[0] * dim, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
                    #     if lp_res.success:
                    #         # print("  Feasible w:", lp_res.x)
                    #         found_reward = True
                    #         reward = lp_res.x
                    #     # else:
                    #         # print("no feasible w!")
                    A_ub = np.array(A_ub, dtype=float) if A_ub else None
                    b_ub = np.array(b_ub, dtype=float) if b_ub else None
                    dim_w = A_ub.shape[1] if (A_ub is not None and len(A_ub) > 0) else 2
                    all_pairs, preferences = mod_check_and_remove_conflicts(all_pairs, preferences, task_description, feature_names, binary_features)
                    result = linprog(c=[0]*dim_w, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
                    if result.success:
                        feasible_w = result.x
                        print("  A feasible weight vector for the updated region is:", feasible_w)
                        found_reward = True
                        reward_func = result.x

                    else:
                        print("  uh oh! No feasible solution found..")
                    break

            if not found_splitting_pair:
                # expand continuous feature ranges by factor of 10
                range_expansions += 1
                new_ranges = []
                for (low, high), is_bin in zip(current_ranges, binary_features):
                    if is_bin:
                        new_ranges.append((0.0, 1.0))  # binary feature
                    else:
                        center = (low + high) / 2
                        half_range = (high - low) / 2 * default_scale
                        new_low = center - half_range
                        new_high = center + half_range
                        new_ranges.append((new_low, new_high))
                current_ranges = new_ranges
                print(f"No splitting pair found. Increasing distance_scale")

        if not found_splitting_pair:
            print("No additional query can further cleave the current region. Stopping.")
            break
        
    return all_pairs, preferences, reward_func

def main():
    all_pairs, prefs, reward = prefs_with_active_query_selection(
        task_description="Drive as far as possible while avoiding crashes.",
        feature_names      = ["distance_km", "crashed"],
        binary_features    = [False,          True],
        feature_ranges     = [(0, 100000),   (0, 1)],   # 0–100 km, crash? (0/1)
        stopping_num       = 15               # ask up to 15 preference queries
    )
    
    # all_pairs, prefs, reward = prefs_with_active_query_selection(
    #     task_description="Successfully pick up an object quickly and without collisions.",
    #     feature_names      = ["time_sec", "collisions", "successful_pickup"],
    #     binary_features    = [False,       False,        True],
    #     feature_ranges     = [(0, 30),     (0, 5),       (0, 1)],
    #     stopping_num       = 20
    # )

    # all_pairs, prefs, reward = prefs_with_active_query_selection(
    #     task_description="Deliver a package using as little energy and time as possible while ensuring it arrives undamaged.",
    #     feature_names      = ["energy_kJ", "delivery_time_min", "package_undamaged"],
    #     binary_features    = [False,        False,               True],
    #     feature_ranges     = [(0, 2_000),   (0, 60),             (0, 1)],
    #     stopping_num       = 25
    # )

    # all_pairs, prefs, reward = prefs_with_active_query_selection(
    #     task_description="Clear the level efficiently, collecting pellets while avoiding death.",
    #     feature_names      = ["pellets_collected", "deaths", "level_time_sec"],
    #     binary_features    = [False,              False,    False],
    #     feature_ranges     = [(0, 240),           (0, 3),   (10, 300)],
    #     stopping_num       = 20
    # )

    # all_pairs, prefs, reward = prefs_with_active_query_selection(
    #     task_description="Drive safely and efficiently, abiding by societal standards.",
    #     feature_names      = ["distance_traveled_km", "object_collision", "pedestrian_collision"],
    #     binary_features    = [False,              True,    True],
    #     feature_ranges     = [(0, 240),           (0, 3),   (10, 300)],
    #     stopping_num       = 20
    # )

    print("Collected", len(prefs), "preferences.")
    print("reward func is", reward)
    
if __name__ == '__main__':
    main()
