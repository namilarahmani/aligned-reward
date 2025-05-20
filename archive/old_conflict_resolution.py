def resolve_conflicts(conflicts,feature_sums,preferences):
    """
    resolves conflicts by querying the LLM to decide which conflicting preferences to flip.

    Args:
        conflicts (List[Tuple[int, ...]]): list of conflict subsets.
        feature_sums (List[Tuple[Tuple[float, float], Tuple[float, float]]]): current list of feature pairs.
        preferences (List[float]): current list of preferences.

    Returns:
        List[float]: updated pairs and preferences after conflict resolution
    """
    client = Groq(api_key='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')
    requests_made = 0
    WAIT_TIME = 60  
    REQUESTS_PER_MINUTE = 60  
    max_iterations = 10  
    iteration = 0

    while conflicts and iteration < max_iterations:
        print(f"\nConflict Resolution Iteration: {iteration + 1}")
        for conflict_set in conflicts:
            options_text = ""
            for idx in conflict_set:
                car1_distance = round(feature_sums[idx][0][0], 2)
                car2_distance = round(feature_sums[idx][1][0], 2)
                car1_crash = "didn’t crash" if feature_sums[idx][0][1] == 0.0 else "crashed"
                car2_crash = "didn’t crash" if feature_sums[idx][1][1] == 0.0 else "crashed"

                option_number = idx + 1  # Assuming options start at 1
                options_text += f"""
Option {option_number}:
    Preference {idx}:
        Car 1: Moved {car1_distance} kilometers toward its goal and {car1_crash}.
        Car 2: Moved {car2_distance} kilometers toward its goal and {car2_crash}.
"""

            # Construct the full prompt
            prompt = (f"Goal: We're trying to drive safely and efficiently on a highway. We have two cars with information about how far each traveled (in kilometers) and whether they crashed.\n\n"
                      f"Preferences have been selected for various car trajectories. You are assisting in conflict resolution for these preferences. Below are some options of preferences that you can change to resolve the conflict:\n\n"
                      f"{options_text}\n"
                      f"Please select the option number that you would like to change to resolve the conflict while adhering best to the goal of safe highway driving. Return only a number (from the options) as your answer - do not include any words.")

            query = prompt.strip()

            # Query the LLM
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

                    # Parse the response
                    response = completion.choices[0].message.content.strip()
                    selected_option = int(response)

                    print(f"LLM selected to change Option {selected_option}.")

                    # Validate the selected option
                    if selected_option < 1 or selected_option > len(feature_sums):
                        raise ValueError(f"Selected option {selected_option} is out of range.")

                    index_to_flip = selected_option - 1  # Convert to 0-based index
                    original_pref = preferences[index_to_flip]
                    flipped_pref = 1.0 - original_pref if original_pref in [0.0, 1.0] else 0.5  # Handle 0.5 appropriately

                    print(f"Flipping preference at index {index_to_flip} from {original_pref} to {flipped_pref}.")

                    preferences[index_to_flip] = flipped_pref

                    requests_made += 1
                    if requests_made >= REQUESTS_PER_MINUTE:
                        print(f"Reached {REQUESTS_PER_MINUTE} requests. Waiting for {WAIT_TIME} seconds...")
                        time.sleep(WAIT_TIME)
                        requests_made = 0

                    break  # Exit the while loop when successful

                except ValueError as ve:
                    print(f"Invalid response from LLM: '{response}'. Error: {ve}. Skipping this conflict.")
                    break  # Skip this conflict
                except Exception as e:
                    print(f"Error occurred while querying LLM: {e}. Retrying in 5 seconds...")
                    time.sleep(5)  # Wait before retrying

        # Re-check for conflicts after flipping
        conflicts = check_and_remove_conflicts(pairs=feature_sums, preferences=preferences)
        iteration += 1

    if conflicts:
        print("\nMaximum iterations reached. Some conflicts could not be resolved.")
    else:
        print("\nAll conflicts resolved successfully.")

    return preferences