import gymnasium as gym
import random
import copy
import sys
from collections import deque
import numpy as np
import pickle

class SegmentPairGenerator:
    def __init__(self):
        self.start_seed = 0
        self.first_time = True
        env = gym.make('highway-v0', render_mode='human')
        env.configure({
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,  # Enable speed control
                "lateral": True  # Enable lane change control
            },
            "offroad_terminal": True,
            "vehicles_count": 3,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted"
            }
        })
        env.reset()
        print("details of action space: type", type(env.action_space), "and full space", env.action_space)
        self.env = env


    def find_terminal_paths_from_start(self, segment_length):
        first_done = False
        trajectories = []

        if self.start_seed > 100000:
            self.first_time = False
            self.start_seed = 1

        #note: add handling for when we don't find something in this time
        for x in range(1000):
            trajectory = []
            state = self.env.reset(seed=self.start_seed)[0]
            for _ in range(segment_length):
                action = self.env.action_space.sample()
                next_state, reward, done, __, info = self.env.step(action)
                trajectory.append((state, self.env.unwrapped.vehicle.crashed, action, reward, next_state, done))
                state = next_state
                if done:
                    break
            if len(trajectory) == segment_length and done:
                if not first_done:
                    first_done = True
                    trajectories.append(trajectory)
                    print("first trajectory found")
                elif not np.array_equal(trajectories[0], trajectory):
                        return trajectories[0], trajectory
        return [],[]

    def get_reward_sum(self, segment):
        # selects the vx, x_displacement, and is_crashed of the controlled car and sums across each step
        reward_sum = np.zeros(2)
        for step in segment:
            reward_features = np.concatenate((step[0][0, [3]], [step[1]]), axis=0)
            reward_sum += reward_features
        # print("reward sum is", reward_sum)
        return reward_sum
        

    def generate_segment_pairs(self, number_of_pairs, segment_length):
        seen_segment_pairs = []
        segment_pair_sums = []
        for i in range(number_of_pairs):
            print("finding pair", i)
            while True:
                self.start_seed += 1
                segment_1, segment_2 = self.find_terminal_paths_from_start(segment_length)
                if self.first_time or (not any(np.array_equal(pair[0], segment_1) and np.array_equal(pair[1], segment_2) for pair in seen_segment_pairs)):
                    seen_segment_pairs.append((segment_1, segment_2))
                    segment_pair_sums.append((self.get_reward_sum(segment_1), self.get_reward_sum(segment_2)))
                    print("pair found")
                    break 
        return seen_segment_pairs, segment_pair_sums

def main():
    if len(sys.argv) != 4:
        print("expected command: python generate_sequences.py <number_of_pairs> <segment_length> <file_name>")
        sys.exit(1)

    number_of_pairs = int(sys.argv[1])
    segment_length = int(sys.argv[2])
    output_file_name = sys.argv[3]

    generator = SegmentPairGenerator()
    pairs, feature_sums = generator.generate_segment_pairs(number_of_pairs, segment_length)
    
    # save to pickle file
    print("saving file")
    with open(output_file_name, 'wb') as f:
        pickle.dump((pairs, feature_sums), f)

    # print("generated pairs:")
    # for i in range(len(pairs)):
    #     print("***segment 1:", pairs[i][0], "with reward sum:", feature_sums[i][0])
    #     print("***segment 2:", pairs[i][1], "with reward sum:", feature_sums[i][1])

if __name__ == "__main__":
    main()