import unittest
import numpy as np
from possible_subspace import find_feasible_weights, check_and_remove_conflicts
from assign_preferences import assign_preferences

class TestPreferenceConsistency(unittest.TestCase):
    
    # size 1 features (number line)
    def test_size1_consistent_preferences(self):
        pairs = np.array([([1],[2]), ([2],[4]), ([5],[6]), ([6],[7]), ([8],[10])])
        preferences = [1,1,1,1,1]  
        result = find_feasible_weights(pairs, preferences)
        self.assertTrue(result.success)  

    def test_size1_inconsistent_preferences(self):
        pairs = np.array([([1],[2]), ([2],[4]), ([6],[4]), ([6],[7]), ([8],[10])])
        preferences = [1,1,1,1,1]  
        result = find_feasible_weights(pairs, preferences)
        self.assertFalse(result.success)  

        conflicts = check_and_remove_conflicts(pairs, preferences)
        self.assertTrue(conflicts.__contains__((2,)), "preference 2 is a conflict")
    
    # size 2 features 
    def test_consistent_preferences(self):
        pairs = np.array([([6, 0], [7, 0]), ([6, 0], [7, 1]), ([2, 0], [70000, 1])])
        preferences = [1, 0, 1]  
        result = find_feasible_weights(pairs, preferences)
        self.assertTrue(result.success)  
    
    def test_size2_inconsistent_2_preferences(self):
        pairs = np.array([([1,1],[2,2]), ([2,2],[4,3]), ([4,3], [1,1])])
        preferences = [1,1,1]  
        result = find_feasible_weights(pairs, preferences)
        self.assertFalse(result.success)  

        conflicts = check_and_remove_conflicts(pairs, preferences)
        self.assertTrue(conflicts.__contains__((0,)))
        self.assertTrue(conflicts.__contains__((1,)))
        self.assertTrue(conflicts.__contains__((2,)))

    def test_larger_inconsistent_2_preferences(self):
        pairs = np.array([
            ([3, 7], [4, 8]),
            ([1, 5], [2, 6]),
            ([7, 2], [5, 9]),
            ([6, 3], [8, 4]),
            ([2, 8], [9, 1]),
            ([4, 4], [7, 7]),
            ([5, 6], [3, 2]),
            ([9, 1], [6, 5]),
            ([8, 1], [1, 9]),
            ([8, 2], [1, 9])
        ])
        preferences = assign_preferences(pairs, [0.2, 0.8])
        preferences[8] = 0
        result = find_feasible_weights(pairs, preferences)
        conflicts = check_and_remove_conflicts(pairs, preferences)
        self.assertFalse(result.success)  
        self.assertTrue(conflicts.__contains__((8,)))

    def test_cycle(self):
        # Test case where the preferences are inconsistent
        pairs = np.array([([1,1], [2,2]), ([2,2], [3,1]), ([3,1], [1,1])])
        preferences = [0, 0, 0]  # These preferences may conflict
        result = find_feasible_weights(pairs, preferences)
        self.assertFalse(result.success)  # Expecting False since this is inconsistent
        
        # check all the options in the cycle can be removed 
        conflicts = check_and_remove_conflicts(pairs, preferences)
        self.assertIsNotNone(conflicts, "Conflicts should not be None")
        self.assertTrue(conflicts.__contains__((0,)))
        self.assertTrue(conflicts.__contains__((1,)))
        self.assertTrue(conflicts.__contains__((2,)))

    def test_larger_cycle(self):
        # nvm i dont think this is even a cycle that can be resolved by removing individuals though
        pairs = np.array([([5, 0], [0, 5]), ([0,5], [2.5,9.33]), ([2.5,9.33], [5,10]), ([5,10], [10,5])])
        preferences = [0, 0, 0, 0] 
        result = find_feasible_weights(pairs, preferences)
        conflicts = check_and_remove_conflicts(pairs, preferences)
        self.assertFalse(result.success)  
        self.assertIsNotNone(conflicts, "Conflicts should not be None")

    def test_conflict_equality(self):
        pairs = np.array([
            ([7, 2], [5, 6]),
            ([1, 1], [2, 3]),
            ([6, 5], [5, 6]),
            ([40, 25], [50, 20]),
            ([40, 25], [45, 20]),
            ([45, 20], [50, 20])
        ])
        preferences = [1, 1, 1, 1, 0, -1]
        conflicts = check_and_remove_conflicts(pairs, preferences)
        self.assertTrue(conflicts.__contains__((5,)))
        self.assertIsNotNone(conflicts, "Conflicts should not be None")
        self.assertGreaterEqual(len(conflicts), 1, "There should be at least 1 possible solutions to this conflict")

    # feature vector size 3
    def test_size3_direct_conflict(self):
        pairs = np.array([
            ([7, 2, 0], [5, 6, 1]),
            ([1, 1, 0], [2, 3, 1]),
            ([6, 5, 1], [5, 6, 0]),
            ([40, 25, 0], [50, 20, 1]),
            ([45, 30, 1], [55, 25, 0]),
            ([45, 30, 1], [55, 25, 0])
        ])
        preferences = assign_preferences(pairs, [0.25, 0.25, 0.5])
        preferences[-1] = 1 if preferences[-1] == 0 else 0
        conflicts = check_and_remove_conflicts(pairs, preferences)
        self.assertIsNotNone(conflicts, "Conflicts should not be None")
        self.assertGreaterEqual(len(conflicts), 1, "There should be at least 1 possible solutions to this conflict")
    
    def test_size3_random_consistent(self):
        pairs = np.array([
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
        ])
        preferences = assign_preferences(pairs, [0.25, 0.25, 0.5])

        conflicts = check_and_remove_conflicts(pairs, preferences)
        self.assertIsNone(conflicts, "Conflicts should be None")

    def test_size3_inconsistent_weights(self):
        pairs = np.array([
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
        ])
        preferences = assign_preferences(pairs, [0.25, 0.25, 0.5])

        conflicting_pairs = np.array([
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
            (np.random.randint(1, 100, size=3), np.random.randint(1, 100, size=3)),
        ])
        conflict_preferences = assign_preferences(conflicting_pairs, [0.25, -0.25, -0.5])
        pairs = np.concatenate((pairs, conflicting_pairs))
        preferences = np.concatenate((preferences, conflict_preferences))
        conflicts = check_and_remove_conflicts(pairs, preferences)
        self.assertIsNotNone(conflicts, "Conflicts should exist")