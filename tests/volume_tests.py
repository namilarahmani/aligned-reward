import unittest
import math
from cdd import Matrix, Polyhedron, RepType
from volume import calculate_polyhedron_volume
from visualize import visualize_polyhedron

'''
basic volume tests
to run: python -m unittest volume_tests.py
note this file should be in main directory (moved into tests folder for rn)
''' 
class TestVolumeCalculation(unittest.TestCase):
    
    def test_unit_cube(self):
        # define h-representation for cube [0,1]^3
        h_rep_matrix = Matrix([
            [0,  1,  0,  0],  # x >= 0
            [0,  0,  1,  0],  # y >= 0
            [0,  0,  0,  1],  # z >= 0
            [1, -1,  0,  0],  # x <= 1
            [1,  0, -1,  0],  # y <= 1
            [1,  0,  0, -1],  # z <= 1
        ], number_type='float')
        
        expected_volume = 1.0
        approx_volume = calculate_polyhedron_volume(h_rep_matrix, n_samples=100000, burn_in=10000)
        
        self.assertTrue(math.isclose(approx_volume, expected_volume, rel_tol=0.05), f"expected {expected_volume}, got {approx_volume}")
    
    def test_rectangular_prism(self):
        # define h-representation for rectangular prism [0,3]^3
        h_rep_matrix = Matrix([
            [0,  1,  0,  0],  # x >= 0
            [0,  0,  1,  0],  # y >= 0
            [0,  0,  0,  1],  # z >= 0
            [3, -1,  0,  0],  # x <= 3
            [3,  0, -1,  0],  # y <= 3
            [3,  0,  0, -1],  # z <= 3
        ], number_type='float')
        
        expected_volume = 27.0
        approx_volume = calculate_polyhedron_volume(h_rep_matrix, n_samples=100000, burn_in=10000)
        
        self.assertTrue(math.isclose(approx_volume, expected_volume, rel_tol=0.05), f"expected {expected_volume}, got {approx_volume}")

    def test_other_rectangular_prism(self):
        # define h-representation for rectangular prism [0,3]^3
        h_rep_matrix = Matrix([
            [0,  1,  0,  0],  # x >= 0
            [0,  0,  1,  0],  # y >= 0
            [0,  0,  0,  1],  # z >= 0
            [3, -1,  0,  0],  # x <= 3
            [4,  0, -1,  0],  # y <= 3
            [5,  0,  0, -1],  # z <= 3
        ], number_type='float')
        
        expected_volume = 60.0
        approx_volume = calculate_polyhedron_volume(h_rep_matrix, n_samples=100000, burn_in=10000)
        
        self.assertTrue(math.isclose(approx_volume, expected_volume, rel_tol=0.05), f"expected {expected_volume}, got {approx_volume}")

    
    def test_unit_square(self):
        # define h-representation for square [0,1]^2
        h_rep_matrix = Matrix([
            [0,  1,  0],  # x >= 0
            [0,  0,  1],  # y >= 0
            [1, -1,  0],  # x <= 1
            [1,  0, -1],  # y <= 1
        ], number_type='float')
        
        expected_volume = 1.0  # area of the square
        approx_volume = calculate_polyhedron_volume(h_rep_matrix, n_samples=100000, burn_in=10000)

        poly = Polyhedron(h_rep_matrix)
        visualize_polyhedron(poly)
        
        self.assertTrue(math.isclose(approx_volume, expected_volume, rel_tol=0.05), f"expected {expected_volume}, got {approx_volume}")
    
  
if __name__ == '__main__':
    unittest.main()
