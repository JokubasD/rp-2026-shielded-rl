import unittest

from src.simulator import Simulator, visualize_grid_gen

class TestGridGenMethods(unittest.TestCase):
    def test_visualize_grid_gen(self):
        simulator = Simulator(50, 50)
        simulator.generate_ground_truth()
        visualize_grid_gen(simulator.ground_truth.traversability.matrix, simulator.ground_truth.agents, simulator.ground_truth.victims)
