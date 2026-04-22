import unittest

from src.simulator import *

class SimulatorTest(unittest.TestCase):
    def test_visualize_grid_gen(self):
        simulator = Simulator(50, 50)
        simulator.generate_ground_truth()
        visualize_grid_gen(simulator.ground_truth.traversability, simulator.ground_truth.agents, simulator.ground_truth.victims)
    
    def test_add_agent(self):
        sim = Simulator(5, 5)
        bob = Agent("Bob", 0, 0, 5, 5, sigma=0.05, scan_accuracy=0.9, scan_radius=0)
        sim.add_agent(bob)

        self.assertEqual(len(sim.agents), 1)
        self.assertEqual(sim.agents[0], bob)
        self.assertEqual(sim.ground_truth.agents[0][0], 1)

    def test_step(self):
        sim = Simulator(4, 3)
        bob = Agent("Bob", 0, 0, 4, 3, sigma=0.05, scan_accuracy=0.9, scan_radius=0)
        sim.add_agent(bob)

        new_state = sim.step()
        # Assumes get_action returns MOVE_RIGHT
        self.assertEqual(new_state.agents[0][0], 0)
        self.assertEqual(new_state.agents[0][1], 1)

    def test_run(self):
        sim = Simulator(4, 3)
        bob = Agent("Bob", 0, 0, 4, 3, sigma=0.05, scan_accuracy=0.9, scan_radius=0)
        sim.add_agent(bob)

        steps = sim.run(3)
        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[0].agents[0][0], 1)
        self.assertEqual(steps[1].agents[0][0], 0)
        self.assertEqual(steps[1].agents[0][1], 1)
        self.assertEqual(steps[2].agents[0][1], 0)
        self.assertEqual(steps[2].agents[0][2], 1)
        self.assertEqual(steps[3].agents[0][2], 0)
        self.assertEqual(steps[3].agents[0][3], 1)