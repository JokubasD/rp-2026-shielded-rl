import unittest

from src.simulator import *

class SimulatorTest(unittest.TestCase):
    def test_add_agent(self):
        sim = Simulator(5, 5)
        bob = Agent("Bob", 0, 0, 5, 5)
        sim.add_agent(bob)

        self.assertEqual(len(sim.agents), 1)
        self.assertEqual(sim.agents[0], bob)
        self.assertEqual(sim.ground_truth.agents.matrix[0][0], 1)

    def test_step(self):
        sim = Simulator(4, 3)
        bob = Agent("Bob", 0, 0, 4, 3)
        sim.add_agent(bob)

        new_state = sim.step()
        # Assumes get_action returns MOVE_RIGHT
        self.assertEqual(new_state.agents.matrix[0][0], 0)
        self.assertEqual(new_state.agents.matrix[0][1], 1)

    def test_run(self):
        sim = Simulator(4, 3)
        bob = Agent("Bob", 0, 0, 4, 3)
        sim.add_agent(bob)

        steps = sim.run(3)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].agents.matrix[0][0], 0)
        self.assertEqual(steps[0].agents.matrix[0][1], 1)
        self.assertEqual(steps[1].agents.matrix[0][0], 0)
        self.assertEqual(steps[1].agents.matrix[0][2], 1)
        self.assertEqual(steps[2].agents.matrix[0][0], 0)
        self.assertEqual(steps[2].agents.matrix[0][3], 1)