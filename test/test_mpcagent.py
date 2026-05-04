import unittest

from src.agents.mpc import MpcAgent
from src.constants import AgentAction
from src.state import State

class MpcAgentTest(unittest.TestCase):

    def test_is_feasible_wall(self):
        agent = MpcAgent("mpc", 0, 0, 2, 2, 0.1, 0.9, 1)
        ground_truth = State(2, 2)
        ground_truth.traversability[0][1] = 1
        agent.scan(ground_truth)
        self.assertFalse(agent._is_feasible(agent.perception, AgentAction.MOVE_RIGHT))

    def test_is_feasible_oob(self):
        agent = MpcAgent("mpc", 0, 0, 1, 1, 0.1, 0.9, 1)
        self.assertFalse(agent._is_feasible(agent.perception, AgentAction.MOVE_RIGHT))
        self.assertFalse(agent._is_feasible(agent.perception, AgentAction.MOVE_UP))
        self.assertFalse(agent._is_feasible(agent.perception, AgentAction.MOVE_LEFT))
        self.assertFalse(agent._is_feasible(agent.perception, AgentAction.MOVE_DOWN))

    def test_is_feasible_valid(self):
        agent = MpcAgent("mpc", 0, 0, 2, 2, 0.1, 0.9, 1)
        ground_truth = State(2, 2)
        ground_truth.traversability[0][1] = 1
        agent.scan(ground_truth)
        self.assertTrue(agent._is_feasible(agent.perception, AgentAction.MOVE_DOWN))
        self.assertTrue(agent._is_feasible(agent.perception, AgentAction.WAIT))

    def test_target_cell_move(self):
        agent = MpcAgent("mpc", 1, 1, 3, 3, 0.1, 0.9, 1)
        self.assertEqual(agent._target_cell(AgentAction.MOVE_UP), (1, 0))
        self.assertEqual(agent._target_cell(AgentAction.MOVE_DOWN), (1, 2))
        self.assertEqual(agent._target_cell(AgentAction.MOVE_LEFT), (0, 1))
        self.assertEqual(agent._target_cell(AgentAction.MOVE_RIGHT), (2, 1))

    def test_target_cell_wait(self):
        agent = MpcAgent("mpc", 1, 1, 3, 3, 0.1, 0.9, 1)
        self.assertEqual(agent._target_cell(AgentAction.WAIT), (1, 1))