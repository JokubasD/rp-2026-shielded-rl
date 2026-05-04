import unittest
import numpy as np

from src.state import State
from src.agent import Agent, AgentAction

class TestAgentMove(unittest.TestCase):
    def test_agent_move_up(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, decay=0.9, scan_falloff=False)
        agnt.move(AgentAction.MOVE_UP)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 0)
        self.assertEqual(agnt.perception.agents[0][1], 1)
        self.assertEqual(agnt.perception.agents[1][1], 0)

    def test_agent_move_down(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, decay=0.9, scan_falloff=False)
        agnt.move(AgentAction.MOVE_DOWN)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 2)
        self.assertEqual(agnt.perception.agents[2][1], 1)
        self.assertEqual(agnt.perception.agents[1][1], 0)

    def test_agent_move_left(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, decay=0.9, scan_falloff=False)
        agnt.move(AgentAction.MOVE_LEFT)
        self.assertEqual(agnt.x, 0)
        self.assertEqual(agnt.y, 1)
        self.assertEqual(agnt.perception.agents[1][0], 1)
        self.assertEqual(agnt.perception.agents[1][1], 0)

    def test_agent_move_right(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, decay=0.9, scan_falloff=False)
        agnt.move(AgentAction.MOVE_RIGHT)
        self.assertEqual(agnt.x, 2)
        self.assertEqual(agnt.y, 1)
        self.assertEqual(agnt.perception.agents[1][2], 1)
        self.assertEqual(agnt.perception.agents[1][1], 0)

    def test_agent_move_wait(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, decay=0.9, scan_falloff=False)
        agnt.move(AgentAction.WAIT)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 1)
        self.assertEqual(agnt.perception.agents[1][1], 1)

    def test_illegal_move(self):
        agnt = Agent("x", x=0, y=0, width=10, height=10, scan_accuracy=0.05, scan_radius=0, decay=0.9, scan_falloff=False)
        agnt.move(AgentAction.MOVE_LEFT)

        self.assertEqual((agnt.x, agnt.y), (0, 0))
        self.assertEqual(agnt.illegal_moves, 1)

    def test_agent_trajectory(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10,
                     scan_accuracy=0.05, scan_radius=0, decay=0.9, scan_falloff=False)
        self.assertEqual(agnt.move_history, [(1, 1)])

        agnt.move(AgentAction.MOVE_UP)
        agnt.move(AgentAction.MOVE_RIGHT)
        self.assertEqual(agnt.move_history, [(1, 1), (1, 0), (2, 0)])

class TestAgentScan(unittest.TestCase):
    def test_scan_basic(self):
        scan_accuracy = 0.9
        ground_truth = State(3, 3)
        ground_truth.traversability[1, 2] = 1
        agent = Agent("x", x=1, y=1, width=3, height=3, scan_accuracy=scan_accuracy, scan_radius=1, decay=0.1, scan_falloff=False)
        
        self.assertEqual(agent.perception.traversability[1, 2], 0)
        self.assertEqual(agent.perception.confidence[1, 2], 0)
        agent.scan(ground_truth)
        self.assertEqual(agent.perception.traversability[1, 2], 1)
        self.assertEqual(agent.perception.confidence[1, 2], scan_accuracy)
    
    def test_scan_decay(self):
        scan_accuracy = 0.9
        decay = 0.1
        ground_truth = State(10, 10)
        ground_truth.traversability[1, 2] = 1
        agent = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=scan_accuracy, scan_radius=1, decay=decay, scan_falloff=False)
        
        self.assertEqual(agent.perception.traversability[1, 2], 0)
        self.assertEqual(agent.perception.confidence[1, 2], 0)
        agent.scan(ground_truth)
        self.assertEqual(agent.perception.traversability[1, 2], 1)
        self.assertEqual(agent.perception.confidence[1, 2], scan_accuracy)

        agent.x, agent.y = 9, 9
        agent.scan(ground_truth)
        self.assertEqual(agent.perception.traversability[1, 2], 1)
        self.assertEqual(agent.perception.confidence[1, 2], scan_accuracy - decay)
        self.assertEqual(agent.perception.confidence[0, 0], 0) # decay doesn't cause negative confidence

    def test_scan_falloff(self):
        scan_accuracy = 1
        decay = 0
        ground_truth = State(9, 9)
        agent = Agent("x", 4, 4, 9, 9, decay, scan_accuracy, 3, True)

        self.assertEqual(agent.perception.confidence[1, 4], 0)
        self.assertEqual(agent.perception.confidence[2, 4], 0)
        self.assertEqual(agent.perception.confidence[3, 4], 0)
        self.assertEqual(agent.perception.confidence[4, 4], 0)
        agent.scan(ground_truth)
        self.assertAlmostEqual(agent.perception.confidence[1, 4], 0.011, 3) # type: ignore
        self.assertAlmostEqual(agent.perception.confidence[2, 4], 0.135, 3) # type: ignore
        self.assertAlmostEqual(agent.perception.confidence[3, 4], 0.607, 3) # type: ignore
        self.assertAlmostEqual(agent.perception.confidence[4, 4], 1.000, 3) # type: ignore

        # pylance is too stupid to figure out the return type of confidence[y, x], I promise this works

    def test_scan_line_of_sight(self):
        scan_accuracy = 1
        decay = 0
        ground_truth = State(9, 9)
        ground_truth.traversability[3, 4] = 1
        agent = Agent("x", 4, 4, 9, 9, decay, scan_accuracy, 3, False)

        self.assertEqual(agent.perception.confidence[1, 4], 0)
        self.assertEqual(agent.perception.confidence[2, 4], 0)
        self.assertEqual(agent.perception.confidence[3, 4], 0)
        self.assertEqual(agent.perception.confidence[4, 4], 0)
        agent.scan(ground_truth)
        self.assertEqual(agent.perception.confidence[1, 4], 0)
        self.assertEqual(agent.perception.confidence[2, 4], 0)
        self.assertEqual(agent.perception.confidence[3, 4], 1)
        self.assertEqual(agent.perception.confidence[4, 4], 1)
