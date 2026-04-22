import unittest

from src.agent import Agent, AgentAction

class TestAgentMove(unittest.TestCase):
    def test_agent_move_up(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, sigma=0.9)
        agnt.move(AgentAction.MOVE_UP)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 0)
        self.assertEqual(agnt.perception.agents[0][1], 1)
        self.assertEqual(agnt.perception.agents[1][1], 0)

    def test_agent_move_down(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, sigma=0.9)
        agnt.move(AgentAction.MOVE_DOWN)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 2)
        self.assertEqual(agnt.perception.agents[2][1], 1)
        self.assertEqual(agnt.perception.agents[1][1], 0)

    def test_agent_move_left(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, sigma=0.9)
        agnt.move(AgentAction.MOVE_LEFT)
        self.assertEqual(agnt.x, 0)
        self.assertEqual(agnt.y, 1)
        self.assertEqual(agnt.perception.agents[1][0], 1)
        self.assertEqual(agnt.perception.agents[1][1], 0)

    def test_agent_move_right(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, sigma=0.9)
        agnt.move(AgentAction.MOVE_RIGHT)
        self.assertEqual(agnt.x, 2)
        self.assertEqual(agnt.y, 1)
        self.assertEqual(agnt.perception.agents[1][2], 1)
        self.assertEqual(agnt.perception.agents[1][1], 0)

    def test_agent_move_bad(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10, scan_accuracy=0.05, scan_radius=0, sigma=0.9)
        agnt.move(AgentAction.WAIT)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 1)
        agnt.move(AgentAction.SCAN)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 1)
        self.assertEqual(agnt.perception.agents[1][1], 1)

    def test_illegal_move(self):
        agnt = Agent("x", x=0, y=0, width=10, height=10, scan_accuracy=0.05, scan_radius=0, sigma=0.9)
        agnt.move(AgentAction.MOVE_LEFT)

        self.assertEqual((agnt.x, agnt.y), (0, 0))
        self.assertEqual(agnt.illegal_moves, 1)

    def test_agent_trajectory(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10,
                     scan_accuracy=0.05, scan_radius=0, sigma=0.9)
        self.assertEqual(agnt.move_history, [(1, 1)])

        agnt.move(AgentAction.MOVE_UP)
        agnt.move(AgentAction.MOVE_RIGHT)
        self.assertEqual(agnt.move_history, [(1, 1), (1, 0), (2, 0)])

# class TestAgentScan(unittest.TestCase):