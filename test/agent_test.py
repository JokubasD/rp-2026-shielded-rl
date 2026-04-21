import unittest

from src.agent import Agent, AgentAction

class TestAgentMove(unittest.TestCase):
    def test_agent_move_up(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10)
        agnt.move(AgentAction.MOVE_UP)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 0)

    def test_agent_move_down(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10)
        agnt.move(AgentAction.down)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 2)

    def test_agent_move_left(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10)
        agnt.move(AgentAction.MOVE_LEFT)
        self.assertEqual(agnt.x, 0)
        self.assertEqual(agnt.y, 1)

    def test_agent_move_right(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10)
        agnt.move(AgentAction.MOVE_RIGHT)
        self.assertEqual(agnt.x, 2)
        self.assertEqual(agnt.y, 1)

    def test_agent_move_bad(self):
        agnt = Agent("x", x=1, y=1, width=10, height=10)
        agnt.move(AgentAction.WAIT)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 1)
        agnt.move(AgentAction.SCAN)
        self.assertEqual(agnt.x, 1)
        self.assertEqual(agnt.y, 1)