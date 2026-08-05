"""AI agents for Singapore Mahjong."""

from mahjong.agents.random_agent import RandomAgent
from mahjong.agents.greedy_agent import GreedyAgent
from mahjong.agents.defensive_agent import DefensiveAgent
from mahjong.agents.hybrid_agent import HybridAgent
from mahjong.agents.learned_agent import LearnedAgent

__all__ = ["RandomAgent", "GreedyAgent", "DefensiveAgent", "HybridAgent",
           "LearnedAgent"]
