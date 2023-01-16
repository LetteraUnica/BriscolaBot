import unittest
from random import randint

from src.agents.RandomAgent import RandomAgent
from src.utils.AgentPool import AgentPool


def gen_ints(n: int, minimum: int = 1, maximum: int = 100) -> list[int]:
    for i in range(n):
        yield randint(minimum, maximum)


class TestPool(unittest.TestCase):
    def test_add_agents(self):
        for n_agents in gen_ints(100):
            pool = AgentPool(512)
            pool.add_agents([RandomAgent(1) for _ in range(n_agents)], [1.] * n_agents)
            self.assertEqual(len(pool.agents), n_agents)
            self.assertEqual(len(pool.ratings), n_agents)

    def test_clean_pool(self):
        pool = AgentPool(512)
        for n_agents in gen_ints(100):
            pool.add_agents([RandomAgent(1)] * n_agents, [1.] * n_agents)
            pool.clean_pool(5)
            self.assertTrue(len(pool.agents) <= 5)
            self.assertTrue(len(pool.ratings) <= 5)

    def test_sample_agents(self):
        n_agents = 100
        for n_samples in gen_ints(100):
            pool = AgentPool(512)
            pool.add_agents([RandomAgent(1)] * n_agents, [10.] + [0.] * (n_agents - 1))
            samples, indexes = pool.sample_agents(n_samples)
            self.assertEqual(len(samples), n_samples)
            self.assertEqual(indexes.size, n_samples)
            self.assertTrue(all([0 <= index < n_agents for index in indexes]))
            self.assertTrue(0 in indexes)  # Player 0 must always be sampled

    def test_max_size(self):
        pool = AgentPool(512)
        for n_agents in range(1, 1024):
            pool.add_agent(RandomAgent(1), 1.)
            self.assertEqual(len(pool.agents), min(n_agents, 512))
            self.assertEqual(len(pool.ratings), min(n_agents, 512))


if __name__ == '__main__':
    print(gen_ints(10))
    unittest.main()
