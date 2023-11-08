from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
from EthicalGatheringGame.wrappers import DummyPeers
import gym

env = gym.make("MultiAgentEthicalGathering-v1", **small)

dummy_mask = [1, 0]
env = DummyPeers(env, dummy_mask)

agents = IPPO.agents_from_file("EGG_DATA/small_dummy/2500_50000_1")
SoftmaxActor.action_selection = no_filter
env.setTrack(True)
env.setStash(True)
env.reset()
for r in range(10):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        env.render()
env.plot_results("median")
env.print_results()