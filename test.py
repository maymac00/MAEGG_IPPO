from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
from EthicalGatheringGame.wrappers import DummyPeers
import gym

env = gym.make("MultiAgentEthicalGathering-v1", **tiny)

agents = IPPO.agents_from_file("EGG_DATA/tiny_ethical_4/2500_30000_1")
SoftmaxActor.action_selection = bottom_filter
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
env.plot_results("line")
env.print_results()
