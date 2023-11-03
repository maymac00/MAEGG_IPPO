from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import gym

env = gym.make("MultiAgentEthicalGathering-v1", **small)

agents = IPPO.agents_from_file("jro/EGG_DATA/small/2500_50000_1")
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
        # env.render()
env.plot_results("median")
env.print_results()