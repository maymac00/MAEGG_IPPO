from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
from EthicalGatheringGame.wrappers import DummyPeers
import gym
import numpy as np

tiny["max_steps"] = 400
env = gym.make("MultiAgentEthicalGathering-v1", **tiny)

agents = IPPO.agents_from_file("EGG_DATA/hall_of_fame/2500_30000_1_2.6")
SoftmaxActor.action_selection = bottom_filter
env.setTrack(True)
env.setStash(True)
env.reset()

mean_acc_reward = np.zeros(env.n_agents)
mean_acc_reward_by_value = np.zeros((env.n_agents,2))

runs = 50

for r in range(runs):
    obs, _ = env.reset()
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        mean_acc_reward += reward
        # env.render()
    for i in range(env.n_agents):
        mean_acc_reward_by_value[i] += env.agents[i].r_vec

mean_acc_reward /= runs
mean_acc_reward_by_value /= runs
print(mean_acc_reward)
print(mean_acc_reward_by_value)

env.plot_results("median")
env.print_results()
