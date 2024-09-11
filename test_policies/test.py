import time

import numpy as np
from EthicalGatheringGame import NormalizeReward, StatTracker
from EthicalGatheringGame.presets import large
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import matplotlib
import os
matplotlib.use('TkAgg')
import gym


eff_rate = 0.4
db = 1000
we = 1.875

large["n_agents"] = 5
large["donation_capacity"] = db
large["efficiency"] = [0.85] * int(5 * eff_rate) + [0.2] * int(5 - eff_rate * 5)
large["we"] = [1, we]
large["color_by_efficiency"] = True
large["objective_order"] = "individual_first"
env = gym.make("MultiAgentEthicalGathering-v1", **large)
# env = NormalizeReward(env)
env = StatTracker(env)


# If root dir is not MAEGG_IPPO, up one level
current_directory = os.getcwd()
directory_name = os.path.basename(current_directory)
while directory_name != "MAEGG_IPPO":
    os.chdir("..")
    current_directory = os.getcwd()
    directory_name = os.path.basename(current_directory)
print(current_directory)
# Loading the agents
agents = IPPO.actors_from_file(f"EGG_DATA/db1000_effrate0.4_we1.675_ECAI_new/db1000_effrate0.4_we1.675_ECAI_new/2500_100000_20")

env.toggleTrack(True)
env.toggleStash(True)
for s in range(50):
    obs, _ = env.reset()
    acc_reward = np.zeros(env.n_agents)
    acc_reward_mo = np.zeros((env.n_agents, 2))
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
        obs, reward, done, info = env.step(actions)
        mo_rewards = np.array([ag.r_vec for ag in env.agents.values()])

        acc_reward += np.array(reward)
        acc_reward_mo += mo_rewards
        env.render(mode="partial_observability", pause=0.4)
    print(f"Episode {s}: {acc_reward} \t Agents (V_0, V_e): ", "\t".join([str(s) for s in acc_reward_mo]))

print(f"\nMean reward per agent: {list(acc_reward / env.max_steps)}")
print(f"\nMean mo reward per agent: {list(acc_reward_mo / env.max_steps)}")
env.plot_results("median")
env.print_results()
