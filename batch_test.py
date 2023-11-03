import os

import numpy as np
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import gym

env = gym.make("MultiAgentEthicalGathering-v1", **small)
folder = "jro/EGG_DATA/tiny_convergence_rate/"

n_runs = 1
agg_data = []

# Read all subfolders in folder
for subfolder in os.listdir(folder):
    if "ckpt" in subfolder:
        print("Skipping checkpoint folder: " + folder + subfolder)
        continue
    # Create a list of agents from each subfolder
    print("Testing agents from folder: " + folder + subfolder)
    print("\n")
    agents = IPPO.agents_from_file(folder + subfolder)
    SoftmaxActor.action_selection = bottom_filter
    env.setTrack(True)
    env.setStash(True)
    env.resetStash()
    env.reset()
    # Run the agents on the environment
    for r in range(n_runs):
        obs, _ = env.reset()
        acc_reward = [0] * env.n_agents
        for i in range(env.max_steps):
            actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

            obs, reward, done, info = env.step(actions)
            # env.render()
    # Plot the results
    # env.plot_results("median")
    env.print_results()
    print("\n")
    agg_data.append(env.get_results())

# Process the aggregated data
all_tags = agg_data[0][0]
data = np.zeros((agg_data[0][1].shape[0], len(all_tags), len(agg_data)))
for i, (tags, results) in enumerate(agg_data):
    data[:, :, i] = results

 # Print the results
print("Aggregated Results:")
print(f"Agent | {' | '.join([tag.ljust(14) for tag in all_tags])}")
print("-" * (15 + 15 * len(data[0])))
for i in range(data.shape[0]):
    print(f"{i}     | {' | '.join([str(str(c.mean().round(2)) + chr(32) + chr(177) + chr(32) + str(c.std().round(2))).ljust(14) for c in data[i]])}")
pass