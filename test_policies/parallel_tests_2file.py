import argparse
import copy
import warnings

from torch.multiprocessing import Manager, Pool
import torch as th
import numpy as np
from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import matplotlib

try:
    matplotlib.use('TkAgg')
except ImportError:
    warnings.warn("Could not set backend to TkAgg. This is not an issue if you are running on a server.")

import gym
import os


def _parallel_rollout(args):
    """
    Function corresponds to the parallel run. It runs a simulation of the environment and stores the results in a shared dictionary.
    We should save: the accumulated rewards as single objective and multi-objective.
    :param args:
    :return:
    """
    th.set_num_threads(1)
    env, d, agents, global_id = args
    obs, _ = env.reset(seed=global_id)
    data = {}
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
        obs, reward, done, info = env.step(actions)
        acc_reward = [acc_reward[i] + reward[i] for i in range(env.n_agents)]
        if all(done):
            break
    data["so"] = acc_reward
    data["mo"] = [env.agents[i].r_vec for i in range(env.n_agents)]
    data["history"] = env.history

    d[global_id] = data


if __name__ == "__main__":
    # Setting up the environment

    eff_rates = [0, 0.2, 0.6, 1]
    dbs = [0, 1, 10, 100]
    wes = [10]

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="EGG_DATA")
    parser.add_argument("--n-sims", type=int, default=50)
    parser.add_argument("--n-cpus", type=int, default=8)
    args = parser.parse_args()

    root = os.getcwd()
    root = os.path.join(root, args.path)
    os.chdir(root)

    env = gym.make("MultiAgentEthicalGathering-v1", **large)
    env.toggleTrack(True)
    env.reset()

    for db in dbs:
        for eff_rate in eff_rates:
            for we in wes:
                try:
                    large["we"] = [1, we]
                    large["efficiency"] = [0.85] * int(5 * eff_rate) + [0.2] * int(5 - eff_rate * 5)
                    large["donation_capacity"] = db
                    env.setStash([])
                    env.setHistory([])
                    env.reset()

                    # Iterate all directories that start with "2500_100000_1"
                    os.chdir(root)
                    os.chdir(f"db{db}_effrate{eff_rate}_we{we}_ECAI/db{db}_effrate{eff_rate}_we{we}_ECAI")
                    dirs = []
                    for file in os.listdir():
                        if file.startswith("2500_100000_1"):
                            # avoid checkpoints
                            if not file.endswith("ckpt"):
                                dirs.append(file)
                    for dir in dirs:
                        agents = IPPO.actors_from_file(dir)

                        n_sims = args.n_sims

                        env.toggleTrack = True
                        env.toggleStash = True
                        stash = []
                        so_rewards = np.zeros((n_sims, env.n_agents))
                        mo_rewards = np.zeros((n_sims, env.n_agents, 2))

                        batch_size = args.n_cpus

                        with Manager() as manager:

                            solved = 0
                            d = manager.dict()
                            while solved < n_sims:
                                d = manager.dict()
                                tasks = [(env, d, agents, global_id) for global_id in
                                         range(solved, solved + batch_size)]
                                with Pool(batch_size) as p:
                                    p.map(_parallel_rollout, tasks)

                                for i in range(solved, solved + batch_size):
                                    stash.append(env.build_history_array(h=d[i]["history"]))
                                    # h = copy.deepcopy(d[i]["history"])
                                    # env.setHistory(h)
                                    so_rewards[i] = d[i]["so"]
                                    mo_rewards[i] = d[i]["mo"]
                                    env.reset()
                                solved += batch_size

                        # Print mean rewards per agent and objective
                        # print("\nMean reward per agent: ", '\t'.join([str(s) for s in so_rewards.mean(axis=0)]))
                        #print(f"\nMean mo reward per agent: ", '\t'.join([str(s) for s in mo_rewards.mean(axis=0)]))

                        # Plotting the results
                        env.toggleTrack = True
                        env.setStash(stash)
                        env.plot_results("median", save_path=dir)

                        # Create txt file with the results
                        fd = open("results.txt", "w")
                        fd.write(f"Mean reward per agent: {so_rewards.mean(axis=0)}\n")
                        fd.write(f"Mean mo reward per agent: {mo_rewards.mean(axis=0)}\n")
                        fd.close()
                        print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} done.")

                except FileNotFoundError as e:
                    print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} not found. Skipping.")
                    continue

                except Exception as e:
                    print(f"Error: {e}")
                    continue
