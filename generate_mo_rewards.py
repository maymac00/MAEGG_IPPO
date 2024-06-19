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
from IndependentPPO.utils.misc import str2bool
import matplotlib
import sys

try:
    matplotlib.use('TkAgg')
except ImportError:
    warnings.warn("Could not set backend to TkAgg. This is not an issue if you are running on a server.")

import gym
import os
import logging as log
import pandas as pd


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
    # If time to survival is -1 we set it to inf
    info["sim_data"]["time_to_survival"] = [np.nan if t == -1 else t for t in info["sim_data"]["time_to_survival"]]
    data["time_to_survival"] = info["sim_data"]["time_to_survival"]
    d[global_id] = data


if __name__ == "__main__":
    # Setting up the environment

    MAEGG.log_level = log.INFO

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="ECAI")
    parser.add_argument("--n-sims", type=int, default=50)
    parser.add_argument("--n-cpus", type=int, default=8)
    parser.add_argument("--write-rewards", type=str2bool, default=False)
    parser.add_argument("--write-t2s", type=str2bool, default=False)
    parser.add_argument("--write-fig", type=str2bool, default=False)
    args = parser.parse_args()

    eff_rates = [0.8]
    dbs = [0, 10]
    wes = [0, 10]

    root = os.getcwd()

    if "test_policies" in root:
        root = os.path.dirname(root)
    root = os.path.join(root, args.path)
    os.chdir(root)
    print(f"Root directory: {root}")

    # Create father numpy db for the results
    header = [f"v0_ag{i}" for i in range(5)] + [f"ve_ag{i}" for i in range(5)]
    for we in wes:
        for eff_rate in eff_rates:
            for db in dbs:
                try:
                    large["we"] = [1, we]
                    large["efficiency"] = [0.85] * int(5 * eff_rate) + [0.2] * int(5 - eff_rate * 5)
                    large["donation_capacity"] = db
                    env = gym.make("MultiAgentEthicalGathering-v1", **large)
                    env.toggleTrack(True)
                    env.reset()

                    # Iterate all directories that start with "2500_100000_1"
                    os.chdir(root)
                    os.chdir(f"db{db}_effrate{eff_rate}_we{we}_ECAI_new/db{db}_effrate{eff_rate}_we{we}_ECAI_new")
                    dirs = []
                    for file in os.listdir():
                        if not file.endswith("ckpt"):
                            dirs.append(file)

                    if len(dirs) == 0:
                        print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} does not have finished "
                              f"experiments. Skipping.")
                        continue

                    for file_num, dir in enumerate(dirs):
                        # Enter dir
                        agents = IPPO.actors_from_file(dir)

                        n_sims = args.n_sims

                        env.toggleTrack = True
                        env.toggleStash = True
                        stash = []
                        so_rewards = np.zeros((n_sims, env.n_agents))
                        mo_rewards = np.zeros((n_sims, env.n_agents, 2))
                        time2survive = np.zeros((n_sims, env.n_agents))

                        batch_size = args.n_cpus

                        with Manager() as manager:

                            solved = 0
                            d = manager.dict()
                            while solved < n_sims:
                                d = manager.dict()
                                tasks = [(env, d, agents, global_id) for global_id in
                                         range(solved, min(solved + batch_size, n_sims))]
                                with Pool(batch_size) as p:
                                    p.map(_parallel_rollout, tasks)

                                for i in range(solved, min(solved + batch_size, n_sims)):
                                    stash.append(env.build_history_array(h=d[i]["history"]))
                                    so_rewards[i] = d[i]["so"]
                                    mo_rewards[i] = d[i]["mo"]
                                    time2survive[i] = d[i]["time_to_survival"]
                                    env.reset()
                                solved += batch_size

                        # Plotting the results
                        th.set_num_threads(args.n_cpus)
                        env.setStash(stash)
                        if args.write_rewards:
                            pd.DataFrame(mo_rewards.reshape(-1, 10)).to_csv(f"mo_rewards.csv", header=header, index=False)
                        if args.write_t2s:
                            pd.DataFrame(time2survive.reshape(-1, 5)).to_csv(f"t2s.csv", header=[f"t2s_ag{i}" for i in range(5)], index=False)
                        if args.write_fig:
                            env.plot_results("median", str(os.path.join(os.getcwd(), "median_plot.png")), show=False)

                        print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} finished.")

                except FileNotFoundError as e:
                    print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} not found. Skipping.")
                    continue

                except Exception as e:
                    raise e
