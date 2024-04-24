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
    info["sim_data"]["time_to_survival"] = [np.inf if t == -1 else t for t in info["sim_data"]["time_to_survival"]]
    data["n_survivors"] = sum([1 if ag.apples >= env.survival_threshold else 0 for ag in env.agents.values()])
    data["time_to_survival"] = info["sim_data"]["time_to_survival"]

    # Calc gini index
    tot_sum = sum([ag.apples for ag in env.agents.values()])
    mean = tot_sum / env.n_agents
    if tot_sum == 0:
        gini = 0
    else:
        gini = 1 - sum([(ag.apples / tot_sum) ** 2 for ag in env.agents.values()])

    # Calc hoover index
    hoover = sum([abs(ag.apples - mean) for ag in env.agents.values()]) / (2 * tot_sum)
    data["gini"] = gini
    data["hoover"] = hoover
    d[global_id] = data


def compute_we(unethical, values: np.ndarray):
    """
    Compute the ethical weight. Values is a numpy array with the values of the policies with shape (simulations, 5, 2)
    :param unethical:
    :param values:
    :return:
    """

    def get_intersection(v01, ve1, v02, ve2):
        if ve1 == ve2:
            return 0
        we = (v01 - v02) / (ve2 - ve1)
        return we

    def compute_wes_prima(eff, db):
        def aggregation_function(wes, method="median"):
            # cut to interval 1-10 else return -inf
            wes = [w for w in wes if w >= 1 and w <= 10]
            if len(wes) == 0:
                return 0
            if method == "median":
                return np.median(wes)
            if method == "mean":
                return np.mean(wes)
            if method == "max":
                return np.max(wes)
            if method == "min":
                return np.min(wes)

        wes_prima = []
        for ag in range(5):
            # Print intersecctions
            p1 = unethical[eff][ag]
            p2 = ref_policies_values[eff][db][ag]
            inter = get_intersection(p1[0], p1[1], p2[0], p2[1])
            wes_prima.append(inter)
        return aggregation_function(wes_prima, method="median")

    return values


if __name__ == "__main__":
    # Setting up the environment

    MAEGG.log_level = log.INFO
    large["color_by_efficiency"] = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="EGG_DATA")
    parser.add_argument("--n-sims", type=int, default=50)
    parser.add_argument("--n-cpus", type=int, default=8)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--testing", type=str2bool, default=False)

    args = parser.parse_args()

    eff_rates = [0.2, 0.6, 0, 1]
    dbs = [1, 10, 100, 0]
    wes = [0, 10]

    root = os.getcwd()

    if "test_policies" in root:
        root = os.path.dirname(root)
    root = os.path.join(root, args.path)
    os.chdir(root)

    # Create father numpy db for the results
    header = (["db", "eff_rate", "we", "dir", "mean_reward"] + ["mean_reward_agent_" + str(i) for i in range(5)] +
              ["mean_mo_reward_agent_v0_" + str(i) for i in range(5)] +
              ["mean_mo_reward_agent_ve_" + str(i) for i in range(5)] +
              ["time_to_survival_agent_" + str(i) for i in range(5)] +
              ["time_to_survival_agent_sd" + str(i) for i in range(5)] +
              ["time_to_survival_agent_median" + str(i) for i in range(5)] +
              ["rate_of_survival" + str(i) for i in range(5)] +
              ["rate_of_survival_std" + str(i) for i in range(5)] +
              ["gini", "hoover", "n_survivors"] +
              ["we_mean", "we_std", "we_median"] +
              ["did_not_donate", "greedy", "picked_apple", "donated", "took_donation", "hungry"])
    father_db = []
    unethical = {}
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
                    os.chdir(f"db{db}_effrate{eff_rate}_we{we}_ECAI/db{db}_effrate{eff_rate}_we{we}_ECAI")
                    dirs = []
                    for file in os.listdir():
                        if file.startswith("2500_100000_1"):
                            # avoid checkpoints
                            if not file.endswith("ckpt"):
                                dirs.append(file)

                    if len(dirs) == 0:
                        print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} does not have finished "
                              f"experiments. Skipping.")
                        continue
                    best_try = None
                    best_mean = -np.infty
                    best_try_mo_rewards = None
                    for file_num, dir in enumerate(dirs):
                        if not args.overwrite:
                            if os.path.exists(dir + "/results.csv"):
                                # Add the results to the father numpy db
                                try:
                                    child_db = np.loadtxt(dir + "/results.csv", delimiter=",", skiprows=1).reshape(
                                        (1, len(header)))
                                    mo_rewards = np.loadtxt(dir + "/mo_rewards.csv", delimiter=",")
                                    mo_rewards = mo_rewards.reshape((mo_rewards.shape[0], 5, 2))

                                except ValueError as e:
                                    print(
                                        f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we}, file {file_num} results does not match the header. Skipping.")
                                    continue
                                if best_mean < child_db[0, 4]:
                                    best_mean = child_db[0, 4]
                                    best_try = copy.deepcopy(child_db)
                                    best_try_mo_rewards = copy.deepcopy(mo_rewards)
                                continue

                        agents = IPPO.actors_from_file(dir)

                        n_sims = args.n_sims

                        env.toggleTrack = True
                        env.toggleStash = True
                        stash = []
                        so_rewards = np.zeros((n_sims, env.n_agents))
                        mo_rewards = np.zeros((n_sims, env.n_agents, 2))
                        gini = np.zeros((n_sims))
                        hoover = np.zeros((n_sims))
                        time_to_survival = np.zeros((n_sims, env.n_agents))
                        rate_of_survival = np.zeros((n_sims, env.n_agents))
                        n_survivors = np.zeros((n_sims))

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
                                    gini[i] = d[i]["gini"]
                                    hoover[i] = d[i]["hoover"]
                                    time_to_survival[i] = d[i]["time_to_survival"]
                                    rate_of_survival[i] = [1 if t != np.inf else 0 for t in d[i]["time_to_survival"]]
                                    n_survivors[i] = d[i]["n_survivors"]
                                    env.reset()
                                solved += batch_size

                        # Plotting the results
                        th.set_num_threads(args.n_cpus)
                        env.setStash(stash)
                        env.plot_results("median", save_path=dir + "/results.png", show=False)

                        # time to survival ignore the inf values
                        time_to_survival = np.array(time_to_survival)
                        time_to_survival = [time_to_survival[:, i] for i in range(env.n_agents)]
                        mean_time_to_survival = []
                        std_time_to_survival = []
                        median_time_to_survival = []
                        for i in range(len(time_to_survival)):
                            m = time_to_survival[i][time_to_survival[i] != np.inf]

                            if m is None or len(m) == 0:
                                mean_time_to_survival.append(np.nan)
                                std_time_to_survival.append(np.nan)
                                median_time_to_survival.append(np.nan)
                                continue
                            mean_time_to_survival.append(m.mean())
                            median_time_to_survival.append(np.median(m))
                            std_time_to_survival.append((time_to_survival[i][time_to_survival[i] != np.inf]).std())

                        mean_time_to_survival = np.array(mean_time_to_survival).reshape((1, env.n_agents))
                        std_time_to_survival = np.array(std_time_to_survival).reshape((1, env.n_agents))
                        median_time_to_survival = np.array(median_time_to_survival).reshape((1, env.n_agents))

                        if we != 0:
                            wes = compute_we(unethical[(0, eff_rate)], mo_rewards)

                        # Build child numpy db
                        child_db = np.zeros((1, len(header)))
                        child_db[0, 0] = db
                        child_db[0, 1] = eff_rate
                        child_db[0, 2] = we
                        child_db[0, 3] = file_num
                        child_db[0, 4] = so_rewards.mean()
                        child_db[0, 5:10] = so_rewards.mean(axis=0)
                        child_db[0, 10:15] = mo_rewards.mean(axis=0)[:, 0]
                        child_db[0, 15:20] = mo_rewards.mean(axis=0)[:, 1]
                        child_db[0, 20:25] = mean_time_to_survival
                        child_db[0, 25:30] = std_time_to_survival
                        child_db[0, 30:35] = median_time_to_survival

                        child_db[0, 35:40] = rate_of_survival.mean(axis=0)
                        child_db[0, 40:45] = rate_of_survival.std(axis=0)

                        child_db[0, 45] = gini.mean()
                        child_db[0, 46] = hoover.mean()
                        child_db[0, 47] = n_survivors.mean()

                        # Save child numpy db as csv
                        if not args.testing:
                            # Create txt file with the results
                            fd = open(dir + "/event_histogram.txt", "w")
                            fd.write(f"Mean reward per agent: {so_rewards.mean(axis=0)}\n")
                            fd.write(f"Mean mo reward per agent: {mo_rewards.mean(axis=0)}\n")
                            fd.write(f"Mean time to survival per agent: {mean_time_to_survival}\n")
                            fd.write(f"Mean gini: {np.median(gini)}\n")
                            fd.write(f"Mean hoover: {np.median(hoover)}\n")
                            stdout = sys.stdout
                            with fd as sys.stdout:
                                h_header, histogram = env.print_results()
                            sys.stdout = stdout
                            fd.close()
                        else:
                            stdout = sys.stdout
                            with open(os.devnull, 'w') as sys.stdout:
                                h_header, histogram = env.print_results()
                            sys.stdout = stdout

                        h_header = list(h_header)
                        grouped_histogram = histogram.mean(axis=0)
                        for it, tag in enumerate(
                                ["did_not_donate", "greedy", "picked_apple", "donated", "took_donation", "hungry"]):
                            child_db[0, -6 + it] = grouped_histogram[h_header.index(tag)] if tag in h_header else 0
                        child_db[0, 4:] = np.round(child_db[0, 4:], 2)
                        child_db = child_db.astype(np.float16)

                        if not args.testing:
                            np.savetxt(dir + "/results.csv", child_db, header=",".join(header), fmt='%.2f', comments="",
                                       delimiter=",")
                            np.savetxt(dir + "/mo_rewards.csv", mo_rewards.reshape(n_sims, 10), fmt='%.2f',
                                       delimiter=",")

                        if best_mean < so_rewards.mean():
                            best_mean = so_rewards.mean()
                            best_try = copy.deepcopy(child_db)

                        print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we}, file {file_num} done.")

                    # Save best try to father numpy db
                    if best_try is not None:
                        father_db.append(best_try)
                    if we == 0:
                        unethical[(db, eff_rate)] = best_try


                except FileNotFoundError as e:
                    print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} not found. Skipping.")
                    continue

                except Exception as e:
                    raise e

    # Save father numpy db as csv
    s = len(father_db)
    father_db = np.array(father_db).reshape((s, len(header)))

    # Timestamp in format YYYY_MM_DD_HH_MM
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    if not args.testing:
        np.savetxt(f"report_{str(timestamp)}.csv", np.array(father_db), header=",".join(header), fmt='%.2f',
                   comments="",
                   delimiter=",")
