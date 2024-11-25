import copy
import sys

import optuna
import pandas as pd
from torch.multiprocessing import Manager, Pool
import torch as th
import numpy as np
from EthicalGatheringGame import MAEGG, NormalizeReward, StatTracker
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import matplotlib

import gymnasium as gym
import os
import argparse


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
    acc_reward = np.zeros(env.unwrapped.n_agents)
    acc_reward_mo = np.zeros((env.unwrapped.n_agents, 2))
    reward_history_mo = np.zeros((env.unwrapped.max_steps, env.unwrapped.n_agents, 2))

    for i in range(env.unwrapped.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
        obs, reward, terminated, truncated, info = env.step(actions)
        mo_rewards = np.array([ag.r_vec for ag in env.unwrapped.agents.values()])
        reward_history_mo[i] = mo_rewards

        acc_reward += np.array(reward)
        acc_reward_mo += mo_rewards

        if terminated or truncated:
            break

    disc_mo = np.zeros((env.unwrapped.max_steps, env.unwrapped.n_agents, 2))
    gamma = 0.8
    for agent in range(env.unwrapped.n_agents):
        for reward_type in range(2):
            # Extract the rewards for the current episode, agent, and reward type
            rewards = reward_history_mo[:, agent, reward_type]  # Shape (500,)
            for t in range(len(rewards)):
                # Apply the discount to the rewards from time step t onward
                disc_mo[t, agent, reward_type] = np.sum(
                    [gamma ** (i - t) * rewards[i] for i in range(t, len(rewards))]
                )

    data["so"] = acc_reward
    data["mo"] = acc_reward_mo
    data["disc_mo"] = disc_mo.sum(axis=0)
    data["history"] = env.unwrapped.history

    info["sim_data"]["time_to_survival"] = [np.inf if t == -1 else t for t in info["sim_data"]["time_to_survival"]]
    data["n_survivors"] = sum([1 if ag.apples >= env.unwrapped.survival_threshold else 0 for ag in env.unwrapped.agents.values()])
    data["time_to_survival"] = info["sim_data"]["time_to_survival"]

    # Print rewards
    print(f"Epsiode {global_id}: {data['so']} \t Agents (V_0, V_e): ", "\t".join([str(s) for s in data["mo"]]))

    d[global_id] = data


if __name__ == "__main__":
    # Setting up the environment

    args = argparse.ArgumentParser()
    args.add_argument("--tag", type=str)
    args.add_argument("--folder", type=str, default="beegfs/EGG_DATA")
    args.add_argument("--n-sims", type=int, default=500)
    args.add_argument("--user", type=str, default="amm")
    args.add_argument("--mode", type=str, default="parallel")
    args.add_argument("--specific-path", type=str, default="")
    args.add_argument("--specific-dir", type=str, default="")

    # Environment parameters
    args.add_argument("--we", type=float, default=10.0)
    args.add_argument("--effrate", type=float, default=0.6)
    args.add_argument("--db", type=float, default=10)
    args.add_argument("--sthld", type=float, default=30)
    args.add_argument("--n-agents", type=int, default=5)
    args = args.parse_args()
    folder = args.folder

    try:
        db = float(args.tag.split("db")[1].split("_")[0])
    except:
        db = args.db
    try:
        eff_rate = float(args.tag.split("effrate")[1].split("_")[0])
    except:
        eff_rate = args.effrate
    try:
        we = float(args.tag.split("we")[1].split("_")[0])
    except:
        we = args.we

    large["we"] = [1, we]
    large["efficiency"] = [0.85] * int(args.n_agents * eff_rate) + [0.15] * int(args.n_agents - eff_rate * args.n_agents)
    large["donation_capacity"] = db
    large["color_by_efficiency"] = True
    large["objective_order"] = "individual_first"
    large["survival_threshold"] = args.sthld
    large["n_agents"] = args.n_agents
    env = gym.make("MultiAgentEthicalGathering-v1", **large)
    if args.mode != "parallel":
        env = StatTracker(env)
    # env = NormalizeReward(env)
    env.unwrapped.toggleTrack(True)
    env.reset()

    # If root dir is not MAEGG_IPPO, up one level
    current_directory = os.getcwd()
    directory_name = os.path.basename(current_directory)
    if args.user == "mrs":
        while directory_name != "arnau":
            os.chdir("..")
            current_directory = os.getcwd()
            directory_name = os.path.basename(current_directory)
    else:
        while directory_name != "MAEGG_IPPO":
            os.chdir("..")
            current_directory = os.getcwd()
            directory_name = os.path.basename(current_directory)
    print(current_directory)

    if args.specific_path == "" and args.specific_dir == "":
        if args.user == "mrs":
            db_path = f"sqlite:////home/csic/mus/{args.user}/arnau/{folder}/{args.tag}/database.db"
        else:
            db_path = f"sqlite:////home/csic/mus/{args.user}/MAEGG_IPPO/{folder}/{args.tag}/database.db"
        try:
            study = optuna.load_study(study_name=f"{args.tag}", storage=db_path)
        except KeyError as e:
            raise ValueError(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} has no DB. Skipping.")

        # Get the best trial
        best_trial = study.best_trial
        dir = best_trial.user_attrs["saved_dir"]
        # trim string to get just the last part
        dir = dir.split("/")[-1]

        print(f"Best trial: {dir}")
        agents = IPPO.actors_from_file(f"{folder}/{args.tag}/{args.tag}/{dir}")
    elif args.specific_dir != "":
        dir = args.specific_dir
        print(f"Specific trial: {dir}")
        agents = IPPO.actors_from_file(f"{folder}/{args.tag}/{args.tag}/{dir}")
    else:
        path = args.specific_path
        print(f"Specific trial: {path}")
        agents = IPPO.actors_from_file(path)

    # Running the simulation. Parallelized on batches of 5 simulations.
    n_sims = args.n_sims

    env.unwrapped.toggleTrack(True)
    env.unwrapped.toggleStash(True)
    stash = []
    final_so_rewards = np.zeros((n_sims, env.unwrapped.n_agents))
    final_mo_rewards = np.zeros((n_sims, env.unwrapped.n_agents, 2))
    time2survive = np.zeros((n_sims, env.unwrapped.n_agents))
    survival = np.zeros((n_sims, env.unwrapped.n_agents))
    discounted_returns = np.zeros((n_sims, env.unwrapped.n_agents, 2))

    batch_size = min(5, n_sims)
    if args.mode == "parallel":
        with Manager() as manager:

            solved = 0
            d = manager.dict()
            while solved < n_sims:
                d = manager.dict()
                tasks = [(env, d, agents, global_id) for global_id in range(solved, solved + batch_size)]
                with Pool(batch_size) as p:
                    p.map(_parallel_rollout, tasks)

                for i in range(solved, solved + batch_size):
                    stash.append(env.unwrapped.build_history_array(h=d[i]["history"]))
                    # h = copy.deepcopy(d[i]["history"])
                    # env.unwrapped.setHistory(h)
                    final_so_rewards[i] = d[i]["so"]
                    final_mo_rewards[i] = d[i]["mo"]
                    discounted_returns[i] = d[i]["disc_mo"]
                    time2survive[i] = d[i]["time_to_survival"]
                    survival[i] = [1 if t != np.inf else 0 for t in d[i]["time_to_survival"]]
                    env.reset()
                solved += batch_size

            env.unwrapped.setStash(stash)
            # redirect stdout to file
            sys.stdout = open(f"{folder}/{args.tag}/{args.tag}/{dir}_results.txt", "w")
            print(f"Results for {args.tag}/{dir}")

            # Print mean rewards per agent and objective
            print("\nMean reward: ", final_so_rewards.mean())
            print("\nMean reward per agent: ", '\t'.join([str(s) for s in final_so_rewards.mean(axis=0)]))
            print("\nMean mo reward per agent: np.array([",
                  ','.join([f'[{round(s[0], 2)}, {round(s[1], 2)}]' for s in final_mo_rewards.mean(axis=0)]), "])")
            print("\nSTD mo reward per agent: np.array([",
                  ','.join([f'[{round(s[0], 2)}, {round(s[1], 2)}]' for s in final_mo_rewards.std(axis=0)]), "])")
            print("Expected discounted mo returns: np.array([", ','.join(
                [f'[{round(s[0], 2)}, {round(s[1], 2)}]' for s in discounted_returns.mean(axis=0)]), "])")


            print("Survival rate: np.array([", ",".join([str(s) for s in survival.mean(axis=0)]), "])")
            print("Std Survival rate: np.array([", ",".join([str(s) for s in survival.std(axis=0)]), "])")
            print("Median Survival rate: np.array([", ",".join([str(s) for s in np.median(survival, axis=0)]), "])")
            print("IQR Survival rate: np.array([", ",".join(
                [str(s) for s in (np.percentile(survival, 75, axis=0) - np.percentile(survival, 25, axis=0))]), "])")

            print("\nMean Time to survive: np.array([", ",".join([f'{round(s, 2)}' for s in time2survive.mean(axis=0)]), "])")
            print("Std Time to survive: np.array([", ",".join([f'{round(s, 2)}' for s in time2survive.std(axis=0)]), "])")
            print("Median Time to survive: np.array([", ",".join([f'{round(np.median(s), 2)}' for s in time2survive.T]), "])")
            print("IQR Time to survive: np.array([",
            ",".join([f'{round(np.percentile(s, 75) - np.percentile(s, 25), 2)}' for s in time2survive.T]), "])")

    else:
        for sim in range(n_sims):
            obs, _ = env.reset()
            acc_reward = np.zeros(env.unwrapped.n_agents)
            acc_reward_mo = np.zeros((env.unwrapped.n_agents, 2))
            reward_history_mo = np.zeros((env.unwrapped.max_steps, env.unwrapped.n_agents, 2))

            for i in range(env.unwrapped.max_steps):
                actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
                obs, reward, terminated, truncated, info = env.step(actions)
                mo_rewards = np.array([ag.r_vec for ag in env.unwrapped.agents.values()])
                reward_history_mo[i] = mo_rewards

                acc_reward += np.array(reward)
                acc_reward_mo += mo_rewards

                if terminated or truncated:
                    break
            disc_mo = np.zeros((env.unwrapped.max_steps, env.unwrapped.n_agents, 2))
            gamma = 0.8
            for agent in range(env.unwrapped.n_agents):
                for reward_type in range(2):
                    # Extract the rewards for the current episode, agent, and reward type
                    rewards = reward_history_mo[:, agent, reward_type]  # Shape (500,)
                    for t in range(len(rewards)):
                        # Apply the discount to the rewards from time step t onward
                        disc_mo[t, agent, reward_type] = np.sum(
                            [gamma ** (i - t) * rewards[i] for i in range(t, len(rewards))]
                        )
            discounted_returns[sim] = disc_mo.sum(axis=0)

            info["sim_data"]["time_to_survival"] = [np.inf if t == -1 else t for t in
                                                    info["sim_data"]["time_to_survival"]]
            time2survive[sim] = info["sim_data"]["time_to_survival"]
            survival[sim] = [1 if t != np.inf else 0 for t in info["sim_data"]["time_to_survival"]]
            final_so_rewards[sim] = acc_reward
            final_mo_rewards[sim] = acc_reward_mo

        # redirect stdout to file
        sys.stdout = open(f"{folder}/{args.tag}/{args.tag}/{dir}_results.txt", "w")
        print(f"Results for {args.tag}/{dir}")
        print("\nMean reward: ", final_so_rewards.mean())
        print("\nMean reward per agent: ", '\t'.join([str(s) for s in final_so_rewards.mean(axis=0)]))
        print("\nMean mo reward per agent: np.array([",
              ','.join([f'[{round(s[0], 2)}, {round(s[1], 2)}]' for s in final_mo_rewards.mean(axis=0)]), "])")
        print("\nSTD mo reward per agent: np.array([",
              ','.join([f'[{round(s[0], 2)}, {round(s[1], 2)}]' for s in final_mo_rewards.std(axis=0)]), "])")

        print("Expected discounted mo returns: np.array([", ','.join(
            [f'[{round(s[0], 2)}, {round(s[1], 2)}]' for s in discounted_returns.mean(axis=0)]), "])")

        print("Survival rate: np.array([", ",".join([str(s) for s in survival.mean(axis=0)]), "])")
        print("Std Survival rate: np.array([", ",".join([str(s) for s in survival.std(axis=0)]), "])")
        print("Median Survival rate: np.array([", ",".join([str(s) for s in np.median(survival, axis=0)]), "])")
        print("IQR Survival rate: np.array([", ",".join([str(s) for s in (np.percentile(survival, 75, axis=0) - np.percentile(survival, 25, axis=0))]), "])")

        print("\nMean Time to survive: np.array([", ",".join([f'{round(s, 2)}' for s in time2survive.mean(axis=0)]),
              "])")
        print("Std Time to survive: np.array([", ",".join([f'{round(s, 2)}' for s in time2survive.std(axis=0)]), "])")

        print("Median Time to survive: np.array([", ",".join([f'{round(np.median(s), 2)}' for s in time2survive.T]), "])")
        print("IQR Time to survive: np.array([", ",".join([f'{round(np.percentile(s, 75) - np.percentile(s, 25), 2)}' for s in time2survive.T]), "])")

    # Plotting the results
    try:
        matplotlib.use("TkAgg")
        env.unwrapped.plot_results("median", show=False,
                                   save_path=f"{folder}/{args.tag}/{args.tag}/{dir}_median_plot.png")
    except ImportError:
        env.unwrapped.plot_results("median", show=False,
                                   save_path=f"{folder}/{args.tag}/{args.tag}/{dir}_median_plot.png")
    env.unwrapped.print_results()

    # Save t2s to csv
    pd.DataFrame(time2survive.reshape(-1, args.n_agents)).to_csv(f"{folder}/{args.tag}/{args.tag}/{dir}_t2s.csv",
                                                     header=[f"t2s_ag{i}" for i in range(args.n_agents)],
                                                     index=False)

    # return stdout to console
    sys.stdout = sys.__stdout__
    print(f"Results saved in {folder}/{args.tag}/{args.tag}/{dir}_median_plot.png")
