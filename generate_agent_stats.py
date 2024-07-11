import argparse
import copy
import warnings

import optuna
from torch.multiprocessing import Manager, Pool
import torch as th
import numpy as np
from EthicalGatheringGame import NormalizeReward, StatTracker, MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
from IndependentPPO.utils.misc import str2bool
import matplotlib
import sys
import gym
import os
import logging as log
import pandas as pd

eff_rates = [0.2, 0.4, 0.6, 0.8]
dbs = [0, 1, 10, 1000]
wes = [0, 2.139, 2.674, 2.656, 2.505, 1.98, 1.275, 7.842, 6.472, 6.76, 8.982, 6.678, 2.556]
wes = [10]
eff_rates = [0.4]
dbs = [1000]

root = os.getcwd()

args = argparse.ArgumentParser()
args.add_argument("--path", type=str, default="EGG_DATA")
args.add_argument("--n-runs", type=int, default=100)
args.add_argument("--n-cpus", type=int, default=8)
args.add_argument("--write-t2s", type=str2bool, default=False)
args.add_argument("--write-stats", type=str2bool, default=False)

args = args.parse_args()

MAEGG.log_level = log.INFO

if "test_policies" in root:
    root = os.path.dirname(root)
root = os.path.join(root, args.path)
os.chdir(root)
print(f"Root directory: {root}")

for we in wes:
    for eff_rate in eff_rates:
        for db in dbs:
            try:
                large["we"] = [1, we]
                large["efficiency"] = [0.85] * int(5 * eff_rate) + [0.2] * int(5 - eff_rate * 5)
                large["donation_capacity"] = db
                large["color_by_efficiency"] = True
                large["objective_order"] = "individual_first"
                env = gym.make("MultiAgentEthicalGathering-v1", **large)
                env = StatTracker(env)
                env.toggleTrack(True)
                env.toggleStash(True)
                env.reset()

                # Iterate all directories that start with "2500_100000_1"
                os.chdir(root)
                os.chdir(f"db{db}_effrate{eff_rate}_we{we}_ECAI_new/db{db}_effrate{eff_rate}_we{we}_ECAI_new")

                db_path = f"sqlite:////home/arnau/PycharmProjects/MAEGG_IPPO/EGG_DATA/db{db}_effrate{eff_rate}_we{we}_ECAI_new/database.db"
                try:
                    study = optuna.load_study(study_name=f"db{db}_effrate{eff_rate}_we{we}_ECAI_new", storage=db_path)
                except KeyError as e:
                    print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} has no DB. Skipping.")
                    continue

                # Get the best trial
                best_trial = study.best_trial
                dir = best_trial.user_attrs["saved_dir"]
                # trim string to get just the last part
                dir = dir.split("/")[-1]

                agents = IPPO.actors_from_file(dir)

                n_runs = args.n_runs

                stash = []
                so_rewards = np.zeros((n_runs, env.n_agents))
                mo_rewards = np.zeros((n_runs, env.n_agents, 2))
                time2survive = np.zeros((n_runs, env.n_agents))

                for sim in range(args.n_runs):

                    acc_reward = np.zeros(env.n_agents)
                    acc_reward_mo = np.zeros((env.n_agents, 2))

                    obs, _ = env.reset()
                    for i in range(env.max_steps):
                        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
                        obs, reward, done, info = env.step(actions)

                        acc_reward += np.array(reward)
                        acc_reward_mo += np.array([ag.r_vec for ag in env.agents.values()])

                        if all(done):
                            break

                    so_rewards[sim] = acc_reward
                    mo_rewards[sim] = acc_reward_mo
                    info["sim_data"]["time_to_survival"] = [np.nan if t == -1 else t for t in
                                                            info["sim_data"]["time_to_survival"]]
                    time2survive[sim] = info["sim_data"]["time_to_survival"]

                if args.write_t2s:
                    pd.DataFrame(time2survive.reshape(-1, 5)).to_csv(f"t2s.csv",
                                                                     header=[f"t2s_ag{i}" for i in range(5)],
                                                                     index=False)
                if args.write_stats:
                    results = env.print_results()
                    pd.DataFrame(results["stats_table"]).to_csv(f"stats.csv", index=False, header=results["stats_header"])
                # Log that the runs were a success
                print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} finished.")




            except FileNotFoundError as e:
                print(f"Experiment with params db:{db}, eff_rate:{eff_rate}, we:{we} not found. Skipping.")
                continue

            except Exception as e:
                raise e
