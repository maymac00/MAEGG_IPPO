import copy

from torch.multiprocessing import Manager, Pool
import torch as th
import numpy as np
from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor, ACTIONS
from IndependentPPO.ActionSelection import *
import matplotlib

matplotlib.use('TkAgg')
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
    env, d, agents, global_id, gamma = args

    obs, _ = env.reset(seed=global_id)
    data = {}
    acc_reward = np.zeros(env.n_agents)
    acc_reward_mo = np.zeros((env.n_agents, 2))
    reward_history_mo = np.zeros((env.max_steps, env.n_agents, 2))
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
        obs, reward, done, info = env.step(actions)
        mo_rewards = np.array([ag.r_vec for ag in env.agents.values()])
        reward_history_mo[i] = mo_rewards

        acc_reward += np.array(reward)
        acc_reward_mo += mo_rewards
        if all(done):
            break
    disc_mo = np.zeros((env.max_steps, env.n_agents, 2))
    gamma = 0.8
    for agent in range(env.n_agents):
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
    data["reward_history_mo"] = reward_history_mo
    data["disc_mo"] = disc_mo.sum(axis=0)
    data["history"] = env.history

    # Print rewards
    print(f"Epsiode {global_id}: {data['so']} \t Agents (V_0, V_e): ", "\t".join([str(s) for s in data["mo"]]))

    d[global_id] = data


if __name__ == "__main__":
    # Setting up the environment

    folder = "EGG_DATA"
    eff_rate = 0.4
    db = 10
    we = 10

    gamma = 0.8
    #SoftmaxActor.eval_action_selection = GreedyActionSelection(ACTIONS)
    large["we"] = [1, we]
    large["efficiency"] = [0.85] * int(5 * eff_rate) + [0.2] * int(5 - eff_rate * 5)
    large["donation_capacity"] = db
    large["color_by_efficiency"] = True
    #large["objective_order"] = "individual_first"
    env = gym.make("MultiAgentEthicalGathering-v1", **large)
    # env = NormalizeReward(env)
    env.toggleTrack(True)
    env.reset()

    # If root dir is not MAEGG_IPPO, up one level
    current_directory = os.getcwd()
    directory_name = os.path.basename(current_directory)
    while directory_name != "MAEGG_IPPO":
        os.chdir("..")
        current_directory = os.getcwd()
        directory_name = os.path.basename(current_directory)
    print(current_directory)
    # Loading the agents
    # agents = IPPO.actors_from_file(f"{folder}/db{db}_effrate{eff_rate}_we{we}_ECAI_special/db{db}_effrate{eff_rate}_we{we}_ECAI_special/5000_60000_18_ckpt")
    #agents = IPPO.actors_from_file(f"{folder}/db{db}_effrate{eff_rate}_we{we}_ECAI_new_arq/db{db}_effrate{eff_rate}_we{we}_ECAI_new_arq/5000_60000_16_ckpt")
    agents = IPPO.actors_from_file("EGG_DATA/db10_effrate0.4_we10_ECAI_new2/db10_effrate0.4_we10_ECAI_new2/2500_120000_15")
    # Running the simulation. Parallelized on batches of 5 simulations.
    n_sims = 200

    env.toggleTrack = True
    env.toggleStash = True
    stash = []
    so_rewards = np.zeros((n_sims, env.n_agents))
    mo_rewards = np.zeros((n_sims, env.n_agents, 2))
    reward_history_mo = np.zeros((n_sims, env.max_steps, env.n_agents, 2))
    disc_mo = np.zeros((n_sims, env.n_agents, 2))

    batch_size = min(25, n_sims)

    with Manager() as manager:

        solved = 0
        d = manager.dict()
        while solved < n_sims:
            d = manager.dict()
            tasks = [(env, d, agents, global_id, gamma) for global_id in range(solved, solved + batch_size)]
            with Pool(batch_size) as p:
                p.map(_parallel_rollout, tasks)

            for i in range(solved, solved + batch_size):
                stash.append(env.build_history_array(h=d[i]["history"]))
                # h = copy.deepcopy(d[i]["history"])
                # env.setHistory(h)
                so_rewards[i] = d[i]["so"]
                mo_rewards[i] = d[i]["mo"]
                disc_mo[i] = d[i]["disc_mo"]
                env.reset()
            solved += batch_size

    # Print mean rewards per agent and objective
    print("\nMean reward per agent: ", '\t'.join([str(s) for s in so_rewards.mean(axis=0)]))
    print(f"\nMean mo reward per agent: ", '\t'.join([str(s) for s in mo_rewards.mean(axis=0)]))
    print(f"\nMean disc mo reward per agent: ", '\t'.join([str(s) for s in disc_mo.mean(axis=0)]))

    # Plotting the results
    env.toggleTrack = True
    env.setStash(stash)
    env.print_results()
    env.plot_results("median")
