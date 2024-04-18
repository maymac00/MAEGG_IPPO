import copy

from torch.multiprocessing import Manager, Pool
import torch as th
import numpy as np
from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import matplotlib

matplotlib.use('TkAgg')
import gym


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

    # Print rewards
    print(f"Epsiode {global_id}: {acc_reward} \t Agents (V_0, V_e): ", "\t".join([str(s) for s in data["mo"]]))

    d[global_id] = data


if __name__ == "__main__":
    # Setting up the environment

    eff_rate = 0.6
    db = 0
    we = 0

    large["we"] = [1, we]
    large["efficiency"] = [0.85] * int(5 * eff_rate) + [0.2] * int(5 - eff_rate * 5)
    large["donation_capacity"] = db
    env = gym.make("MultiAgentEthicalGathering-v1", **large)
    # env = NormalizeReward(env)
    env.toggleTrack(True)
    env.reset()

    # Loading the agents
    agents = IPPO.actors_from_file(f"../EGG_DATA/db{db}_effrate{eff_rate}_we{we}_ECAI/db{db}_effrate{eff_rate}_we{we}_ECAI/2500_100000_1")
    #agents = IPPO.actors_from_file("../EGG_DATA/db1_effrate0.6_we10_ECAI/db1_effrate0.6_we10_ECAI/2500_100000_1_(1)")

    # Running the simulation. Parallelized on batches of 5 simulations.
    n_sims = 100

    env.toggleTrack = True
    env.toggleStash = True
    stash = []
    so_rewards = np.zeros((n_sims, env.n_agents))
    mo_rewards = np.zeros((n_sims, env.n_agents, 2))

    batch_size = 25

    with Manager() as manager:

        solved = 0
        d = manager.dict()
        while solved < n_sims:
            d = manager.dict()
            tasks = [(env, d, agents, global_id) for global_id in range(solved, solved + batch_size)]
            with Pool(batch_size) as p:
                p.map(_parallel_rollout, tasks)

            for i in range(solved, solved + batch_size):
                stash.append(env.build_history_array(h=d[i]["history"]))
                #h = copy.deepcopy(d[i]["history"])
                #env.setHistory(h)
                so_rewards[i] = d[i]["so"]
                mo_rewards[i] = d[i]["mo"]
                env.reset()
            solved += batch_size

    # Print mean rewards per agent and objective
    print("\nMean reward per agent: ", '\t'.join([str(s) for s in so_rewards.mean(axis=0)]))
    print(f"\nMean mo reward per agent: ", '\t'.join([str(s) for s in mo_rewards.mean(axis=0)]))

    # Plotting the results
    env.toggleTrack = True
    env.setStash(stash)
    env.print_results()
    env.plot_results("mean")
