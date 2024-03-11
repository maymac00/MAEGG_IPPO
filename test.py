import numpy as np
from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import matplotlib

matplotlib.use('TkAgg')
import gym

medium["we"] = [1, 0.91]
env = gym.make("MultiAgentEthicalGathering-v1", **medium)
# env = NormalizeReward(env)

agents = IPPO.actors_from_file("EGG_DATA/ethical_medium_we10_try3/ethical_medium_we10_try3/2500_60000_1_(28)")
env.setTrack(True)
env.setStash(True)
env.reset()
history = []
mo_history = []
db_full = 0
for r in range(100):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        acc_reward = [acc_reward[i] + reward[i] for i in range(env.n_agents)]
        # env.render(mode="partial_observability")
    print(f"Epsiode {r}: {np.round(acc_reward, 2)} \t Agents (V_0, V_e): ", "\t".join([f"{np.round(env.agents[i].r_vec, 2)}" for i in range(env.n_agents)]))
    mo_history.append([env.agents[i].r_vec for i in range(env.n_agents)])
    history.append(acc_reward)
    if info["sim_data"]["donation_box_full"] != -1:
        db_full += 1
print("Donation box full: ", db_full)
mo_history = np.array(mo_history)
history = np.array(history)
# Print history mean
print(f"\nMean reward per agent: {list(history.mean(axis=0))}")
print(f"\nMean mo reward per agent: {list(mo_history.mean(axis=0))}")
env.plot_results("median")
env.print_results()
