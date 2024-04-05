import numpy as np
from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO import IPPO
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import matplotlib

matplotlib.use('TkAgg')
import gym

eff_rate = 0.8
large["efficiency"] = [0.85]*int(5*eff_rate) + [0.2]*int(5 - eff_rate * 5)
large["we"] = [1, 0.91]
env = gym.make("MultiAgentEthicalGathering-v1", **large)
# env = NormalizeReward(env)

agents = IPPO.actors_from_file("../EGG_DATA/CIPPO_large_db10_try2_effrate0.4/CIPPO_large_db10_try2_effrate0.4/2500_60000_1_(1)")
env.setTrack(True)
env.setStash(True)
env.reset()
aux_cont = [0] * env.n_agents
aux_cost = [0] * env.n_agents
last_apple_value = [0] * env.n_agents
history = []
mo_history = []
for r in range(100):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]

        obs, reward, done, info = env.step(actions)
        acc_reward = [acc_reward[i] + reward[i] for i in range(env.n_agents)]

        apple_diff = [env.agents[i].apples - last_apple_value[i] for i in range(env.n_agents)]
        aux_cont = [aux_cont[i] + (1 if (env.agents[i].apples > env.survival_threshold and env.donation_box < env.donation_capacity and apple_diff[i] >=0) else 0) for i in range(env.n_agents)]
        aux_cost = [aux_cost[i] + info["R'_E"][i] for i in range(env.n_agents)]
        last_apple_value = [env.agents[i].apples for i in range(env.n_agents)]
        # env.render(mode="partial_observability")

    print(f"Epsiode {r}: {acc_reward} \t Agents (V_0, V_e): ", "\t".join([f"({env.agents[i].r_vec})" for i in range(env.n_agents)]))
    mo_history.append([env.agents[i].r_vec for i in range(env.n_agents)])
    history.append(acc_reward)

print(f"Percentage of non praiseworthy actions {(np.array(aux_cost) / np.array(aux_cont))* 100}%")
mo_history = np.array(mo_history)
history = np.array(history)
# Print history mean
print(f"\nMean reward per agent: {list(history.mean(axis=0))}")
print(f"\nMean mo reward per agent: {list(mo_history.mean(axis=0))}")
env.plot_results("median")
env.print_results()
